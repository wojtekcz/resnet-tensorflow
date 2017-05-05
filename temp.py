import numpy as np
import tensorflow as tf

def avgpool(inputs, ksize, strides, padding='VALID', name=None):
    with tf.name_scope(name, 'avgpool'):
        ksize = [1] + ksize + [1]
        strides = [1] + strides + [1]
        return tf.nn.avg_pool(inputs, ksize, strides, padding)


def maxpool(inputs, ksize, strides, padding='VALID', name=None):
    with tf.name_scope(name, 'maxpool'):
        ksize = [1] + ksize + [1]
        strides = [1] + strides + [1]
        return tf.nn.max_pool(inputs, ksize, strides, padding)

def flatten(inputs, name=None):
    prev_shape = inputs.get_shape().as_list()
    fan_in = np.prod(prev_shape[1:])
    with tf.name_scope(name, 'flatten'):
        return tf.reshape(inputs, [-1, fan_in])

def conv(inputs, depth, ksize, strides=[1, 1], padding='SAME',
         bval=0.01, activation_fn=tf.nn.relu, scope=None):
    prev_shape = inputs.get_shape().as_list()
    prev_depth = prev_shape[-1]
    kshape = ksize + [prev_depth, depth]
    strides = [1] + strides + [1]
    fan_in = np.prod(prev_shape[1:], dtype=np.float32)
    with tf.variable_scope(scope, 'conv_layer'):
        xavier_stddev = tf.sqrt(tf.constant(2.0, dtype=tf.float32) / fan_in, name='xavier_stddev')
        w = tf.Variable(tf.truncated_normal(kshape, stddev=xavier_stddev), name='kernel')
        conv = tf.nn.conv2d(inputs, w, strides, padding, name='conv')
        if bval:
            b = tf.Variable(tf.constant(bval, shape=[depth]), name='bias')
            z = tf.nn.bias_add(conv, b)
        return z if activation_fn is None else activation_fn(z)

def fully_connected_layer(inputs, depth, bval=0.01, activation_fn=tf.nn.relu,
                          keep_prob=None, scope=None):
    inputs = tf.convert_to_tensor(inputs)
    prev_shape = inputs.get_shape().as_list()
    fan_in = prev_shape[-1]
    with tf.variable_scope(scope, 'fully_connected'):
        xavier_stddev = tf.sqrt(tf.constant(2.0, dtype=tf.float32) / fan_in, name='xavier_stddev')
        w = tf.Variable(tf.truncated_normal([fan_in, depth], stddev=xavier_stddev), name='W')
        b = tf.Variable(tf.constant(bval, shape=[depth]), name='bias')
        z = tf.matmul(inputs, w) + b
        a = z if activation_fn is None else activation_fn(z)
        return a if keep_prob is None else tf.nn.dropout(a, keep_prob)

def residual_block(inputs, bottleneck_depth, output_depth, downsample=False, scope=None):
    with tf.variable_scope(scope, 'residual_block'):
        if downsample:
            inputs = maxpool(inputs=inputs,
                           ksize=[2, 2],
                           strides=[2, 2],
                           padding='VALID',
                           name='downsample')
        prev_depth = inputs.get_shape()[3]
        relu1 = conv(inputs=inputs,
                     depth=bottleneck_depth,
                     ksize=[1, 1],
                     strides=[1, 1],
                     padding='SAME',
                     scope='conv1')
        relu2 = conv(inputs=relu1,
                     depth=bottleneck_depth,
                     ksize=[3, 3],
                     strides=[1, 1],
                     padding='SAME',
                     scope='conv2')
        conv3 = conv(inputs=relu2,
                     depth=output_depth,
                     ksize=[1, 1],
                     strides=[1, 1],
                     padding='SAME',
                     scope='conv3',
                     activation_fn=None)
        if inputs.get_shape() != conv3.get_shape():
            inputs = conv(inputs=inputs,
                        depth=output_depth,
                        ksize=[1, 1],
                        strides=[1, 1],
                        padding='SAME',
                        scope='shortcut',
                        activation_fn=None)
        add = tf.add(conv3, inputs)
        return tf.nn.relu(add)

# # Test
# graph = tf.Graph()
# with graph.as_default():
#     inputs = tf.placeholder(tf.float32, shape=[10, 224, 224, 3], name='inputs')
#     block = inputs
#     for i in range(3):
#         ds = True if i == 0 else False  # down-sample first block only
#         block = residual_block(inputs=block,
#                                bottleneck_depth=128,
#                                output_depth=512,
#                                scope='block_{}'.format(i),
#                                downsample=ds)
#     writer = tf.summary.FileWriter('tbout/residual_stack', graph=graph)

def resnet(inputs, keep_prob):
    input_depth = inputs.get_shape()[3]
    c = conv(inputs=inputs, depth=64, ksize=[7, 7], strides=[2, 2], padding='SAME', scope='conv1')
    p = maxpool(inputs=c, ksize=[3, 3], strides=[2, 2], padding='SAME', name='maxpool_3x3')
    block = p  # makes below loops more semantic
    with tf.variable_scope('stack_1'):
        for i in range(3):
            block = residual_block(inputs=block, bottleneck_depth=64, output_depth=256, scope='block_{}'.format(i))
    with tf.variable_scope('stack_2'):
        for i in range(8):
            ds = True if i == 0 else False  # down-sample first block only
            block = residual_block(inputs=block, bottleneck_depth=128, output_depth=512,
                                   scope='block_{}'.format(i), downsample=ds)
    with tf.variable_scope('stack_3'):
        for i in range(36):
            ds = True if i == 0 else False  # down-sample first block only
            block = residual_block(inputs=block, bottleneck_depth=256, output_depth=1024,
                                          scope='block_{}'.format(i), downsample=ds)
    with tf.variable_scope('stack_4'):
        for i in range(3):
            ds = True if i == 0 else False  # down-sample first block only
            block = residual_block(inputs=block, bottleneck_depth=512, output_depth=2048,
                                          scope='block_{}'.format(i), downsample=ds)
    p = avgpool(inputs=block, ksize=[7, 7], strides=[1, 1], padding='VALID', name='avgpool_7x7')
    flat = flatten(p)
    fc = fully_connected_layer(flat, 1000, activation_fn=None, scope='linear')
    softmax = tf.nn.softmax(fc)
    return softmax

# Test module: Run once you're ready to check your work
graph = tf.Graph()
with graph.as_default():
    inputs = tf.random_normal([10, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)
    output = resnet(inputs, keep_prob)
    writer = tf.summary.FileWriter('tbout/resnet', graph=graph)
    writer.close()


