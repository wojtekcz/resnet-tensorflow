from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def avgpool(self,inputs, ksize, strides, padding='VALID', name=None):
        with tf.name_scope(name, 'avgpool'):
            ksize = [1] + ksize + [1]
            strides = [1] + strides + [1]
            return tf.nn.avg_pool(inputs, ksize, strides, padding)

    def maxpool(self,inputs, ksize, strides, padding='VALID', name=None):
        with tf.name_scope(name, 'maxpool'):
            ksize = [1] + ksize + [1]
            strides = [1] + strides + [1]
            return tf.nn.max_pool(inputs, ksize, strides, padding)

    def flatten(self,inputs, name=None):
        prev_shape = inputs.get_shape().as_list()
        fan_in = np.prod(prev_shape[1:])
        with tf.name_scope(name, 'flatten'):
            return tf.reshape(inputs, [-1, fan_in])

    def conv(self,inputs, depth, ksize, strides=[1, 1], padding='SAME',
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

    def fully_connected_layer(self,inputs, depth, bval=0.01, activation_fn=tf.nn.relu,
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

    def residual_block(self,inputs, bottleneck_depth, output_depth, downsample=False, scope=None):
        with tf.variable_scope(scope, 'residual_block'):
            if downsample:
                inputs = self.maxpool(inputs=inputs,
                                 ksize=[2, 2],
                                 strides=[2, 2],
                                 padding='VALID',
                                 name='downsample')
            prev_depth = inputs.get_shape()[3]
            relu1 = self.conv(inputs=inputs,
                         depth=bottleneck_depth,
                         ksize=[1, 1],
                         strides=[1, 1],
                         padding='SAME',
                         scope='conv1')
            relu2 = self.conv(inputs=relu1,
                         depth=bottleneck_depth,
                         ksize=[3, 3],
                         strides=[1, 1],
                         padding='SAME',
                         scope='conv2')
            conv3 = self.conv(inputs=relu2,
                         depth=output_depth,
                         ksize=[1, 1],
                         strides=[1, 1],
                         padding='SAME',
                         scope='conv3',
                         activation_fn=None)
            if inputs.get_shape() != conv3.get_shape():
                inputs = self.conv(inputs=inputs,
                              depth=output_depth,
                              ksize=[1, 1],
                              strides=[1, 1],
                              padding='SAME',
                              scope='shortcut',
                              activation_fn=None)
            add = tf.add(conv3, inputs)
            return tf.nn.relu(add)

    def resnet(self, inputs, keep_prob):
        input_depth = inputs.get_shape()[3]
        c = self.conv(inputs=inputs, depth=64, ksize=[7, 7], strides=[2, 2], padding='SAME', scope='conv1')
        p = self.maxpool(inputs=c, ksize=[3, 3], strides=[2, 2], padding='SAME', name='maxpool_3x3')
        block = p  # makes below loops more semantic
        with tf.variable_scope('stack_1'):
            for i in range(3):
                block = self.residual_block(inputs=block, bottleneck_depth=64, output_depth=256, scope='block_{}'.format(i))
        with tf.variable_scope('stack_2'):
            for i in range(8):
                ds = True if i == 0 else False  # down-sample first block only
                block = self.residual_block(inputs=block, bottleneck_depth=128, output_depth=512,
                                       scope='block_{}'.format(i), downsample=ds)
        with tf.variable_scope('stack_3'):
            for i in range(36):
                ds = True if i == 0 else False  # down-sample first block only
                block = self.residual_block(inputs=block, bottleneck_depth=256, output_depth=1024,
                                       scope='block_{}'.format(i), downsample=ds)
        with tf.variable_scope('stack_4'):
            for i in range(3):
                ds = True if i == 0 else False  # down-sample first block only
                block = self.residual_block(inputs=block, bottleneck_depth=512, output_depth=2048,
                                       scope='block_{}'.format(i), downsample=ds)
        p = self.avgpool(inputs=block, ksize=[7, 7], strides=[1, 1], padding='VALID', name='avgpool_7x7')
        flat = self.flatten(p)
        fc = self.fully_connected_layer(flat, 1000, activation_fn=None, scope='linear')
        softmax = tf.nn.softmax(fc)
        return softmax, fc

    def _build_model(self):
        """Build the core model within the graph."""

        # cut here
        x = self._images
        # x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        self.predictions, logits = self.resnet(inputs=x, keep_prob=0.5)

        # cut here
        # output? = self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.cast(tf.argmax(logits, 1), tf.int32), self.labels)
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')

            tf.summary.scalar('accuracy', self.acc)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)



    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

