import numpy as np
import tensorflow as tf


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


# Test module: Run once you're ready to check your work
graph = tf.Graph()
with graph.as_default():
    inputs = tf.random_normal([10, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)
    output = resnet(inputs, keep_prob)
    writer = tf.summary.FileWriter('tbout/resnet', graph=graph)
    writer.close()


