#coding=utf-8

from __future__ import print_function
from __future__ import division


import tensorflow as tf
import numpy as np
import math
slim = tf.contrib.slim
from model import resnet_v1_beta
from model import resnet_utils

#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# for dataset
CLASSES = 19
HEIGHT = 768
WIDTH = 768

#for network
N = 4
tf_initial_checkpoint = '/home/zhulifa/code/PSPNet/resnet_v1_101/model.ckpt'


def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)

    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(inputs, training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def resnet_101(input, is_training):
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        input, endpoints = resnet_v1_beta.resnet_v1_101_beta(input, is_training=is_training)

    return input, endpoints
def pyramid_pooling(input, is_training):
    with tf.name_scope("pyramid_pooling"):
        input_shape = input.get_shape().as_list()

        num_output_features = input_shape[-1] // N
        with tf.name_scope("pool_1"):
            pool_bin_1 = tf.nn.avg_pool(input, [1, input_shape[1], input_shape[2], 1],
                                        [1, input_shape[1], input_shape[2], 1], padding='VALID')
            weight_1 = weight_variable([1, 1, input_shape[-1], num_output_features])

            output_1 = tf.nn.conv2d(pool_bin_1, weight_1, [1, 1, 1, 1], padding='SAME')
            output_1 = batch_norm(output_1, is_training)
            output_1 = tf.nn.relu(output_1)

            output_1 = tf.image.resize_bilinear(output_1, [input_shape[1], input_shape[2]])

        with tf.name_scope("pool_2"):
            pool_bin_2 = tf.nn.avg_pool(input, [1, input_shape[1] // 2, input_shape[2] // 2, 1],
                                        [1, input_shape[1] // 2, input_shape[2] // 2, 1], padding='VALID')

            weight_2 = weight_variable([1, 1, input_shape[-1], num_output_features])

            output_2 = tf.nn.conv2d(pool_bin_2, weight_2, [1, 1, 1, 1], padding='SAME')
            output_2 = batch_norm(output_2, is_training)
            output_2 = tf.nn.relu(output_2)

            output_2 = tf.image.resize_bilinear(output_2, [input_shape[1], input_shape[2]])

        with tf.name_scope("pool_3"):
            pool_bin_3 = tf.nn.avg_pool(input, [1, input_shape[1] // 3, input_shape[2] // 3, 1],
                                        [1, input_shape[1] // 3, input_shape[2] // 3, 1], padding='VALID')

            weight_3 = weight_variable([1, 1, input_shape[-1], num_output_features])

            output_3 = tf.nn.conv2d(pool_bin_3, weight_3, [1, 1, 1, 1], padding='SAME')
            output_3 = batch_norm(output_3, is_training)
            output_3 = tf.nn.relu(output_3)

            output_3 = tf.image.resize_bilinear(output_3, [input_shape[1], input_shape[2]])


        with tf.name_scope("pool_6"):
            pool_bin_6 = tf.nn.avg_pool(input, [1, input_shape[1] // 6, input_shape[2] // 6, 1],
                                        [1, input_shape[1] // 6, input_shape[2] // 6, 1], padding='VALID')

            weight_6 = weight_variable([1, 1, input_shape[-1], num_output_features])

            output_6 = tf.nn.conv2d(pool_bin_6, weight_6, [1, 1, 1, 1], padding='SAME')
            output_6 = batch_norm(output_6, is_training)
            output_6 = tf.nn.relu(output_6)

            output_6 = tf.image.resize_bilinear(output_6, [input_shape[1], input_shape[2]])


        input = tf.concat([input, output_1], axis=-1)
        input = tf.concat([input, output_2], axis=-1)
        input = tf.concat([input, output_3], axis=-1)
        input = tf.concat([input, output_6], axis=-1)

    return input

def PSPNet(input, is_training):

    input, endpoints = resnet_101(input, is_training=is_training)


    with tf.name_scope("auli_logits"):

        key = 'resnet_v1_101/block3'
        auxi_logits = endpoints[key]

        auli_shape = auxi_logits.get_shape().as_list()
        weight_3 = weight_variable([3, 3, auli_shape[-1], auli_shape[-1] // 4])
        auxi_logits = tf.nn.conv2d(auxi_logits, weight_3, [1, 1, 1, 1], padding='SAME')
        auxi_logits = batch_norm(auxi_logits, is_training)
        auxi_logits = tf.nn.relu(auxi_logits)

        weight_1 = weight_variable([1, 1, auli_shape[-1] // 4, CLASSES])
        bias = bias_variable([CLASSES])
        auxi_logits = tf.nn.conv2d(auxi_logits, weight_1, [1, 1, 1, 1], padding='SAME') + bias

        auxi_logits = tf.image.resize_bilinear(auxi_logits, [HEIGHT, WIDTH])


    #print(len(variables_to_restore))

    input = pyramid_pooling(input, is_training)

    with tf.name_scope("segmentation"):
        input_shape = input.get_shape().as_list()
        weight_3 = weight_variable([3, 3, input_shape[-1], input_shape[-1] // 4])
        input = tf.nn.conv2d(input, weight_3, [1, 1, 1, 1], padding='SAME')
        input = batch_norm(input, is_training)
        input = tf.nn.relu(input)

        weight_1 = weight_variable([1, 1, input_shape[-1] // 4, CLASSES])
        bias = bias_variable([CLASSES])
        input = tf.nn.conv2d(input, weight_1, [1, 1, 1, 1], padding='SAME') + bias

        logits = tf.image.resize_bilinear(input, [HEIGHT, WIDTH])
    
    if is_training:

        exclude_list = ['global_step']
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)

        init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
            tf_initial_checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)


        return init_op, init_feed_dict, auxi_logits, logits
    else:
        return logits


if __name__ == '__main__':
    input = tf.constant(0.1, shape=[2, 768, 768, 3])
    init_op, init_feed_dict, auxi_logits, logits = PSPNet(input, True)

    print(auxi_logits)
    print(logits)
    #auxi_logits, endpoints = resnet_101(input, is_training=True)
    #print(endpoints)


    '''
    a = 'resnet_v1_101/block3'
    b = 'resnet_v1_101/block3/unit_23/bottleneck_v1'
    for key in endpoints:
        if key == a:
            print(key)
            print(endpoints[a])
        if key == b:
            print(key)
            print(endpoints[b])
            
    '''



