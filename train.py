#coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import tensorflow as tf
import numpy as np
import os
import cv2
slim = tf.contrib.slim
import model.pspnet as pspnet
from model import resnet_utils
from model import resnet_v1_beta
import input_data
import utils.utils as Utils


BATCH_SIZE = 2
CROP_HEIGHT = input_data.CROP_HEIGHT
CROP_WIDTH = input_data.CROP_WIDTH
CLASSES = pspnet.CLASSES
CHANNELS = 3
MAX_STEPS = 80*6000
# 6000 steps for one epoch
KEEP_PROB = 1.0

initial_lr = 1e-3
weight_decay = 1e-5

saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'

def weighted_loss(logits, labels, num_classes, head=None, ignore=19):
    """re-weighting"""
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        #if head == None:
        #    cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), axis=[1])
        #else:
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def cal_loss(logits, labels):


    #CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


    loss_weight = [3.045384, 12.862123, 4.509889, 38.15694, 35.25279, 31.482613, \
                    45.792305, 39.694073, 6.0639296, 32.16484, 17.109228, 31.563286, \
                    47.333973, 11.610675, 44.60042, 45.23716, 45.283024, 48.14782, 41.924667]
    loss_weight = np.array(loss_weight)

    labels = tf.cast(labels, tf.int32)

    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=CLASSES, head=loss_weight)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CROP_HEIGHT, CROP_WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, CROP_HEIGHT, CROP_WIDTH], name='ground_truth')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

init_op, init_feed_dict, auxi_logits, logits = pspnet.PSPNet(x, is_training=True)

with tf.name_scope('regularization'):
    #regularizer = tf.contrib.layers.l2_regularizer(scale)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer)
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.trainable_variables()
        ])

with tf.name_scope('loss'):
    #reshaped_logits = tf.reshape(logits, [BATCH_SIZE, -1])
    #reshape_y = tf.reshape(y, [BATCH_SIZE, -1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshape_y, logits=reshaped_logits), name='loss')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    loss_1 = cal_loss(logits, y)
    tf.summary.scalar('loss', loss_1)
    loss_2 = cal_loss(auxi_logits, y)
    loss_all = loss_1 + 0.4 * loss_2 + l2_loss
    #loss_all = loss
    tf.summary.scalar('loss_all', loss_all)

with tf.name_scope('learning_rate'):
    lr = tf.Variable(initial_lr, dtype=tf.float32)
    tf.summary.scalar('learning_rate', lr)

optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all)

with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(logits, axis=-1, name='predictions')

    train_mIoU = tf.Variable(0, dtype=tf.float32)
    tf.summary.scalar('train_mIoU', train_mIoU)
    test_mIoU = tf.Variable(0, dtype=tf.float32)
    tf.summary.scalar('test_mIoU',test_mIoU)

merged = tf.summary.merge_all()

image_batch_0, image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type = 'train')
_, image_batch_test, anno_batch_test, filename_test = input_data.read_batch(BATCH_SIZE, type = 'val')

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(init_op, init_feed_dict)

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/denseASPP.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):

        b_image_0, b_image, b_anno, b_filename = sess.run([image_batch_0, image_batch, anno_batch, filename])

        b_image_test, b_anno_test, b_filename_test = sess.run([image_batch_test, anno_batch_test, filename_test])


        _ = sess.run(optimizer, feed_dict={x: b_image, y: b_anno, keep_prob: KEEP_PROB})

        train_summary = sess.run(merged, feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})
        test_summary_writer.add_summary(test_summary, i)

        pred_train, train_loss_val_all, train_loss_val = sess.run([predictions, loss_all, loss_1], feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})
        pred_test, test_loss_val_all, test_loss_val = sess.run([predictions, loss_all, loss_1], feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})



        learning_rate = sess.run(lr)

        if i % 10 == 0:
            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, test loss all: %f, test loss: %f," % (
                    i, learning_rate, train_loss_val_all, train_loss_val, test_loss_val_all, test_loss_val))

        if i % 200 == 0:

            train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, b_anno, CLASSES)
            test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, b_anno_test, CLASSES)

            sess.run(tf.assign(train_mIoU, train_mIoU_val))
            sess.run(tf.assign(test_mIoU, test_mIoU_val))

            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, train mIoU: %f, test loss all: %f, test loss: %f, test mIoU: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
            print(train_IoU_val)
            print(test_IoU_val)
            #prediction = tf.argmax(logits, axis=-1, name='predictions')

        if i % 1000 == 0:
            for j in range(BATCH_SIZE):
                cv2.imwrite('images/img_%s' % b_filename[j], b_image_0[j])

        if i % 5000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'pspnet.model'), global_step=i)


        if i == 10000 or i == 40000 or i == 100000:
            sess.run(tf.assign(lr, 0.1 * lr))

    coord.request_stop()
    coord.join(threads)




if __name__ == '__main__':

    with tf.Session() as sess:
        input = tf.constant(0.1, shape=[2, 768, 768, 3])
        sess.run(tf.global_variables_initializer())

        sess.run(init_op, init_feed_dict)
        for i in range(2):
            print(sess.run(auxi_logits)[0, 0, 0])
            print(sess.run(logits)[0, 0, 0])
