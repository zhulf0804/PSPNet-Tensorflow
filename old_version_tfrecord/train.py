#coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
slim = tf.contrib.slim
import model.pspnet as pspnet

import input_data
import utils.utils as Utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# for dataset
flags.DEFINE_integer('height', 1024, 'The height of raw image.')
flags.DEFINE_integer('width', 2048, 'The width of raw image.')
flags.DEFINE_integer('crop_height', 768, 'The height of cropped image used for training.')
flags.DEFINE_integer('crop_width', 768, 'The width of cropped image used for training.')
flags.DEFINE_integer('channels', 3, 'The channels of input image.')
flags.DEFINE_integer('ignore_label', 255, 'The ignore label value.')
flags.DEFINE_integer('classes', 19, 'The ignore label value.')
#flags.DEFINE_multi_float('rgb_mean', [123.15,115.90,103.06], 'RGB mean value of ImageNet.')
flags.DEFINE_multi_float('rgb_mean', [72.39239876,82.90891754,73.15835921], 'RGB mean value of ImageNet.')

# for augmentation
flags.DEFINE_boolean('train_random_scales', True, 'whether to random scale.')
flags.DEFINE_multi_float('scales', [0.5,0.75,1.0,1.25,1.5,1.75,2.0], 'Scales for random scale.')
flags.DEFINE_boolean('train_random_mirror', True, 'whether to random mirror.')

flags.DEFINE_boolean('val_random_scales', False, 'whether to random scale.')
flags.DEFINE_boolean('val_random_mirror', False, 'whether to random mirror.')

# for training configuration
flags.DEFINE_integer('batch_size', 4, 'The number of images in each batch during training.')
flags.DEFINE_integer('max_epoches', 40, 'The max epoches to train the model.')
flags.DEFINE_integer('samples', 2975, 'The number of images used to train.')

MAX_STEPS = FLAGS.max_epoches * FLAGS.samples // FLAGS.batch_size

# for network configration
flags.DEFINE_integer('output_stride', 16, 'output stride in the resnet model.')


# network hyper-parameters
flags.DEFINE_float('initial_lr', 1e-2, 'The initial learning rate.')
flags.DEFINE_float('end_lr', 1e-6, 'The end learning rate.')
flags.DEFINE_integer('decay_steps', 50000, 'Used for poly learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'The weight decay value for l2 regularization.')
flags.DEFINE_float('power', 0.9, 'Used for poly learning rate.')

# for saved configration
flags.DEFINE_string('saved_ckpt_path', './checkpoint/', 'Path to save training checkpoint.')
flags.DEFINE_string('saved_summary_train_path', './summary/train/', 'Path to save training summary.')
flags.DEFINE_string('saved_summary_test_path', './summary/test/', 'Path to save test summary.')
flags.DEFINE_string('pretrained_model_path', './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt', 'Path to save test summary.')

'''

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
'''

def cal_loss(logits, y, loss_weight=1.0):
    '''
    raw_prediction = tf.reshape(logits, [-1, CLASSES])
    raw_gt = tf.reshape(y, [-1])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    '''

    y = tf.reshape(y, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(y,
                                               FLAGS.ignore_label)) * loss_weight
    one_hot_labels = tf.one_hot(
        y, FLAGS.classes, on_value=1.0, off_value=0.0)
    logits = tf.reshape(logits, shape=[-1, FLAGS.classes])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, weights=not_ignore_mask)

    return tf.reduce_mean(loss)


with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.crop_height, FLAGS.crop_width, FLAGS.channels], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.crop_height, FLAGS.crop_width], name='ground_truth')

auxi_logits, logits = pspnet.PSPNet(x, is_training=True, output_stride=FLAGS.output_stride, pre_trained_model=FLAGS.pretrained_model_path, classes=FLAGS.classes)


with tf.name_scope('regularization'):
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]
    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        l2_loss = FLAGS.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])

with tf.name_scope('loss'):
    #reshaped_logits = tf.reshape(logits, [BATCH_SIZE, -1])
    #reshape_y = tf.reshape(y, [BATCH_SIZE, -1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshape_y, logits=reshaped_logits), name='loss')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    loss_1 = cal_loss(logits, y)
    tf.summary.scalar('loss', loss_1)
    loss_2 = cal_loss(auxi_logits, y)
    loss_all = loss_1 + l2_loss
    #loss_all = loss
    tf.summary.scalar('loss_all', loss_all)

with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(
        learning_rate=FLAGS.initial_lr,
        global_step=global_step,
        decay_steps=FLAGS.decay_steps,
        end_learning_rate=FLAGS.end_lr,
        power=FLAGS.power,
        cycle=False,
        name=None
    )
    tf.summary.scalar('learning_rate', lr)

with tf.name_scope("opt"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all, var_list=train_var_list, global_step=global_step)


with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(softmax, axis=-1, name='predictions')

    train_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('train_mIoU', train_mIoU)
    test_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('test_mIoU',test_mIoU)

merged = tf.summary.merge_all()

image_batch_0, image_batch, anno_batch, filename = input_data.read_batch(FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.crop_height, FLAGS.crop_width, FLAGS.train_random_scales, FLAGS.scales, FLAGS.train_random_mirror, FLAGS.rgb_mean, type='train')


_, image_batch_test, anno_batch_test, filename_test = input_data.read_batch(FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.crop_height, FLAGS.crop_width, FLAGS.val_random_scales, FLAGS.scales, FLAGS.val_random_mirror, FLAGS.rgb_mean, type='val')

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/PSPNet.model-30000')

    train_summary_writer = tf.summary.FileWriter(FLAGS.saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(FLAGS.saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):

        b_image_0, b_image, b_anno, b_filename = sess.run([image_batch_0, image_batch, anno_batch, filename])

        b_image_test, b_anno_test, b_filename_test = sess.run([image_batch_test, anno_batch_test, filename_test])


        _ = sess.run(optimizer, feed_dict={x: b_image, y: b_anno})

        train_summary = sess.run(merged, feed_dict={x: b_image, y: b_anno})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: b_image_test, y: b_anno_test})
        test_summary_writer.add_summary(test_summary, i)

        pred_train, train_loss_val_all, train_loss_val = sess.run([predictions, loss_all, loss_1], feed_dict={x: b_image, y: b_anno})
        pred_test, test_loss_val_all, test_loss_val = sess.run([predictions, loss_all, loss_1], feed_dict={x: b_image_test, y: b_anno_test})



        learning_rate = sess.run(lr)

        if i % 200 == 0:
            print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d | Train loss all: %f" % (i, train_loss_val_all))

        if i % 1000 == 0:

            train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, b_anno, FLAGS.classes)
            test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, b_anno_test, FLAGS.classes)

            sess.run(tf.assign(train_mIoU, train_mIoU_val))
            sess.run(tf.assign(test_mIoU, test_mIoU_val))

            print('------------------------------')

            print(
                "Step: %d | Lr: %f | Train loss all: %f | Train loss: %f | Train mIoU: %f | Test loss all: %f | Test loss: %f | Test mIoU: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
            print('------------------------------')
            print(train_IoU_val)
            print(test_IoU_val)
            print('------------------------------')
            #prediction = tf.argmax(logits, axis=-1, name='predictions')

        if i % 1000 == 0:
            for j in range(FLAGS.batch_size):
                cv2.imwrite('images/img_%s' % b_filename[j], b_image_0[j])

        if i % 5000 == 0:
            saver.save(sess, os.path.join(FLAGS.saved_ckpt_path, 'pspnet.model'), global_step=i)


    coord.request_stop()
    coord.join(threads)




if __name__ == '__main__':

    with tf.Session() as sess:
        input = tf.constant(0.1, shape=[2, 768, 768, 3])
        sess.run(tf.global_variables_initializer())

        for i in range(2):
            print(sess.run(auxi_logits)[0, 0, 0])
            print(sess.run(logits)[0, 0, 0])
