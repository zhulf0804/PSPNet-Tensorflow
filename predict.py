# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import model.pspnet as PSPNet
import input_data
import utils.utils as Utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# for dataset
flags.DEFINE_integer('height', 1024, 'The height of raw image.')
flags.DEFINE_integer('width', 2048, 'The width of raw image.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('classes', 19, 'The number of classes')
flags.DEFINE_integer('ignore_label', 255, 'The ignore label value.')


# for checkpoint
flags.DEFINE_string('pretrained_model_path', './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt', 'Path to save pretrained model.')
flags.DEFINE_string('saved_ckpt_path', './checkpoint/', 'Path to load training checkpoint.')

# for network configration
flags.DEFINE_integer('output_stride', 8, 'output stride in the resnet model.')

# for saved configration
flags.DEFINE_enum('prediction_on', 'val', ['train', 'val', 'test'], 'Which dataset to predict.')
flags.DEFINE_string('saved_prediction', './pred/', 'Path to save predictions.')

cmap = input_data.label_colours

def color_gray(image):
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] == FLAGS.ignore_label:
                return_img[i, j, :] = (0, 0, 0)
            else:
                return_img[i, j, :] = cmap[image[i, j]]

    return return_img

val_data = input_data.read_val_data()

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.height, FLAGS.width, 3], name='x_input')
    y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.height, FLAGS.width], name='ground_truth')

_, logits = PSPNet.PSPNet(x, is_training=False, output_stride=FLAGS.output_stride, pre_trained_model=FLAGS.pretrained_model_path, classes=FLAGS.classes)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #saver.restore(sess, './checkpoint/pspnet.model-2000')

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    print("predicting on %s set..." % FLAGS.prediction_on)


    for i in range(1):
        b_image_0, b_image, b_anno, b_filename = val_data.next_batch(FLAGS.batch_size, is_training=False)

        pred = sess.run(prediction, feed_dict={x: b_image, y: b_anno})

        print(pred.shape, b_anno.shape)

        mIoU_val, IoU_val = Utils.cal_batch_mIoU(pred, b_anno, FLAGS.classes)
        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)
        b_anno = b_anno.astype(np.uint8)
        pred_color = color_gray(pred[0, :, :])
        b_anno_color = color_gray(b_anno[0, :, :])

        b_image_0 = b_image_0.astype(np.uint8)

        img = Image.fromarray(b_image_0[0])
        anno = Image.fromarray(b_anno_color)
        pred = Image.fromarray(pred_color)

        basename = b_filename.split('.')[0]
        #print(basename)

        if not os.path.exists(FLAGS.saved_prediction):
            os.mkdir(FLAGS.saved_prediction)
        img.save(os.path.join(FLAGS.saved_prediction, basename + '.png'))
        anno.save(os.path.join(FLAGS.saved_prediction, basename + '_anno.png'))
        pred.save(os.path.join(FLAGS.saved_prediction, basename + '_pred.png'))

        print("%s.png: prediction saved in %s, mIoU value is %.2f" % (basename, FLAGS.saved_prediction, mIoU_val))
        print(IoU_val)