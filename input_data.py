# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import sys
from PIL import Image
import cv2
import random
from random import choice
import to_tfrecord as TFRecord

tfrecord_file = TFRecord.tfrecord_file
_NUM_SHARDS = TFRecord._NUM_SHARDS
HEIGHT = 1024
WIDTH = 2048
CROP_HEIGHT = 768
CROP_WIDTH = 768

RANDOM_SCALE = [0.75, 1.0, 1.25, 1.5, 1.8]


def flip_randomly_left_right_image_with_annotation(image_0_tensor, image_tensor, annotation_tensor):
    # Reference https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/augmentation.py
    # Random variable: two possible outcomes (0 or 1)
    # with a 1 in 2 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img_0 = tf.cond(pred=tf.equal(random_var, 0),
                                   fn1=lambda: tf.image.flip_left_right(image_0_tensor),
                                   fn2=lambda: image_0_tensor)

    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                                 fn1=lambda: tf.image.flip_left_right(image_tensor),
                                                 fn2=lambda: image_tensor)

    randomly_flipped_annotation = tf.cond(pred=tf.equal(random_var, 0),
                                                        fn1=lambda: tf.image.flip_left_right(tf.expand_dims(annotation_tensor, -1)),
                                                        fn2=lambda: tf.expand_dims(annotation_tensor, -1))
    randomly_flipped_annotation = tf.squeeze(randomly_flipped_annotation, -1)
    return randomly_flipped_img_0, randomly_flipped_img, randomly_flipped_annotation

def random_resize(image_0, image, anno, scales):

    rand_var =choice(scales)
    scaled_shape = [tf.cast(tf.round(rand_var * HEIGHT), tf.int32), tf.cast(tf.round(rand_var * WIDTH), tf.int32)]

    batch_image_0 = tf.expand_dims(image_0, 0)
    batch_image = tf.expand_dims(image, 0)
    batch_anno = tf.expand_dims(anno, 0)
    batch_anno = tf.expand_dims(batch_anno, -1)

    batch_image_0 = tf.image.resize_bilinear(batch_image_0, scaled_shape)
    batch_image_0 = tf.cast(batch_image_0, tf.uint8)

    batch_image = tf.image.resize_bilinear(batch_image, scaled_shape)


    batch_anno = tf.image.resize_nearest_neighbor(batch_anno, scaled_shape)

    image_0 = tf.squeeze(batch_image_0, 0)
    image = tf.squeeze(batch_image, 0)
    anno = tf.squeeze(batch_anno, -1)
    anno = tf.squeeze(anno, 0)

    return image_0, image, anno

def random_crop(image_0, image, anno):

    '''
    seed = random.randint(0, 1e10)
    input_shape = batch_image.get_shape().as_list()
    batch_image_0 = tf.random_crop(batch_image_0, [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3], seed=seed)
    batch_image = tf.random_crop(batch_image, [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3], seed=seed)
    batch_anno = tf.random_crop(batch_anno, [input_shape[0], CROP_HEIGHT, CROP_WIDTH], seed=seed)
    return batch_image_0, batch_image, batch_anno
    '''

    input_shape = image.get_shape().as_list()
    max_h = input_shape[0] - CROP_HEIGHT
    max_w = input_shape[1] - CROP_WIDTH
    x_st = np.random.randint(low=0, high=max_h + 1)
    y_st = np.random.randint(low=0, high=max_w + 1)


    crop_image_0 = tf.slice(image_0, [x_st, y_st, 0], [CROP_HEIGHT, CROP_WIDTH, 3])
    crop_image = tf.slice(image, [x_st, y_st, 0], [CROP_HEIGHT, CROP_WIDTH, 3])
    crop_anno = tf.slice(anno, [x_st, y_st], [CROP_HEIGHT, CROP_WIDTH])

    return crop_image_0, crop_image, crop_anno

def augmentation_standardization(image_0, image, anno, type):

    image = tf.cast(image, tf.float32)
    #print(id(image_0))
    if type == 'train' or type == 'trainval':
        image_0, image, anno = random_resize(image_0, image, anno, scales=RANDOM_SCALE)
    #print(id(image_0))
    image_0, image, anno = random_crop(image_0, image, anno)

    if type == 'train' or type == 'trainval':
        image_0, image, anno = flip_randomly_left_right_image_with_annotation(image_0, image, anno)
        image = tf.image.random_brightness(image, max_delta=10)

    image = tf.image.per_image_standardization(image)
    #image /= 255
    #image -= 0.5
    image_0 = tf.reshape(image_0, [CROP_HEIGHT, CROP_WIDTH, 3])
    image = tf.reshape(image, [CROP_HEIGHT, CROP_WIDTH, 3])
    anno = tf.reshape(anno, [CROP_HEIGHT, CROP_WIDTH])

    return image_0, image, anno

def read_and_decode(filelist):
    filename_queue = tf.train.string_input_producer(filelist)
    reader = tf.TFRecordReader()
    _, serialized_exampe = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_exampe,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/anno': tf.FixedLenFeature([], tf.string),
                                           'image/filename': tf.FixedLenFeature([], tf.string),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    anno = tf.decode_raw(features['image/anno'], tf.uint8)
    filename = features['image/filename']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    anno = tf.reshape(anno, [HEIGHT, WIDTH])

    return image, anno, filename

def read_batch(batch_size, type='train'):
    filelist_train = [ os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS - 1)) for
        shard_id in range(_NUM_SHARDS)]
    filelist_val = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('val', shard_id, _NUM_SHARDS - 1))
                    for shard_id in range(_NUM_SHARDS)]
    filelist_test = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('test', shard_id, _NUM_SHARDS - 1))
                     for shard_id in range(_NUM_SHARDS)]

    filelist = []
    if type == 'train':
        filelist = filelist + filelist_train
    elif type == 'val':
        filelist = filelist + filelist_val
    elif type == 'test':
        filelist = filelist + filelist_test
    elif type == 'trainval':
        filelist = filelist + filelist_train + filelist_val
    else:
        raise Exception('data set name not exits')

    print(filelist)
    image, anno, filename = read_and_decode(filelist)
    image_0 = image

    ## data augmentation and standardation

    image_0_aug, image_aug, anno_aug = augmentation_standardization(image_0, image, anno, type)

    image_0_batch, image_batch, anno_batch, filename = tf.train.shuffle_batch([image_0_aug, image_aug, anno_aug, filename], batch_size=batch_size,
                                                               capacity=128, min_after_dequeue=64, num_threads=2)

    # print(image_batch, anno_batch)
    #image_0_batch, image_batch, anno_batch = augmentation_scale(image_0_batch, image_batch, anno_batch, mmin=0.5, mmax=2.0, type=type)
    return image_0_batch, image_batch, anno_batch, filename

if __name__ == '__main__':
    BATCH_SIZE = 4
    image_0, image_batch, anno_batch, filename = read_batch(BATCH_SIZE, type='train')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        b_image_0, b_image, b_anno, b_filename = sess.run([image_0, image_batch, anno_batch, filename])
        print(b_filename)
        '''
        print(b_image_0.shape)
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        print(b_image_0)
        print(b_image)
        print(b_anno)
        print(np.unique(b_anno))
        '''
        print(np.unique(b_anno))
        for i in range(BATCH_SIZE):
            cv2.imwrite('test/%d_img.png'%i, b_image_0[i])
            cv2.imwrite('test/%d_img_2.png' % i, 255 * (0.5 + b_image[i]))
            cv2.imwrite('test/%d_anno.png' % i, 10*b_anno[i])

        coord.request_stop()

        coord.join(threads)
