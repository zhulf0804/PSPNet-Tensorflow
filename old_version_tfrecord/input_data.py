# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import sys
from PIL import Image
from random import choice
import cv2
import random
from random import choice
import to_tfrecord as TFRecord

tfrecord_file = TFRecord.tfrecord_file
_NUM_SHARDS = TFRecord._NUM_SHARDS

def scale_image_anno(image, anno, scale):

    image_shape = tf.shape(image)
    new_dim = tf.cast(
        tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale,
        tf.int32)

    image = tf.squeeze(tf.image.resize_bilinear(
        tf.expand_dims(image, 0),
        new_dim,
        align_corners=True), [0])

    anno = tf.squeeze(tf.squeeze(tf.image.resize_nearest_neighbor(
        tf.expand_dims(tf.expand_dims(anno, 0), 3),
        new_dim,
        align_corners=True), 3), 0)

    return image, anno

def random_flip(processed_image, anno, prob=0.5):
    random_value = np.random.random()
    is_flipped = random_value <= prob
    if is_flipped:
        return tf.image.flip_left_right(processed_image), tf.squeeze(tf.image.flip_left_right(tf.expand_dims(anno, 2)), 2)
    else:
        return processed_image, anno

def padding(processed_image, anno, crop_height, crop_width, rgb_mean, num_channels=3):

    # padding
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    padding_img = [[0, target_height - image_height], [0, target_width - image_width], [0, 0]]
    padding_anno = [[0, target_height - image_height], [0, target_width - image_width]]

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=processed_image)

    #processed_image = tf.convert_to_tensor(np.zeros([target_height, target_width, 3], dtype=np.float32))
    for i in range(num_channels):
         if i == 0:
             processed_image = tf.pad(channels[i], padding_img, mode='CONSTANT', constant_values=rgb_mean[i])
         else:
             processed_image = tf.concat([processed_image, tf.pad(channels[i], padding_img, mode='CONSTANT', constant_values=rgb_mean[i])], -1)

    anno = tf.pad(anno, padding_anno, mode='CONSTANT', constant_values=255)

    return processed_image, anno

def random_crop(processed_image, anno, crop_height, crop_width, channels=3):

    shape = tf.shape(anno)
    image_height = shape[0]
    image_width = shape[1]

    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])

    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    processed_image = tf.slice(processed_image, [offset_height, offset_width, 0], [crop_height, crop_width, channels])
    anno = tf.slice(anno, [offset_height, offset_width], [crop_height, crop_width])

    return processed_image, anno

def mean_substraction(image, rgb_mean, num_channels = 3):
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= rgb_mean[i]
    return tf.concat(axis=2, values=channels)

def preprocess(image, anno, crop_height, crop_width, random_scales, scales, random_mirror, rgb_mean):
    original_image = image
    processed_image = tf.cast(image, tf.float32)

    if random_scales:
        scale = choice(scales)
        processed_image, anno = scale_image_anno(processed_image, anno, scale)


    processed_image, anno = padding(processed_image, anno, crop_height, crop_width, rgb_mean)

    processed_image, anno = random_crop(processed_image, anno, crop_height, crop_width)

    if random_mirror:
        processed_image, anno = random_flip(processed_image, anno)

    processed_image = mean_substraction(processed_image, rgb_mean=rgb_mean)

    processed_image = tf.reshape(processed_image, [crop_height, crop_width, 3])
    anno = tf.reshape(anno, [crop_height, crop_width])


    return original_image, processed_image, anno

def read_and_decode(filelist, HEIGHT, WIDTH):
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

def read_batch(batch_size, HEIGHT, WIDTH, crop_height, crop_width, random_scales, scales, random_mirror, rgb_mean, type='train'):
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
    image, anno, filename = read_and_decode(filelist, HEIGHT, WIDTH)

    original_image, image_aug, anno_aug = preprocess(image, anno, crop_height, crop_width, random_scales, scales, random_mirror, rgb_mean)

    image_0_batch, image_batch, anno_batch, filename = tf.train.shuffle_batch([original_image, image_aug, anno_aug, filename], batch_size=batch_size,
                                                               capacity=128, min_after_dequeue=64, num_threads=2)

    # print(image_batch, anno_batch)
    #image_0_batch, image_batch, anno_batch = augmentation_scale(image_0_batch, image_batch, anno_batch, mmin=0.5, mmax=2.0, type=type)
    return image_0_batch, image_batch, anno_batch, filename

if __name__ == '__main__':

    image_0, image_batch, anno_batch, filename = read_batch(4, 1024, 2048, 768, 768, True, [0.5,0.75,1.0,1.25,1.5,1.75,2.0], True, [123.15,115.90,103.06], type='train')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        b_image_0, b_image, b_anno, b_filename = sess.run([image_0, image_batch, anno_batch, filename])
        #print(b_filename)
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
        print(np.unique(b_image))




        for i in range(4):
            cv2.imwrite('test/%d_img.png'%i, b_image_0[i])
            cv2.imwrite('test/%d_img_2.png' % i, b_image[i])
            cv2.imwrite('test/%d_anno.png' %i, 10*b_anno[i])

        coord.request_stop()

        coord.join(threads)
