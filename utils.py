# -*- coding:utf-8 -*-
import scipy.misc
import numpy as np
import os
from glob import glob
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from keras.datasets import cifar10, mnist
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tensorflow.contrib.layers import batch_norm
from tflearn.layers.conv import global_avg_pool

import utilsForTF


def get_image_label_batch(config, shuffle, name):
    with tf.name_scope('get_batch'):
        Data = utilsForTF.Data_set(config, shuffle=shuffle, name=name)
        image_batch, label_batch = Data.read_processing_generate_image_label_batch()
    return image_batch, label_batch

def count_trainable_params():
    total_parameters = 0
    a = []
    for variable in tf.trainable_variables():
        a.append(variable)
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))
    return total_parameters

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    images = np.nan_to_num(images)
    h, w = images.shape[1], images.shape[2]
    h, w = 128, 128
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = cv2.resize(image,(h,w))
            #img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = cv2.resize(image,(h,w))
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')
