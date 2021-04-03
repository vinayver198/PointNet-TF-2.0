# -*- coding: utf-8 -*-
# @Author: vinay
# @Date:   2021-04-03 12:03:25
# @Last Modified by:   vinay
# @Last Modified time: 2021-04-03 12:03:27

import numpy as np
import h5py
import tensorflow as tf


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f["data"][:]
    label = f["label"][:]
    return data, label


def read_filename(filename):
    with open(filename, 'rb') as file:
        lines = file.readlines()
    pc, labels = np.array([]), np.array([])
    for i, line in enumerate(lines):
        if i == 0:
            data, label = load_h5(line[:-1])
            pc = data
            labels = label
        else:
            data, label = load_h5(line[:-1])
            pc = np.vstack((pc, data))
            labels = np.vstack((labels, label))
    dataset_length = pc.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((pc, labels))
    return dataset,dataset_length


def create_label_map(filename):
    with open(filename,'r') as file:
        labels = file.readlines()
    class2index = {}
    index2class = {}
    for i,label in enumerate(labels):
        label = label[:-1]
        class2index[label] = i
        index2class[i] = label
    return class2index,index2class


if __name__ == "__main__":
    
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset, train_length= read_filename('data/modelnet40_ply_hdf5_2048/train_files.txt')
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    validation_dataset, validation_length= read_filename('data/modelnet40_ply_hdf5_2048/test_files.txt')
    validation_dataset = validation_dataset.batch(batch_size=BATCH_SIZE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    class2index, index2class = create_label_map('data/modelnet40_ply_hdf5_2048/shape_names.txt')
    print(train_length)
    print(validation_length)
    #print(class2index)
    #print(index2class)