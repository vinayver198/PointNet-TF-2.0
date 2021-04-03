# -*- coding: utf-8 -*-
# @Author: vinay
# @Date:   2021-04-03 12:02:20
# @Last Modified by:   vinay
# @Last Modified time: 2021-04-03 12:02:23

import numpy as np
from model import get_model
import argparse
import tensorflow as tf
from utils import create_label_map
from visualization import visualize



parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, help='Filename on which inference is to be performed')
parser.add_argument('--weights_path', type=str, help='Weights file location')

FLAGS = parser.parse_args()

n_classes = 40


# Activation function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def get_bn_momentum(step):
    return min(0.99, 0.5 + 0.0002 * step)


# Generate the data
pc = np.load(FLAGS.filepath)
pc = np.expand_dims(pc, axis=0)
print(pc.shape)

# Get the model
bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)
model = get_model(40, bn_momentum=bn_momentum)
model.load_weights(FLAGS.weights_path)
#print(model.summary())

logits = model(pc, training=False)
logits = tf.squeeze(logits,axis=0)
probabilities = sigmoid(logits)
class2index, index2class = create_label_map('data/modelnet40_ply_hdf5_2048/shape_names.txt')


predicted_index = np.argmax(probabilities)
predicted_class = index2class[predicted_index]
predicted_prob = np.max(probabilities)

print('PC predicted as {} with probability {}'.format(predicted_class, predicted_prob))


visualize(pc[0])
