# -*- coding: utf-8 -*-
# @Author: vinay
# @Date:   2021-04-03 11:58:38
# @Last Modified by:   vinay
# @Last Modified time: 2021-04-03 11:58:52


import tensorflow as tf
from tensorflow import keras
import numpy as np


class CustomConvolution(keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID', bn_momentum=0.99,
                 activation=None, apply_bn=False):
        super(CustomConvolution, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.convolution = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.bn = keras.layers.BatchNormalization(momentum=bn_momentum)

    def call(self, inputs, training=None):
        x = self.convolution(inputs)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConvolution, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'bn_momentum': self.bn_momentum,
                'activation': self.activation,
                'apply_bn': self.apply_bn

            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(keras.layers.Layer):
    def __init__(self, units=256, activation=tf.nn.relu, bn_momentum=0.99, apply_bn=False):
        super(CustomDense, self).__init__()
        self.filters = units
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.dense = keras.layers.Dense(units)
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.bn = keras.layers.BatchNormalization(momentum=bn_momentum)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update(
            {
                'units': self.units,
                'activation': self.activation,
                'bn_momentum': self.bn_momentum,
                'apply_bn': self.apply_bn

            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformNet(keras.layers.Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, name=None, **kwargs):
        super(TransformNet, self).__init__(name=name, **kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConvolution(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)
        self.conv1 = CustomConvolution(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=self.bn_momentum,apply_bn=True)
        self.conv2 = CustomConvolution(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=self.bn_momentum,apply_bn=True)
        self.dense0 = CustomDense(units=512, activation=tf.nn.relu, bn_momentum=self.bn_momentum,apply_bn=True)
        self.dense1 = CustomDense(units=256, activation=tf.nn.relu, bn_momentum=self.bn_momentum,apply_bn=True)

    def build(self, input_shape):
        self.K = input_shape[2]
        self.w = self.add_weight(name='w', shape=[256, self.K ** 2], initializer=tf.zeros_initializer, dtype=tf.float32)
        self.b = self.add_weight(name='b', shape=(self.K, self.K), initializer=tf.zeros_initializer, dtype=tf.float32)
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, inputs, training=None):
        x = inputs  # BxNx3
        x = tf.expand_dims(x, axis=2)  # BxNx1x3
        x = self.conv0(x, training=training)  # BxNx1x64
        x = self.conv1(x, training=training)  # BxNx1x128
        x = self.conv2(x, training=training)  # BxNx1x1024
        x = tf.squeeze(x, axis=2)  # BxNx1024
        x = tf.reduce_max(x, axis=1)  # Bx1024sss
        x = self.dense0(x, training=training)  # Bx512
        x = self.dense1(x, training=training)  # Bx256

        x = tf.expand_dims(x, axis=1)  # Bx1x256
        x = tf.matmul(x, self.w)  # Bx1xK**2
        x = tf.squeeze(x, axis=1)  # BxK**2
        x = tf.reshape(x, (-1, self.K, self.K))  # BxKxK
        x += self.b

        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_transpose = tf.transpose(x, perm=[0, 2, 1])
            x_xt = tf.matmul(x, x_transpose)
            reg = tf.nn.l2_loss(eye - x_xt)
            self.add_loss(1e-3 * reg)

        return tf.matmul(inputs, x)

    def get_config(self):
        config = super(TransformNet, self).get_config()
        config.update(
            {
                'add_regularization': self.add_regularization,
                'bn_momentum': self.bn_momentum,
            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_model(n_classes, bn_momentum):
    add_regularization = True
    # Initialize the input layer
    input_layer = keras.Input(shape=(None, 3),
                              name='input_points')  # MxNx3 where M is batch size,N is number of points and 3 for channels

    # Input transform layer
    input_transform = TransformNet(add_regularization=add_regularization, bn_momentum=bn_momentum,
                                   name='input_transformation')(input_layer)  # BxNx3 --> BxNx3
    input_transform = tf.expand_dims(input_transform, axis=2)  # BxNx3 --> BxNx1x3

    shared_layer_1 = CustomConvolution(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)(
        input_transform)  # BxNx1x3 --> BxNx1x64
    shared_layer_2 = CustomConvolution(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)(
        shared_layer_1)  # BxNx1x64 --> BxNx1x64
    shared_layer_2 = tf.squeeze(shared_layer_2, axis=2)

    feature_transform = TransformNet(add_regularization=add_regularization, bn_momentum=bn_momentum,
                                     name='feature_transformation')(shared_layer_2)  # BxNx64 --> BxNx64
    feature_transform = tf.expand_dims(feature_transform, axis=2)  # BxNx64 --> BxNx1x64

    shared_layer_3 = CustomConvolution(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)(
        feature_transform)  # BxNx1x64 --> BxNx1x64
    shared_layer_4 = CustomConvolution(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)(
        shared_layer_3)  # BxNx1x64 --> BxNx1x128
    shared_layer_5 = CustomConvolution(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                                       bn_momentum=bn_momentum, activation=tf.nn.relu,apply_bn=True)(
        shared_layer_4)  # BxNx1x128 --> BxNx1x1024

    shared_layer_5 = tf.squeeze(shared_layer_5, axis=2)  # BxNx1x1024 --> BxNx1024

    global_features = tf.reduce_max(shared_layer_5, axis=1)  # BxNx1024 --> Bx1024

    dense0 = CustomDense(units=512, activation=tf.nn.relu, bn_momentum=bn_momentum,apply_bn=True)(global_features)
    dense0 = keras.layers.Dropout(0.7)(dense0)
    dense1 = CustomDense(units=256, activation=tf.nn.relu, bn_momentum=bn_momentum,apply_bn=True)(dense0)
    dense1 = keras.layers.Dropout(0.7)(dense1)

    output_layer = CustomDense(units=n_classes, apply_bn=False)(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


inputs = tf.random.uniform((3, 1024, 3))
n_classes = 40

model = get_model(n_classes, 0.99)
model.summary()
