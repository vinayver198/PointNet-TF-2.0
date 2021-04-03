# -*- coding: utf-8 -*-
# @Author: vinay
# @Date:   2021-04-03 12:00:27
# @Last Modified by:   vinay
# @Last Modified time: 2021-04-03 12:00:28

import os
import sys
import argparse
from glob import glob
import tensorflow as tf
from model import get_model
from utils import read_filename
import time
from tensorflow.keras.utils import plot_model
import numpy as np

tf.random.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

log_dir = FLAGS.log_dir
if not os.path.exists(log_dir): os.mkdir(log_dir)

# Initialize the metrics
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Initialize the loss function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Initialize the callback
callback = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)


# Initialize the batch normalization
def get_bn_momentum(step):
    return min(0.99, 0.5 + 0.0002 * step)


bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)

"""
# Initialize the learning rate
def get_learning_rate(step):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,  # Base learning rate.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    return learning_rate


# Initialize the optimizer
#lr = tf.Variable(get_learning_rate(step=0), trainable=False)
lr = get_learning_rate(step=0)
"""


# Instantiate optimizer and loss function
def get_lr(initial_learning_rate, decay_steps, decay_rate, step, staircase=False, warm_up=True):
    if warm_up:
        coeff1 = min(1.0, step / 2000)
    else:
        coeff1 = 1.0

    if staircase:
        coeff2 = decay_rate ** (step // decay_steps)
    else:
        coeff2 = decay_rate ** (step / decay_steps)

    current = initial_learning_rate * coeff1 * coeff2
    return current


LR_ARGS = {'initial_learning_rate': BASE_LEARNING_RATE, 'decay_steps': DECAY_STEP,
           'decay_rate': DECAY_RATE, 'staircase': False, 'warm_up': True}
lr = tf.Variable(get_lr(**LR_ARGS, step=0), trainable=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_val = loss_function(labels, logits) + sum(model.losses)
    gradients = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return logits, loss_val, model.losses[0]


@tf.function
def validation_step(inputs, labels):
    logits = model(inputs, training=False)
    loss_val = loss_function(labels, logits)
    return logits, loss_val


n_classes = 40
# Get the train and validation dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset, train_length = read_filename('data/modelnet40_ply_hdf5_2048/train_files.txt')
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset, validation_length = read_filename('data/modelnet40_ply_hdf5_2048/test_files.txt')
validation_dataset = validation_dataset.batch(batch_size=BATCH_SIZE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

model = get_model(n_classes, bn_momentum)
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

# Model training
start_time = time.time()
print('Model training started : \n')
print('Steps per epoch : {}'.format(train_length // BATCH_SIZE))
print('Total Number of steps while training : {}'.format((train_length // BATCH_SIZE) * MAX_EPOCH))
writer = tf.summary.create_file_writer(log_dir)
step = 0
for epoch in range(MAX_EPOCH):
    print("Epoch started : {}".format(epoch))
    train_accuracy.reset_states()
    validation_accuracy.reset_states()
    total_loss = []
    for i, (x_train, y_train) in enumerate(train_dataset):
        logits, loss_value, matrix_loss = train_step(x_train, y_train)
        train_accuracy.update_state(y_train, logits)
        total_loss.append(loss_value)
        step += 1
        bn_momentum.assign(get_bn_momentum(step))
        lr.assign(get_lr(**LR_ARGS, step=step))
    train_loss = np.mean(total_loss)
    val_loss = []
    for x_val, y_val in validation_dataset:
        val_logits, val_loss_value = validation_step(x_val, y_val)
        val_loss.append(val_loss_value)
        validation_accuracy.update_state(y_val, val_logits)
    validation_loss = np.mean(val_loss)
    with writer.as_default():
        tf.summary.scalar("training_loss", train_loss, step=epoch)
        tf.summary.scalar("validation_loss", validation_loss, step=epoch)
        tf.summary.scalar("training_accuracy ", train_accuracy.result().numpy(), step=epoch)
        tf.summary.scalar("validation_accuracy ", validation_accuracy.result().numpy(), step=epoch)
        tf.summary.scalar("learning_rate", lr.numpy(), step=epoch)
        tf.summary.scalar("batch_normalization", bn_momentum.numpy(), step=epoch)
        tf.summary.scalar("matrix_reg_loss", matrix_loss.numpy(), step=epoch)
    writer.flush()
    model.save_weights('model/checkpoints/' + 'iter-' + str(epoch), save_format='tf')

end_time = time.time()
time_lapsed = end_time - start_time
print('Training Completed')
print('Total time lapsed in seconds {}'.format(time_lapsed))
print('Total time lapsed in minutes {}'.format(time_lapsed / 60.))
print('Total time lapsed in hrs {}'.format(time_lapsed / (60 * 60)))