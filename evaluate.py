import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import os
import pickle
from skimage.transform import resize
import random
import sys
from matplotlib.patches import Rectangle, Circle
import csv
from tensorflow import keras
import tensorflow_addons as tfa

# this is where my data is
#data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'
data_dir_path = 'X:/Mannu/sf2022-new/data'

#DEFINE MODEL
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Flatten
from keras.layers import Dropout, MaxPooling2D, concatenate
from keras.layers import Input
from keras.models import Model

# cnn model
from tensorflow.keras import datasets, layers, models

model_name = 'detection_model_1'
train = False
dev = False
test = True
upsample = False
print(f"train = {train}")
print(f"dev = {dev}")


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
  # Function to add 2 convolutional layers with the parameters passed to it
  # first layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
             kernel_initializer='he_normal', padding='same')(input_tensor)
  if batchnorm:
    x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # second layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
             kernel_initializer='he_normal', padding='same')(x)
  if batchnorm:
    x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x

# Now load the train data
#with open(data_dir_path + '/saved_data/data_train_1.pickle', 'rb') as f:
  #x_train, y_train, pid_train = pickle.load(f)
#with open(data_dir_path + '/saved_data/data_dev_1.pickle', 'rb') as f:
  #x_dev, y_dev, pid_dev = pickle.load(f)
if dev:
  with open(data_dir_path + '/saved_data/data_dev_2.pickle', 'rb') as f:
    x_dev, y_dev_det, pid_dev = pickle.load(f)

if test:
#with open(data_dir_path + '/saved_data/data_test_1.pickle', 'rb') as f:
  #x_test, y_test, pid_test = pickle.load(f)
  with open(data_dir_path + '/saved_data/data_test_2.pickle', 'rb') as f:
    x_test, y_test_det, pid_test = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

print(x_test.shape)
print(y_test_det.shape)

if dev:
  y_dev_det = np.array(y_dev_det)
if test:
  y_test_det = np.array(y_test_det)

# just use first n
"""n = 2000
x_train = x_train[0:n]
y_train = y_train[0:n]
pid_train = pid_train[0:n]"""

## up-sample positive examples
if upsample and dev:
  upsample_by_x = 10
  n = x_dev.shape[0]
  new_x_dev = []
  new_y_dev = []
  new_pid_dev = []
  num_pos_samples = 0
  for i in range(n):
    r = 1
    """if np.sum(y_dev[i]) > 0:
      r = upsample_by_x
      num_pos_samples += 1

    for j in range(r):
      new_x_dev.append(x_dev[i])
      new_y_dev.append(y_dev[i])
      new_pid_dev.append(pid_dev[i])"""
    if np.sum(y_dev_det[i]) > 0:
      r = upsample_by_x
      num_pos_samples += 1

    for j in range(r):
      new_x_dev.append(x_dev[i])
      new_y_dev.append(y_dev_det[i])
      new_pid_dev.append(pid_dev[i])

  x_dev = np.array(new_x_dev)
  y_dev = np.array(new_y_dev)
  pid_dev = np.array(new_pid_dev)
  print(f"number of positive samples = {num_pos_samples}")

  del new_x_dev, new_y_dev, new_pid_dev

if upsample and test:
  upsample_by_x = 10
  n = x_test.shape[0]
  new_x_test = []
  new_y_test = []
  new_pid_test = []
  num_pos_samples = 0
  """for i in range(n):
      r = 1
      if np.sum(y_test[i]) > 0:
          r = upsample_by_x
          num_pos_samples += 1

      for j in range(r):
          new_x_test.append(x_test[i])
          new_y_test.append(y_test[i])
          new_pid_test.append(pid_test[i])"""
  for i in range(n):
    r = 1
    if np.sum(y_test_det[i]) > 0:
      r = upsample_by_x
      num_pos_samples += 1

    for j in range(r):
      new_x_test.append(x_test[i])
      new_y_test.append(y_test_det[i])
      new_pid_test.append(pid_test[i])

  x_test = np.array(new_x_test)
  y_test = np.array(new_y_test)
  pid_test = np.array(new_pid_test)
  print (f"number of positive samples = {num_pos_samples}")

  del new_x_test, new_y_test, new_pid_test

#NORMALIZE DATA
if dev:
  x_dev /= 255.0
  x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))
  #y_dev = y_dev.reshape((y_dev.shape[0], y_dev.shape[1], y_dev.shape[2], 1))
  print(x_dev.shape)
  #print(y_dev.shape)
if test:
  x_test /= 255.0
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
  #y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 1))
  print(x_test.shape)
  #print(y_test.shape)

#print(x_train.shape)
#print(y_train_det.shape)

model = tf.keras.models.load_model(model_name + '.h5')
print(f"loading model {model_name}")
model.summary()
optimizer = tf.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.BinaryCrossentropy()
#loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
#model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(),
#                                             tf.keras.metrics.BinaryIoU(target_class_ids=[1])])
model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
if dev:
  #results = model.evaluate(x_dev, y_dev)
  results = model.evaluate(x_dev, y_dev_det)
  print(results)
  y_hats = model.predict(x_dev)

if test:
  #results = model.evaluate(x_test, y_test, batch_size=8)
  results = model.evaluate(x_test, y_test_det, batch_size=8)
  print(results)
  y_hats = model.predict(x_test)

y_hats = np.array(y_hats)

"""def f1_score(y, y_hat):
  n, h, w, _ = y.shape
  y_hat = np.reshape(y_hat, (n, h, w, 1))
  p = y_hat >= 0.5
  tp = np.sum((p == 1) * (y == 1))
  fp = np.sum((p == 1) * (y == 0))
  fn = np.sum((p == 0) * (y == 1))
  tn = np.sum((p == 0) * (y == 0))

  p = tp * 1.0 / (tp + fp)
  r = tp * 1.0 / (tp + fn)
  f1 = 2 * p * r / (p + r)

  return (tp, fp, fn, tn, p, r, f1)

y_train = np.array(y_train)"""

def f1_score(y, y_hat):
  print(y.shape)
  print(y_hat.shape)
  #n = y.shape
  #y_hat = np.reshape(y_hat, (n, 1))
  p = y_hat >= 0.5
  tp = np.sum((p == 1) * (y == 1))
  fp = np.sum((p == 1) * (y == 0))
  fn = np.sum((p == 0) * (y == 1))
  tn = np.sum((p == 0) * (y == 0))

  p = tp * 1.0 / (tp + fp)
  r = tp * 1.0 / (tp + fn)
  f1 = 2 * p * r / (p + r)

  return (tp, fp, fn, tn, p, r, f1)

"""if dev:
  tp, fp, fn, tn, p, r, f1 = f1_score(y_dev, y_hats)
if train:
  tp, fp, fn, tn, p, r, f1 = f1_score(y_train, y_hats)
if test:
  tp, fp, fn, tn, p, r, f1 = f1_score(y_test, y_hats)"""
if dev:
  tp, fp, fn, tn, p, r, f1 = f1_score(y_dev_det, y_hats)
if test:
  tp, fp, fn, tn, p, r, f1 = f1_score(y_test_det, y_hats)


print(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}, p = {p}, r = {r}, f1 = {f1}")
