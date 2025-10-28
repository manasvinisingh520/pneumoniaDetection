import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import os
import pickle
from skimage.transform import resize
import random
import sys
from matplotlib.patches import Rectangle
import csv
from tensorflow import keras
import tensorflow_addons as tfa

# this is where my data is
data_dir_path = 'D:/Manasvini-2022/sf2022/data'

#DEFINE MODEL
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import Dropout, MaxPooling2D, concatenate
from keras.layers import Input
from keras.models import Model

model_name = 'unet_focal_drop_p2'
train = True
dev = False
print(train)
print(dev)

# Now load the train data
with open(data_dir_path + '/saved_data/data_train_1.pickle', 'rb') as f:
  x_train, y_train, pid_train = pickle.load(f)

with open(data_dir_path + '/saved_data/data_dev_1.pickle', 'rb') as f:
  x_dev, y_dev, pid_dev = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

# just use first n
n = 2000
x_train = x_train[0:n]
y_train = y_train[0:n]
pid_train = pid_train[0:n]

## up-sample positive examples
if train:
  upsample_by_x = 10
  n = x_train.shape[0]
  new_x_train = []
  new_y_train = []
  new_pid_train = []
  num_pos_samples = 0
  for i in range(n):
      r = 1
      if np.sum(y_train[i]) > 0:
          r = upsample_by_x
          num_pos_samples += 1

      for j in range(r):
          new_x_train.append(x_train[i])
          new_y_train.append(y_train[i])
          new_pid_train.append(pid_train[i])

  x_train = np.array(new_x_train)
  y_train = np.array(new_y_train)
  pid_train = np.array(new_pid_train)
  print (f"number of positive samples = {num_pos_samples}")

  del new_x_train, new_y_train, new_pid_train

if dev and 0:
  upsample_by_x = 10
  n = x_dev.shape[0]
  new_x_dev = []
  new_y_dev = []
  new_pid_dev = []
  num_pos_samples = 0
  for i in range(n):
    r = 1
    if np.sum(y_dev[i]) > 0:
      r = upsample_by_x
      num_pos_samples += 1

    for j in range(r):
      new_x_dev.append(x_dev[i])
      new_y_dev.append(y_dev[i])
      new_pid_dev.append(pid_dev[i])

  x_train = np.array(new_x_dev)
  y_train = np.array(new_y_dev)
  pid_train = np.array(new_pid_dev)
  print(f"number of positive samples = {num_pos_samples}")

  del new_x_dev, new_y_dev, new_pid_dev


#NORMALIZE DATA
if train:
  x_train /= 255.0
  x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
  y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))
  print(x_train.shape)
  print(y_train.shape)
if dev:
  x_dev /= 255.0
  x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))
  y_dev = y_dev.reshape((y_dev.shape[0], y_dev.shape[1], y_dev.shape[2], 1))
  print(x_dev.shape)
  print(y_dev.shape)

model = tf.keras.models.load_model(model_name + '.h5')
print(f"loading model {model_name}")
optimizer = tf.optimizers.Adam(learning_rate = 0.001)
#loss = tf.keras.losses.BinaryCrossentropy()
loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(),
                                             tf.keras.metrics.BinaryIoU(target_class_ids=[1])])
if dev:
  results = model.evaluate(x_dev, y_dev)
  print(results)
  y_hats = model.predict(x_dev)

if train:
  results = model.evaluate(x_train, y_train, batch_size=8)
  print(results)
  y_hats = model.predict(x_train)

def f1_score(y, y_hat):
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

tp, fp, fn, tn, p, r, f1 = f1_score(y_dev, y_hats)
print(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}, p = {p}, r = {r}, f1 = {f1}")
