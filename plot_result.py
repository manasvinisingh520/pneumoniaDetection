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
data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'

#DEFINE MODEL
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import Dropout, MaxPooling2D, concatenate
from keras.layers import Input
from keras.models import Model

#model_name = '../trained_models/unet_focal_dropout_p0'
model_name = '../trained_models/model1'
train = False
dev = True
test = False
upsample = False
print(f"train = {train}")
print(f"dev = {dev}")
print(f"test = {test}")

# Now load the train data
with open(data_dir_path + '/saved_data/data_train.pickle', 'rb') as f:
  x_train, y_train, pid_train = pickle.load(f)

with open(data_dir_path + '/saved_data/data_dev.pickle', 'rb') as f:
  x_dev, y_dev, pid_dev = pickle.load(f)

with open(data_dir_path + '/saved_data/data_dev.pickle', 'rb') as f:
  x_test, y_test, pid_test = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

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
if test:
  x_test /= 255.0
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
  y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 1))
  print(x_test.shape)
  print(y_test.shape)

model = tf.keras.models.load_model(model_name + '.h5')
print(f"loading model {model_name}")
optimizer = tf.optimizers.Adam(learning_rate = 0.001)
#loss = tf.keras.losses.BinaryCrossentropy()
loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(),
                                             tf.keras.metrics.BinaryIoU(target_class_ids=[1])])
# Lets plot one of the image and its prediction

for i in range(20):
    index = 20 + i
    figure = plt.figure()
    ax = figure.add_subplot(121)
    if train:
        pid = pid_train[index]
        img = x_train[index, :, :] * 255
    if dev:
        pid = pid_dev[index]
        img = x_dev[index, :, :] * 255
    if test:
        pid = pid[index]
        img = x_test[index, :, :] * 255
    bboxes = mdata[pid]

    if len(bboxes) > 0:
      for x, y, w, h in bboxes:
        ax.add_patch( Rectangle((x, y),
                                w, h,
                              fc ='none',
                              ec ='r',
                              lw = 2) )

    ax.imshow(img, cmap='gray')

    # plotting the prediction
    ax = figure.add_subplot(122)
    if train:
        y_hat = model.predict(np.reshape(x_train[index], (1, 128, 128, 1)))
    if dev:
        y_hat = model.predict(np.reshape(x_dev[index], (1, 128, 128, 1)))
    if test:
        y_hat = model.predict(np.reshape(x_test[index], (1, 128, 128, 1)))

    print (y_hat.shape)
    pred = y_hat[0, :, :, 0] >= 0.5
    print (pred.shape)
    ax.imshow(img, cmap='gray')
    Y, X = np.where(pred)
    xys = []
    if len(Y) > 0:
      for y, x in zip(Y, X):
        xys.append((x,y))

    for x,y in xys:
      ax.add_patch(Circle((x,y), radius=0.5, color='red'))

    plt.show()

# How many images are true positives
if train:
    y = y_train
    x = x_train
if dev:
    y = y_dev
    x = x_dev
if test:
    y = y_test
    x = x_test

y_hats = model.predict(x, batch_size=8)

def f1_score(y, y_hats):
  n, h, w, _ = y.shape
  y_hat = np.reshape(y_hats, (n, h, w, 1))
  p = y_hat >= 0.5

  tp = 0
  fp = 0
  fn = 0
  tn = 0
  for i in range(n):
      if np.sum(p[i]) > 0 and np.sum(y[i]) > 0:
          tp += 1
      if np.sum(p[i]) > 0 and np.sum(y[i]) == 0:
          fn += 1
      if np.sum(p[i]) == 0 and np.sum(y[i]) > 0:
          fp += 1
      if np.sum(p[i]) == 0 and np.sum(y[i]) == 0:
          tn += 1

  p = tp * 1.0 / (tp + fp)
  r = tp * 1.0 / (tp + fn)
  f1 = 2 * p * r / (p + r)

  return (tp, fp, fn, tn, p, r, f1)


tp, fp, fn, tn, p, r, f1 = f1_score(y, y_hats)
print(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}, p = {p}, r = {r}, f1 = {f1}")

