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

# this is where my data is
data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'

#DEFINE MODEL
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import Dropout, MaxPooling2D, concatenate
from keras.layers import Input
from keras.models import Model


with open(data_dir_path + '/saved_data/data_dev.pickle', 'rb') as f:
  x_dev, y_dev, pid_dev = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

#NORMALIZE DATA
x_dev /= 255.0
print (x_dev.shape)
x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))
print (x_dev.shape)

model = tf.keras.models.load_model('my_trained_model.h5')

model.evalu

optimizer = tf.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])

def F1(y_hat, y):
