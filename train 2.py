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

reset_model = True
model_name = 'model2_new'
no_epochs = 20
batch_size = 8
alpha = 0.25
gamma = 2.0


# this is where my data is
data_dir_path = 'D:/Manasvini-2022/sf2022/data'

#DEFINE MODEL
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import Dropout, MaxPooling2D, concatenate
from keras.layers import Input
from keras.models import Model

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


def get_unet(input_img, n_filters=16, dropout=0.4, batchnorm=False):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c1 = Dropout(dropout)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c2 = Dropout(dropout)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    c3 = Dropout(dropout)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    c4 = Dropout(dropout)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


inputs = Input(shape=(128, 128, 1))

if reset_model:
  model = get_unet(inputs)
else:
  print('loading ' + model_name)
  model = tf.keras.models.load_model(model_name + '.h5')

optimizer = tf.optimizers.Adam(learning_rate = 0.0001)
#loss = tf.keras.losses.BinaryCrossentropy()
loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
#model.compile(optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
model.compile(optimizer, loss=loss, metrics=tf.keras.metrics.BinaryIoU(target_class_ids=[1]))


# Now load the train data
with open(data_dir_path + '/saved_data/data_train_1.pickle', 'rb') as f:
  x_train, y_train, pid_train = pickle.load(f)

# just to speed up
"""n = 2000
x_train = x_train[0:n]
y_train = y_train[0:n]
pid_train = pid_train[0:n]"""

## up-sample positive examples
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

print (x_train.shape)
print (y_train.shape)
print (pid_train.shape)

with open(data_dir_path + '/saved_data/data_dev_1.pickle', 'rb') as f:
  x_dev, y_dev, pid_dev = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

#NORMALIZE DATA
x_train /= 255.0
x_dev /= 255.0
print (x_train.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))

print (x_train.shape)
print (x_dev.shape)

# reshape y to (None, 128, 128, 1)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))
y_dev = np.reshape(y_dev, (y_dev.shape[0], y_dev.shape[1], y_dev.shape[2], 1))

# tensorboard stuff

#MODEL TRAINING
model.fit(x_train, y_train, validation_data=(x_dev, y_dev),
          batch_size=batch_size, epochs=no_epochs, callbacks=[])
model.save(model_name + '.h5')

# Load and predict again
model.evaluate(x_train, y_train, batch_size=batch_size)
