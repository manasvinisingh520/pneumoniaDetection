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

data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'

y_train_det = []
y_dev_det = []
y_test_det = []

with open(data_dir_path + '/saved_data/data_train_1.pickle', 'rb') as f:
  x_train, y_train, pid_train = pickle.load(f)

with open(data_dir_path + '/saved_data/data_dev_1.pickle', 'rb') as f:
  x_dev, y_dev, pid_dev = pickle.load(f)

with open(data_dir_path + '/saved_data/data_test_1.pickle', 'rb') as f:
  x_test, y_test, pid_test = pickle.load(f)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)


for pid in pid_train:
  if len(mdata[pid]) > 0:
    y_train_det.append(1)
  else:
    y_train_det.append(0)

for pid in pid_dev:
  if len(mdata[pid]) > 0:
    y_dev_det.append(1)
  else:
    y_dev_det.append(0)

for pid in pid_test:
  if len(mdata[pid]) > 0:
    y_test_det.append(1)
  else:
    y_test_det.append(0)

y_train_det = np.array(y_train_det)
y_dev_det = np.array(y_dev_det)
y_test_det = np.array(y_test_det)

print(x_train.shape)
print(y_train_det.shape)
print(x_dev.shape)
print(y_dev_det.shape)
print(x_test.shape)
print(y_test_det.shape)

train_pos = 0
dev_pos = 0
test_pos = 0
for y in y_train_det:
  if y == 1:
    train_pos += 1
for y in y_dev_det:
  if y == 1:
    dev_pos += 1
for y in y_test_det:
  if y == 1:
    test_pos += 1

print(int(21347/train_pos))

with open(data_dir_path + '/saved_data/data_train_2.pickle', 'wb') as f:
  pickle.dump((x_train, y_train_det, pid_train), f, pickle.HIGHEST_PROTOCOL)

with open(data_dir_path + '/saved_data/data_dev_2.pickle', 'wb') as f:
  pickle.dump((x_dev, y_dev_det, pid_dev), f, pickle.HIGHEST_PROTOCOL)

with open(data_dir_path + '/saved_data/data_test_2.pickle', 'wb') as f:
  pickle.dump((x_test, y_test_det, pid_test), f, pickle.HIGHEST_PROTOCOL)