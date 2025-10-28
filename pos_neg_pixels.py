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

data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'
with open(data_dir_path + '/saved_data/data_mdata.pickle', 'rb') as f:
  mdata = pickle.load(f)

pos = 0
for k in mdata:
  d = mdata[k]
  for x, y, w, h in d:
    pos = pos + (w*h)

total_pixels = 128*128*26684
neg = total_pixels - pos

print(total_pixels)
print(pos)
print(neg)