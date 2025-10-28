import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import matplotlib.pyplot as mlab
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

print(mdata)