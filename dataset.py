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

from skimage.transform import resize
num_bins = 10
Xm_all = [[] for i in range(num_bins)]
Ym_all = [[] for i in range(num_bins)]
Pidm_all = [[] for i in range(num_bins)]

Xf_all = [[] for i in range(num_bins)]
Yf_all = [[] for i in range(num_bins)]
Pidf_all = [[] for i in range(num_bins)]

pid_gender = {}
new_data = True

age_male = []
age_female = []
m_images = [[] for i in range(10)]
f_images = [[] for i in range(10)]

def get_class(pid):
    return 0

path = '/Users/manasvinisingh/Projects/sf2022/data/stage_2_train_images/'
#count = 0
# browse all the files
if new_data:
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".dcm")):
                #count += 1
                print (f"reading file {name}")
                ds = pydicom.dcmread(path + "/" + name)
                img = ds.pixel_array
                print (img[100,100])
                img = resize(img, (128, 128), preserve_range=True)
                print(img[100, 100])
                pid = name.split(".")[0]
                gender = ds.PatientSex
                age = int(ds.PatientAge)
                if age > 100:
                    continue
                print(age)
                #somehow find out if this is pneumonia or not
                y = 0
                if pid in mdata and len(mdata[pid]) > 0:
                    y = 1

                if ds.PatientSex == 'M':
                  age_male.append(age)
                  Pidm_all[age//num_bins].append(pid)
                  Xm_all[age//num_bins].append(img)
                  Ym_all[age//num_bins].append(y)
                else:
                  age_female.append(ds.PatientAge)
                  Pidf_all[age//num_bins].append(pid)
                  Xf_all[age//num_bins].append(img)
                  Yf_all[age//num_bins].append(y)

            #if count == 100:
              #break

    age_male = [int(age) for age in age_male]
    age_female = [int(age) for age in age_female]


    with open('../data/saved_data/age_m_or_f.pickle', 'wb') as f:
      pickle.dump((age_male, age_female, Pidm_all, Xm_all, Ym_all, Pidf_all, Xf_all, Yf_all), f, pickle.HIGHEST_PROTOCOL)

else:
    with open('../data/saved_data/age_m_or_f.pickle', 'rb') as f:
        age_male, age_female, Pidm_all, Xm_all, Ym_all, Pidf_all, Xf_all, Yf_all = pickle.load(f)

bin_list = [i*10 for i in range(10)]
n, bins, patches = plt.hist(age_male, bins = bin_list, facecolor='blue', alpha=0.5)
plt.xlabel('Male Age')
plt.ylabel('No. of Patients')
plt.show()

bin_list = [i*10 for i in range(10)]
n, bins, patches = plt.hist(age_female, bins = bin_list, facecolor='blue', alpha=0.5)
plt.xlabel('Female Age')
plt.ylabel('No. of Patients')
plt.show()

for age_group, x in enumerate(Xm_all):
    print (f" Male Age group {age_group} = {len(x)}, positives = {sum(Ym_all[age_group])}")

for age_group, x in enumerate(Xf_all):
    print(f" Female Age group {age_group} = {len(x)}, positives = {sum(Yf_all[age_group])}")