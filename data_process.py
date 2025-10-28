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


## set the seed so that we can repeat the data generation
random.seed(100023232)

# READ ALL FILES AND CREATE X'S
X_all= []
Pid_all = []
pid_gender = {}
data_dir_path = '/Users/manasvinisingh/Projects/sf2022/data'

count = 0
# browse all the files

## lets create estimate of total work, so we can show the progress
total_work = 0
for root, dirs, files in os.walk(data_dir_path + "/stage_2_train_images/"):
    for name in files:
        if name.endswith((".dcm")):
            total_work += 1

## Reading and re-sizing all Images to 128x128
work_done = 0
for root, dirs, files in os.walk(data_dir_path + "/stage_2_train_images/"):
    for name in files:
        if name.endswith((".dcm")):
            count += 1
            ds = pydicom.dcmread(data_dir_path + "/stage_2_train_images/" + "/" + name)
            X_all.append(resize(ds.pixel_array, (128,128), preserve_range=True))
            pid = name.split(".")[0]
            Pid_all.append(pid)
            pid_gender[pid] = ds.PatientSex
            work_done += 1
            if (work_done % 10 == 0):
                sys.stdout.write("\r")
                sys.stdout.write("[ %s ] %s" % (work_done, total_work))
                sys.stdout.flush()

        if 0 and count == 1000:
          break

X_all = np.array(X_all)
Pid_all = np.array(Pid_all)

print (X_all.shape)
print (Pid_all.shape)

#CREATING BOUNDING BOXES
mdata = {}
with open(data_dir_path + '/stage_2_train_labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    is_first = True
    for row in reader:
        if is_first:
            is_first = 0
            continue
        pid = row[0]
        target = int(row[-1])
        if not pid in mdata:
            mdata[pid] = []

        if target:
            x = int(float(row[1]) / 8)
            y = int(float(row[2]) / 8)
            w = int(float(row[3]) / 8)
            h = int(float(row[4]) / 8)
            mdata[pid].append((x, y, w, h))


# Printing image with the bounding box
index = 42
figure = plt.figure()
ax = figure.add_subplot(111)

print(Pid_all.shape)

pid = Pid_all[index]
bboxes = mdata[pid]

if len(bboxes) > 0:
  for x, y, w, h in bboxes:
    ax.add_patch( Rectangle((x, y),
                            w, h,
                          fc ='none',
                          ec ='r',
                          lw = 2) )
    x = X_all[index, :, :]


print (x.shape)
print (pid)
print (bboxes)
ax.imshow(x, cmap='gray')
plt.show()

# GENDERS IN DATASET
positive = 0
male_count = 0
for p in Pid_all:
  if p in mdata:
    if len(mdata[p]) > 0:
      positive += 1

  if pid_gender[p] == 'M':
    male_count += 1

print(male_count)
print(positive)

gender = ds.PatientSex
print(gender)

# CREATE Y'S
Y_all = []
for pid in Pid_all:
  aaa = np.zeros((128,128))
  bboxes = mdata[pid]
  if len(bboxes) > 0:
    for x, y, w, h in bboxes:
      aaa[y:y+h, x:x+w] = np.ones((h, w))
  Y_all.append(aaa)

Y_all = np.array(Y_all)
print (Y_all.shape)

# Plotting X and Y for a given pid to see if it make sense
index = 42
figure = plt.figure()
ax = figure.add_subplot(121)
print(Pid_all.shape)
pid = Pid_all[index]
bboxes = mdata[pid]

if len(bboxes) > 0:
  for x, y, w, h in bboxes:
    ax.add_patch( Rectangle((x, y),
                            w, h,
                          fc ='none',
                          ec ='r',
                          lw = 2) )
x = X_all[index, :, :]

print (x.shape)
ax.imshow(x, cmap='gray')

ax = figure.add_subplot(122)
ax.imshow(Y_all[index], cmap='gray')
plt.show()

# Split the data into train, dev, and test
# train - 80%, dev - 10%, test - 10%
n = len(X_all)
indices = [i for i in range(n)]
random.shuffle(indices)
dev_start_index = int(n * 0.8)
test_start_index = int(n * 0.9)
x_train = X_all[indices[0:dev_start_index], :, :]
y_train = Y_all[indices[0:dev_start_index], :, :]
pid_train = Pid_all[indices[0:dev_start_index]]

x_dev = X_all[indices[dev_start_index : test_start_index], :, :]
y_dev = Y_all[indices[dev_start_index : test_start_index], :, :]
pid_dev = Pid_all[indices[dev_start_index : test_start_index]]

x_test = X_all[indices[test_start_index : n], :, :]
y_test = Y_all[indices[test_start_index : n], :, :]
pid_test = Pid_all[indices[test_start_index: n]]

## Save all the data to a file
with open(data_dir_path + '/saved_data/data_train_1.pickle', 'wb') as f:
  pickle.dump((x_train, y_train, pid_train), f, pickle.HIGHEST_PROTOCOL)

with open(data_dir_path + '/saved_data/data_dev_1.pickle', 'wb') as f:
  pickle.dump((x_dev, y_dev, pid_dev), f, pickle.HIGHEST_PROTOCOL)

with open(data_dir_path + '/saved_data/data_test_1.pickle', 'wb') as f:
  pickle.dump((x_test, y_test, pid_test), f, pickle.HIGHEST_PROTOCOL)

with open(data_dir_path + '/saved_data/data_mdata.pickle', 'wb') as f:
  pickle.dump(mdata, f, pickle.HIGHEST_PROTOCOL)

