import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import os
import pickle

# READ ALL FILES AND CREATE X'S

from skimage.transform import resize
X_all= []
Pid_all = []
pid_gender = {}
path = '/content/gdrive/MyDrive/sf2022/data/stage_2_train_images/'
#count = 0
# browse all the files
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".dcm")):
            #count += 1
            ds = pydicom.dcmread(path + "/" + name)
            X_all.append(resize(ds.pixel_array, (128,128)))
            pid = name.split(".")[0]
            Pid_all.append(pid)
            pid_gender[pid] = ds.PatientSex

        #if count == 10000:
          #break

X_all = np.array(X_all)
Pid_all = np.array(Pid_all)

print (X_all.shape)
print (Pid_all.shape)
print(pid_gender)

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

index = 13

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

ax.imshow(x, cmap='gray')

ax = figure.add_subplot(122)
ax.imshow(Y_all[index], cmap='gray')

print(X_all.shape)
print(Y_all.shape)