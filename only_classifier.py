#!/usr/bin/env python
# coding: utf-8

# In[76]:


# import moduels
import h5py
import numpy as np
#import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
#import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier


#from PIL import Image
#import random
#import math

import os

from sklearn.ensemble import RandomForestRegressor

# In[77]:


# access the data
# this part have to be replaced when the images from the neuronal networks are available
dset = h5py.File("dataset_train.h5","r")

#Access to the input data
RGB = dset["RGB"]
NIR = dset["NIR"]

#Access groundtruth data
GT = dset["GT"]
CLD = dset["CLD"]

print("Thes shape of the images is:")
print(RGB.shape)


# get size and numbers of the dataset
# number of images
n_images = RGB.shape[0]

# imagesize x_direction
sx_image = RGB.shape[1]
#imageszize y_direction
sy_image = RGB.shape[2]


# In[78]:


# function to create groundtruth

def groundtruth_generator(img_number, x_ind, y_ind,win_x,win_y):
    #create temp image
    cout_out = GT[img_number, x_ind:x_ind+win_x,y_ind:y_ind+win_y]
    temp_img = np.where(cout_out==99,3,cout_out)

    #insert the clouds into the temp groundtruth
    temp_cloud = CLD[img_number, x_ind:x_ind+win_x,y_ind:y_ind+win_y]
    
    #cloud_positions = np.where(CLD[img_number] > 10) # check if this number makes sense
    cloud_positions = np.where(temp_cloud > 10) # check if this number makes sense
    temp_img[cloud_positions] = 2
    
    #cout_out = temp_img[x_ind:x_ind+win_x,y_ind:y_ind+win_y]

    final_ground = temp_img.reshape(win_x*win_y)
    return final_ground


# In[79]:


#Access the data from the neuronal network part
# path where the files of the networkpart are 
path = './output_test_one_channel_resnet_50/'
files = os.listdir(path)
#count how many files there are
num_files = len(files)

#seperate the data into training data and validation data
X_train_n, X_val_n, y_train_n, y_val_n = train_test_split(np.arange(num_files),np.arange(num_files),test_size=0.3,random_state=0)


# In[80]:


# create the X_train etc. arrays
#test = np.load('./output_network/'+str(files[X_train[0]]))
#test.shape

s_patch = 256 # TODO integrate this information into the metadata line
n_cat = 1# TODO integrate this information into the metadata line

X_train = np.zeros([len(X_train_n)*np.square(s_patch),n_cat])
y_train = np.zeros([len(y_train_n)*np.square(s_patch)])

X_val = np.zeros([len(X_val_n)*np.square(s_patch),n_cat])
y_val = np.zeros([len(y_val_n)*np.square(s_patch)])

step_size = np.square(s_patch)

# TODO check if everything is correctly implemented
for i in range(1,len(X_train_n)+1):
    temp = np.load(path+str(files[X_train_n[i-1]])) # last three line contains metadata
    X_train[i*step_size-step_size:i*step_size,:] = temp[:-3,:]
    y_train[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1,0].astype(np.int64),temp[-3,0].astype(np.int64),temp[-2,0].astype(np.int64),s_patch,s_patch)
    
    
for i in range(1,len(X_val_n)+1):
    temp = np.load(path+str(files[X_val_n[i-1]])) # last three line contains metadata
    X_val[i*step_size-step_size:i*step_size,:] = temp[:-3,:]
    y_val[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1,0].astype(np.int64),temp[-3,0].astype(np.int64),temp[-2,0].astype(np.int64),s_patch,s_patch)
    
    


# # Test dataset
#  ist praktisch 1:1 von only network uebernommen

# In[139]:


# access the data
# this part have to be replaced when the images from the neuronal networks are available
dset_test = h5py.File("dataset_test.h5","r")

#Access to the input data
RGB_test = dset_test["RGB"]
NIR_test = dset_test["NIR"]

#Access groundtruth data
GT_test = dset_test["GT"]
CLD_test = dset_test["CLD"]

print("Thes shape of the images is:")
print(RGB_test.shape)

n_images_test = RGB_test.shape[0]


# In[140]:


#Access the data from the neuronal network part
# path where the files of the networkpart are 
path_test = './output_test_one_channel_resnet_50_test_data/'
files_test = os.listdir(path_test)
#count how many files there are
num_files_test = len(files_test)


# In[158]:


X_test = np.zeros([len(files_test)*np.square(s_patch),1])
y_test = np.zeros([len(files_test)*np.square(s_patch)])

for i in range(1,num_files_test+1):
    temp = np.load(path_test+str(files_test[i-1]))#last three line contains metadata
    X_test[i*step_size-step_size:i*step_size,:] = temp[:-3,:]
    y_test[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1,0].astype(np.int64),temp[-3,0].astype(np.int64),temp[-2,0].astype(np.int64),s_patch,s_patch)
    


# # Classifier part

# ## Decision Tree

# In[147]:


# Decision Tree
# train classifier and validate the classifier
dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(X_train, y_train)
dt_clf.score(X_val, y_val)


# In[148]:


# make predictions
dt_pred = dt_clf.predict(X_test)


# In[149]:


#validation of output

tn, fp, fn, tp = confusion_matrix(y_test, dt_pred)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
print("background, oil palm, clouds, no label")
precision = tp/(tp+fp)
print("precision \n"+str(precision))
recall = tp/(tp+fn)
print("recall \n" + str(recall))
f1 = 2*(precision*recall)/(precision+recall)
print("f1 \n" + str(f1))


# ## Random Forest

# In[83]:


# Import the model we are using

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
# Train the model on training data
rf.fit(X_train, y_train.ravel());


# In[159]:


# Use the forest's predict method on the test data
rf_pred = np.round(rf.predict(X_test))

tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, dt_pred)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
print("background, oil palm, clouds, no label")
precision_rf = tp/(tp+fp)
print("precision \n"+str(precision))
recall_rf = tp/(tp+fn)
print("recall \n" + str(recall))
f1_rf = 2*(precision*recall)/(precision+recall)
print("f1 \n" + str(f1))


# # Visualization
# TODO implement

# In[127]:


#print(sys.getsizeof(X_train)/1000000000)
#print(sys.getsizeof(dt_pred)/1000000000)

