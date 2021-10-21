# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:00:25 2021

@author: paimhof
"""
####################################################################################
####################################################################################
# import 
import h5py
import numpy as np
#import tensorflow as tf
#import matplotlib.pylab as plt

#from sklearn.model_selection import train_test_split
from sklearn import tree
#import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier


#from PIL import Image
#import random
import math

#import os

from tqdm import tqdm
####################################################################################
####################################################################################
# access data
dset = h5py.File("dataset_train.h5","r")
dset_test = h5py.File("dataset_test.h5","r")

#Access to the input data
RGB = dset["RGB"]
NIR = dset["NIR"]

RGB_test = dset_test["RGB"]
NIR_test = dset_test["NIR"]

#Access groundtruth data
GT = dset["GT"]
CLD = dset["CLD"]

GT_test = dset_test["GT"]
CLD_test = dset_test["CLD"]


#add the NIR to the training images
input_img = np.concatenate([RGB[:10],np.expand_dims(NIR[:10], axis = -1)], axis = -1)

#add the NIR to the test images
test_img =  np.concatenate([RGB_test[:],np.expand_dims(NIR_test[:], axis = -1)], axis = -1)

print("Done NIR")

####################################################################################
####################################################################################

#set vairalbes 

# allgemein
# imagesize x_direction
sx_image = input_img.shape[1]
#imageszize y_direction
sy_image = input_img.shape[2]

# number of images in x direction
nx = 4
#number of images in y direction 
ny = 4

s_patch = 256 
n_cat = 4

#################### train images##########################
# number of trainng images
n_images_train = input_img.shape[0]
# number of trainng channels 
n_channels = input_img.shape[3]


#################### test images##########################
# number of test images
n_images_test = test_img.shape[0]
# number of channels in the test images
test_channel = test_img.shape[3]



#####################################################################################################
#####################################################################################################
# create meshgrid

x_max = sx_image-s_patch
y_max = sy_image-s_patch 

stepsize_x = math.floor(x_max/nx)
stepsize_y = math.floor(y_max/ny)

#create arrays with the corrdinates
x_coordinates = np.arange(0,x_max,stepsize_x)
y_coordinates = np.arange(0,y_max,stepsize_y)

# create grid
xx , yy = np.meshgrid(x_coordinates,y_coordinates)


#####################################################################################################
#####################################################################################################

# function to create groundtruth

def groundtruth_generator(img_number, x_ind, y_ind,win_x,win_y):
    #create temp image
    cout_out = GT[img_number, x_ind:x_ind+win_x,y_ind:y_ind+win_y]
    temp_img = np.where(cout_out==99,3,cout_out)

    #insert the clouds into the temp groundtruth
    temp_cloud = CLD[img_number, x_ind:x_ind+win_x,y_ind:y_ind+win_y]
    
    cloud_positions = np.where(temp_cloud > 10) 
    temp_img[cloud_positions] = 2
    

    final_ground = temp_img.reshape(win_x*win_y)
    return final_ground

#####################################################################################################
#####################################################################################################

# create arrays to train the classifier



###################################### create training data ##########################################
X_train= np.zeros([n_images_train,nx+1,ny+1,np.square(s_patch),n_cat])
y_train = np.zeros([n_images_train,nx+1,ny+1,np.square(s_patch)])


for n in tqdm(range(n_images_train)):
    for i in range(len(yy)):
        for j in range(len(xx)):
            temp = (input_img[n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:]).reshape(np.square(s_patch),n_cat)
            X_train[n,i,j,:,:]=temp
            y_train[n,i,j,:] = groundtruth_generator(n,xx[i][j],yy[i][j],s_patch,s_patch)
            
###################################### create testing data ##########################################            
X_test= np.zeros([n_images_test,nx+1,ny+1,np.square(s_patch),n_cat])
y_test = np.zeros([n_images_test,nx+1,ny+1,np.square(s_patch)])


for n in tqdm(range(n_images_test)):
    for i in range(len(yy)):
        for j in range(len(xx)):
            temp = (test_img[n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:]).reshape(np.square(s_patch),n_cat)
            X_test[n,i,j,:,:]=temp
            y_test[n,i,j,:] = groundtruth_generator(n,xx[i][j],yy[i][j],s_patch,s_patch)




################################## delete the no label part in the arrays#############################
#reshape all tensors to array/matrix
X_train = X_train.reshape([n_images_train*(nx+1)*(ny+1)*np.square(s_patch),n_cat])
y_train = y_train.reshape([n_images_train*(nx+1)*(ny+1)*np.square(s_patch)])

X_test = X_test.reshape([n_images_test*(nx+1)*(ny+1)*np.square(s_patch),n_cat])
y_test = y_test.reshape([n_images_test*(nx+1)*(ny+1)*np.square(s_patch)])

print("Done reshape")


# delete no label
background_train = np.where(y_train != 3)
background_test = np.where(y_test != 3)

X_train_def = X_train[background_train]
y_train_def = y_train[background_train]

X_test_def = X_test[background_test]
y_test_def = y_test[background_test]

print("Done no label dedection")

#####################################################################################################
#####################################################################################################

# # Decision Tree
# train classifier and validate the classifier
dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(X_train_def, y_train_def)
score = dt_clf.score(X_test_def, y_test_def)

print(score)

dt_pred = dt_clf.predict(X_test_def)



conf_matrix = confusion_matrix(y_test_def,dt_pred)
print(conf_matrix)



