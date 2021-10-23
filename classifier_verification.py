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

from sklearn.model_selection import train_test_split
from sklearn import tree
#import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier


#from PIL import Image
#import random
import math

import os

from tqdm import tqdm
####################################################################################

#all_counter = 64

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
#input_img = np.concatenate([RGB[:all_counter],np.expand_dims(NIR[:all_counter], axis = -1)], axis = -1)

#add the NIR to the test images
#test_img =  np.concatenate([RGB_test[:],np.expand_dims(NIR_test[:], axis = -1)], axis = -1)

#print("Done NIR")

####################################################################################
####################################################################################

#set vairalbes 

# allgemein
# imagesize x_direction
sx_image = RGB.shape[1]
#imageszize y_direction
sy_image = RGB.shape[2]

# number of images in x direction
nx = 3
#number of images in y direction 
ny = 3

s_patch = 256 
n_cat = 4

#################### train images##########################
# number of trainng images
n_images_train = RGB.shape[0]
# number of trainng channels 
n_channels = RGB.shape[3]


#################### test images##########################
# number of test images
n_images_test = RGB_test.shape[0]
# number of channels in the test images
test_channel = RGB_test.shape[3]



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




# function to cutt out the specific part out of the RGB and NIR image and concatenate to one image

def image_cutter_train(img_number, x_ind, y_ind,win_x,win_y):
    cut_RGB = RGB[img_number, x_ind:x_ind+win_x, y_ind:y_ind+win_y, :]
    cut_NIR = NIR[img_number, x_ind:x_ind+win_x, y_ind:y_ind+win_y]
    cut = np.concatenate([cut_RGB,np.expand_dims(cut_NIR, axis = -1)], axis = -1)
    return (cut.reshape(np.square(s_patch),n_cat))
       
def image_cutter_test(img_number, x_ind, y_ind,win_x,win_y):
    cut_RGB = RGB_test[img_number, x_ind:x_ind+win_x, y_ind:y_ind+win_y, :]
    cut_NIR = NIR_test[img_number, x_ind:x_ind+win_x, y_ind:y_ind+win_y]
    cut = np.concatenate([cut_RGB,np.expand_dims(cut_NIR, axis = -1)], axis = -1)
    return (cut.reshape(np.square(s_patch),n_cat))
       

# for loop for different experiments in the baseline

steps = [15]
for step in range(3):
    nx = steps[step]
    ny = steps[step]


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
    #####################################################################################################
    
    # create arrays to train/test the classifier
    
    
    ###################################### create training data ##########################################
    
    
    
    X_train= np.zeros([n_images_train,nx+1,ny+1,np.square(s_patch),n_cat])
    y_train = np.zeros([n_images_train,nx+1,ny+1,np.square(s_patch)])
    
    
    for n in tqdm(range(n_images_train)):
        for i in range(len(yy)):
            for j in range(len(xx)):
                #temp = (input_img[n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:]).reshape(np.square(s_patch),n_cat)
                temp = image_cutter_train(n,xx[i][j],yy[i][j],s_patch,s_patch)
                X_train[n,i,j,:,:]=temp
                y_train[n,i,j,:] = groundtruth_generator(n,xx[i][j],yy[i][j],s_patch,s_patch)
                
             
       
    ###################################### create testing data ##########################################            
    
    
    X_test= np.zeros([n_images_test,nx+1,ny+1,np.square(s_patch),n_cat])
    y_test = np.zeros([n_images_test,nx+1,ny+1,np.square(s_patch)])
    
    
    for n in tqdm(range(n_images_test)):
        for i in range(len(yy)):
            for j in range(len(xx)):
                #temp = (test_img[n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:]).reshape(np.square(s_patch),n_cat)
                temp = image_cutter_test(n,xx[i][j],yy[i][j],s_patch,s_patch)
                X_test[n,i,j,:,:]=temp
                y_test[n,i,j,:] = groundtruth_generator(n,xx[i][j],yy[i][j],s_patch,s_patch)
    
    
    
    
    ################################## delete the no label part in the arrays#############################
    #reshape all tensors to array/matrix
    X_train = X_train.reshape([n_images_train*(nx+1)*(ny+1)*np.square(s_patch),n_cat])
    y_train = y_train.reshape([n_images_train*(nx+1)*(ny+1)*np.square(s_patch)])
    
    
    X_test = X_test.reshape([n_images_test*(nx+1)*(ny+1)*np.square(s_patch),n_cat])
    y_test = y_test.reshape([n_images_test*(nx+1)*(ny+1)*np.square(s_patch)])
    
    
    print("Done reshape")
    
    
    #####################################################################################################
    #####################################################################################################
    
    # add here code to include the information of the neuronal network 
    # path = './resnet_50_donnerstag_train/'
    # files = os.listdir(path)
    # #count how many files there are
    # num_files = len(files)
    
    # X_temp_train = np.zeros([n_images_train*(nx+1)*(ny+1)*np.square(s_patch),1])
    
    # for i in range(n_images_train):
    #      temp = np.load(path+str(files[i]))
    #      temp = np.expand_dims(temp, axis = 1)
    #      X_temp_train[i*256*256:i*256*256+256*256] = temp[:-3]
        
    # X_train = np.append(X_train,X_temp_train,axis = 1)
    
    
    
    # #### check if that is necessary
    # path = './resnet_50_donnerstag_test/'
    # files = os.listdir(path)
    # #count how many files there are
    # num_files = len(files)
    
    # X_temp_test = np.zeros([3*(nx+1)*(ny+1)*np.square(s_patch),1])
    
    # for i in range(n_images_test):
    #      temp = np.load(path+str(files[i]))
    #      temp = np.expand_dims(temp, axis = 1)
    #      X_temp_test[i*256*256:i*256*256+256*256] = temp[:-3]
        
    # X_test = np.append(X_test,X_temp_test,axis = 1)
    
    
    #####################################################################################################
    #####################################################################################################
    
    
    # delete no label
    no_label_train = np.where(y_train != 3)
    no_label_test = np.where(y_test != 3)
    
    X_train_def = X_train[no_label_train]
    y_train_def = y_train[no_label_train]
    
    
    X_test_def = X_test[no_label_test]
    y_test_def = y_test[no_label_test]
    
    print("Done no label dedection")
    
    #####################################################################################################
    #####################################################################################################
    
    # # Decision Tree
    # train classifier and validate the classifier
    dt_clf = tree.DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_train_def, y_train_def)
    score = dt_clf.score(X_test_def, y_test_def)
    
    print("Results from test data set: ")
    
    print("Test score: " +str(score))
    
    dt_pred = dt_clf.predict(X_test_def)
    
    conf_matrix = confusion_matrix(y_test_def,dt_pred)
    print("confusion matrix:")
    print(conf_matrix)
    
    
    
    #####################################################################################################
    #####################################################################################################
    # results test
    n = 3
    
    sum_matrix = np.sum(conf_matrix)
    sum_diag = sum(conf_matrix[i][i] for i in range(n))
    
    # overall accuracy
    accuracy = sum_diag/sum_matrix
    print("Overall accuracy testing: "+str(accuracy))
    
    # background
    precision_background = conf_matrix[0][0]/np.sum(np.array(conf_matrix)[:,0])
    recall_background = conf_matrix[0][0]/np.sum(np.array(conf_matrix)[0,:])
    f_background = 2/(1/recall_background+1/precision_background)
    print("Precision background: "+str(precision_background))
    print("Recall background: "+str(recall_background))
    print("F1 background: "+str(f_background))
    
    # palm oil trees
    precision_palm = conf_matrix[1][1]/np.sum(np.array(conf_matrix)[:,1])
    recall_palm = conf_matrix[1][1]/np.sum(np.array(conf_matrix)[1,:])
    f_palm = 2/(1/recall_palm+1/precision_palm)
    print("Precision palm oil: "+str(precision_palm))
    print("Recall palm oil: "+str(recall_palm))
    print("F1 palm oil: "+str(f_palm))
    
    
    # clouds
    precision_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[:,2])
    recall_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[2,:])
    f_clouds = 2/(1/recall_clouds+1/precision_clouds)
    
    print("Precision clouds: "+str(precision_clouds))
    print("Recall clouds: "+str(recall_clouds))
    print("F1 palm clouds: "+str(f_clouds))
    
