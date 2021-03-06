# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:00:25 2021

@author: paimhof
"""
####################################################################################
####################################################################################
#%%
# import 
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import math
import os
from tqdm import tqdm
#%%
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

#%%
#set vairalbes 

# set paths
# path where the files of the networkpart are 
path = './resnet_8_8_train/'
# path where the testing images are
path_test = './resnet_8_8_test/'

# allgemein
# imagesize x_direction
sx_image = RGB.shape[1]
#imageszize y_direction
sy_image = RGB.shape[2]

# number of images in x direction
nx = 7
#number of images in y direction 
ny = 7

s_patch = 256 
n_cat = 4

# train images
# number of trainng images
n_images_train = RGB.shape[0]
# number of trainng channels 
n_channels = RGB.shape[3]

# test images
# number of test images
n_images_test = RGB_test.shape[0]
# number of channels in the test images
test_channel = RGB_test.shape[3]

#%%

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
       
#%%

#####################################################################################################
# for loop for different experiments in the baseline
# depth_steps = [10,20,30,40,None]
# for steps in range(5):
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

#%%
# create arrays to train/validate/test the classifier

#Access the data from the neuronal network part

files = os.listdir(path)
#count how many files there are
num_files = len(files)

#seperate the data into training data and validation data
X_train_n, X_val_n, y_train_n, y_val_n = train_test_split(np.arange(num_files),np.arange(num_files),test_size=0.3,random_state=0)
      
         
X_train = np.zeros([len(X_train_n)*np.square(s_patch),n_cat+1])
y_train = np.zeros([len(y_train_n)*np.square(s_patch)])
step_size = np.square(s_patch)

for i in tqdm(range(1,len(X_train_n)+1)):
    temp = np.load(path+str(files[X_train_n[i-1]]))
    X_train[i*step_size-step_size:i*step_size,:4] = image_cutter_train(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)
    X_train[i*step_size-step_size:i*step_size,-1] = temp[:-3]
    y_train[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)
 
X_val = np.zeros([len(X_val_n)*np.square(s_patch),n_cat+1])
y_val = np.zeros([len(y_val_n)*np.square(s_patch)])


for i in tqdm(range(1,len(X_val_n)+1)):
    temp = np.load(path+str(files[X_val_n[i-1]]))
    X_val[i*step_size-step_size:i*step_size,:4] = image_cutter_train(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)
    X_val[i*step_size-step_size:i*step_size,-1] = temp[:-3]
    y_val[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)

###################################### create testing data ##########################################            

files = os.listdir(path_test)
#count how many files there are
num_files = len(files)

X_test = np.zeros([len(np.arange(num_files))*np.square(s_patch),n_cat+1])
y_test = np.zeros([len(np.arange(num_files))*np.square(s_patch)])

for i in tqdm(range(1,len(np.arange(num_files))+1)):
    temp = np.load(path_test+str(files[i-1]))
    X_test[i*step_size-step_size:i*step_size,:4] = image_cutter_train(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)
    X_test[i*step_size-step_size:i*step_size,-1] = temp[:-3]
    y_test[i*step_size-step_size:i*step_size] = groundtruth_generator(temp[-1].astype(np.int64),temp[-3].astype(np.int64),temp[-2].astype(np.int64),s_patch,s_patch)
#%%
# delete no label
no_label_train = np.where(y_train != 3)
no_label_val = np.where(y_val != 3)
no_label_test = np.where(y_test != 3)

X_train_def = X_train[no_label_train]
y_train_def = y_train[no_label_train]

X_val_def = X_val[no_label_val]
y_val_def = y_val[no_label_val]

X_test_def = X_test[no_label_test]
y_test_def = y_test[no_label_test]

print("Done no label dedection")
#%%
# Decision Tree training
dt_clf = tree.DecisionTreeClassifier(max_depth = 10)
dt_clf.fit(X_val_def, y_val_def)
score = dt_clf.score(X_val_def, y_val_def)

print("Results from validation data set: ")
print("Test score: " +str(score))

# make predictions
dt_pred = dt_clf.predict(X_val_def)
# confusion matrix
conf_matrix = confusion_matrix(y_val_def,dt_pred)
print("confusion matrix:")
print(conf_matrix)


#%%
# results validation
n = 3

sum_matrix = np.sum(conf_matrix)
sum_diag = sum(conf_matrix[i][i] for i in range(n))

# overall accuracy
accuracy = sum_diag/sum_matrix
print("Overall accuracy validation: "+str(accuracy))

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

#%%
# results test
print("Results from validation data set: ")
dt_pred_test = dt_clf.predict(X_test_def)

conf_matrix_test = confusion_matrix(y_test_def,dt_pred_test)
print("confusion matrix: ")
print(conf_matrix_test)

n = 3

sum_matrix_test = np.sum(conf_matrix_test)
sum_diag_test = sum(conf_matrix_test[i][i] for i in range(n))

# overall accuracy
accuracy_test = sum_diag_test/sum_matrix_test
print("Overall accuracy test data: "+str(accuracy_test))

# background
precision_background_test = conf_matrix_test[0][0]/np.sum(np.array(conf_matrix_test)[:,0])
recall_background_test = conf_matrix_test[0][0]/np.sum(np.array(conf_matrix_test)[0,:])
f_background_test = 2/(1/recall_background_test+1/precision_background_test)
print("Precision background: "+str(precision_background_test))
print("Recall background: "+str(recall_background_test))
print("F1 background: "+str(f_background_test))

# palm oil trees
precision_palm_test = conf_matrix_test[1][1]/np.sum(np.array(conf_matrix_test)[:,1])
recall_palm_test = conf_matrix_test[1][1]/np.sum(np.array(conf_matrix_test)[1,:])
f_palm_test = 2/(1/recall_palm_test+1/precision_palm_test)
print("Precision palm oil: "+str(precision_palm_test))
print("Recall palm oil: "+str(recall_palm_test))
print("F1 palm oil: "+str(f_palm_test))


# clouds
precision_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[:,2])
recall_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[2,:])
f_clouds = 2/(1/recall_clouds+1/precision_clouds)

print("Precision clouds: "+str(precision_clouds))
print("Recall clouds: "+str(recall_clouds))
print("F1 palm clouds: "+str(f_clouds))