#!/usr/bin/env python
# coding: utf-8

#%%
# import
import h5py
import numpy as np
import math
from PIL import Image
#import torch
from torchvision import models
import torchvision.transforms as T
from tqdm import tqdm
#%%
# access the data
dset = h5py.File("dataset_train.h5","r")

#Access to the input data
RGB = dset["RGB"]
NIR = dset["NIR"]
print("The shape of the images is:")
print(RGB.shape)

#add the NIR to the training images and remove the blue channel
input_img = np.concatenate([RGB[:,:,:,:2],np.expand_dims(NIR[:], axis = -1)], axis = -1)
# only use the RGB information 
# input_img = RGB
biggest_NIR = np.max(NIR[:]) # to normalize the NIR layer
print("Done NIR")
#%%
# settings
# get size and numbers of the dataset
# number of images
n_images = RGB.shape[0]
# imagesize x_direction
sx_image = RGB.shape[1]
#imageszize y_direction
sy_image = RGB.shape[2]

# number of images in x direction
nx = 7
#number of images in y direction 
ny = 7

# size of patch
s_patch = 256

# numbers of categories in the pretrained network Pascal Voc --> 21
n_cat = 21

#%%
# create meshgrid

# compute the coordinates of all top left pixels of each patch
# is for each image the same

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
# Network part

#FCN ResNet50
model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)


# function to prepare the input image into the correct form (for all segementation models of torchvision the same normalistation and standard deviation can be used!)
# more information: https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
# or https://pytorch.org/vision/stable/models.html
trf = T.Compose([T.Resize(s_patch),
                 T.CenterCrop(s_patch),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])

#%%
# run over each image patch and store it on the computer

# zero_vector = np.zeros([1,21])
zero_vector = np.zeros([3])

# iterate over images and meshgrid
for img_n in tqdm(range(n_images)):
    for i in range(len(yy)):
        for j in range(len(xx)):
            
            temp_img = input_img[img_n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:3]
            
            # only use when the NIR layer is used
            norm_NIR = temp_img[:,:,2]/biggest_NIR*255 #normalize NIR values to grey values
            
            temp_img[:,:,2] = norm_NIR
            
            
            temp_img2 = Image.fromarray((temp_img).astype(np.uint8))
            
            inp = trf(temp_img2).unsqueeze(0) #check what unsqueeze does
            out = model(inp)['out']


            temp1 = np.amax(out.squeeze().detach().numpy().reshape([np.square(256),21]),1)
            
            # add metadata to vector 
            zero_vector[0] = xx[i][j]
            zero_vector[1] = yy[i][j]
            zero_vector[2] = img_n
            
            temp3 = np.append(temp1, zero_vector, axis = 0)

            
            # save array
            np.save('./resnet_8_8_train_RGB/'+str(img_n)+'_'+str(xx[i][j])+'_'+str(yy[i][j])+'.npy',temp3)




