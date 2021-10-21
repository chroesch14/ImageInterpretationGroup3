#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


# import moduels
import h5py
import numpy as np
import matplotlib.pylab as plt
import math
from PIL import Image


import torch

# # settings

# In[2]:


# access the data
# this part have to be replaced when the images from the neuronal networks are available
dset = h5py.File("dataset_test.h5","r")

#Access to the input data
RGB = dset["RGB"]
NIR = dset["NIR"]
print("Thes shape of the images is:")
print(RGB.shape)


# get size and numbers of the dataset
# number of images
n_images = RGB.shape[0]

# imagesize x_direction
sx_image = RGB.shape[1]
#imageszize y_direction
sy_image = RGB.shape[2]


# In[3]:


# set number of images in x and y direction

# number of images in x direction
nx = 15
#number of images in y direction 
ny = 15

# size of patch
# alle sind quadratisch
s_patch = 256

# numbers of categories in the pretrained network Pascal Voc --> 21
n_cat = 21


# In[4]:


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


# # Network part

# In[5]:


# import networks modules
from torchvision import models
import torchvision.transforms as T


# In[11]:


# select model
# available more models are avialable at: https://pytorch.org/vision/stable/models.html
# check if mdoel.eval() is needed
# check how to add the pretrained weights from the BigEarthNet


#FCN ResNet50
model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)

#FCN ResNet101
#model = models.segmentation.fcn_resnet101(pretrained=False, progress=True)


# In[7]:


# function to prepare the input image into the correct form (for all segementation models of torchvision the same normalistation and standard deviation can be used!)
# more information: https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
# or https://pytorch.org/vision/stable/models.html
trf = T.Compose([T.Resize(s_patch),
                 T.CenterCrop(s_patch),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])


# In[15]:


# run over each image patch and store it

# zero_vector = np.zeros([1,21])
zero_vector = np.zeros([3,1])

# laeuft im Moment noch ueber das test datenset
for img_n in range(3):
    for i in range(len(yy)):
        for j in range(len(xx)):
            temp_img = Image.fromarray(RGB[img_n,xx[i][j]:xx[i][j]+s_patch, yy[i][j]:yy[i][j]+s_patch,:3])
            inp = trf(temp_img).unsqueeze(0) #check what unsqueeze does
            out = model(inp)['out']

            # reshape the output of the network to use it in the classifier 
            # size (image_size x image_size) x 21 (all of the torchvision segmentation models are trained with the Pascal Voc dataseet --> contains 21 categories) more info: https://pytorch.org/vision/stable/models.html


###############################################################################
###############################################################################

            # TODO check if the reshape is done in the correct way
            # temp = out.detach().numpy()
            # temp1 = temp[0]
            # temp2 = temp1.reshape(np.square(s_patch),n_cat)

            # zero_vector[0][0] = xx[i][j]
            # zero_vector[0][1] = yy[i][j]
            # zero_vector[0][2] = img_n

            # temp3 = np.append(temp2,zero_vector,axis =0)

###############################################################################
###############################################################################


###############################################################################
###############################################################################            
            temp = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
            temp1 = temp.reshape(np.square(s_patch),1)
            
            zero_vector[0][0] = xx[i][j]
            zero_vector[1][0] = yy[i][j]
            zero_vector[2][0] = img_n
            
            temp3 = np.append(temp1, zero_vector, axis = 0)
###############################################################################
###############################################################################
            
            # save array
            np.save('./output_test_one_channel_resnet_50_test_data/'+str(img_n)+'_'+str(xx[i][j])+'_'+str(yy[i][j])+'.npy',temp3)




