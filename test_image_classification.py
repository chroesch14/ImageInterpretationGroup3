# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:48:54 2021

@author: Pascal
"""

import numpy as np
import matplotlib.pylab as plt

import h5py

#Gain access to the data.
#Note: This does *not* load the entire data set into memory.
dset = h5py.File("dataset_test.h5","r")

#Access to the input data
RGB = dset["RGB"]
NIR = dset["NIR"]

# concatenting the RGB and NIR channel

input_image = np.concatenate([RGB,np.expand_dims(NIR,axis=-1)],axis=-1)
print(np.shape(input_image))

