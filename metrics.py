# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:22:29 2021

@author: chrig
"""
import numpy as np

n = 3
conf_matrix = [[10, 2, 2], [3, 9, 3],[4,4,8]]
sum_matrix = np.sum(conf_matrix)
sum_diag = sum(conf_matrix[i][i] for i in range(n))

# overall accuracy
accuracy = sum_diag/sum_matrix

# background
precision_background = conf_matrix[0][0]/np.sum(np.array(conf_matrix)[:,0])
recall_background = conf_matrix[0][0]/np.sum(np.array(conf_matrix)[0,:])
f_background = 2/(1/recall_background+1/precision_background)

# palm oil trees
precision_palm = conf_matrix[1][1]/np.sum(np.array(conf_matrix)[:,1])
recall_palm = conf_matrix[1][1]/np.sum(np.array(conf_matrix)[1,:])
f_palm = 2/(1/recall_palm+1/precision_palm)

# clouds
precision_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[:,2])
recall_clouds = conf_matrix[2][2]/np.sum(np.array(conf_matrix)[2,:])
f_clouds = 2/(1/recall_clouds+1/precision_clouds)