# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:38:35 2021

@author: chrig
"""

import numpy as np
import h5py
from scipy.stats import wilcoxon
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from mlxtend.evaluate import mcnemar_table, mcnemar

h5 = h5py.File("dataset_test.h5", 'r')

RGB = h5["RGB"]
NIR = h5["NIR"]

CLD = h5["CLD"]
GT = h5["GT"] # 0, 1 , 99

X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)
#3 x 10980 x 1... x 4 

### correct gt data ###
# first assign gt at the positions of clouds
cloud_positions = np.where(CLD_sample > 10)
GT[cloud_positions] = 2 

# second remove gt where no data is available - where the max of the input channel is zero
idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training







## evaluation of classifiers

# Prepare models and select CV method
model1 = ExtraTreesClassifier()
model2 = RandomForestClassifier()
kf = KFold(n_splits=20, random_state=42)

# Extract results for each model on the same folds
results_model1 = cross_val_score(model1, data, target, cv=kf)
results_model2 = cross_val_score(model2, data, target, cv=kf)

# Calculate p value
stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit'); p

# The correct target (class) labels
y_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# Class labels predicted by model 1
y_model1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                     0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
# Class labels predicted by model 2
y_model2 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
                     1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

# Calculate p value
tb = mcnemar_table(y_target=y_target, 
                   y_model1=y_model1, 
                   y_model2=y_model2)
chi2, p = mcnemar(ary=tb, exact=True)

print('chi-squared:', chi2)
print('p-value:', p)