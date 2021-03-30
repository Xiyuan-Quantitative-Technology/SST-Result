# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:17:54 2020

@author: tanzheng
"""

import numpy as np

def meta_X_trans(X, pred_y):
    meta_X = X
    pred_y = np.array(pred_y).T
    row, col = pred_y.shape[0], pred_y.shape[1]
    trans_meta_X = []
    for i in range(row):
        sub_meta_X = meta_X[i]
        sub_pred_y = np.tile(pred_y[i,:].T.reshape(1, col),(sub_meta_X.shape[0],1))
        sub_stack_meta_X = np.hstack((sub_meta_X, sub_pred_y))
        trans_meta_X.append(sub_stack_meta_X)  
        
    return trans_meta_X

