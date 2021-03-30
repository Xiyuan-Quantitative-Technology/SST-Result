# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:17:54 2020

@author: tanzheng
"""

import numpy as np

def meta_X_trans(X, pred_y):
    meta_X = X
    
    for i in range(len(pred_y)):
        meta_X = np.hstack((meta_X, pred_y[i]))
    
    
    return meta_X

