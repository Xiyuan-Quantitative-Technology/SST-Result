# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:17:54 2020

@author: tanzheng
"""

import numpy as np

def meta_X_trans(X, pred_y):
    trans_y_meta_var = np.array(pred_y)
    trans_y_meta_var = trans_y_meta_var.T

    meta_X = np.hstack((X,trans_y_meta_var))
    
    return meta_X

