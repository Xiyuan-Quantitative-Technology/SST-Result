# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:18:19 2020

@author: tanzheng
"""

from xgboost import XGBRegressor
import xlrd

#xgboost

n_estimators_v = [350, 100, 390, 390, 390, 200,	390, 60, 120, 120, 120, 110]
max_depth_v = [10, 8, 10, 10, 10, 3, 3, 3, 3, 3, 3, 3]
learning_rate_v = [0.2, 0.1, 0.2, 0.2, 0.15, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]

# build property models
prop_models = []

for i in range(len(n_estimators_v)):
    model = XGBRegressor(n_estimators = n_estimators_v[i],
                     max_depth = max_depth_v[i],
                     learning_rate = learning_rate_v[i],
                     objective = 'reg:squarederror',
                     booster = 'gbtree',
                     n_jobs = -1,
                     gamma = 0,
                     random_state = 0,
                     importance_type = 'gain')
    
    prop_models.append(model)


