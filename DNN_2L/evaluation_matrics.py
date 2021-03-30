# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:39:34 2020

@author: tanzheng
"""

import pickle
import numpy as np


with open('DNN_2L_SST_predict.pkl', 'rb') as f:
    MT_predict_result = pickle.load(f)
    f.close()
    
first_pred_out_y, second_pred_out_y, out_prop_y, tasks = MT_predict_result

No_samples = out_prop_y.shape[0]
np_fir_pred_out_y = np.empty(shape=(No_samples, 0))
np_sec_pred_out_y = np.empty(shape=(No_samples, 0))

for i in range(len(first_pred_out_y)):
    np_fir_pred_out_y = np.hstack((np_fir_pred_out_y, first_pred_out_y[i]))
    np_sec_pred_out_y = np.hstack((np_sec_pred_out_y, second_pred_out_y[i]))



# target RRMSE
# single target
single_task_RRMSE = []

for i in range(len(tasks)):
    temp_ST_RRMSE = sum(np.square(out_prop_y[:,i]-np_fir_pred_out_y[:,i])) / sum(np.square(out_prop_y[:,i]-np.mean(out_prop_y[:,i])))
    temp_ST_RRMSE = np.sqrt(temp_ST_RRMSE)
    single_task_RRMSE.append(temp_ST_RRMSE)
    

# multi target
multi_task_RRMSE = []

for i in range(len(tasks)):
    temp_MT_RRMSE = sum(np.square(out_prop_y[:,i]-np_sec_pred_out_y[:,i])) / sum(np.square(out_prop_y[:,i]-np.mean(out_prop_y[:,i])))
    temp_MT_RRMSE = np.sqrt(temp_MT_RRMSE)
    multi_task_RRMSE.append(temp_MT_RRMSE)
    
