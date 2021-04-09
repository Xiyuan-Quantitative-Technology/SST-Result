# -*- coding: utf-8 -*-
"""

@author: tanzheng
"""
def main():
    import pickle
    import XGBregress
    import matplotlib.pyplot as plt
    import meta_X_transform as mXt
    import numpy as np
    import time
    
    #fetch data
    with open('ECFP_modified.pkl', 'rb') as f:
        data_list = pickle.load(f)
        f.close()
    
    tasks, train_X, train_y, test_X, test_y, val_X, val_y = data_list
    #########################################################################
    # first stage training
    X = train_X
    prop_y = train_y
    
    start_time = time.time()
    First_stage_models = [0]*len(tasks)
    for i in range(len(tasks)):
        model = XGBregress.prop_models[i]
        model.fit(X, prop_y[:,i])
        First_stage_models[i] = model 
    end_time = time.time()
    print(end_time - start_time)
    #########################################################################
    # meta-variable generation
    y_meta_var = [0]*len(tasks)
    
    for i in range(len(tasks)):
        y_meta_var[i] = First_stage_models[i].predict(X)
        
    
    ################################################################
    # first stage out-sample prediction
    out_X = test_X
    out_prop_y = test_y
    
    first_pred_out_y = []
    for i in range(len(tasks)):
        first_pred_out_y.append(First_stage_models[i].predict(out_X))
    
    
    
    ################################################################
    # second stage training
    # meta-feature generation
    meta_X = mXt.meta_X_trans(X, y_meta_var)
    
    start_time = time.time()
    Second_stage_models = [0]*len(tasks)
    for i in range(len(tasks)):
        model = XGBregress.prop_models[i]
        model.fit(meta_X, prop_y[:,i])
        Second_stage_models[i] = model 
    end_time = time.time()
    print(end_time - start_time)
    
    # meta-feature generation
    meta_out_X = mXt.meta_X_trans(out_X, first_pred_out_y)
    
    second_pred_out_y = []
    for i in range(len(tasks)):
        second_pred_out_y.append(Second_stage_models[i].predict(meta_out_X))
    
    ################################################################
    #RRMSE PLOT DATA
    No_samples = out_prop_y.shape[0]
    np_fir_pred_out_y = np.empty(shape=(No_samples, 0))
    np_sec_pred_out_y = np.empty(shape=(No_samples, 0))
    
    for i in range(len(first_pred_out_y)):
        np_fir_pred_out_y = np.hstack((np_fir_pred_out_y, first_pred_out_y[i].reshape(No_samples,1)))
        np_sec_pred_out_y = np.hstack((np_sec_pred_out_y, second_pred_out_y[i].reshape(No_samples,1)))
    
    
    
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
        
    #bar figure
     
    tasks = ('mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298')
    
    bar_width = 0.2
    index_single_task_RRMSE = np.arange(len(tasks)) # single_task_RRMSE
    index_multi_task_RRMSE= index_single_task_RRMSE + bar_width # multi_task_RRMSE
     
    
    plt.bar(index_single_task_RRMSE, height=single_task_RRMSE, width=bar_width, color='b', label='ST')
    plt.bar(index_multi_task_RRMSE, height=multi_task_RRMSE, width=bar_width, color='y', label='SST')
     
    plt.legend()
    plt.xticks(index_single_task_RRMSE + bar_width/2, tasks, rotation=45) #
    plt.title('XGBoost')
    plt.show()

if __name__ == '__main__':
    main()
