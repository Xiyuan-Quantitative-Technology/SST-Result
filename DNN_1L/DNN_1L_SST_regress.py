# -*- coding: utf-8 -*-
"""

@author: tanzheng
"""
def main():
    import torch
    import numpy as np
    import pickle
    import time
    from one_layer_method import One_layer_meth
    import meta_X_transform as mXt
    import matplotlib.pyplot as plt
    import deepchem
    ##########################################################
    # epoch parameters
    epoch_No = [180, 350, 250, 340, 330, 370, 380, 340,	330, 320, 340, 370]
    print('1')
    ##########################################################
    
    #set seed and device
    seed=12
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Python random module.
    torch.manual_seed(seed) 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(48)
    
    ##########################################################
    
    #fetch data
    with open('ECFP_modified.pkl', 'rb') as f:
        data_list = pickle.load(f)
        f.close()
    
    tasks, train_X, train_y, test_X, test_y, val_X, val_y = data_list
    
    ##########################################################
    
    # first stage training
        
    X = torch.Tensor(train_X).to(device)
    prop_y = torch.Tensor(train_y).to(device)
    
    start_time = time.time()
    First_stage_models = []
    for i in range(len(tasks)):
        prop_y_train = prop_y[:,i].reshape(prop_y.shape[0],1)
        
        neural_nets = One_layer_meth()
        neural_nets.set_epoch(epoch_No[i])
        neural_nets.set_Net(1024, 1024, 1)
        
        model = neural_nets.one_layer_net(X, prop_y_train)
        First_stage_models.append(model)
    end_time = time.time()
    print(end_time - start_time)
    # meta-variable generation
    y_meta_var = []
    
    for i in range(len(tasks)):
        y_train_pred = First_stage_models[i](X)
        y_train_pred = y_train_pred.detach().numpy()
        
        y_meta_var.append(y_train_pred)
    
        
    # first stage out-sample prediction
    out_X = test_X
    out_prop_y = test_y
    
    out_X = torch.Tensor(out_X).to(device)
    #out_prop_y = torch.Tensor(out_prop_y).to(device)
    
    first_pred_out_y = []
    for i in range(len(tasks)):
        y_test_pred = First_stage_models[i](out_X)
        y_test_pred = y_test_pred.detach().numpy()
        
        first_pred_out_y.append(y_test_pred)
        
    
    ##########################################################
    
    # second stage training
    # meta-feature generation
    meta_X = mXt.meta_X_trans(train_X, y_meta_var)
    meta_X = torch.Tensor(meta_X).to(device)
    
    start_time = time.time()
    Second_stage_models = []
    for i in range(len(tasks)):
        prop_y_train = prop_y[:,i].reshape(prop_y.shape[0],1)
        
        neural_nets = One_layer_meth()
        neural_nets.set_epoch(epoch_No[i])
        neural_nets.set_Net(1036, 1036, 1)
        
        model = neural_nets.one_layer_net(meta_X, prop_y_train)
        Second_stage_models.append(model)
    end_time = time.time()
    print(end_time - start_time)
    
    # second stage out-sample prediction
    # meta-feature generation
    meta_out_X = mXt.meta_X_trans(test_X, first_pred_out_y)
    meta_out_X = torch.Tensor(meta_out_X).to(device)
    
    second_pred_out_y = []
    for i in range(len(tasks)):
        sec_y_test_pred = Second_stage_models[i](meta_out_X)
        sec_y_test_pred = sec_y_test_pred.detach().numpy()
        
        second_pred_out_y.append(sec_y_test_pred)
        
        
    ##########################################################
    #RRMSE PLOT DATA
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
        
    #bar figure
     
    tasks = ('mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298')
    
    bar_width = 0.2 
    index_single_task_RRMSE = np.arange(len(tasks)) # single_task_RRMSE
    index_multi_task_RRMSE= index_single_task_RRMSE + bar_width # multi_task_RRMSE
     
    
    plt.bar(index_single_task_RRMSE, height=single_task_RRMSE, width=bar_width, color='b', label='ST')
    plt.bar(index_multi_task_RRMSE, height=multi_task_RRMSE, width=bar_width, color='y', label='SST')
     
    plt.legend()
    plt.xticks(index_single_task_RRMSE + bar_width/2, tasks, rotation=45) #
    plt.title('DNN_1L')
    plt.show()

if __name__ == '__main__':
    main()

