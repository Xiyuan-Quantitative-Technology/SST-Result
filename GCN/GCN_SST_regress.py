# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:48:10 2020

@author: tanzheng
"""
def main():
    
    import torch
    import numpy as np
    import pickle
    import meta_X_transform as mXt
    import time
    from GCN_module import MolecularGCN, Trainer
    import matplotlib.pyplot as plt
    import padding_adj_mat as PAM
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ##########################################################
    #epoch parameters from excel
    
    epoch_No = [8 ,25 ,10 ,16 ,4 ,27 ,26 ,24 ,28 ,28 ,29 ,29]
    ##########################################################
    #fetch data
    with open('qm9_GCN_input.pkl', 'rb') as f:
        input_data = pickle.load(f)
        f.close()
    
    train_X_atom_feat,train_X_adj_list,train_y,\
                    val_X_atom_feat,val_X_adj_list,val_y,\
                      test_X_atom_feat,test_X_adj_list,test_y = input_data
    
    ##########################################################
    
    #set seed and device
    seed=1
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Python random module.
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(256)
    
    ##########################################################
    
    # first stage training
    
    #GCN parameters
    n_atom_feat = 75
    n_gcn_L1 = 64
    n_gcn_L2 = 64
    nod_mlp_L = 512
    gra_mlp_L = 512
    n_output = 1
    
    #training parameters
    lr, batch_size = 0.001, 32
    
    #GCN training
    all_train_data_preparation = []
    
    #training set option
    index_id = list(range(0,len(train_X_atom_feat)))
    np.random.shuffle(index_id)
    loader = [index_id[i:i+batch_size] for i in range(0,len(index_id),batch_size)]
    #val set shuffle option
    val_index_id = list(range(0,len(val_X_atom_feat)))
    np.random.shuffle(val_index_id)
    val_loader = [val_index_id[i:i+batch_size] for i in range(0,len(val_index_id),batch_size)]
    #test
    test_index_id = list(range(0,len(test_X_atom_feat)))
    np.random.shuffle(test_index_id)
    test_loader = [test_index_id[i:i+batch_size] for i in range(0,len(test_index_id),batch_size)]
    
    #train data preparation
    for ii in range(len(epoch_No)):
        train_y_target = train_y[:,ii]######################
        train_batch_input_all = []
        #training data prepare
        for i in loader:
            BX_atom_feat, BX_adj_list, By_prop = [train_X_atom_feat[int(x)] for x in i],\
                    [train_X_adj_list[int(x)] for x in i], np.array([train_y_target[int(x)] for x in i])
    
            #doing batch staking and padding
            BX_all_feat = np.vstack(BX_atom_feat)
            BX_adj_padmat, molecular_sizes = PAM.pad_adjacency_matrix(BX_adj_list, 0)
            
            #input and output
            BX_all_feat = torch.Tensor(BX_all_feat)
            BX_adj_padmat = torch.Tensor(BX_adj_padmat)
            
            By_prop = torch.Tensor(By_prop)
            By_prop = By_prop.reshape(By_prop.shape[0], 1)
            
            train_batch_input_all.append([BX_all_feat, BX_adj_padmat, molecular_sizes, By_prop])
        all_train_data_preparation.append(train_batch_input_all)    
            
    start_time = time.time()
    First_stage_models = []
    for i in range(len(epoch_No)):
        used_train_batch_input_all = all_train_data_preparation[i]
        # train_y_target = train_y[:,i]
        EPOCH_ = range(epoch_No[i])
        #model generation
        model =  MolecularGCN(n_atom_feat, n_gcn_L1, n_gcn_L2, nod_mlp_L, gra_mlp_L, n_output).to(device)
        trainer = Trainer(model, lr, batch_size)
        for epoch in EPOCH_:
            # start_time = time.time()
            loss_train, model = trainer.train(used_train_batch_input_all) 
        First_stage_models.append(model)
    end_time = time.time()
    print(end_time - start_time)
        
    # meta-variable generation
    y_meta_var = []
    y_train_out = []
    for ii in range(len(epoch_No)):
        predicted_ = []
        batch_ = []
        used_model = First_stage_models[ii]
        used_train_batch_input_all = all_train_data_preparation[ii]
        for i in used_train_batch_input_all:
                By_prop = i[-1].to(device)
                sub_X_all_feat = i[0].to(device)
                sub_X_adj_padmat = i[1].to(device)
                sub_molecular_sizes = i[2]
                batch_input = [sub_X_all_feat, sub_X_adj_padmat, sub_molecular_sizes]
                # batch_input = i[0:3]
                loss, predicted_output, batch_output = used_model.forward_regressor(batch_input, By_prop)
                predicted_.extend(predicted_output.cpu().detach().numpy())
                batch_.extend(batch_output.cpu().detach().numpy())
        # y_train_pred = [b for a in predicted_ for b in a]
        # y_train_true = [c for aa in batch_ for c in aa]    
        y_train_pred = np.array(predicted_).reshape(-1)
        y_train_true = np.array(batch_).reshape(-1)
        y_meta_var.append(y_train_pred)
        y_train_out.append(y_train_true)
        
    # first stage out-sample prediction
    
    all_test_data_preparation = []
    #test set shuffle option
    # test_index_id = list(range(0,len(test_X_atom_feat)))
    # np.random.shuffle(test_index_id)
    # test_loader = [test_index_id[i:i+batch_size] for i in range(0,len(test_index_id),batch_size)]
    
    #test data preparation
    for ii in range(len(epoch_No)):
        test_y_target = test_y[:,ii]##########################
        test_batch_input_all = []
        for i in test_loader:
            BX_atom_feat, BX_adj_list, By_prop = [test_X_atom_feat[int(x)] for x in i],\
                    [test_X_adj_list[int(x)] for x in i], np.array([test_y_target[int(x)] for x in i])
    
            #doing batch staking and padding
            BX_all_feat = np.vstack(BX_atom_feat)
            BX_adj_padmat, molecular_sizes = PAM.pad_adjacency_matrix(BX_adj_list, 0)
            
            #input and output
            BX_all_feat = torch.Tensor(BX_all_feat)
            BX_adj_padmat = torch.Tensor(BX_adj_padmat)
            
            By_prop = torch.Tensor(By_prop)
            By_prop = By_prop.reshape(By_prop.shape[0], 1)
            
            test_batch_input_all.append([BX_all_feat, BX_adj_padmat, molecular_sizes, By_prop])
        all_test_data_preparation.append(test_batch_input_all)
    
    
    first_pred_out_y = []
    first_true_out_y = []
    for ii in range(len(epoch_No)):
        predicted_ = []
        batch_ = []
        used_model = First_stage_models[ii]
        used_test_batch_input_all = all_test_data_preparation[ii]
        for i in used_test_batch_input_all:
                By_prop = i[-1].to(device)
                sub_test_X_all_feat = i[0].to(device)
                sub_test_X_adj_padmat = i[1].to(device)
                sub_molecular_sizes = i[2]
                batch_input = [sub_test_X_all_feat, sub_test_X_adj_padmat, sub_molecular_sizes]
                # batch_input = i[0:3]
                loss, predicted_output, batch_output = used_model.forward_regressor(batch_input, By_prop)
                predicted_.extend(predicted_output.cpu().detach().numpy())
                batch_.extend(batch_output.cpu().detach().numpy())
        y_test_pred = np.array(predicted_).reshape(-1)
        y_test_true = np.array(batch_ ).reshape(-1)
        first_pred_out_y.append(y_test_pred)
        first_true_out_y.append(y_test_true)
    
    ##########################################################
    
    # second stage training
    # meta-feature generation
    shuffle_train_X_adj_list = [train_X_adj_list[int(x)] for x in index_id]
    
    shuffle_train_X_atom_feat = [train_X_atom_feat[int(x)] for x in index_id]
    meta_X = mXt.meta_X_trans(shuffle_train_X_atom_feat, y_meta_var)
    
    #meta-training set option
    meta_index_id = list(range(0,len(meta_X)))
    meta_loader = [meta_index_id[i:i+batch_size] for i in range(0,len(meta_index_id),batch_size)]
    
    all_meta_train_data_preparation = []
    #train data preparation
    for ii in range(len(epoch_No)):
        #changed y order data
        train_y_target = y_train_out[ii]
        train_batch_input_all = []
        #training data prepare
        for i in meta_loader:
            BX_atom_feat, BX_adj_list, By_prop = [meta_X[int(x)] for x in i],\
                    [shuffle_train_X_adj_list[int(x)] for x in i], np.array([train_y_target[int(x)] for x in i])
    
            #doing batch staking and padding
            BX_all_feat = np.vstack(BX_atom_feat)
            BX_adj_padmat, molecular_sizes = PAM.pad_adjacency_matrix(BX_adj_list, 0)
            
            #input and output
            BX_all_feat = torch.Tensor(BX_all_feat)
            BX_adj_padmat = torch.Tensor(BX_adj_padmat)
            
            By_prop = torch.Tensor(By_prop)
            By_prop = By_prop.reshape(By_prop.shape[0], 1)
            
            train_batch_input_all.append([BX_all_feat, BX_adj_padmat, molecular_sizes, By_prop])
        all_meta_train_data_preparation.append(train_batch_input_all)    
    
    meta_n_atom_feat = len(epoch_No) + n_atom_feat 
    Second_stage_models = []
    Second_loss = []
    start_time = time.time()
    for i in range(len(epoch_No)):
        meta_used_train_batch_input_all = all_meta_train_data_preparation[i]
        # train_y_target = train_y[:,i]
        EPOCH_ = range(epoch_No[i])
        loss_=[]
        #model generation
        meta_model =  MolecularGCN(meta_n_atom_feat, n_gcn_L1, n_gcn_L2, nod_mlp_L, gra_mlp_L, n_output).to(device)
        meta_trainer = Trainer(meta_model, lr, batch_size)
        for epoch in EPOCH_:
            # start_time = time.time()
            loss_train, meta_model = meta_trainer.train(meta_used_train_batch_input_all) 
            loss_.append(loss_train)
        Second_stage_models.append(meta_model)
        Second_loss.append(np.array(loss_).reshape(-1))
    end_time = time.time()
    print(end_time - start_time)
    
    # second stage out-sample prediction
    # meta-feature generation
    shuffle_test_X_adj_list = [test_X_adj_list[int(x)] for x in test_index_id]
    
    shuffle_test_X_atom_feat = [test_X_atom_feat[int(x)] for x in test_index_id]
    meta_out_X = mXt.meta_X_trans(shuffle_test_X_atom_feat, first_pred_out_y)
    
    #meta-test set
    meta_test_index_id = list(range(0,len(meta_out_X)))
    meta_test_loader = [meta_test_index_id[i:i+batch_size] for i in range(0,len(meta_test_index_id),batch_size)]
    
    all_meta_test_data_preparation = []
    #test data preparation
    for ii in range(len(epoch_No)):
        test_y_target = first_true_out_y[ii]
        test_batch_input_all = []
        for i in meta_test_loader:
            BX_atom_feat, BX_adj_list, By_prop = [meta_out_X[int(x)] for x in i],\
                    [shuffle_test_X_adj_list[int(x)] for x in i], np.array([test_y_target[int(x)] for x in i])
    
            #doing batch staking and padding
            BX_all_feat = np.vstack(BX_atom_feat)
            BX_adj_padmat, molecular_sizes = PAM.pad_adjacency_matrix(BX_adj_list, 0)
            
            #input and output
            BX_all_feat = torch.Tensor(BX_all_feat)
            BX_adj_padmat = torch.Tensor(BX_adj_padmat)
            
            By_prop = torch.Tensor(By_prop)
            By_prop = By_prop.reshape(By_prop.shape[0], 1)
            
            test_batch_input_all.append([BX_all_feat, BX_adj_padmat, molecular_sizes, By_prop])
        all_meta_test_data_preparation.append(test_batch_input_all)
    
    
    second_pred_out_y = []
    y_test_out = []
    for ii in range(len(epoch_No)):
        predicted_ = []
        batch_ = []
        used_model = Second_stage_models[ii]
        meta_used_test_batch_input_all = all_meta_test_data_preparation[ii]
        for i in meta_used_test_batch_input_all:
                By_prop = i[-1].to(device)
                sub_X_all_feat = i[0].to(device)
                sub_X_adj_padmat = i[1].to(device)
                sub_molecular_sizes = i[2]
                batch_input = [sub_X_all_feat, sub_X_adj_padmat, sub_molecular_sizes]
                # batch_input = i[0:3]
                loss, predicted_output, batch_output = used_model.forward_regressor(batch_input, By_prop)
                predicted_.extend(predicted_output.cpu().detach().numpy())
                batch_.extend(batch_output.cpu().detach().numpy())
        y_test_pred = np.array(predicted_).reshape(-1)
        y_test_true = np.array(batch_ ).reshape(-1)    
        second_pred_out_y.append(y_test_pred)
        y_test_out.append(y_test_true)
        
    ##########################################################
    #RRMSE PLOT DATA
    out_prop_y = np.array(y_test_out).T
    No_samples = out_prop_y.shape[0]
    np_fir_pred_out_y = np.empty(shape=(No_samples, 0))
    np_sec_pred_out_y = np.empty(shape=(No_samples, 0))
    tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298']
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
    
    bar_width = 0.2
    index_single_task_RRMSE = np.arange(len(tasks)) # single_task_RRMSE
    index_multi_task_RRMSE= index_single_task_RRMSE + bar_width # multi_task_RRMSE
     
    
    plt.bar(index_single_task_RRMSE, height=single_task_RRMSE, width=bar_width, color='b', label='ST')
    plt.bar(index_multi_task_RRMSE, height=multi_task_RRMSE, width=bar_width, color='y', label='SST')
     
    plt.legend()
    plt.xticks(index_single_task_RRMSE + bar_width/2, tasks, rotation=45) #
    plt.title('GCN')
    plt.show()

if __name__ == '__main__':
    main()