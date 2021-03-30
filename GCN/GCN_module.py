# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:00:39 2020

@author: 86178
"""

import numpy as np
import padding_adj_mat as PAM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MolecularGCN(nn.Module):
    def __init__(self, n_atom_feat, n_gcn_L1, n_gcn_L2, nod_mlp_L, gra_mlp_L, n_output):
        super(MolecularGCN, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gcn_W1 = torch.randn((n_atom_feat, n_gcn_L1), requires_grad=True).to(self.device)
        self.gcn_W2 = torch.randn((n_gcn_L1, n_gcn_L2), requires_grad=True).to(self.device)
        
        self.gcn_input_batnorm = nn.BatchNorm1d(n_atom_feat).to(self.device)
        # self.gcn_input_batnorm.to(self.device)
        
        self.gcn_output_batnorm = nn.BatchNorm1d(n_gcn_L2).to(self.device)
        # self.gcn_output_batnorm.to(self.device)
        
        self.node_mlp_module = nn.Sequential(nn.Linear(n_gcn_L2, nod_mlp_L),
                                             nn.BatchNorm1d(nod_mlp_L),
                                             nn.Softmax(dim=1)).to(self.device)
        # self.node_mlp_module.to(self.device)
        
        self.pool_batnorm = nn.BatchNorm1d(nod_mlp_L).to(self.device)
        # self.pool_batnorm.to(self.device)
        
        self.graph_mlp_module = nn.Sequential(nn.Linear(nod_mlp_L, gra_mlp_L), 
                                              nn.ReLU(),
                                              nn.Linear(gra_mlp_L, int(gra_mlp_L/2)),
                                              nn.ReLU(),
                                              nn.Linear(int(gra_mlp_L/2), n_output)).to(self.device)
        # self.graph_mlp_module.to(self.device)
        
    def sum_(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean_(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)
        
    
    def gcn(self, X_all_feat, X_adj_padmat):
        #input BN
        #X_all_feat = self.gcn_input_batnorm(X_all_feat)
        
        gcn_layer_1 = F.relu(X_adj_padmat.mm(X_all_feat).mm(self.gcn_W1)).to(self.device)
        gcn_layer_2 = F.relu(X_adj_padmat.mm(gcn_layer_1).mm(self.gcn_W2)).to(self.device)
        
        #output BN
        gcn_layer_2 = self.gcn_output_batnorm(gcn_layer_2).to(self.device)
        
        return gcn_layer_2
    
    
    def node_mlp(self, gcn_layer):
        """mlp transformation for gcn layer on each node"""
        node_mlp_layer = self.node_mlp_module(gcn_layer)
        return node_mlp_layer
    
    
    def pooling(self, node_mlp_layer, molecular_sizes):
        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum_(node_mlp_layer, molecular_sizes)
        '''normalization'''
        molecular_vectors = self.pool_batnorm(molecular_vectors).to(self.device)
        return molecular_vectors
        
    
    def graph_mlp(self, molecular_vectors):
        """regressor based on multilayer perceptron."""
        outputs = self.graph_mlp_module(molecular_vectors)
        return outputs
    
    
    def forward_regressor(self, batch_input, batch_output):
        X_all_feat, X_adj_padmat, molecular_sizes = batch_input
        
        gcn_layer = self.gcn(X_all_feat, X_adj_padmat)
        node_mlp_layer = self.node_mlp(gcn_layer)
        molecular_vectors = self.pooling(node_mlp_layer, molecular_sizes)
        
        predicted_output = self.graph_mlp(molecular_vectors)
        loss = F.mse_loss(predicted_output, batch_output).cuda()
        # loss = F.l1_loss(predicted_output, batch_output).cuda()
        
        return loss, predicted_output, batch_output
    
    
    

class Trainer():
    def __init__(self, model, lr, batch_size):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        # self.model.to(self.device)
        self.lr = lr
        self.bs = batch_size
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.99))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train(self, train_batch_input):
        loss_total = 0
        for i in train_batch_input:
            sub_X_all_feat = i[0].to(self.device)
            sub_X_adj_padmat = i[1].to(self.device)
            sub_molecular_sizes = i[2]
            batch_input = [sub_X_all_feat, sub_X_adj_padmat, sub_molecular_sizes]
            By_prop = i[-1].to(self.device)
            # loss = self.model.forward_regressor(batch_input, By_prop)
            loss, P, B = self.model.forward_regressor(batch_input, By_prop)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
                
            #print(i/self.bs+1, 'batch finishes')
            
        return loss_total, self.model
    
                                                
                                                
        
    