# -*- coding: utf-8 -*-

class Three_layer_meth:
    def __init__(self):
#       self.batchsize=512
#		self.USE_CUDA=False
        self.n_feature = 20

        self.n_hidden_1 = 1024

        self.n_hidden_2 = 512

        self.n_hidden_3= 256
        
        self.n_output=1
        
        self.lr = 0.001
	
        self.epoch = 100

#    def cuda_check(self):
#		import torch

#		if torch.cuda.is_available():
#			self.USE_CUDA=True
#		self.USE_CUDA=False
    def set_Net(self,n_feature,n_hidden_1,n_hidden_2,n_hidden_3,n_output):
        self.n_feature = int(n_feature)
        self.n_hidden_1 = int(n_hidden_1) 
        self.n_hidden_2 = int(n_hidden_2)
        self.n_hidden_3= int(n_hidden_3)
        self.n_output=int(n_output)
        
    def set_epoch(self,epoch):
        self.epoch = int(epoch)
        
    def set_lr(self,lr):
        self.lr = lr   
    
    def three_layer_net(self,X_data,y_data):
        import torch
        import torch.nn as nn
        from three_layer_module import Net
        
        model = Net(self.n_feature,self.n_hidden_1,self.n_hidden_2,self.n_hidden_3,self.n_output)
    
        optimizers = torch.optim.Adam(model.parameters(),lr=self.lr,betas=(0.9,0.99))

        loss_func = nn.MSELoss()

        # losses = []
        for epoch_ in range(self.epoch):  

            output = model(X_data) # get output for every net
            loss = loss_func(output,y_data) # compute loss for every net
            optimizers.zero_grad() # claer gradients for net train
            loss.backward() # backpropagation, compute gradients
            optimizers.step() # apply gradients

            
        return model
    
