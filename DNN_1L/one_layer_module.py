import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_features,n_hidden_1,n_output):
        super(Net, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_features,n_hidden_1), 
                                   nn.ReLU(),
                                   nn.Linear(n_hidden_1, n_output))
        
    def forward(self, input1):
        output = self.model(input1)
        return output

