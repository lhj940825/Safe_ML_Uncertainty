'''
 * User: Hojun Lim
 * Date: 2020-06-05
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class pu_fc(nn.Module):

    def __init__(self, input_dim):
        super(pu_fc, self).__init__()
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 2) # one unit for predicting mean and the other for standard deviation
        self.loss_fn = custom_NLL()

        self.bias = nn.Parameter(torch.rand(1, 1))
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        #out[:, 1] = out[:, 1] + self.bias
        out[:,1] = torch.exp(out[:,1])
        # print('bias: ', self.bias)

        return out

class pu_fc2(nn.Module):

    def __init__(self, input_dim):
        super(pu_fc2, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 2) # one unit for predicting mean and the other for standard deviation
        self.loss_fn = custom_NLL()

        self.bias = nn.Parameter(torch.rand(1, 1))

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        #out[:, 1] = out[:, 1] + self.bias
        out[:,1] = torch.exp(out[:,1])

        return out


class custom_NLL():
    def __init__(self):
        pass

    def __call__(self, label, mean: torch.autograd.Variable, output2: torch.autograd.Variable):
        #output2 = output2 + bias
        label = label.view(-1)
        std = output2 # exp has already been applied in forward path of the network.
        #std = torch.exp(output2)
        var = torch.pow(std,2)

        NLL = torch.log(var)*0.5 + torch.div(torch.pow((label-mean),2), 2*var)
        NLL = torch.clamp(NLL, min = -100) # cap NLL values at -100
        NLL = torch.mean(NLL) # average the computed NLL

        return NLL
"""
def custom_NLL(label, mean: torch.autograd.Variable, std: torch.autograd.Variable):
    var = torch.pow(std,2)
    NLL = torch.log(var)*0.5 + torch.div(torch.pow((label-mean),2), 2*var)
    NLL = torch.clamp(NLL, -100) # cap NLL values at -100

    return NLL
    
"""
