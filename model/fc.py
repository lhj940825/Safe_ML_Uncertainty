import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_dim, dprop):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.dprop = dprop

        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = F.dropout(out, p=self.dprop, training=True, inplace=True)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)

        return out

class FC2(nn.Module):
    def __init__(self, input_dim, dprop):
        super(FC2, self).__init__()
        self.input_dim = input_dim
        self.dprop = dprop

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = F.dropout(out, p=self.dprop, training=True, inplace=True)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)

        return out