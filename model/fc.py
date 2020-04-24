import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_dim):
        super(FC, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)
        # self.log_softmax = nn.LogSoftmax(dim=1)

        # self.loss_fn = nn.NLLLoss()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        # out = self.log_softmax(out)
        return out