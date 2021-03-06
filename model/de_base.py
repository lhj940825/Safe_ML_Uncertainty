import torch
import torch.nn as nn
import torch.nn.functional as F

class DE_base(nn.Module):
    def __init__(self, input_dim, init_factor=1.0):
        super(DE_base, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

        self.loss_fn = torch.nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=init_factor)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)

        return out