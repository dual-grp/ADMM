import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN2(nn.Module):
    def __init__(self, input_dim = 784, mid_dim_in = 100, mid_dim_out= 100, output_dim = 10):
        super(DNN2, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim_in)
        self.fc2 = nn.Linear(mid_dim_in, mid_dim_out)
        self.fc3 = nn.Linear(mid_dim_out, output_dim)
    
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
