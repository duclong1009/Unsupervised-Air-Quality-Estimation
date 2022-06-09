from builtins import breakpoint
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft, hid_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=hid_ft)
        self.linear = nn.Linear(hid_ft, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, h, x, x_c):
        
        ret1 = self.relu(self.fc(h, x)) 
        ret1 = self.linear(ret1)
        ret2 = self.relu(self.fc(h, x_c))
        ret2 = self.linear(ret2)
        ret = torch.cat((ret1, ret2), 2)
        return self.sigmoid(ret)

