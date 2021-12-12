import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=6)
        self.linear = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        for m in self.modules():
            self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h, x, x_c):
        # import pdb; pdb.set_trace()
        ret1 = self.relu(self.fc(h, x))
        ret1 = self.linear(ret1)
        ret1 = self.sigmoid(ret1)
        ret2 = self.relu(self.fc(h, x_c))
        ret2 = self.linear(ret2)
        ret2 = self.sigmoid(ret2)
        ret = torch.cat((ret1, ret2), 1)
        return ret
