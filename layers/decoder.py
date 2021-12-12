import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.gru = nn.GRU(in_ft, in_ft, batch_first=True)
        self.cnn = nn.Conv1d(
            in_channels=in_ft, out_channels=128, kernel_size=5, padding=2, stride=1
        )
        self.linear = nn.Linear(in_features=128, out_features=64)
        self.linear2 = nn.Linear(64, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, h, l):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x (n1-1) x latent_dim
        l.shape = (n1-1) x 1
        """
        x_ = torch.cat((x, h), dim=-1)  # batch_size x nodes x hidden feat
        x_ = torch.transpose(x_, 1, 2)
        ret = self.cnn(x_)
        ret = torch.bmm(ret, l)
        ret = torch.transpose(ret, 1, 2)
        # ret = ret.reshape(ret.shape[0], -1)
        ret = self.linear(ret)
        ret = self.relu(ret)
        ret = self.linear2(ret)
        ret = self.relu(ret)
        return ret
