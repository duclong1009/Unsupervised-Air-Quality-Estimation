from builtins import breakpoint
from operator import imod
import torch
import torch.nn as nn
import torch
from src.layers.attention import AttentionLSTM
from src.layers.temporal_gcn import TemporalGCN, GCN

class BaseEncoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(BaseEncoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.gcn = GCN(hid_ft1, hid_ft2, "relu")
        self.gcn2 = GCN(hid_ft2, out_ft, "relu")
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.fc(x)
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

class Encoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.gcn = GCN(hid_ft1, hid_ft2, "relu")
        self.gcn2 = GCN(hid_ft2, out_ft, "relu")
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.fc(x)
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.fc2 = nn.Linear(hid_ft1, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # print(x.shape)
        # breakpoint()
        # breakpoint()
        x = self.relu(self.fc(x)) 
        x = self.relu(x.unsqueeze(0)) # # 1, 12, 19, 200
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1) # 1, 19, 200 
            h = self.rnn_gcn(x_i, adj, h)
        return h

class WoGCN_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(WoGCN_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(20 * hid_ft1,20 * hid_ft1, batch_first=False, num_layers=1)
        self.fc2 = nn.Linear(hid_ft1, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # breakpoint()
        x = self.relu(self.fc(x)) 
        raw_shape = x.shape
        x = torch.reshape(x,(raw_shape[0],1,-1))
        x, h = self.rnn(x)
        x = torch.reshape(x,(raw_shape[0],raw_shape[1],raw_shape[2]))
        x = x[-1]
        x = self.relu(x.unsqueeze(0))
        x = self.fc2(x)
        return x


class GCN_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(GCN_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(13 * hid_ft1,13 * hid_ft1, batch_first=False, num_layers=1)
        self.gcn = GCN(hid_ft1, hid_ft2, act)
        self.gcn2 = GCN(hid_ft2, out_ft, act)
        self.fc2 = nn.Linear(hid_ft1, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x)) 
        x = x[-1]
        x = self.relu(x.unsqueeze(0))
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

class InterpolateAttentionEncoder(nn.Module):
    def __init__(self,in_ft, hid_ft1, hid_ft2, out_ft, act="relu"): #in_ft = 28
        super(InterpolateAttentionEncoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(hid_ft1, hid_ft1, batch_first=False, num_layers=1)
        self.attn = AttentionLSTM(hid_ft1, 120, hid_ft1, 12, 0.1)
        self.gcn = GCN(hid_ft1, hid_ft2, act)
        self.gcn2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()

    def forward(self, x_inp, adj, l):
        l_ = l / l.sum()

        x = self.relu(self.fc(x_inp))
        x,h = self.rnn(x)
        
        x = self.attn(x)
        x = self.relu(x.unsqueeze(0))

        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x, h