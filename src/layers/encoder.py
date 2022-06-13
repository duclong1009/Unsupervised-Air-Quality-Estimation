# from builtins import breakpoint
# from builtins import breakpoint
from operator import imod
import torch
import torch.nn as nn
import torch
from src.layers.attention import AttentionLSTM
from src.layers.temporal_gcn import TemporalGCN, GCN
from src.layers.EGCN import TemporalEGCN
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

# class TemporalEGCNEncoder(nn.Module):
#     def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, p,n_layer, batch_size,act="relu"):
#         super(TemporalEGCNEncoder, self).__init__()
#         self.fc = nn.Linear(in_ft, hid_ft1)
#         self.rnn_egcn = TemporalEGCN(in_channels=hid_ft1, out_channels=out_ft, hidden_dim=hid_ft2, p=p, n_layer=n_layer, batch_size=batch_size)
#         self.out_dim = out_ft
#         self.relu = nn.ReLU()
#     def forward(self, x, e):
#         x = self.relu(self.fc(x)) # 32, 12, 19, 200 
#         raw_shape = x.shape
#         h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
#         hs = []
#         breakpoint()
#         for i in range(raw_shape[1]):
#             x_i = x[:,i,:,:].squeeze(1)
#             e_i = e[:,i,:,:].squeeze(1)
#             h, _ =  self.rnn_egcn(x_i, e_i, h)
#             hs.append(h.unsqueeze(1))
#         out =  torch.cat(hs, dim=1)
#         return out 

class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x)) # 12, 19, 200 
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
        # breakpoint()
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1) # 1, 19, 200 
            e = adj[:,i,:,:].squeeze(1) # 1, 19, 19
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_

class WoGCN_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, num_input_station,act="relu", device="cuda"):
        super(WoGCN_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.GRU(hid_ft1 * num_input_station,hid_ft2 * num_input_station, batch_first=True, num_layers=1)
        self.fc2 = nn.Linear(hid_ft2, out_ft)
        self.relu = nn.ReLU()
        self.num_input_station = num_input_station
        self.device = device 
        
    def forward(self, x, adj):
        x_ = self.relu(self.fc(x)) 
        raw_shape = x_.shape
        if self.device == 'cuda':
            h = torch.zeros(1, raw_shape[0], self.hid_dim * self.num_input_station).cuda()
        else:
            h = torch.zeros(1, raw_shape[0], self.hid_dim * self.num_input_station).to(self.device)
        x_ =  torch.reshape(x_, (raw_shape[0], raw_shape[1], -1))
        res, h = self.rnn(x_, h) # 32, 1 ,hid_ft * num_input
        final_emb = torch.reshape(res, (raw_shape[0], raw_shape[1], self.num_input_station, -1))
        res = self.relu(final_emb)
        out = self.fc2(res)
        return out[:,-1,:,:]

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
        x = x[:,-1,:,:]
        adj = adj[:,-1,:,:]
        x = self.relu(self.fc(x)) 
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

