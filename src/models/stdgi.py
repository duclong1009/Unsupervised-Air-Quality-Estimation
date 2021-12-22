# from layers.encoder import Encoder
# from layers.discriminator import Discriminator
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, infea, outfea, act="relu", bias=True):
        super(GCN, self).__init__()
        # define cac lop fc -> act
        self.fc = nn.Linear(infea, outfea, bias=False)
        self.act = nn.ReLU() if act == "relu" else nn.ReLU()

        # init bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outfea))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        # init weight
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # neu la lop fully connectedd
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(
                torch.bmm(adj, torch.squeeze(adj, torch.squeeze(seq_fts, 0)))
            )
        else:
            out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        return self.act(out)


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


class BaseSTDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu"):
        super(BaseSTDGI, self).__init__()
        self.encoder = BaseEncoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)

    def forward(self, x, x_k, adj):
        # x_ = x(t+k)
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        # print(f"shape x_c : {x_c.shape}")
        ret = self.disc(h, x_k, x_c)
        # print(f"shape h : {ret.shape}")
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(2, 4) * shuf_fts

    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h

from src.layers.attention import AttentionLSTM
class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(hid_ft1,hid_ft1, batch_first=False, num_layers=1)
        self.attn = AttentionLSTM(hid_ft1,120,hid_ft1,12,0.1)
    
        self.gcn = GCN(hid_ft1, hid_ft2, act)
        self.gcn2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x))
        x,h = self.rnn(x)
        x = self.attn(x)
        print(x.shape)
        x = self.relu(x.unsqueeze(0))
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x


class Encoder(nn.Module):
    # hshape = 64*27*128
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(hid_ft1,hid_ft1, batch_first=False, num_layers=1)
        self.gcn = GCN(hid_ft1, hid_ft2, act)
        self.gcn2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x))
        x,h = self.rnn(x)
        
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x


class STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu"):
        super(STDGI, self).__init__()
        self.encoder = Encoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)

    def forward(self, x, x_k, adj):
        # x_ = x(t+k)
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        # print(f"shape x_c : {x_c.shape}")
        ret = self.disc(h, x_k, x_c)
        # print(f"shape h : {ret.shape}")
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(2, 4) * shuf_fts

    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h

class Attention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu"):
        super(Attention_STDGI, self).__init__()
        self.encoder = Attention_Encoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)

    def forward(self, x, x_k, adj):
        # x_ = x(t+k)
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        # print(f"shape x_c : {x_c.shape}")
        ret = self.disc(h, x_k, x_c)
        # print(f"shape h : {ret.shape}")
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(2, 4) * shuf_fts

    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h


if __name__ == "__main__":
#   atten_layer = AttentionLSTM(1,60,120,12,0.1)
#   print(atten_layer(torch.rand(12,27,1)).shape)
  atten_encoder = Attention_Encoder(1,60,120,200,'relu')
  print(atten_encoder(torch.rand(12,27,1),torch.rand(27,27)).shape)