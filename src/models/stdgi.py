# from layers.encoder import Encoder
# from layers.discriminator import Discriminator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.layers.discriminator import Discriminator
from src.layers.encoder import BaseEncoder, Attention_Encoder, Encoder




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
        ret = self.disc(h, x_k[-1].unsqueeze(0), x_c[-1].unsqueeze(0))
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
    atten_encoder = Attention_Encoder(1, 60, 120, 200, "relu")
    model = Attention_STDGI(1, 60, 120, 240, 10, "relu")
    # print(
    #     model.embedd(torch.rand(12, 27, 1), torch.rand(1, 27, 27)).shape
    # )
