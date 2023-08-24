# from layers.encoder import Encoder
# from layers.discriminator import Discriminator
# from builtins import breakpoint
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.layers.encoder import BaseEncoder, Attention_Encoder, Encoder, GCN_Encoder, WoGCN_Encoder
from src.layers.discriminator import Discriminator

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
        print(shuf_fts)
        print(np.random.uniform(2, 4) * shuf_fts)
        # return np.random.uniform(2, 4) * shuf_fts
        return shuf_fts

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
        # return np.random.uniform(2, 4) * shuf_fts
        return shuf_fts
        
    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h

class Attention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu", stdgi_noise_min=0.4, stdgi_noise_max=0.7,model_type="gede",num_input_station= 0):
        super(Attention_STDGI, self).__init__()
        if model_type == "gede" or model_type == "woclimate":
            print("Init Attention_Encoder model ...")
            self.encoder = Attention_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
            )
        elif model_type == "wogcn":
            print("Init WoGCN_Encoder model ...")
            self.encoder = WoGCN_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en,num_input_station=num_input_station
            )
        elif model_type == "wornnencoder":
            print("Init WoGCN_Encoder model ...")
            self.encoder = GCN_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
            )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max 

    def forward(self, x, x_k, adj):
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h[:,-1,:,:], x_k[:,-1,:,:], x_c[:,-1,:,:])
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts
        
    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h