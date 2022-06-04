# from layers.encoder import Encoder
# from layers.discriminator import Discriminator
# from builtins import breakpoint
import numpy as np
from src.layers.EGCN import EGCN
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.layers.encoder import BaseEncoder, Attention_Encoder, Encoder, GCN_Encoder, WoGCN_Encoder, TemporalEGCNEncoder
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
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu", stdgi_noise_min=0.4, stdgi_noise_max=0.7,model_type="gede"):
        super(Attention_STDGI, self).__init__()
        if model_type == "gede" or model_type == "woclimate":
            print("Init Attention_Encoder model ...")
            self.encoder = Attention_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
            )
        elif model_type == "wogcn":
            print("Init WoGCN_Encoder model ...")
            self.encoder = WoGCN_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
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
        # breakpoint()
        x_c = self.corrupt(x_k)
        ret = self.disc(h, x_k[-1].unsqueeze(0), x_c[-1].unsqueeze(0))
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        # print('Noise min: {}'.format(self.stdgi_noise_min))
        # print('Noise max: {}'.format(self.stdgi_noise_max))
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts
        # return shuf_fts
    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h

class EGCN_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, config, act_en="relu", stdgi_noise_min=0.4, stdgi_noise_max=0.7,p=2, model_type="gede"):
        super(EGCN_STDGI, self).__init__()
        bs = config['batch_size']
        self.model_type = model_type
        num_input_station = len(config['train_station']) -1  # tru di mot tram target 
        if model_type == "wogcn":
            print("Init WoGCN_Encoder model ...")
            self.encoder = WoGCN_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, num_input_station=num_input_station,act=act_en
            )
        elif model_type == "wornnencoder":
            print("Init EGCN model ...")
            self.encoder = EGCN(
                in_channels=in_ft, hid_dim=en_hid1, output_dim=out_ft, p=p, n_layer=2
            )
        else:
            print("Init GEDE model ...")
            self.encoder = TemporalEGCNEncoder(
                in_ft, en_hid1, en_hid2, out_ft,p,2, bs, act=act_en
            )
        
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max 

    def forward(self, x, x_k, adj):
        # breakpoint()
        if self.model_type == 'wornnencoder':
            h, _ = self.encoder(x, adj)
            # import pdb; pdb.set_trace()
        elif self.model_type == 'wogcn':
            h = self.encoder(x, adj)
            x_k = x_k[:,-1, :, :]
        else: 
            h = self.encoder(x, adj)
            x_k = x_k[:,-1, :, :]
        x_c = self.corrupt(x_k)
        ret = self.disc(h, x_k, x_c)
        # ret = self.disc(h, x_k_[-1].unsqueeze(0), x_c[-1].unsqueeze(0))
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts

    def embedd(self, x, adj):
        if self.model_type == 'wornnencoder':
            h, _ = self.encoder(x, adj)
        else:    
            h = self.encoder(x, adj)
        return h

from src.layers.encoder import InterpolateAttentionEncoder

class InterpolateAttention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu"):
        super(InterpolateAttention_STDGI, self).__init__()
        self.encoder = InterpolateAttentionEncoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)

    def forward(self, x, x_k, adj, l):
        # x_ = x(t+k)
        x_ = get_interpolate(x, l)
        h, enc_out = self.encoder(x_, adj, l)

        x_k_ = get_interpolate(x_k, l)
        x_c = self.corrupt(x_k_)

        # print(f"shape x_c : {x_c.shape}")
        ret = self.disc(h, x_k_[-1].unsqueeze(0), x_c[-1].unsqueeze(0))
        # print(f"shape h : {ret.shape}")
        return ret, enc_out # encoder output, hidden state 

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(2, 4) * shuf_fts

    def embedd(self, x, adj, l):
        x_ = get_interpolate(x, l)
        h, enc_out = self.encoder(x_, adj, l)
        return h, enc_out

def get_interpolate(inp_feat, lst_rev_distance):  # inp: 12, 27, 1  lst_rev_distance: 1, 27
    inp_feat_ = inp_feat.reshape(inp_feat.shape[0], inp_feat.shape[2], inp_feat.shape[1]) # (seq_len,station,feat) -> (seq_len, feat, station)  =  12,6,27
    # print(inp_feat_.shape)
    if len(lst_rev_distance.shape) == 1:
        lst_rev_distance = torch.unsqueeze(lst_rev_distance, 0)
    add_feat =  torch.matmul(inp_feat_, lst_rev_distance.T) # (12, 6, 27 ) * (27,1) -> (12,6,1)
    if len(add_feat.shape) == 2:
        add_feat= torch.unsqueeze(add_feat, 1)
    # import pdb; pdb.set_trace()
    
    add_feat_ = add_feat.reshape(add_feat.shape[0], add_feat.shape[2], add_feat.shape[1]) # 12, 1, 6
    total_feat = torch.cat((inp_feat, add_feat_), dim=1) # 12, 28, 6
    return total_feat


if __name__ == "__main__":
    # atten_layer = AttentionLSTM(1,60,120,12,0.1)
    # print(atten_layer(torch.rand(12,27,1)).shape)
    atten_encoder = InterpolateAttentionEncoder(1, 60, 120, 200, "relu")
    model = Attention_STDGI(1, 60, 120, 240, 10, "relu")
    print(
        model.embedd(torch.rand(12, 27, 1), torch.rand(1, 28, 28)).shape
    )
