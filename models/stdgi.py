from layers.discriminator import Discriminator
from layers.encoder import Encoder
import numpy as np
import torch.nn as nn


class STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, gconv="gcn"):
        super(STDGI, self).__init__()
        self.gconv = gconv
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.encoder = Encoder(in_ft, out_ft, gconv)
        self.disc = Discriminator(x_ft=12, h_ft=out_ft)

    def forward(self, x, x_k, adj=None, edge_idx=None, edge_attr=None):
        # x_ = x(t+k)
        if self.gconv == "gcn":
            h = self.encoder(x, edge_idx=edge_idx, edge_attr=edge_attr)
        elif self.gconv == "dcn":
            h = self.encoder(x, edge_idx=edge_idx, edge_attr=edge_attr)
        elif self.gconv == "gat":
            h = self.encoder(x, edge_idx=edge_idx)
        x_c = self.corrupt(x_k)
        # import pdb; pdb.set_trace()
        ret = self.disc(h, x_k, x_c)
        # print(f"shape h : {ret.shape}")
        # import pdb; pdb.set_trace()
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[0]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[idx, :]
        return np.random.uniform(2, 4) * shuf_fts

    def embedd(self, x, adj=None, edge_idx=None, edge_attr=None):
        if self.gconv == "gcn":
            # h = self.encoder(x, adj=adj)
            h = self.encoder(x, edge_idx=edge_idx, edge_attr=edge_attr)
        elif self.gconv == "dcn":
            h = self.encoder(x, edge_idx=edge_idx)
        elif self.gconv == "gat":
            h = self.encoder(x, edge_idx=edge_idx)
        return h
