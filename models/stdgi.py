import numpy as np 

import torch 
import torch.nn as nn 

from layers.discriminator import Discriminator
from layers.ed import Encoder

class STDGI(nn.Module):
    def __init__(self,in_ft, out_ft):
      super(STDGI, self).__init__()
      self.encoder = Encoder(in_ft, out_ft)
      self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft)

    def forward(self, x, x_k, adj):
        # x_ = x(t+k)
        h = self.encoder(x, adj)
        # print(f"shape h : {h.shape}")
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