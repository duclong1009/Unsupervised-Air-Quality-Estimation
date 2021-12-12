import scipy
import torch
import torch.nn as nn
import numpy as np


def get_dist(location_dt, y):
    from scipy.spatial.distance import cdist

    return cdist(location_dt, y)


# Build distance graph
def get_G1(location_dt, n_node=28, time_steps=8):
    G1 = get_dist(location_dt, location_dt)
    G = G1 / G1.sum(axis=0)
    G = G * 0.7 + np.eye(n_node) * 0.3
    G = torch.Tensor(G)
    G = torch.unsqueeze(G, 0)
    # print(G.shape)
    G = G.expand(time_steps, G.shape[1], G.shape[2])
    return G
