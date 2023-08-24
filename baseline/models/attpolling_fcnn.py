import functools
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import yaml 

if torch.cuda.is_available():
    dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} 
else:
    dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 


def get_iterator(x, n, forced=False):
    r"""If x is int, copy it to a list of length n
    Cannot handle a special case when the input is an iterable and len(x) = n, 
    but we still need to copy it to a list of length n
    """
    if forced:
        return [x] * n
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x] * n
    # Note: np.array, list are always iterable
    if len(x) != n:
        x = [x] * n
    return x

class GraphAttentionLayer(nn.Module):
    r"""Attention layer
    
    Args:
        in_dim: int, dimension of input
        out_dim: int, dimension of output
        out_indices: torch.LongTensor, the indices of nodes whose representations are 
                     to be computed
                     Default None, calculate all node representations
                     If not None, need to reset it every time model is run
        feature_subset: torch.LongTensor. Default None, use all features
        kernel: 'affine' (default), use affine function to calculate attention 
                'gaussian', use weighted Gaussian kernel to calculate attention
        k: int, number of nearest-neighbors used for calculate node representation
           Default None, use all nodes
        graph: a list of torch.LongTensor, corresponding to the nearest neighbors of nodes 
               whose representations are to be computed
               Make sure graph and out_indices are aligned properly
        use_previous_graph: only used when graph is None
                            if True, to calculate graph use input
                            otherwise, use newly transformed output
        nonlinearity_1: nn.Module, non-linear activations followed by linear layer 
        nonlinearity_2: nn.Module, non-linear activations followed after attention operation
    
    Shape:
        - Input: (N, in_dim) graph node representations
        - Output: (N, out_dim) if out_indices is None 
                  else (len(out_indices), out_dim)
        
    Attributes:
        weight: (out_dim, in_dim)
        a: out_dim if kernel is 'gaussian' 
           out_dim*2 if kernel is 'affine'
           
    Examples:
    
        >>> m = GraphAttentionLayer(2,2,feature_subset=torch.LongTensor([0,1]), 
                        graph=torch.LongTensor([[0,5,1], [3,4,6]]), out_indices=[0,1], 
                        kernel='gaussian', nonlinearity_1=None, nonlinearity_2=None)
        >>> x = Variable(torch.randn(10,3))
        >>> m(x)
    """
    def __init__(self, in_dim, out_dim, k=None, graph=None, out_indices=None, 
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, reset_graph_every_forward=False,
                no_feature_transformation=False, rescale=True, layer_norm=False, layer_magnitude=100,
                key_dim=None, feature_selection_only=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.graph = graph
        if graph is None:
            self.cal_graph = True
        else:
            self.cal_graph = False
        self.use_previous_graph = use_previous_graph
        self.reset_graph_every_forward = reset_graph_every_forward
        self.no_feature_transformation = no_feature_transformation
        if self.no_feature_transformation:
            assert in_dim == out_dim
        else:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            # initialize parameters
            std = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-std, std)
        self.rescale = rescale
        self.k = k
        self.out_indices = out_indices
        self.feature_subset = feature_subset
        self.kernel = kernel
        self.nonlinearity_1 = nonlinearity_1
        self.nonlinearity_2 = nonlinearity_2
        self.layer_norm = layer_norm
        self.layer_magnitude = layer_magnitude
        self.feature_selection_only = feature_selection_only

        if kernel=='affine':
            self.a = nn.Parameter(torch.Tensor(out_dim*2))
        elif kernel=='gaussian' or kernel=='inner-product' or kernel=='avg_pool' or kernel=='cosine':
            self.a = nn.Parameter(torch.Tensor(out_dim))
        elif kernel=='key-value':
            if key_dim is None:
                self.key = None
                key_dim = out_dim
            else:
                if self.use_previous_graph:
                    self.key = nn.Linear(in_dim, key_dim)
                else:
                    self.key = nn.Linear(out_dim, key_dim)
            self.key_dim = key_dim
            self.a = nn.Parameter(torch.Tensor(out_dim))
        else:
            raise ValueError('kernel {0} is not supported'.format(kernel))
        self.a.data.uniform_(0, 1)
    
    def reset_graph(self, graph=None):
        self.graph = graph
        self.cal_graph = True if self.graph is None else False
        
    def reset_out_indices(self, out_indices=None):
        self.out_indices = out_indices
    
    def forward(self, x):
        if self.reset_graph_every_forward:
            self.reset_graph()
            
        N = x.size(0)
        out_indices = dtype['long'](range(N)) if self.out_indices is None else self.out_indices
        if self.feature_subset is not None:
            x = x[:, self.feature_subset]
        assert self.in_dim == x.size(1)
         
        if self.no_feature_transformation:
            out = x
        else:
            out = nn.functional.linear(x, self.weight)
        
        feature_weight = nn.functional.softmax(self.a, dim=0) 
        if self.rescale and self.kernel!='affine':
            out = out*feature_weight
            if self.feature_selection_only:
                return out

        if self.nonlinearity_1 is not None:
            out = self.nonlinearity_1(out)
        k = N if self.k is None else min(self.k, out.size(0))

        if self.kernel=='key-value':
            if self.key is None:
                keys = x if self.use_previous_graph else out
            else:
                keys = self.key(x) if self.use_previous_graph else self.key(out)
            norm = torch.norm(keys, p=2, dim=-1)
            att = (keys[out_indices].unsqueeze(-2) * keys.unsqueeze(-3)).sum(-1) / (norm[out_indices].unsqueeze(-1)*norm)
            att_, idx = att.topk(k, -1)
  
            a = Variable(torch.zeros(att.size()).fill_(float('-inf')).type(dtype['float']))
            a.scatter_(-1, idx, att_)
            a = nn.functional.softmax(a, dim=-1)

            y = (a.unsqueeze(-1)*out.unsqueeze(-3)).sum(-2)
            if self.nonlinearity_2 is not None:
                y = self.nonlinearity_2(y)
            if self.layer_norm:
                y = nn.functional.relu(y)  # maybe redundant; just play safe
                y = y / y.sum(-1, keepdim=True) * self.layer_magnitude # <UncheckAssumption> y.sum(-1) > 0
            return y

        # The following line is BUG: self.graph won't update after the first update
        # if self.graph is None
        # replaced with the following line
        if self.cal_graph:
            if self.kernel != 'key-value':
                features = x if self.use_previous_graph else out
                dist = torch.norm(features.unsqueeze(1)-features.unsqueeze(0), p=2, dim=-1)
                _, self.graph = dist.sort()
                self.graph = self.graph[out_indices]               
        y = Variable(torch.zeros(len(out_indices), out.size(1)).type(dtype['float']))
        

        neighbor_indices = []
        attentions = []

        for i, idx in enumerate(out_indices):
            neighbor_idx = self.graph[i][:k]
            if self.kernel == 'gaussian':
                if self.rescale: # out has already been rescaled
                    a = -torch.sum((out[idx] - out[neighbor_idx])**2, dim=1)
                else:
                    a = -torch.sum((feature_weight*(out[idx] - out[neighbor_idx]))**2, dim=1)
            elif self.kernel == 'inner-product':
                if self.rescale: # out has already been rescaled
                    a = torch.sum(out[idx]*out[neighbor_idx], dim=1)
                else:
                    a = torch.sum(feature_weight*(out[idx]*out[neighbor_idx]), dim=1)
            elif self.kernel == 'cosine':
                if self.rescale: # out has already been rescaled
                    norm = torch.norm(out[idx]) * torch.norm(out[neighbor_idx], p=2, dim=-1)
                    a = torch.sum(out[idx]*out[neighbor_idx], dim=1) / norm
                else:
                    norm = torch.norm(feature_weight*out[idx]) * torch.norm(feature_weight*out[neighbor_idx], p=2, dim=-1)
                    a = torch.sum(feature_weight*(out[idx]*out[neighbor_idx]), dim=1) / norm
            elif self.kernel == 'affine':
                a = torch.mv(torch.cat([(out[idx].unsqueeze(0) 
                                         * Variable(torch.ones(len(neighbor_idx)).unsqueeze(1)).type(dtype['float'])), 
                                        out[neighbor_idx]], dim=1), self.a)
            elif self.kernel == 'avg_pool':
                a = Variable(torch.ones(len(neighbor_idx)).type(dtype['float']))
            a = nn.functional.softmax(a, dim=0)
            # since sum(a)=1, the following line should torch.sum instead of torch.mean
            y[i] = torch.sum(out[neighbor_idx]*a.unsqueeze(1), dim=0)

            neighbor_indices.append(neighbor_idx)
            attentions.append(a)
        if self.nonlinearity_2 is not None:
            y = self.nonlinearity_2(y)
        if self.layer_norm:
            y = nn.functional.relu(y)  # maybe redundant; just play safe
            y = y / y.sum(-1, keepdim=True) * self.layer_magnitude # <UncheckAssumption> y.sum(-1) > 0
        return y, torch.stack(neighbor_indices), torch.stack(attentions)

class AttPollingFCNN(nn.Module):
    def __init__(self, config, device):
        super(AttPollingFCNN, self).__init__()
        self.config = config 
        self.device = device
        self.k = config['num_neigh']
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(config['meteo_dim']+1,16)
        self.fc2 = nn.Linear(config['meteo_dim'], 16)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.prediction_layer = nn.Sequential(
            self.fc3, 
            nn.ReLU(),
            self.fc4,
            nn.ReLU(),
            self.fc5, 
            nn.ReLU(), 
            self.fc6,
            nn.ReLU(),
            self.fc_out
        )

        self.att_polling_layer = GraphAttentionLayer(
            in_dim=config['meteo_dim'],
            out_dim=config['meteo_dim'], 
            k=self.k+1,
            out_indices=None,
            kernel='cosine', 
            nonlinearity_1=None, 
            nonlinearity_2=None,
        )

    def forward(self, x, meteo, input_stat, target_stat):
        '''
            x (N-1,l): pm2.5 information from all stats (except target station)
            meteo(N, h): meterology information from all stats 
            target_id (int): id of target station needed for prediction 
        '''
        target_stat = target_stat.squeeze(-1)
        meteo = meteo.squeeze(0)
        x = x.squeeze(0).unsqueeze(-1)
        input_stat = input_stat.squeeze(0)
        target_id = input_stat.tolist().index(int(target_stat.item()))

        _, id_neighbour, att = self.att_polling_layer(meteo) # list(), att(B,k)
        id_neighbour_target = id_neighbour[target_id,:].tolist() # (B, N) -> (B,k)

        if target_stat in id_neighbour_target:
            idx_target  = id_neighbour_target.index(int(target_stat.item()))
            idx_neigh_chosen = [ i for i in range(len(id_neighbour_target)) if i != idx_target ]
            id_neighbour_target.pop(idx_target)       
            att_target = att[target_id,idx_neigh_chosen].unsqueeze(0) # (B,k) -> (1, k)
        else:
            id_neighbour_target = id_neighbour_target[:self.k]
            idx_neigh_chosen  = [i for i in range(self.k)]
            att_target = att[target_id,:self.k].unsqueeze(0) # (B,k) -> (1, k)

        meteo_target =  meteo[target_id, :].unsqueeze(0)
        meteo_neigh = meteo[idx_neigh_chosen,:]
        x_neigh = x[idx_neigh_chosen, :] # (B, k, 1)

        node_info = torch.cat([x_neigh, meteo_neigh], axis=-1) #(B,k, H+1)
        att_node_info = torch.matmul(att_target, node_info) # ( 1, k) * ( k , H+1) = (1, H+1)
        att_node_info_embed = self.leakyrelu(self.fc1(att_node_info)) #(1, H+1) -> (1, 16)
        meteo_embed = self.leakyrelu(self.fc2(meteo_target)) # (1, H) -> (1,16)
        node_concat = torch.cat([att_node_info_embed, meteo_embed], axis=-1)
        out = self.prediction_layer(node_concat).squeeze(-1)
        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = './config/'
    config_file = config_path + 'fcnn' + '.yml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    x = Variable(torch.randn(147,1)).to(device)
    meteo = Variable(torch.randn(147,2)).to(device)
    m= AttPollingFCNN(config, device).to(device)
    a = m(x,meteo, 0)
    print(a)
