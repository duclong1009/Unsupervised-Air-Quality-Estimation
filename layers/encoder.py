import torch
from torch_geometric.nn import GATConv
from layers.gconv import GCNConv2
import torch.nn as nn
import torch


class Encoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, out_ft, gconv="gcn"):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, 400)
        self.gconv = gconv
        self.in_ft = in_ft
        self.out_ft = out_ft
        if gconv == "gcn":
            self.gcn = GCNConv2(400, 200)
            self.gcn2 = GCNConv2(200, out_ft)
        elif gconv == "dcn":
            self.dcn = DCN(400, 200, 3, "relu")
            self.dcn2 = DCN(200, out_ft, 3, "relu")
        elif gconv == "gat":
            self.gcn = GATConv(400, 200)
            self.gcn2 = GATConv(200, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, adj=None, edge_idx=None, edge_attr=None):
        if self.gconv == "gat":
            x = self.fc(x)
            x = self.relu(x)
            # import pdb; pdb.set_trace()
            x = self.gcn(x, edge_index=edge_idx)
            x = self.relu(x)
            x = self.gcn2(x, edge_index=edge_idx)
        elif self.gconv == "dcn":
            x = self.fc(x)
            x = self.relu(x)
            x = self.dcn(x, edge_index=edge_idx)
            # x = self.relu(x)
            # x = self.dcn2(x, edge_index=edge_idx, edge_weight=edge_weight)
        elif self.gconv == "gcn":
            x = self.fc(x)
            x = self.relu(x)
            # import pdb; pdb.set_trace()
            x = self.gcn(x, edge_index=edge_idx, edge_weight=edge_attr)
            x = self.relu(x)
            x = self.gcn2(x, edge_index=edge_idx, edge_weight=edge_attr)
        x = self.relu(x)
        return x
