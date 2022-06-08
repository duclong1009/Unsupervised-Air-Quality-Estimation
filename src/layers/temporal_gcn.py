import torch 
from torch import nn 


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
        # import pdb; pdb.set_trace()
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

class GCN_2_layers(torch.nn.Module):
    def __init__(self, hid_ft1, hid_ft2, out_ft, act='relu') -> None:
        super(GCN_2_layers, self).__init__()
        self.gcn_1 = GCN(hid_ft1, hid_ft2, act)
        self.gcn_2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj, sparse=False):
        
        x = self.gcn_1(x, adj)
        x  = self.gcn_2(x, adj)
        return x 

class TemporalGCN(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, batch_size: int=1, improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TemporalGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        # self.config = config
        # breakpoint()
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(self.batch_size,X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, adj, H):
        # import pdb; pdb.set_trace()
        h = self.conv_z(X, adj)
        Z = torch.cat([h, H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, adj, H):
        conv = self.conv_r(X, adj)
        R = torch.cat([conv, H], axis=2) # (b, 207, 64)
        # import pdb;pdb.set_trace()
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, adj, H, R):
        H_tilde = torch.cat([self.conv_h(X, adj), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, adj: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, adj, H)
        R = self._calculate_reset_gate(X, adj, H)
        H_tilde = self._calculate_candidate_state(X, adj, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H