# from builtins import breakpoint
import torch
import torch.nn as nn

class EGCN_layer(nn.Module):
    def __init__(self, in_channels, hid_dim,output_dim,p) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hid_dim)
        self.activation = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.ModuleList([nn.Linear(in_channels, hid_dim)  for i in range(p)])
        self.linear3 = nn.ModuleList([nn.Linear(in_channels, hid_dim)for i in range(p)])
        self.linear4 = nn.ModuleList([nn.Linear(hid_dim * 2, 1)for i in range(p)])
        self.p = p
        self.last_liner = nn.Linear(hid_dim *p, output_dim)

    def forward(self, x, e):
        """
        :param x: [batch,num_nodes, n_fts]
        :param e: [batch,num_nodes, num_nodes,n_channels]
        """
        list_x = []
        n_channels = e.shape[-1]
        gl = self.linear1(x) # (8)
        new_e = torch.zeros_like(e)
        for i in range(n_channels):
            # breakpoint()
            coef = self._atten(x,i) # (11)
            new_e[:,:,:,i] = coef * e[:,:,:,i] # (9)
            x_i = torch.bmm(coef,gl)
            # x_i = torch.bmm(new_e[:,:,:,i],gl) 
            list_x.append(x_i)
        res_x = torch.concat(list_x,dim=-1) #(7)
        new_e = self.DS(new_e) # (10)
        res_x = self.last_liner(res_x)
        return res_x, new_e

    def _atten(self, x,idx): # fl( Xi(l-1), Xj(l-1) )
        n_nodes = x.shape[1]
        x1 = self.linear2[idx](x).unsqueeze(1).expand(-1,n_nodes, -1, -1)
        x2 = self.linear3[idx](x).unsqueeze(2).expand(-1,-1, n_nodes, -1)
        x3 = self.LeakyReLU(self.linear4[idx](torch.cat([x1, x2], dim=-1))) # (seq_len, n_nodes, n_nodes, n_channels)     
        x3 = torch.squeeze(x3,-1) # (seq_len, n_nodes, n_nodes)
        return self.sigmoid(x3)

    def DS(self,e):
        """
        :param e: [batch_size,num_nodes, num_nodes,n_channels]"""
        raw_shape = e.shape
        # breakpoint()
        n_nodes = e.shape[1]
        e = torch.permute(e,(0,3,1,2))
        e = e.reshape(-1,n_nodes,n_nodes)
        e_ = torch.sum(e,dim=2)
        e_ = e_.unsqueeze(2).expand(-1,-1,n_nodes)
        new_e = e/e_
        new_e_ = torch.sum(new_e,dim=1)
        new_e_ = new_e_.unsqueeze(1).expand(-1,n_nodes,-1)
        new_e = new_e/ new_e_
        new_e_T = torch.transpose(new_e,1,2)
        # breakpoint()
        new_e = torch.bmm(new_e,new_e_T)
        new_e = new_e.reshape(raw_shape[0],raw_shape[3],raw_shape[1],raw_shape[2])
        new_e = torch.permute(new_e,(0,2,3,1))
        return new_e
        
class EGCN(nn.Module):
    def __init__(self, in_channels, hid_dim, output_dim,p,n_layer) -> None:
        super().__init__()
        list_egcn = [EGCN_layer(in_channels, hid_dim, output_dim,p)]
        for i in range(n_layer-1):
            list_egcn.append(EGCN_layer(output_dim, hid_dim, output_dim,p))
        self.egcn = nn.ModuleList(list_egcn)
        self.n_layer = n_layer

    def forward(self, x, e):
        """
            :param x: [num_nodes, n_fts]
            :param e: [num_nodes, num_nodes,n_channels]
        """
        # breakpoint()
        e = self.DS(e)
        for i in range(self.n_layer):
            x,e = self.egcn[i](x,e)
        return x, e

    def DS(self,e):
        """
        :param e: [batch_size,num_nodes, num_nodes,n_channels]"""
        raw_shape = e.shape
        # breakpoint()
        n_nodes = e.shape[1]
        # (1)
        # import pdb; pdb.set_trace()
        e = torch.permute(e,(0,3,1,2))
        e = e.reshape(-1,n_nodes,n_nodes)
        e_ = torch.sum(e,dim=2)
        e_ = e_.unsqueeze(2).expand(-1,-1,n_nodes)
        new_e = e/e_
        # (2)
        new_e_ = torch.sum(new_e,dim=1)
        new_e_ = new_e_.unsqueeze(1).expand(-1,n_nodes,-1)
        new_e = new_e/ new_e_
        new_e_T = torch.transpose(new_e,1,2)
        new_e = torch.bmm(new_e,new_e_T)
        new_e = new_e.reshape(raw_shape[0],raw_shape[3],raw_shape[1],raw_shape[2])
        new_e = torch.permute(new_e,(0,2,3,1))
        return new_e

class TemporalEGCN(torch.nn.Module):
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

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, p: int, n_layer: int, batch_size: int=1, improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TemporalEGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_dim = hidden_dim
        self.p = p
        self.n_layer = n_layer
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        # self.config = config
        # breakpoint()
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = EGCN(in_channels=self.in_channels,hid_dim=self.hid_dim, output_dim=self.out_channels, p=self.p, n_layer=self.n_layer)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = EGCN(in_channels=self.in_channels,hid_dim=self.hid_dim, output_dim=self.out_channels, p=self.p, n_layer=self.n_layer)
        # GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = EGCN(in_channels=self.in_channels,hid_dim=self.hid_dim, output_dim=self.out_channels, p=self.p, n_layer=self.n_layer)
        # GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
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
        h, E = self.conv_z(X, adj)
        Z = torch.cat([h, H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)
        return Z, E

    def _calculate_reset_gate(self, X, adj, H):
        conv, _ = self.conv_r(X, adj)
        R = torch.cat([conv, H], axis=2) # (b, 207, 64)
        # import pdb;pdb.set_trace()
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, adj, H, R):
        conv, _ = self.conv_h(X, adj)
        H_tilde = torch.cat([conv, H * R], axis=2) # (b, 207, 64)
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
        Z, E = self._calculate_update_gate(X, adj, H)
        R = self._calculate_reset_gate(X, adj, H)
        H_tilde = self._calculate_candidate_state(X, adj, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H, E

if __name__ == '__main__':
    model = TemporalEGCN(in_channels=10,hid_dim=30,output_dim=60,p=4,n_layer=2)
    x = torch.rand(12,6,10)
    e = torch.rand(12,6,6,4)
    model(x,e)
