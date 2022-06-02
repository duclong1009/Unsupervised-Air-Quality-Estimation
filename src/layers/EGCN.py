from builtins import breakpoint
import torch
import torch.nn as nn


class EGCN_layer(nn.Module):
    def __init__(self, in_channels, hid_dim,output_dim,p) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hid_dim)
        self.activation = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
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
        # breakpoint()
        list_x = []
        n_channels = e.shape[-1]
        gl = self.linear1(x)
        new_e = torch.zeros_like(e)
        for i in range(n_channels):
            # breakpoint()
            coef = self._atten(x,i)
            new_e[:,:,:,i] = coef * e[:,:,:,i]#(9)
            x_i = torch.bmm(coef,gl)
            # x_i = torch.bmm(new_e[:,:,:,i],gl) #(7)
            list_x.append(x_i)
        res_x = torch.concat(list_x,dim=-1)
        new_e = self.DS(new_e)
        res_x = self.last_liner(res_x)
        return res_x,new_e

    def _atten(self, x,idx):
        n_nodes = x.shape[1]
        x1 = self.linear2[idx](x).unsqueeze(1).expand(-1,n_nodes, -1, -1)
        x2 = self.linear3[idx](x).unsqueeze(2).expand(-1,-1, n_nodes, -1)
        x3 = self.LeakyReLU(self.linear4[idx](torch.cat([x1, x2], dim=-1))) # (seq_len, n_nodes, n_nodes, n_channels)     
        x3 = torch.squeeze(x3,-1) # (seq_len, n_nodes, n_nodes)
        return nn.ReL(x3)

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
        return x

    def DS(self,e):
        """
        :param e: [batch_size,num_nodes, num_nodes,n_channels]"""
        raw_shape = e.shape
        # breakpoint()
        n_nodes = e.shape[1]
        # (1)
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
        # breakpoint()
        new_e = torch.bmm(new_e,new_e_T)
        new_e = new_e.reshape(raw_shape[0],raw_shape[3],raw_shape[1],raw_shape[2])
        new_e = torch.permute(new_e,(0,2,3,1))
        return new_e
        
if __name__ == '__main__':
    model = EGCN(in_channels=10,hid_dim=30,output_dim=60,p=4,n_layer=2)
    x = torch.rand(12,6,10)
    e = torch.rand(12,6,6,4)
    model(x,e)
    breakpoint()