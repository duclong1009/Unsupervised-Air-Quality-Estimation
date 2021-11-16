import torch 
import torch.nn as  nn 
from gconv import GCN

class Encoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, out_ft):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, 400)
        self.gcn = GCN(400, 200,'relu')
        self.gcn2 = GCN(200,out_ft,'relu')
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(in_ft) # them batch norm 

    #   for m in self.modules:
    #     self.weight_init(m)
    # def weight_init(self, m):
    #   if isinstance(m, nn.Linear):
    #     torch.nn.init.xavier_uniform_(m.weight.data)
    #     if m.bias is not None:
    #       m.bias.data.fill_(0.0)
        
    def forward(self, x, adj):
        x = self.fc(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

class Decoder(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.gru = nn.GRU(in_ft, in_ft, batch_first=False)
        self.cnn = nn.Conv1d(
            in_channels=in_ft, out_channels=128, kernel_size=5, padding=2, stride=1
        )
        self.linear = nn.Linear(in_features=128, out_features=64)
        self.linear2 = nn.Linear(64, out_ft)
        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(num_features=128)

    #     for m in self.modules():
    #         self.weight_init(m)

    # def weight_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)

    def forward(self, x, h, l):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x (n1-1) x latent_dim
        l.shape = (n1-1) x 1
        """
        
        batch_size = x.shape[1]
        l_ = l / l.sum()
        l_ = l_.T
        l_ = l_.reshape(1, 27, 1)
        x_ = torch.cat((x, h), dim=2)  # timestep x nodes x hidden feat
        # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
        output, hid_state = self.gru(x_)
        x_ = output[-1]
        x_ = x_.reshape(1, x_.shape[1], x_.shape[0])
        ret = self.cnn(x_)
        ret = torch.bmm(ret, l_)
        ret = ret.reshape(ret.shape[0], -1)
        ret = self.linear(ret)
        ret = self.relu(ret)
        ret = self.linear2(ret)
        ret = self.relu(ret)
        return ret