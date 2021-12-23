import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self, in_ft, out_ft, n_layers_rnn=1, rnn="GRU", cnn_hid_dim=128, fc_hid_dim=64
    ):
        super(Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        if rnn == "GRU":
            self.rnn = nn.GRU(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        else:
            self.rnn = nn.RNN(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        self.cnn = nn.Conv1d(
            in_channels=in_ft,
            out_channels=cnn_hid_dim,
            kernel_size=5,
            padding=2,
            stride=1,
        )
        self.linear = nn.Linear(in_features=cnn_hid_dim, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()

    def forward(self, x, h, l):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """

        l_ = l / l.sum()
        l_ = l_.T
        l_ = l_.reshape(1, 27, 1)
        x_ = torch.cat((x, h), dim=2)  # timestep x nodes x hidden feat
        # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
        output, hid_state = self.rnn(x_) # hidden_state = (seq_len,60, 27)
        x_ = output[-1] # (1, 27, 60)
        x_ = x_.reshape(1, x_.shape[1], x_.shape[0]) # output = (1, 60, 27)
        ret = self.cnn(x_) # (input_size, hidden_dim) = (60, 128)
        ret = torch.bmm(ret, l_) # ret= (1, 128, 27) * (1, 27, 1) = (1, 128, 1)
        ret = ret.reshape(ret.shape[0], -1) # output =  
        ret = self.linear(ret) # (128, 1)
        ret = self.relu(ret) # (128,64)
        ret = self.linear2(ret) # (64,1)
        ret = self.relu(ret) # (1)
        return ret

class InterpolateAttentionDecoder(nn.Module):
    def __init__(
        self, in_ft, out_ft, n_layers_rnn=1, rnn="GRU", cnn_hid_dim=128, fc_hid_dim=64
    ):
        super(InterpolateAttentionDecoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        if rnn == "GRU":
            self.rnn = nn.GRU(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        else:
            self.rnn = nn.RNN(in_ft, in_ft, batch_first=False, num_layers=n_layers_rnn)
        self.cnn = nn.Conv1d(
            in_channels=in_ft,
            out_channels=cnn_hid_dim,
            kernel_size=5,
            padding=2,
            stride=1,
        )
        self.linear = nn.Linear(in_features=cnn_hid_dim, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)

        self.relu = nn.ReLU()

    def forward(self, x, h, l):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x n1 x latent_dim
        l.shape = n1 x 1 = (27* 1)
        """ 
        # import pdb; pdb.set_trace()
        x_inter = get_interpolate(x, l)

        x_ = torch.cat((x_inter, h), dim=2)  # timestep x nodes x hidden feat (1,28,1) (1,28,2)
        # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
        output, hid_state = self.rnn(x_) # output = (seq_len, 28, 60)
        x_ = output[-1] # (1, 28, 60)
        x_ = x_.reshape(1, x_.shape[1], x_.shape[0]) # output = (1, 60, 28)
        ret = self.cnn(x_) # CNN: (input_size, hidden_dim) = (60, 128), ret = (1, 128, 28)
        ret = ret[:,:,-1] # size = (1, 128,1)
        
        ret = ret.reshape(ret.shape[0], -1) # output = (1,128)
        ret = self.linear(ret) # (128, 1)
        ret = self.relu(ret) # (128,64)
        ret = self.linear2(ret) # (64,1)
        ret = self.relu(ret) # (1)
        return ret 

def get_interpolate(inp_feat, lst_rev_distance):  # inp: 12, 27, 6  lst_rev_distance: 1, 27
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
    # print(total_feat.shape)
    return total_feat


# class AttneDecoder(nn.Module):
#     def __init__(
#         self, in_ft, out_ft, n_layers_rnn=1, cnn_hid_dim=128, fc_hid_dim=64
#     ):
#         super(BaseDecoder, self).__init__()
#         self.in_ft = in_ft
#         self.out_ft = out_ft
#         self.n_layers_rnn = n_layers_rnn
#         self.cnn = nn.Conv1d(
#             in_channels=in_ft,
#             out_channels=cnn_hid_dim,
#             kernel_size=5,
#             padding=2,
#             stride=1,
#         )
#         self.linear = nn.Linear(in_features=cnn_hid_dim, out_features=fc_hid_dim)
#         self.linear2 = nn.Linear(fc_hid_dim, out_ft)
#         self.relu = nn.ReLU()

#     def forward(self, x, h, l):
#         """
#         x.shape = steps x (n1-1) x num_ft
#         h.shape = steps x (n1-1) x latent_dim
#         l.shape = (n1-1) x 1
#         """

#         batch_size = x.shape[1]
#         l_ = l / l.sum()
#         l_ = l_.T
#         l_ = l_.reshape(1, 27, 1)
#         x_ = torch.cat((x, h), dim=2)  # timestep x nodes x hidden feat
#         # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
#         x_ = x_.reshape(1, x_.shape[1], x_.shape[0])
#         ret = self.cnn(x_)
#         ret = torch.bmm(ret, l_)
#         ret = ret.reshape(ret.shape[0], -1)
#         ret = self.linear(ret)
#         ret = self.relu(ret)
#         ret = self.linear2(ret)
#         ret = self.relu(ret)
#         return ret

if __name__ == "__main__":
    decoder = Decoder(61,1,1,128,64)
    print(decoder(torch.rand(12,27,1),torch.rand(12,27,60),torch.rand(27,1)).shape)