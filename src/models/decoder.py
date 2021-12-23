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
        l.shape = (n1-1) x 1
        """

        batch_size = x.shape[1]
        time_st = x.shape[0]
        l_ = l / l.sum()
        l_ = l_.T
        a = l_.shape[0]
        # print(a)
        l_ = l_.reshape(time_st, batch_size, 1)
        x_ = torch.cat((x, h), dim=2)  # timestep x nodes x hidden feat
        # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
        output, hid_state = self.rnn(x_)
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