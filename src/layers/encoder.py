import torch
import torch.nn as nn
import torch
from src.layers.attention import AttentionLSTM

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


class BaseEncoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(BaseEncoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.gcn = GCN(hid_ft1, hid_ft2, "relu")
        self.gcn2 = GCN(hid_ft2, out_ft, "relu")
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.fc(x)
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x


class Encoder(nn.Module):
    # hshape = 64*28*128
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.gcn = GCN(hid_ft1, hid_ft2, "relu")
        self.gcn2 = GCN(hid_ft2, out_ft, "relu")
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.fc(x)
        x = self.relu(x)
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn = nn.LSTM(hid_ft1, hid_ft1, batch_first=False, num_layers=1)
        self.attn = AttentionLSTM(hid_ft1, 120, hid_ft1, 12, 0.1)
        self.gcn = GCN(hid_ft1, hid_ft2, act)
        self.gcn2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x))
        x, h = self.rnn(x)
        x = self.attn(x)
        x = self.relu(x.unsqueeze(0))
        x = self.gcn(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x
