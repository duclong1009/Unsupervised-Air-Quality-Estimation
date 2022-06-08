# from builtins import breakpoint
from builtins import breakpoint
from src.layers.attention import DotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=7,
        num_input_stat=7,
    ):
        super(Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        if rnn == "GRU":
            self.rnn = nn.GRU(
                in_ft * (self.num_input_stat),
                in_ft * (self.num_input_stat),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        else:
            self.rnn = nn.RNN(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim * 2, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)

    def forward(self, x, h, l, climate):
        """
        x.shape = batch_size x (n1-1) x num_ft
        h.shape = batch_size x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """
        x = x[:, -1, :, :]
        batch_size = x.shape[0]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        x_size = x_.shape
        if len(x_.shape) == 4:
            x_ = x_.reshape(x_size[0], x_size[1], -1)
        else:
            x_ = x_.reshape(x_size[0], 1, -1)
        # hid_state = torch.zeros(1, batch_size, self.in_ft).to(DEVICE)
        output, hid_state = self.rnn(x_)  # output = (bs, 12, 426)
        # output = self.relu(self.fc(x_))
        output = output.reshape(
            output.shape[0], output.shape[1], self.num_input_stat, -1
        )
        x_ = output[:, -1]
        ret = self.fc(x_)
        ret = ret.permute(0, 2, 1)
        ret = torch.bmm(ret, l_)  # ret= (1, 128, 27) * (1, 27, 1) = (1, 128, 1)
        ret = ret.reshape(ret.shape[0], -1)  # output =
        embed = self.embed(climate)
        ret = torch.cat((ret, embed), dim=-1)
        ret = self.linear(ret)  # (128, 1)
        ret = self.relu(ret)  # (128,64)
        ret = self.linear2(ret)  # (64,1)
        return ret


class WoCli_Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=7,
        num_input_stat=7,
    ):
        super(WoCli_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        if rnn == "GRU":
            self.rnn = nn.GRU(
                in_ft * (num_input_stat - 1),
                in_ft * (num_input_stat - 1),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(
                in_ft * (num_input_stat - 1),
                in_ft * (num_input_stat - 1),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        else:
            self.rnn = nn.RNN(
                in_ft * (num_input_stat - 1),
                in_ft * (num_input_stat - 1),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        self.num_input_stat = num_input_stat - 1
        # self.selector2 = torch.nn.Parameter(torch.ones(1,3))
        # self.selector1 = torch.nn.Parameter(torch.ones(1,9))
        # self.cnn = nn.Conv1d(
        #     in_channels=in_ft,
        #     out_channels=cnn_hid_dim,
        #     kernel_size=5,
        #     padding=2,
        #     stride=1,
        # )
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)

    def forward(self, x, h, l, climate):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """
        batch_size = x.shape[0]
        l_ = l / l.sum()
        l_ = l_.T
        l_ = l_.reshape(batch_size, -1, 1)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        x_size = x_.shape
        if len(x_.shape) == 4:
            x_ = x_.reshape(x_size[0], x_size[1], -1)
        else:
            x_ = x_.reshape(x_size[0], 1, -1)

        output, hid_state = self.rnn(x_)  # output = (bs, 12, 426)
        # output = self.relu(self.fc(x_))
        # import pdb; pdb.set_trace()
        output = output.reshape(
            output.shape[0], output.shape[1], self.num_input_stat, -1
        )
        x_ = output[:, -1]
        # x_ = torch.unsqueeze(x_,0) # (1, 27, 60)
        # breakpoint()
        ret = self.fc(x_)
        ret = ret.permute(0, 2, 1)
        ret = torch.bmm(ret, l_)  # ret= (1, 128, 27) * (1, 27, 1) = (1, 128, 1)
        ret = ret.reshape(ret.shape[0], -1)  # output =
        ret = self.linear(ret)  # (128, 1)
        ret = self.relu(ret)  # (128,64)
        ret = self.linear2(ret)  # (64,1)
        # ret = self.relu(ret) # (1)
        return ret


class Global_Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=7,
        num_input_stat=7,
    ):
        super(Global_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        if rnn == "GRU":
            self.rnn = nn.GRU(
                in_ft * (self.num_input_stat),
                in_ft * (self.num_input_stat),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        else:
            self.rnn = nn.RNN(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim * 2, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.query = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key = nn.Linear(cnn_hid_dim, cnn_hid_dim)
        self.value = nn.Linear(cnn_hid_dim, cnn_hid_dim)
        self.atten = DotProductAttention()

    def forward(self, x, h, l, climate):
        """
        x.shape = batch_size x (n1-1) x num_ft
        h.shape = batch_size x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """
        x = x[:, -1, :, :]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        # output = self.relu(self.fc(x_))
        ret = self.relu(self.fc(x_))
        ret_ = ret.permute(0, 2, 1)
        interpolation_ = torch.bmm(ret_, l_)
        interpolation_ = interpolation_.reshape(ret.shape[0], -1)
        embed = self.embed(climate)
        query = self.query(torch.cat((interpolation_, embed), dim=-1))
        value = self.value(ret)
        key = self.key(ret)
        atten_weight = self.atten(key, query)
        atten_vector = torch.bmm(atten_weight.unsqueeze(1), value).squeeze()
        ret = self.linear(
            torch.cat((atten_vector, embed), dim=-1)
        )  # (128, 1) # (128, 1)
        ret = self.relu(ret)  # (128,64)
        ret = self.linear2(ret)  # (64,1)
        return ret


class Local_Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=7,
        num_input_stat=7,
    ):
        super(Local_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        if rnn == "GRU":
            self.rnn = nn.GRU(
                in_ft * (self.num_input_stat),
                in_ft * (self.num_input_stat),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        else:
            self.rnn = nn.RNN(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim*2, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.query = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.value = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.atten = DotProductAttention()

    def forward(self, x, h, l, climate):
        """
        x.shape = batch_size x (n1-1) x num_ft
        h.shape = batch_size x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """
        x = x[:, -1, :, :]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        # output = self.relu(self.fc(x_))
        ret = self.relu(self.fc(x_))
        ret_ = ret.permute(0, 2, 1)
        interpolation_ = torch.bmm(ret_, l_)
        interpolation_ = interpolation_.reshape(ret.shape[0], -1)
        embed = self.embed(climate)
        query = self.query(torch.cat((interpolation_, embed), dim=-1))
        # ret [batch, hidden_ft, n_node]
        value = self.value(ret_)
        key = self.key(ret_)
        atten_weight = self.atten(key, query)
        atten_vector = torch.bmm(atten_weight.unsqueeze(1), value).squeeze()
        ret = self.linear(torch.cat((atten_vector, embed), dim=-1))  # (128, 1)
        ret = self.relu(ret)  # (128,64)
        ret = self.linear2(ret)  # (64,1)
        return ret

class Local_Global_Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=7,
        num_input_stat=7,
    ):
        super(Local_Global_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        if rnn == "GRU":
            self.rnn = nn.GRU(
                in_ft * (self.num_input_stat),
                in_ft * (self.num_input_stat),
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        else:
            self.rnn = nn.RNN(
                in_ft * self.num_input_stat,
                in_ft * self.num_input_stat,
                batch_first=True,
                num_layers=n_layers_rnn,
            )
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim*3, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.query_local = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.value_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.atten = DotProductAttention()

        self.query = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key = nn.Linear(cnn_hid_dim, cnn_hid_dim)
        self.value = nn.Linear(cnn_hid_dim, cnn_hid_dim)
    def forward(self, x, h, l, climate):
        """
        x.shape = batch_size x (n1-1) x num_ft
        h.shape = batch_size x (n1-1) x latent_dim
        l.shape = (n1-1) x 1 = (27* 1)
        """
        x = x[:, -1, :, :]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        # output = self.relu(self.fc(x_))
        ret = self.relu(self.fc(x_))
        ret_ = ret.permute(0, 2, 1) # ret_ [batch, hidden_ft, n_node]
        interpolation_ = torch.bmm(ret_, l_)
        interpolation_ = interpolation_.reshape(ret.shape[0], -1)
        embed = self.embed(climate)

        query_local = self.query_local(torch.cat((interpolation_, embed), dim=-1))
        value_local = self.value_local(ret_)
        key_local = self.key_local(ret_)
        atten_weight_local = self.atten(key_local, query_local)
        atten_vector_local = torch.bmm(atten_weight_local.unsqueeze(1), value_local).squeeze()

        query = self.query(torch.cat((interpolation_, embed), dim=-1))
        value = self.value(ret)
        key = self.key(ret)
        atten_weight = self.atten(key, query)
        atten_vector = torch.bmm(atten_weight.unsqueeze(1), value).squeeze()     

        ret = self.linear(torch.cat((atten_vector_local,atten_vector, embed), dim=-1))  # (128, 1)
        ret = self.relu(ret)  # (128,64)
        ret = self.linear2(ret)  # (64,1)
        return ret
if __name__ == "__main__":
    decoder = Decoder(61, 1, 1, 128, 64)
    print(
        decoder(torch.rand(12, 27, 1), torch.rand(12, 27, 60), torch.rand(27, 1)).shape
    )
