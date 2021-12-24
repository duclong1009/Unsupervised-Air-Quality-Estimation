import torch
import torch.nn as nn
import torch.nn.functional as F 

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
class InterpolateDecoder(nn.Module):
    def __init__(
        self, in_ft, out_ft, n_layers_rnn=1, rnn="GRU", cnn_hid_dim=128, fc_hid_dim=64
    ):
        super(InterpolateDecoder, self).__init__()
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

        x_inter = get_interpolate(x, l) #(1, 28, 7)

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


class InterpolateAttentionDecoder(nn.Module):
    def __init__(
        self, in_ft ,out_ft, num_stat, en_hid1=200,stdgi_out_dim=2, n_layers_rnn=1, rnn="GRU", cnn_hid_dim=128, fc_hid_dim=64, drop_prob=0.5
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

        self.embedding = nn.Linear(in_ft-stdgi_out_dim, in_ft-stdgi_out_dim) # (28,1)
        self.dropout = nn.Dropout(drop_prob)
        self.linear3 = nn.Linear(en_hid1 * 2, in_ft - stdgi_out_dim ) # linear of catted encoder hidden [h,c] 
        self.linear4 = nn.Linear(num_stat, 1)

        self.attn = nn.Linear((in_ft-stdgi_out_dim) * 2, in_ft-stdgi_out_dim)

    def forward(self, x, h, enc_hidd,l):
        """
        x.shape = steps x (n1-1) x num_ft
        h.shape = steps x n1 x latent_dim
        l.shape = n1 x 1 = (27* 1)
        enc_hidden = steps * n1 * en_hid1(200)
        """ 
        x_inter = get_interpolate(x, l) #(1, 20, 6)
        # x_ = x_inter.reshape(x_inter.shape[0], x_inter.shape[2], x_inter.shape[1]) # (1,28,6) -> (1, 6, 28)
        
        embedded = self.embedding(x_inter ) # (1, 20,6) -> (1, 20, 6)
        embedded = self.dropout(embedded)  

        encoder_hidden = enc_hidd[0] # h
        encoder_context = enc_hidd[1] # c

        cated_encoder = torch.cat((encoder_hidden, encoder_context), 2) #(1,20,200) + (1,20,200) -> (1, 20,400)
        cated_encoder= self.linear3(cated_encoder) # (1, 20, 400) -> (1, 20, 6)

        cated_encoder_embedded = torch.cat((cated_encoder, embedded) ,2) # (1, 20, 6) * 2 -> (1, 20, 12)
        cated_encoder_embedded = cated_encoder_embedded.reshape(cated_encoder_embedded.shape[0], cated_encoder_embedded.shape[2], cated_encoder_embedded.shape[1]) # (1, 12, 20)
        cated_encoder_embedded = self.linear4(cated_encoder_embedded) # (1, 12, 1)
        cated_encoder_embedded = torch.squeeze(cated_encoder_embedded, 2) # (1, 12)
        # import pdb; pdb.set_trace()
        attn_weights = F.softmax(self.attn(cated_encoder_embedded), dim=1) # embeded (1,12) => attn (1,6) 
        dim_2 = attn_weights.shape[1]
        attn_weights = attn_weights.unsqueeze(1).repeat(1, dim_2, 1)
        attn_applied = torch.bmm(x_inter, attn_weights) # x_inter=(1, 20, 6) attn_weights=(1, 6,6) => (1, 20, 6)

        x_ = torch.cat((attn_applied, h), dim=2)  # timestep x nodes x hidden feat (1,20,6) (1,20,2)
        
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


if __name__ == "__main__":
    decoder = Decoder(61,1,1,128,64)
    print(decoder(torch.rand(12,27,1),torch.rand(12,27,60),torch.rand(27,1)).shape)