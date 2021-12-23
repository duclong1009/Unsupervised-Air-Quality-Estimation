import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, drop_prob=0.5):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.drop_prob = drop_prob
        self.seq_len = seq_len

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.seq_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.bn_1 = nn.BatchNorm1d(self.hidden_size)
        self.mlp_1 = nn.Linear(self.hidden_size, output_size)
        self.act_1 = nn.ReLU()
        self.mlp_2 = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        transposed_inputs = torch.transpose(inputs, 0, 1)
        output, h = self.lstm1(transposed_inputs)
        enc_output, (h_s, c_s) = self.lstm2(output, h)

        # Dùng torch zeros để khởi tạo input cũng được. Hoặc dùng time step trước đó là inputs[:, -1, :]
        # embedded = self.embedding(torch.zeros(inputs.size(0), 1, inputs.size(2)))
        embedded = self.embedding(transposed_inputs[:, -1:, :])
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, h_s[-1].unsqueeze(1)), 2)), dim=2
        )
        attn_applied = torch.bmm(attn_weights, enc_output)
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, _ = self.lstm3(output, (h_s, c_s))
        output = output[:, -1, :]
        output = self.bn_1(
            output
        )  # batch norm cần nhiều hơn 1 giá trị. (batch_size != 1)
        output = self.dropout(output)
        output = self.mlp_1(output)
        output = F.relu(output)
        output = self.mlp_2(output)
        return output


# if __name__ == "__main__":
#   atten_layer = AttentionLSTM(1,60,120,12,0.1)
#   print(atten_layer(torch.rand(12,27,1)).shape)
