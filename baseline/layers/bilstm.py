import torch 
import torch.nn as nn 

class BiLSTM(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, 
        lstm_num_layers, linear_hidden_size 
    ):
        super().__init__()
        self.blstm = nn.LSTM(
            lstm_input_size, lstm_hidden_size, 
            lstm_num_layers, bidirectional=True
        )
        self.batchnorm1 = nn.BatchNorm1d(lstm_hidden_size)
        self.linear = nn.Linear(lstm_hidden_size, linear_hidden_size)
        self.relu = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(linear_hidden_size)
        
    def forward(self, input):
        """
        input shape: (batch, seq_length)
        output shape: (batch, linear_hidden_output)
        """
        # BLSTM layer
        input = input.T.unsqueeze(-1)      # shape (window, batch, 1)
        lstm_output, _ = self.blstm(input)
        lstm_hidden_size = int(lstm_output.shape[-1] / 2)
        lstm_output_forward = lstm_output[-1, :, :lstm_hidden_size]
        lstm_output_backward = lstm_output[0, :, lstm_hidden_size:]
        lstm_output = lstm_output_forward + lstm_output_backward
        lstm_output = self.batchnorm1(lstm_output)
        
        # FC+BN Layer
        linear_output = self.linear(lstm_output)
        linear_output = self.relu(linear_output)
        output = self.batchnorm2(linear_output)
        return output
