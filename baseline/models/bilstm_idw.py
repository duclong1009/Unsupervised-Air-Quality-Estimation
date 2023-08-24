import torch 
import torch.nn as nn 
from layers.bilstm import BiLSTM
from layers.idw import IDWLayer

class IDW_BiLSTM_Model(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, 
        lstm_num_layers, linear_hidden_size, 
        idw_hidden_size, idweight, device
    ):
        super().__init__()
        num_train = idweight.shape[1]
        self.num_train = num_train
        self.idweight = idweight
        self.blstm = nn.ModuleList(
            [BiLSTM(
                lstm_input_size, lstm_hidden_size, 
                lstm_num_layers, linear_hidden_size
            ) for _ in range(num_train)])
        self.idw = IDWLayer( 
            linear_hidden_size * num_train,
            idw_hidden_size, idweight, device 
        )
        self.linear = nn.Linear(idw_hidden_size, 1)
        
    def forward(self, X):
        """
        X shape: (batch, seq_len, num_train)
        output shape: (batch, 1)
        """
        blstm_output = []
        # iterate over stations
        num_train = self.num_train
        # import pdb; pdb.set_trace()

        for i in range(num_train):
            Xi = X[:, :, i]      # data of ith station
            bltsm_output_i = self.blstm[i](Xi)
            blstm_output.append(bltsm_output_i)
        blstm_output = torch.cat(blstm_output, axis=1)
        
        idw_output = self.idw(blstm_output)
        output = self.linear(idw_output)
        return output

