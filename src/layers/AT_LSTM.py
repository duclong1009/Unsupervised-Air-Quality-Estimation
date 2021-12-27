import torch
import torch.nn as nn

class AT_LSTM(nn.Module):
    #input_shape : (12,27,8)
    def __init__(self,n_ts,n_fts,hidden_size):
        self.LSTM = nn.LSTM(n_ts,hidden_size,batch_first=False)
    def forward(self,x):
        x_ = torch.transpose(x,0,2)
        print(x_.shape)

if __name__ == '__main__':
    model = AT