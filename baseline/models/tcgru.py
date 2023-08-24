import torch
import torch.nn as nn
import numpy as np

from layers.timedistributed import TimeDistributed

import torch
import torch.nn as nn
import numpy as np
from layers.timedistributed import TimeDistributed

class TCGRU(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.h, self.w = params[-2:]
        self.conv2d = TimeDistributed(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8)
        )
        self.selu1 = TimeDistributed(nn.ReLU())
        self.flatten = TimeDistributed(nn.Flatten())
        self.dropout = nn.Dropout(p=0.6)
        # 7488 -uk
        # 15232
        # 6528 - beijing 
        self.gru = nn.GRU(
            7488, self.h*self.w, 2,
            dropout=0.6,
            bias=True
        )
        self.linear = nn.Linear(self.h*self.w, self.h * self.w)
        self.selu2 = nn.SELU()
        self.batchnorm1d = nn.BatchNorm1d(self.h * self.w)
        self.dropout = nn.Dropout()

    def forward(self, input):
        """
        input: (batch, seq_len, 1, H, W)
        output: (batch, 1, H, W)
        """
        # breakpoint()
        # import pdb; pdb.set_trace()
        input = input.squeeze()
        conv_out  = self.conv2d(input)
        conv_out = self.selu2(conv_out)
        # conv_out_ = self.selu1(self.flatten(input))
        conv_out_ = self.flatten(conv_out)
        conv_out_ = torch.swapaxes(conv_out_, 0, 1)
        gru_out, _ = self.gru(conv_out_)
        gru_out = gru_out[-1]

        lin_out = self.linear(gru_out)
        lin_out = self.dropout(lin_out)
        lin_out_ = self.selu2(lin_out)
        output = lin_out_.view(lin_out.shape[0], self.h, self.w)
        # print(output)
        return output
    