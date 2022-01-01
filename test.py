from src.layers.encoder import BaseEncoder, Attention_Encoder, Encoder, InterpolateAttentionEncoder
from src.layers.discriminator import Discriminator
from src.models.stdgi import Attention_STDGI, InterpolateAttention_STDGI
from src.models.decoder import InterpolateAttentionDecoder
import torch
import torch.nn as nn 
import numpy as np 

if __name__ == "__main__":
    # atten_layer = AttentionLSTM(1,60,120,12,0.1)
    # print(atten_layer(torch.rand(12,27,1)).shape)


    # atten_encoder = InterpolateAttentionEncoder(1, 60, 120, 200, "relu")

    # model = InterpolateAttentionEncoder(
    #     in_ft=6, out_ft=60, hid_ft1=120, hid_ft2=240, act='relu'
    # )
    # print(
    #     model.forward(torch.rand(12,28,6), torch.rand(1,28,28), torch.rand(1,27) ).shape
    # )

    # model_disc = Discriminator(x_ft=6, h_ft=60, hid_ft=10)
    # print(
    #     model_disc.forward(torch.rand(1,28,60), torch.rand(1,28,6), torch.rand(1,28,6)).shape
    # # )
    # model = Attention_STDGI(6, 60, 120, 240, 10, "relu")    
    # print(
    #     model.forward(torch.rand(12, 27, 6), torch.rand(12, 27, 6), torch.rand(1, 27, 27)).shape # (1, 28, 60)
    # )


    # model = InterpolateAttention_STDGI(6, 60, 120, 240, 10, "relu")
    # print(
    #     model.forward(torch.rand(12, 27, 6), torch.rand(12, 27, 6), torch.rand(1, 28, 28),  torch.rand(1,27)  ).shape
    # )

    model = InterpolateAttentionDecoder(in_ft=8, out_ft=1)
    print(model.forward(torch.rand(1,28,6), torch.rand(1,28,2) ).shape )
