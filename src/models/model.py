import torch.nn as nn

class End2End(nn.Module):
    def __init__(self,encoder,decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,x,adj,l,climate):
        # breakpoint()
        h = self.encoder(x,adj)
        x = x[-1].unsqueeze(0)
        ret = self.decoder(x,h,l,climate)
        return ret