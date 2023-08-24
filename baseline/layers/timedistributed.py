import torch 
import torch.nn as nn 

class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input):
        """
        input: shape (batch, time, *)
        output: shape (batch, time, *)
        """
        output = []
        # breakpoint()
        for t in range(input.shape[1]):
            # import pdb; pdb.set_trace()
            output_t = self.layer(input[:, t].unsqueeze(1))
            output_t = output_t.unsqueeze(1)
            output.append(output_t)
        # breakpoint()
        output = torch.cat(output, axis=1)
        return output
