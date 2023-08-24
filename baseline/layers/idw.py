import torch 
import torch.nn as nn 

class IDWLayer(nn.Module):
    def __init__(self, input_size, hidden_size, idweight, device):
        super().__init__()
        self.idweight = idweight
        self.input_size = input_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device 
        
    def forward(self, input):
        """
        input shape: (batch, input_size)
        output shape: (batch, hidden_size)
        """
        # import pdb; pdb.set_trace()

        num_train = self.idweight.shape[1]
        weight = self.idweight.T * torch.ones(self.idweight.shape[0],
            int(self.input_size / num_train), 
            device=self.device
        )
        weight = weight.reshape((1, -1))
        input = input * weight
        output = self.linear(input)
        output = self.sigmoid(output)
        return output