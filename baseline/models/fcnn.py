import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, config, device):
        super(FCNN, self).__init__()
        
        self.config = config  
        self.device = device    
        self.fc1=  nn.Linear(config['num_neigh'], 16)
        self.fc2 = nn.Linear(config['meteo_dim'], 16)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        self.prediction_layer = nn.Sequential(
                self.fc4,
                nn.ReLU(),
                self.fc5,
                nn.ReLU(),
                self.fc6, 
                nn.ReLU(), 
                self.fc7,
                nn.ReLU(),
                self.fc_out,
            )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x, meteo):
        x_embed = self.leakyrelu(self.fc1(x))
        meteo_embed = self.leakyrelu(self.fc2(meteo))
        embed = torch.cat([x_embed, meteo_embed], axis=-1)
        out = self.prediction_layer(embed)
        # out = self.fc_out(self.relu(self.fc5(self.relu(self.fc4(embed)))))
        return out  
