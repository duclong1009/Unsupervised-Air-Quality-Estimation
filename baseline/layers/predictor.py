import torch 
import torch.nn as nn 
from modules.utils.interpolate import interpolate_pm

class Predictor(nn.Module): 
    def __init__(self, loc_df, test_idx, net_loc, params, device, train=False):
        self.test_loc = loc_df.iloc[test_idx].to_numpy()
        self.net_loc = net_loc
        self.params = params
        self.device = device
        self.train = train
        
    def __call__(self, net_pm):
        """
            Interpolate test_pm from predicted net_pm.
            net_pm: tensor (batch, 1, H, W)
        """
        if self.train:
            net_pm = net_pm.cpu().detach().numpy()
            net_pm = net_pm.reshape(net_pm.shape[0], -1)
            # import pdb; pdb.set_trace()
            pred = interpolate_pm(
                self.test_loc, self.net_loc,
                net_pm, self.params, reshape=False 
            )
            return torch.tensor(pred, requires_grad=True, dtype=torch.float32).to(self.device)

        else:
            net_pm = net_pm.cpu().numpy().reshape(net_pm.shape[0], -1)
            pred = interpolate_pm(
                self.test_loc, self.net_loc,
                net_pm, self.params, reshape=False 
            )
            return torch.from_numpy(pred).to(self.device)