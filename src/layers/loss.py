import torch

def linex_loss(y_gt,y_pred,alpha= -0.1):
    diff = y_pred - y_gt
    exp = torch.exp(alpha * diff)
    linear = alpha * diff
    return torch.clip(exp - linear -1, 1e-8, 1e8)