import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def cal_acc(y_prd,y_grt):
    mae = mean_absolute_error(y_grt,y_prd)
    mse = mean_squared_error(y_grt,y_prd)
    corr = np.corrcoef(np.reshape(y_grt,(-1)),np.reshape(y_prd,(-1)))[0][1]
    return mae,mse,corr

def test_atten_decoder_fn(stdgi, decoder, dataloader, device, interpolate=False):
    decoder.eval()
    stdgi.eval()
    epoch_loss = 0
    list_prd = []
    list_grt = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            for index in range(data["X"].shape[0]):
                y_grt = data["Y"][index].to(device).float()
                x = data["X"][index].to(device).float()
                G = data["G"][index][0].to(device).float()
                l = data["l"][index].to(device).float()
                if not interpolate:
                    h = stdgi.embedd(x, G.unsqueeze(0))
                    y_prd = decoder(x[-1].unsqueeze(0), h, l)  # 3x1x1
                else:
                    h, enc_hidd = stdgi.embedd(x, G.unsqueeze(0), l)
                    y_prd = decoder(x[-1].unsqueeze(0), h, enc_hidd, l)
                y_prd = torch.squeeze(y_prd).cpu().detach().numpy()
                y_grt = torch.squeeze(y_grt).cpu().detach().numpy()
                list_prd.append(y_prd)
                list_grt.append(y_grt)
    return  list_prd, list_grt
