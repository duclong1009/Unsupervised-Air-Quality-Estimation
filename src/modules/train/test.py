import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)


def cal_acc(y_prd, y_grt):
    mae = mean_absolute_error(y_grt, y_prd)
    mse = mean_squared_error(y_grt, y_prd, squared=True)
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    rmse = mean_squared_error(y_grt, y_prd, squared=False)
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    return mae, mse, mape, rmse, r2, corr


def test_atten_decoder_fn(
    stdgi, decoder, dataloader, device,criterion, interpolate=False, scaler=None,test=True
):
    decoder.eval()
    stdgi.eval()
    
    list_prd = []
    list_grt = []
    # breakpoint()
    epoch_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            batch_loss = 0
            for index in range(data["X"].shape[0]):
                y_grt = data["Y"][index].to(device).float()
                x = data["X"][index].to(device).float()
                G = data["G"][index][0].to(device).float()
                l = data["l"][index].to(device).float()
                cli = data["climate"][index].to(device).float()
                if not interpolate:
                    h = stdgi.embedd(x, G.unsqueeze(0))
                    y_prd = decoder(x[-1].unsqueeze(0), h, l,cli)  # 3x1x1
                else:
                    h, enc_hidd = stdgi.embedd(x, G.unsqueeze(0), l)
                    y_prd = decoder(x[-1].unsqueeze(0), h, l)
                batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
                
                y_prd = torch.squeeze(y_prd).cpu().detach().numpy()
                y_grt = torch.squeeze(y_grt).cpu().detach().numpy()
                list_prd.append(y_prd)
                list_grt.append(y_grt)
            batch_loss = batch_loss / data["X"].shape[0]
            epoch_loss += batch_loss.item()
    if test:
        a_max = scaler.data_max_[0]
        a_min = scaler.data_min_[0]
        list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
        list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
        list_grt_ = [float(i) for i in list_grt]
        list_prd_ = [float(i) for i in list_prd]
        return list_prd_, list_grt_, epoch_loss / len(dataloader)
    else:
        return epoch_loss / len(dataloader)