import sched
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy
from modules.utils.utilities import mape_loss
from sklearn.metrics import r2_score
import pandas as pd 
import time 
class BlstmIDW:
    def __init__(self, log_dir, params):
        self.log_dir = log_dir
        self.params = params 

    def test():
        pass 

def train_loop_bilstm_idw(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0 
    for batch, (input, target) in enumerate((dataloader)):
        # forward pass
        input = input.to(device)
        target = target.to(device)
        pred = model(input)
        loss = torch.sqrt(loss_fn(pred, target)) # rmse loss
        total_loss += loss.item()
    
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # print loss
    train_loss = total_loss / len(dataloader)
    print(f'Train: {train_loss:>.4f}')
    return train_loss
import numpy as np
def eval_loop_bilstm_idw(model, dataloader, loss_fn, device, scheduler): 
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.eval()
        total_loss = 0 
        for batch, (input, target) in enumerate((dataloader)):
            # forward pass
            input = input.to(device)
            target = target.to(device)
            pred = model(input)
            loss = torch.sqrt(loss_fn(pred, target)) # rmse loss 
            total_loss += loss.item()
        # print loss
        # import pdb; pdb.set_trace()
        val_loss = total_loss / len(dataloader)
        scheduler.step(val_loss)
        print(f'Validation: {val_loss:>.4f}')
        return val_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mae_func(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
def mse_func(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)
def mape_func(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mdape_func(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100

def test_loop_bilstm_idw(model, dataloader, idw_matrix, device, output_dir, test_idx, train_time): 
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.eval()
        rmse, mse, mae, mape, mdape, r2 = {}, {}, {}, {} , {}, {}
        predict, gt = {}, {}

        test_time = 0.0
        for input, target in dataloader:
            # forward pass
            # import pdb; pdb.set_trace()
            input = input.to(device)
            target = target.to(device)
            num_test_stations = target.shape[1]

            for i in range(num_test_stations):
                # Change idweight of another test station.
                # test_model.idw.idweight = idw_matrix[[i]]
                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))

                pred_i = test_model(input).squeeze(1)

                torch.cuda.synchronize()
                total_inference_time = int(round(time.time()*1000)) - inference_time_start 
                test_time += total_inference_time

                target_i = target[:, i]
                if i not in mse.keys():
                    rmse[i], mse[i], mae[i], mape[i], mdape[i], r2[i] = 0, 0, 0, 0, 0, 0
                    predict[i], gt[i] = [], [] 
                # agregate metrics                     
                
                # rmse[i] += torch.sqrt(F.mse_loss(pred_i, target_i, reduction='sum'))
                # mse[i] += F.mse_loss(pred_i, target_i, reduction='sum')
                # mae[i] += F.l1_loss(pred_i, target_i, reduction='sum')
                # mape[i] += mape_loss(pred_i, target_i, device, reduction='sum')

                predict[i] += pred_i.tolist()
                gt[i] += target_i.tolist()
            
        # num_entries = len(dataloader.dataset) * len(idw_matrix)
        num_entries = len(dataloader.dataset)

        model_params =  sum(p.numel() for p in test_model.parameters())
        
        results = [] 
        for i, stat in enumerate(test_idx):
            mse[i] = mse_func(gt[i], predict[i])
            mae[i] = mae_func(gt[i], predict[i])
            mape[i] = mape_func(gt[i], predict[i])
            mdape[i] = mdape_func(gt[i], predict[i]) 
            r2[i] = r2_score(gt[i], predict[i])
            rmse[i] = np.sqrt(mse[i])
            # rmse[i] = (rmse[i] / num_entries).item()
            # mse[i] = (mse[i] / num_entries).item()
            # mae[i] = (mae[i] / num_entries).item()
            # mape[i] = (mape[i] / num_entries).item()

            results.append({"Station": stat, "RMSE": rmse[i], "MSE": mse[i], "MAE": mae[i], "MAPE": mape[i], "MDAPE": mdape[i], "R2": r2[i], 'No params': model_params, 'Train time': train_time, 'Test time': test_time})
            pred_gt = {"Groundtruth": gt[i], "Predict": predict[i]}
            df = pd.DataFrame(data=pred_gt, columns=['Groundtruth', "Predict"])
            df.to_csv(output_dir + 'Station_{}.csv'.format(stat))        
        return results
