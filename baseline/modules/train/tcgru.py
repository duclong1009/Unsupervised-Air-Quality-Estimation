import torch
from tqdm.notebook import tqdm
import copy
import torch.nn.functional as F
from modules.utils.utilities import mape_loss
import numpy as np
import pandas as pd
import time 

def train_loop_tcgru(model, dataloader, predictor, loss_fn, optimizer, device, scheduler=None):
    model.train()
    total_loss = []
    mse = mae = mape = 0.
    for input, target in dataloader:
        optimizer.zero_grad()
        # forward pass
        input = input.to(device)
        target = target.to(device)
        net_pm = model(input)
        # import pdb
        # pdb.set_trace()
        # pred = predictor(net_pm)
        # import pdb
        # pdb.set_trace()

        loss = loss_fn(net_pm, target.squeeze())
        total_loss.append(loss)
        # backward pass
        loss.backward()
        optimizer.step()

    total_loss = torch.tensor(total_loss).mean().sqrt().item()
    scheduler.step(total_loss)
    print(f'train loss: {total_loss:>.4f}')
    return total_loss

    #     loss = loss_fn(pred, target)
    #     total_loss.append(loss)

    #     # backward pass
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # # print loss
    # total_loss = torch.tensor(total_loss).mean().sqrt().item()
    # print(f'train loss: {total_loss:>.4f}')
    # return total_loss

def mdape_func(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100


def eval_loop_tcgru(model, dataloader, predictor, device, scheduler=None):
    mse = mae = mape =mdape = 0.
    test_model = copy.deepcopy(model)
    test_model.eval()

    with torch.no_grad():
        for input, target in dataloader:
            # forward pass
            input = input.to(device)
            target = target.to(device)
            net_pm = test_model(input)
            pred = predictor(net_pm)

            mse += F.mse_loss(pred, target, reduction='sum')
            mae += F.l1_loss(pred, target, reduction='sum')
            mape += mape_loss(pred, target, device, reduction='sum')
            mdape += mdape_func(target, pred)
    num_entries = len(dataloader.dataset) * len(predictor.test_loc)
    rmse = torch.sqrt(mse / num_entries).item()
    mae = (mae / num_entries).item()
    mape = (mape / num_entries).item()
    # scheduler.step(mse)

    print(f'val loss  : {rmse:>.4f} | {mae:>.4f} | {mape:>.4f}')
    return rmse, mae, mape,mdape
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

from sklearn.metrics import r2_score
def test_loop(model, test_dataloader, predictor, device, args, output_dir, scaler,train_time):
    test_stats = args.test_station
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.eval()
        mse = mae = mape =mdape= 0
        lst_mse = {}  # mse error of 12 station
        lst_mae = {}  # mae error of 12 station
        lst_mape = {}  # mape error of 12 station
        lst_mdape = {}
        lst_r2 = {} # r2 error of 12 station
        predict, gt = {}, {}
        test_time = 0.0 

        for input, target in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            torch.cuda.synchronize()
            test_it_start = int(round(time.time()*1000))

            net_pm = test_model(input)
            pred = predictor(net_pm)

            torch.cuda.synchronize()
            time_elapsed = int(round(time.time()*1000)) - test_it_start
            test_time += time_elapsed
            # breakpoint()
            tp_pred = np.transpose(pred.cpu().numpy())  # 32,12 -> 12,32
            tp_target = np.transpose(target.cpu().numpy())

            num_stat_shape = tp_pred.shape 
            for idx in range(num_stat_shape[0]):
                tp_pred[idx, :] = np.squeeze(scaler.inverse_transform(tp_pred[idx, :].reshape(-1,1)), 1)
                tp_target[idx, :] = np.squeeze(scaler.inverse_transform(tp_target[idx, :].reshape(-1,1)), 1)

            # tp_pred = scaler.inverse_transform(tp_pred)
            # tp_target = scaler.inverse_transform(tp_target)
            for i, stat in enumerate(test_stats):
                if str(stat) not in lst_mse.keys():
                    lst_mse.update({str(stat): 0})
                    lst_mae.update({str(stat): 0})
                    lst_mape.update({str(stat): 0})
                    lst_mdape.update({str(stat): 0})
                    lst_r2.update({str(stat): 0})
                    predict.update({str(stat): []})
                    gt.update({str(stat): []})
                lst_mse[str(stat)] += F.mse_loss(torch.from_numpy(tp_pred[i,:]),
                                                 torch.from_numpy(tp_target[i,:]), reduction='sum')
                lst_mae[str(stat)] += F.l1_loss(torch.from_numpy(tp_pred[i,:]),
ch.from_numpy(tp_target[i,:]), reduction='sum')
                lst_mape[str(stat)] += mape_loss(torch.from_numpy(tp_pred[i,:]).to(device),
                                                 torch.from_numpy(tp_target[i,:]).to(device), device, reduction='sum')
                # breakpoint()
                # lst_mdape[str(stat)] += mdape_func(tp_target[i], tp_pred[i])
                # lst_r2[str(stat)] += r2_score(tp_target[i], tp_pred[i])
                # predict[str(stat)] += scaler.inverse_transform(tp_pred[i].reshape(-1,1) )
                # gt[str(stat)] += scaler.inverse_transform(tp_target[i].reshape(-1,1) )
                predict[str(stat)] += tp_pred[i, :].tolist()
                gt[str(stat)] += tp_target[i,:].tolist()
            
            # mse += F.mse_loss(pred, target, reduction='sum')
            # mae += F.l1_loss(pred, target, reduction='sum')
            # mape += mape_loss(pred, target, device, reduction='sum')

            # mdape += mdape_func(target, pred)
        lst_pred = []
        for key in lst_mse.keys():
            lst_mse[key] = mse_func(gt[key], predict[key]) 
            lst_mae[key] = mae_func(gt[key], predict[key])
            lst_mape[key] = mape_func(gt[key], predict[key])
            lst_mdape[key] = mdape_func(gt[key], predict[key]) 
            lst_r2[key] = r2_score(gt[key], predict[key])
            rmse = np.sqrt(lst_mse[key])
            lst_pred.append({'Station': key, 'MSE': lst_mse[key].item(), 'RMSE': np.sqrt(
                lst_mse[key].item()), 'MAE': lst_mae[key].item(), 'MAPE': lst_mape[key].item(),'MDAPE': lst_mdape[key],"R2" : lst_r2[key],
                'Train time': train_time, 'Test time': test_time})
            print("Station {}: {} | {} | {} | {}".format(
                key, lst_mse[key], rmse, lst_mae[key], lst_mape[key]))

            pred_gt = {"Groundtruth": gt[key], "Predict": predict[key]}
            df = pd.DataFrame(data=pred_gt, columns=['Groundtruth', "Predict"])
            df.to_csv(output_dir + 'Station_{}.csv'.format(key))

        # df_pred = pd.DataFrame(data=lst_pred)
        # df_pred.to_csv(output_dir + 'result.csv')
        return lst_pred

        # num_entries = len(test_dataloader.dataset) * len(predictor.test_loc)
        # rmse = torch.sqrt(mse / num_entries).item()
        # mae = (mae / num_entries).item()
        # mape = (mape / num_entries).item()
        # print(f'val loss  : {rmse:>.4f} | {mae:>.4f} | {mape:>.4f}')
