import argparse
from multiprocessing import reduction
from operator import index
from statistics import mode
import numpy as np 
import random 
from tqdm.notebook import tqdm
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import yaml
import math
from .modules.utils.utilities import *
from .modules.utils.loader import *
from .modules.utils.interpolate import *
from .modules.utils.early_stop import * 
from .models.idw_knn import *
import torch.nn.functional  as F 
from .modules.utils.utilities import mape_loss

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def  parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        type=int,
                        default=0)
    parser.add_argument('--test',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')   
    parser.add_argument('--target_station',
                        default=14,
                        type=int)
    parser.add_argument('--test_station',
                        default=[29,30,31,32,33,34,35,36,37,38,39,40],
                        type=list)                                             
    parser.add_argument('--model_type',
                        default='knn_idw',
                        type=str)
    return parser.parse_args()
def mdape(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100


if __name__=="__main__":
    args = parse_args()
    seed(args)
    file_path = "data/new_hanoi2/"
    data_path = file_path + 'pm2_5.csv'
    loc_path  = file_path + 'location.csv'
    config_path = './config/'

    device ='cpu'

    df_data = pd.read_csv(data_path, index_col=False)
    # df_data = df_data.where(df_data >= 160, 160)
    df_data = df_data.fillna(100)
    df_loc = pd.read_csv(loc_path, index_col=False)
    arr = df_data.values
    # stt =0 
    # for k in arr:
    #     for j in k:
    #         if type(j) == str:
    #             print(j)
    #             stt += 1
    # breakpoint()
    model_type = 'knn_idw'
    config_file = config_path + model_type + '.yml'

    # test_case_1 = {
    #     'train': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    #     'valid': [14,15, 16,17,18,19, 20, 21, 22, 23],
    #     'test': [24,25,26,27,28,29]
    # }
    # test_case_2 = {
    #     'train': [0,8,12,21,23,24,20,29,1,2,5,14,15,28],
    #     'valid': [4,6,7,19,22,16,13,9,11,27],
    #     'test': [17,25,26,3,10,18]
    # }
    # test_case_1 = {
    #     'train': [0, 4, 6, 7, 8, 12, 17, 19, 21, 22, 23, 24, 25, 26],
    #     'valid': [1, 2, 5, 9, 10, 11, 14, 15, 18, 27, 28],
    #     'test': [3, 13, 16, 20, 29]
    # }
    # beijing 
    # test_case_1 = {
    #     'train': [18, 11, 3, 15, 8, 1, 9],
    #     'valid': [12, 7, 2, 10, 13],
    #     'test': [0, 4, 5, 6]
    # }
    # hanoi
    # test_case_1 = {
    #     'train': [55, 53, 69, 49, 10, 37, 64, 41, 45, 19, 60, 2, 7, 40, 32, 52, 54, 17, 0, 18, 35, 56, 67, 33, 22, 44, 61, 30, 72, 16, 65, 24, 39, 29, 71,6, 74, 58,36, 5, 9, 70],
    #     'valid': [50, 46, 62, 31, 14, 25, 11, 26, 3, 66, 68, 63, 57, 20, 8, 34, 21,42, 13,43, 73],
    #     'test': [1, 4, 38, 12, 47, 48, 51, 23, 28,59,27, 15]
    # }
    # uk 

    test_case_1 = {
        # 'train': [15, 17, 19, 21, 48, 73, 96, 114, 131],
        'train': [8, 15, 18, 0, 16, 9, 3],
        'valid': [5, 7, 12, 14],
        'test': [6, 17, 1, 2, 13, 4, 10, 11]
    }

    # test_case_1 = {
    #     # 'train': [15, 17, 19, 21, 48, 73, 96, 114, 131],
    #     'train': range(9),
    #     'valid': range(9,14),
    #     'test': range(14,19)
    # }
    # test_case_1 = {
    #     'train':[21, 25, 26, 23, 8, 10, 11, 7, 27, 0, 15, 9, 5, 12],
    #     'valid': [ 1, 2, 4, 6, 14, 17, 19, 22, 24, 28],
    #     'test': [13,20,29,3,16,18]
    # }
    # test_case_2 = {
    #     'train': [8, 12,21, 26, 25, 15,28,2,10, 18, 11,14, 20,29],
    #     'valid': [6, 24, 19, 17, 23, 1, 5, 13, 16, 27],
    #     'test': [0, 7, 4, 22, 3,9]
    # }
    # test_case_2 = {
    #     'train': [9, 11,13, 16,0, 1, 2, 4, 5, 6, 7, 8, 12,29],
    #     'valid': [14,27,20, 15, 19, 21, 22, 23, 24, 28],
    #     'test': [17,25,26,3,10,18]
    # }
    # test_case_3 = {
    #     'train': [4,7,17,25,26,13,16,3,15,28,1,5,10,2],
    #     'valid': [0,6,8,12,19],
    #     'test': [0,6,8,12,19]
    # }
    tests = [test_case_1]

    for idx, test in enumerate(tests):
        output_dir =  './result/' + model_type +'/'  + 'test_{}'.format(idx+1) +'/'
        train_idx = test['train'] 
        test_idx = test['test']
        # breakpoint()
        lst_stats_train = df_loc.iloc[train_idx, 0].tolist() #n,2
        lst_stats_test = df_loc.iloc[test_idx, 0].tolist() #m,2
        print(lst_stats_train)
        print("--------------")
        print(lst_stats_test)
        train_loc = df_loc.iloc[train_idx, [2,1]].astype(float)
        test_loc = df_loc.iloc[test_idx, [2,1]].astype(float)
        train_data = df_data[lst_stats_train] #k,n
        test_data = df_data[lst_stats_test] # k,m
        test_data = test_data.to_numpy()

        lst_pred = []

        for idx, row in train_data.iterrows(): # for each day calculate test value for corresponding day    
            train_data = np.transpose(row.to_numpy())
            idw_tree = tree(train_loc, train_data)
            pred_data = idw_tree(test_loc).tolist()
            lst_pred.append(pred_data)

        pred_data = np.array(lst_pred)
        pred_data[np.isnan(pred_data)] = 10
        print(pred_data.shape)
        len_data = pred_data.shape[0]
        for i in range(pred_data.shape[1]):
            test_id = int(len_data * 0.85)
            mse = mean_squared_error(test_data[test_id:,i], pred_data[test_id:,i])
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(test_data[test_id:,i], pred_data[test_id:,i] )
            mdape_ = mdape(test_data[test_id:,i], pred_data[test_id:,i])
            mape = mape_loss(torch.from_numpy(pred_data[test_id:,i]).float(), torch.from_numpy(test_data[test_id:,i]).float(),device, reduction='mean')
            r2 = r2_score(test_data[test_id:,i], pred_data[test_id:,i])
            print('Station: {} RMSE: {} - MAE: {} - MDAPE:  {}- MAPE: {} - R2: {}'.format(i, rmse, mae, mdape_,mape, r2))
            with open(output_dir + 'result.txt', 'w') as f:
                f.write('Station: {} RMSE: {} - MAE: {} - MDAPE: {} MAPE: {} - R2: {}'.format(i, rmse, mae, mdape_, mape, r2))
        
        mse = mean_squared_error(test_data[test_id:,:], pred_data[test_id:,:])
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(test_data[test_id:,:], pred_data[test_id:,:])
        mdape_ = mdape(test_data[test_id:,:], pred_data[test_id:,:])
        mape = mape_loss(torch.from_numpy(pred_data[test_id:,:]).float(), torch.from_numpy(test_data[test_id:,:]).float(),device, reduction='mean')
        r2 = r2_score(test_data[test_id:,:], pred_data[test_id:,:])
        print('MAE: {} - MSE: {} - MDAPE: {} - MAPE: {} - R2: {}'.format(mae, mse, mdape_,  mape, r2))
        with open(output_dir + 'result.txt', 'w') as f:
            f.write('MAE: {} - MSE: {} - MDAPE: {} - MAPE: {} - R2: {}'.format(mae, mse, mdape_,  mape, r2))

        for idx in range(0,len(lst_pred[0])):
            print(idx)
            stat = test_idx[idx]
            len_data = test_data.shape[0]
            test_id = int(len_data * 0.85)
            data = {
                'groundtruth': test_data[test_id:, idx].tolist(),
                'predict': pred_data[test_id:, idx].tolist(),
            }
            df = pd.DataFrame(data=data)
            df.to_csv(output_dir + '{}.csv'.format(stat))
