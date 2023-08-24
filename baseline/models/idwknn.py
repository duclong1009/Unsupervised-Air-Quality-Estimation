import numpy as np 
import pandas as pd 
from sklearn import metrics

from modules.utils.interpolate import  haverside_dist
from modules.utils.loader import create_training_data_interpolate
from modules.utils.error import *

def interpolate_pm(
        unknown_loc, known_loc, known_pm, num_neighbors):
    """
    Interpolate the PM2.5 values of unknown locations, 
    using k nearest known stations. modules
    
    unkown_pm: array (batch, 1, H, W) if reshape=True,
    else shape (batch, num_unknown)
    """
    distance = haverside_dist(unknown_loc, known_loc)
    distance = np.where(distance < 1e-6, 1e-6, distance)     
    bound = np.partition(
        distance, num_neighbors - 1, axis=1
    )[:, [num_neighbors - 1]]
    neighbor_mask = np.where(distance <= bound, 1., np.nan)
        
    neighbor_dist = distance * neighbor_mask
    R = 1 / neighbor_dist
    weight = R / np.nansum(R, axis=1, keepdims=True)
    weight = np.nan_to_num(weight, nan=0.)
    
    unknown_pm = known_pm @ weight.T
    return unknown_pm

def interpolator(file_path, args):
    train_pm, test_pm, train_loc, test_loc = create_training_data_interpolate(file_path, args)
    pred_pm = interpolate_pm(test_loc, train_loc, train_pm, num_neighbors=args.neighbours)

    pred_df = pd.DataFrame(data=pred_pm)
    pred_df.to_csv("../output/log/pred.csv")
    test_df = pd.DataFrame(data=test_pm)
    test_df.to_csv("../output/log/test.csv")

    mse = mean_squared_error(pred_pm, test_pm)
    mae = mean_absolute_error(pred_pm, test_pm)
    mape = mean_absolute_percentage_error(pred_pm, test_pm)


    lst_dict =[]
    for i, station_num in enumerate(args.test_station):
        print(f'station {station_num} val loss  : {mse[i]:>.4f} | {mae[i]:>.4f} | {mape[i]:>.4f}')
        lst_dict.append({"Station": station_num, "MSE": mse[i], "MAE": mae[i], "MAPE": mape[i]})
        print("-------------------------------------------------------")
    df_error = pd.DataFrame( data=lst_dict)
    df_error.to_csv("../output/log/error.csv")
    return pred_pm, test_pm 

