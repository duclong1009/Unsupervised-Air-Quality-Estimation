import pandas as pd
import numpy as np

def get_data_array(file_path):
    location_df = pd.read_csv(file_path + "location.csv")
    station = location_df['location'].values
    location = location_df.values[:,1:]
    location_ = location[:,[1,0]]
    
    list_arr = []
    for i in station:
        df = pd.read_csv(file_path  + f"{i}.csv")
        df = df.fillna(5)
        df = df.fillna(method='ffill')
        arr = df.values[:,1:]
        arr = np.expand_dims(arr,axis=1)
        list_arr.append(arr)
    list_arr = np.concatenate(list_arr,axis=1)
    for i in range(14):
        print(type(list_arr[0,0,i]))
    return list_arr,location_,station

get_data_array("data/Beijing2/")