import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

from modules.utils.interpolate import interpolate_pm
from modules.utils.utilities import *


def get_columns(file_path):
    """
      Get List Stations 
      Return Dict {"Numerical Name": "Japanese Station Name"}
    """
    fl = file_path + "pm2_5.csv"

    df = pd.read_csv(fl)
    cols = df.columns.to_list()

    res, res_rev = {}, {}

    for i, col in enumerate(cols):

        # if (i == 0):
        #   res.update({"Date": col})
        #   res_rev.update({col: "Date"})
        # elif (i==1):
        #   res.update({"Month": col})
        #   res_rev.update({col: "Month"})
        # elif (i == 2):
        #   res.update({"Day": col})
        #   res_rev.update({col: "Day"})
        # elif (i == 3):
        #   res.update({"Hour": col})
        #   res_rev.update({col: "Hour"})
        # else:
        # i -= 1
        stat_name = i
        res.update({stat_name: col})
        # print({stat_name: col })
        res_rev.update({col: stat_name})

#   lst_col = list( set(list(res.keys())) -set(["Date"]) )
    
    lst_col = res.keys()
    pm_df = df.rename(columns=res_rev)
    pm_df = pm_df.loc[:, lst_col]

    w, h = pm_df.shape
    scaler = MinMaxScaler()
    pm_df_ = pm_df.values.reshape(-1, 1)

    pm_df_ = scaler.fit_transform(pm_df_)
    pm_df_ = pm_df_.reshape(w,h)
    pm_df_ = pd.DataFrame(pm_df_)
    return res, res_rev, pm_df_, scaler 

def to_numeric(x):
    # non-numeric -> nan
    x_1 = x.apply(pd.to_numeric, errors="coerce")
    # nan thi lay gia tri lien truoc
    x_2 = x_1.fillna(method='ffill')
    # set threshold cho gia tri pm2.5
    res = x_2.clip(lower=1, upper=45)
    return res


def preprocess_pipeline(df):
    # loc null va replace = most frequent
    
    lst_cols = list(set(list(df.columns)) -
                    set(['Year', 'Month', 'Day', 'Hour']))

    # type_transformer = FunctionTransformer(to_numeric)
    # num_pl = Pipeline(
    #     steps=[
    #         ('numeric_transform', type_transformer),
    #     ],
    # )
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', num_pl, lst_cols)
    #     ]
    # )
    # breakpoint()
    # breakpoint()
    # res = preprocessor.fit_transform(df)
    # trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    columns = lst_cols
    w, h = df.shape
    # scaler = MinMaxScaler()
    # trans_df = df.values.reshape(-1,1)
    # trans_df = scaler.fit_transform(trans_df)
    # trans_df = trans_df.reshape(w,h)
    # trans_df = pd.DataFrame(trans_df,columns=columns)
    # trans_df.to_csv("./result/kidw_tcgru/log/transform_df.csv")
    return df


def split_dataset_tcgru(filepath):
    """
    Implement a 60:20:20 contiguous split.
    """
    _, _, pm_df, scaler = get_columns(filepath)
    # import pdb; pdb.set_trace()
    # pm_df = preprocess_pipeline(pm_df)

    length = len(pm_df)
    train_df = pm_df.loc[: int(0.6 * length)]
    valid_df = pm_df.loc[int(0.6 * length): int(0.8 * length)]
    test_df = pm_df.copy()
    return train_df, valid_df, test_df, scaler


def split_dataset_bilstm_idw(filepath, train_pct, valid_pct):
    """
    Implement a 60:20:20 contiguous split.
    """
    pm_df = pd.read_csv(
        filepath, header=None, skiprows=1
    )
    length = len(pm_df)
    train_df = pm_df.loc[: int(train_pct * length)]
    valid_df = pm_df.loc[int(train_pct * length): int((train_pct + valid_pct) * length)]
    test_df = pm_df.loc[int((train_pct + valid_pct) * length):]
    return train_df, valid_df, test_df

class PMDataset(Dataset):
    def __init__(
        self,config, pm_df, loc_df, target_idx, test_idx,
        network_loc, params, train_idx, training=True
    ):
        """
            pm_df, loc_df: dataframe
            target_idx: integer, test_idx: list
            training: False if test dataset
        """
        self.config = config 
        self.test_pct = config['data']['test_pct']
        self.training = training
        self.window = params[1]
        # breakpoint()
        
    
        if not self.training:
            train_pm, train_loc = self.get_station_data(pm_df, loc_df, train_idx, self.test_pct, is_train=False)
            
            self.input_data = torch.tensor(
                interpolate_pm(network_loc, train_loc, train_pm, params),
                dtype=torch.float32
            )
            test_pm, _ = self.get_station_data(pm_df, loc_df, test_idx, self.test_pct, is_train=False)
            self.target_data = torch.tensor(test_pm, dtype=torch.float32)
        else: 
            target_idx = target_idx + train_idx
            train_pm, train_loc = self.get_station_data(pm_df, loc_df, train_idx, self.test_pct)
            
            self.input_data = torch.tensor(
                interpolate_pm(network_loc, train_loc, train_pm, params),
                dtype=torch.float32
            )
            target_pm, target_loc = self.get_station_data(pm_df, loc_df, target_idx, self.test_pct)
            self.target_data = torch.tensor(
            interpolate_pm(network_loc, target_loc, target_pm, params),
            dtype=torch.float32
        )

    @staticmethod
    def get_station_data(pm_df, loc_df, station_idx, test_pct, is_train=True):
        """
        Retrieve pm2.5 data and locations of given stations.
        """
        idx_test = int(len(pm_df) * (1 - test_pct))
        if is_train:
            station_pm = pm_df.iloc[:idx_test, station_idx].to_numpy()
            station_loc = loc_df.iloc[station_idx].to_numpy()
        else:
            station_pm = pm_df.iloc[idx_test:, station_idx].to_numpy()
            station_loc = loc_df.iloc[station_idx].to_numpy()
        return station_pm, station_loc

    def __len__(self):
        return len(self.input_data) - self.window + 1

    def __getitem__(self, idx):
        """
        input shape: (seq_len, 1, H, W)
        target shape: (1, H, W) if testing, else 
        (num_test,)
        """
        input = self.input_data[idx: idx + self.window]
        target = self.target_data[idx + self.window - 1]
        return input, target


class PMDatasetBiLSTMIDW(Dataset):
    def __init__(
        self, pm_df, train_idx, target_idx,
        test_idx, window, training=False
    ):
        """
        pm_df: dataframe
        target_idx: integer index of target station
        test_idx: list of indices of test stations
        training: False if test dataset
        """

        self.input_data = torch.tensor(
            pm_df.drop(columns=train_idx).to_numpy(),
            dtype=torch.float32
        )
        if not training:
            self.target_data = torch.tensor(
                pm_df.iloc[:, test_idx].to_numpy(),
                dtype=torch.float32
            )
        else:
            self.target_data = torch.tensor(
                pm_df.iloc[:, target_idx].to_numpy(),
                dtype=torch.float32
            )
        self.window = window

    def __len__(self):
        return len(self.input_data) - self.window + 1

    def __getitem__(self, idx):
        """
        input shape: (seq_len, num_train_stations)
        target shape: (1, ), if testing (num_test_stations, )
        """
        input = self.input_data[idx: idx + self.window]
        target = self.target_data[idx + self.window - 1]
        return input, target

def get_loc_df(filepath):
    loc_df = pd.read_csv(filepath + 'location.csv')
    modified_loc_df = loc_df.rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})
    modified_loc_df = modified_loc_df[['lat', 'lon']]
    return modified_loc_df

def get_dataloader(dataset_class, batch_size, shuffle=False):
    return DataLoader(
        dataset_class,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

def create_required_data(file_path, args, config):
    # target_idx = args.target_station
    target_idx = args.target_station
    test_idx = args.test_station

    # train_idx = list(set(range(args.num_station)) -
    #                  set(test_idx) - set(target_idx))
    train_idx = args.train_station
    _, _, pm_df, scaler  = get_columns(file_path)
    # print(pm_df.max())
    # train_df, valid_df, test_df = split_dataset_tcgru(file_path)
    loc_df = get_loc_df(file_path)
    params = get_params(loc_df, args.spatial_res,
                        args.temporal_res, args.neighbours)
    net_loc = create_network(params, loc_df)
    train_dataset = PMDataset(
        config, pm_df, loc_df, target_idx, test_idx, net_loc, params, train_idx, training=True)
    # valid_dataset = PMDataset(valid_df, loc_df, target_idx, test_idx, net_loc, params, train_idx, training=False)
    test_dataset = PMDataset(config, pm_df, loc_df, target_idx,
                             test_idx, net_loc, params, train_idx, training=False)
    train_dataloader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_dataloader = get_dataloader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
#   return train_dataloader, valid_dataloader, test_dataloader, loc_df, params, net_loc, target_idx, test_idx
    return train_dataloader, test_dataloader, loc_df, params, net_loc, target_idx, test_idx, scaler

def create_training_data_interpolate(file_path, args):
    loc_df = get_loc_df(file_path)
    test_idx = args.test_station
    train_idx = list(set(range(args.num_station)) - set(test_idx))

    train_loc = loc_df.iloc[train_idx].to_numpy()
    test_loc = loc_df.iloc[test_idx].to_numpy()

    _, _, pm_df, scaler = get_columns(file_path)
    pm_df,scaler = preprocess_pipeline(pm_df)

    train_pm = pm_df.iloc[:, train_idx].to_numpy()
    test_pm = pm_df.iloc[:, test_idx].to_numpy()

    return train_pm, test_pm, train_loc, test_loc,scaler

def create_data_file():
    data_path = './data/BeijingSSA/data/'
    list_data_files = os.listdir(data_path)
    output_file = './data/BeijingSSA/pm2_5.csv'
    loc_path = './data/BeijingSSA/location.csv'

    loc_df = pd.read_csv(loc_path, header=0, usecols=['location'])
    loc_lst = loc_df['location'].tolist()
    # import pdb; pdb.set_trace()
    data_df = pd.DataFrame()
    for idx, loc in enumerate(loc_lst):
        loc_df = pd.read_csv(data_path + loc + '.csv')
        loc_df_pm2_5 = loc_df['PM2.5'].tolist()
        data_df[str(idx+1)] = loc_df_pm2_5
    data_df.to_csv(output_file, index=False)
