# from builtins import breakpoint
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    List,
    Iterable,
    Tuple,
    NamedTuple,
    Callable,
)
import torch
import random
import numpy as np
import pandas as pd

# chi su dung data PM truoc
def get_columns(file_path):
    """
    Get List Stations
    Return Dict {"Numerical Name": "Japanese Station Name"}
    """
    fl = file_path + "PM2.5.csv"
    df = pd.read_csv(fl)
    df = df.fillna(5)
    cols = df.columns.to_list()
    res, res_rev = {}, {}
    for i, col in enumerate(cols):
        if i == 0:
            pass
        else:
            i -= 1
            stat_name = "Station_" + str(i)
            res.update({stat_name: col})
            res_rev.update({col: stat_name})

    pm_df = df.rename(columns=res_rev)
    return res, res_rev, pm_df


# preprocess pipeline
def to_numeric(x):
    x_1 = x.apply(pd.to_numeric, errors="coerce")
    res = x_1.clip(lower=0)
    return res


def remove_outlier(x):
    # remove 97th->100th percentile
    pass


def rolling(x):
    res = []
    for col in list(x.columns):
        ans = x[col].rolling(2, min_periods=1)
        res.append(ans)
    # pdb.set_trace()
    ans = np.array(res)
    return ans  # rolling va lay mean trong 3 timeframe gan nhat


from sklearn.impute import KNNImputer, SimpleImputer


def preprocess_pipeline(df, args):
    # 800,35,17
    scaler = MinMaxScaler((-1, 1))
    # import pdb; pdb.set_trace()
    # breakpoint()
    (a, b, c) = df.shape
    res = np.reshape(df, (-1, c))
    for i in range(c):
        threshold = np.percentile(res[:, i], 95)
        res[:, i] = np.where(res[:, i] > threshold, threshold, res[:, i])
    # res = np.reshape(res, (-1, b,c))
    # breakpoint()
    res = scaler.fit_transform(res)
    res = np.reshape(res, (-1, b, c))
    # breakpoint()
    trans_df = res[:, :, :]

    idx_climate = 5 if args.dataset == 'uk' else 7
    climate_df = res[:, :, idx_climate:]
    del res
    return trans_df, climate_df, scaler

def get_list_file(folder_path):
    from os import listdir
    from os.path import isfile, join

    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return onlyfiles
def comb_df(file_path, pm_df, res):
    list_file = get_list_file(file_path)
    list_file.remove("PM2.5.csv")
    list_file.remove("location.csv")
    column = [res[i] for i in list(pm_df.columns)[1:]]
    comb_arr = pm_df.iloc[:, 1:].to_numpy()
    comb_arr = np.expand_dims(comb_arr, -1)
    for file_name in list_file:
        df = pd.read_csv(file_path + file_name)
        # preprocess()
        df = df.fillna(5)
        df = df[column]
        arr = df.to_numpy()
        arr = np.expand_dims(arr, -1)
        comb_arr = np.concatenate((comb_arr, arr), -1)
    del arr
    return comb_arr, column


from torch.utils.data import Dataset


def location_arr(file_path, res):
    location_df = pd.read_csv(file_path + "location.csv")
    list_location = []
    for i in res.keys():
        loc = location_df[location_df["location"] == res[i]].to_numpy()[0, 1:]
        list_location.append([loc[1], loc[0]])
    del loc
    return np.array(list_location)


def get_data_array(args, file_path):
    # columns1 = ["PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]
    # import pdb; pdb.set_trace()
    if args.dataset == 'uk':
        columns1 = ["PM2.5", "PM10", "O3", "SO2", "NO2"]
    elif args.dataset == 'beijing':
        columns1 = ['PM2.5','AQI','PM10','CO','NO2','O3','SO2']
    columns2 = args.climate_features
    columns = columns1 + columns2
    location_df = pd.read_csv(file_path + "location.csv")
    # breakpoint()
    station = location_df["station"].values
    location = location_df.values[:, 1:]
    location_ = location[:, [1, 0]]

    list_arr = []
    for i in station:
        df = pd.read_csv(file_path + f"{i}.csv")[columns]
        df = df.fillna(method="ffill")
        df = df.fillna(10)
        arr = df.astype(float).values
        arr = np.expand_dims(arr, axis=1)
        list_arr.append(arr)
    list_arr = np.concatenate(list_arr, axis=1)
    # print(list_arr.shape)
    pm2_5 = list_arr[:,:,0]
    # pm_dff = pd.DataFrame(pm2_5)
    # pm_dff.to_csv("AQ_pm2_5.csv")
    # breakpoint()
    corr = pd.DataFrame(pm2_5).corr().values
    del df 
    del arr
    del location
    del location_df
    # breakpoint()
    return list_arr, location_, station, columns, corr


# from torchvision import transforms
class AQDataSet(Dataset):
    def __init__(
        self,
        data_df,
        climate_df,
        location_df,
        list_train_station,
        input_dim,
        test_station=None,
        test=False,
        interpolate=False,
        corr=None,
        args=None
    ) -> None:
        super().__init__()
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (
            test_station in list_train_station
        ), "tram test khong trong tram train"
        # self.list_cols_train = ["Station_{}".format(i) for i in list_train_station]
        self.list_cols_train_int = list_train_station
        self.input_len = input_dim
        self.test = test
        self.data_df = data_df
        self.location = location_df
        self.interpolate = interpolate
        self.climate_df = climate_df
        self.n_st = len(list_train_station) - 1
        self.corr =corr
        self.train_cpt = args.train_pct
        self.valid_cpt = args.valid_pct
        self.test_cpt = args.test_pct

        idx_test = int(len(data_df) * (1- self.test_cpt))
        # phan data train thi khong lien quan gi den data test 
        self.X_train = data_df[:idx_test,:, :]
        self.climate_train = climate_df[:idx_test,:, :]
        
        # test data
        if self.test:
            # phan data test khong lien quan gi data train 
            test_station = int(test_station)
            lst_cols_input_test_int = list(
                set(self.list_cols_train_int) - set([self.list_cols_train_int[-1]])
            )

            self.X_test = data_df[idx_test:, lst_cols_input_test_int,:]
            self.l_test = self.get_reverse_distance_matrix(
                lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df[idx_test:, test_station, :]
            self.climate_test = climate_df[idx_test:, test_station, :]
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int)
            if self.corr is not None:
                self.corr_matrix_test = self.get_corr_matrix(lst_cols_input_test_int)

    def get_distance(self, coords_1, coords_2):
        import geopy.distance
        return geopy.distance.geodesic(coords_1, coords_2).km

    def get_distance_matrix(self, list_col_train_int, target_station):
        matrix = []
        for i in list_col_train_int:
            matrix.append(
                self.get_distance(self.location[i], self.location[target_station])
            )
        res = np.array(matrix)
        return res
    def get_corr_matrix(self, list_station):
        # breakpoint()
        # print(list_station)
        # breakpoint()
        corr_mtr = self.corr[np.ix_(list_station,list_station)]
        corr_mtr_ = np.expand_dims(corr_mtr.sum(-1),-1)
        corr_mtr_ = np.repeat(corr_mtr_,corr_mtr_.shape[0],-1)
        corr_mtr = corr_mtr/corr_mtr_
        corr_mtr = np.expand_dims(corr_mtr,0)
        corr_mtr = np.repeat(corr_mtr, self.input_len,0)
        return corr_mtr

    def get_reverse_distance_matrix(self, list_col_train_int, target_station):
        distance_matrix = self.get_distance_matrix(list_col_train_int, target_station)
        reverse_matrix = 1 / distance_matrix
        return reverse_matrix / reverse_matrix.sum()

    def get_adjacency_matrix(self, list_col_train_int, target_station_int=None):
        adjacency_matrix = []
        for j, i in enumerate(list_col_train_int):
            distance_matrix = self.get_distance_matrix(list_col_train_int, i)
            distance_matrix[j] += 15
            reverse_dis = 1 / distance_matrix
            adjacency_matrix.append(reverse_dis / reverse_dis.sum())
        adjacency_matrix = np.array(adjacency_matrix)
        adjacency_matrix = np.expand_dims(adjacency_matrix, 0)
        adjacency_matrix = np.repeat(adjacency_matrix, self.input_len, 0)
        return adjacency_matrix

    def __getitem__(self, index: int):
        list_G = []
        if self.test:
            x = self.X_test[index : index + self.input_len, :]
            y = self.Y_test[index + self.input_len - 1, 0]
            G = self.G_test
            l = self.l_test
            climate = self.climate_test[index + self.input_len - 1, :]
            if self.corr is not None:
                list_G = [G,self.corr_matrix_test]
        else:
            # chon 1 tram ngau  nhien trong 28 tram lam target tai moi sample
            # import pdb; pdb.set_trace()
            picked_target_station_int = random.choice(self.list_cols_train_int)
            # print(picked_target_station_int)
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            x = self.X_train[index : index + self.input_len, lst_col_train_int, :]
            # x = np.expand_dims(x, -1)
            y = self.X_train[index + self.input_len - 1, picked_target_station_int, 0]
            climate = self.climate_train[
                index + self.input_len - 1, picked_target_station_int, :
            ]
            G = self.get_adjacency_matrix(
                lst_col_train_int, picked_target_station_int
            )
            if self.corr is not None:
                corr_matrix = self.get_corr_matrix(lst_col_train_int)
                list_G = [G,corr_matrix]
            l = self.get_reverse_distance_matrix(
                lst_col_train_int, picked_target_station_int
            )
            
        sample = {
            "X": x,
            "Y": np.array([y]),
            # "G": np.array(G),
            "l": np.array(l),
            "climate": climate,
        }
        if self.corr is None:
            sample["G"] = G
        else:
            sample["G"] = np.stack(list_G,-1)
        # breakpoint()
        return sample

    def __len__(self) -> int:
        if self.test:
            return self.X_test.shape[0] - self.input_len
        return self.X_train.shape[0] - (self.input_len)


from utils.ultilities import config_seed
from torch.utils.data import DataLoader


if __name__ == "__main__":
    file_path = "../data/"
    # Preprocess and Load data
    location = pd.read_csv(file_path + "locations.csv").to_numpy()
    location = location[:, 1:]
    res, res_rev, pm_df = get_columns(file_path)
    trans_df, scaler = preprocess_pipeline(pm_df)
    train_dataset = AQDataSet(
        data_df=trans_df[:50],
        location_df=location,
        list_train_station=[i for i in range(28)],
        input_dim=12,
        interpolate=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for v in train_dataloader:
        print("X: ")
        print(v["X"].size())
        print("Y: ")
        print(v["Y"].size())
        print("G: ")
        print(v["G"].size())
        break
