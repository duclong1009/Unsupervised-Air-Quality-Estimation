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
            # print({stat_name: col })
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

# def preprocess_pipeline(df, threshold=50):
#     sta_ = ["Station_{}".format(i) for i in range(65)]
#     lst_cols = list(set(list(df.columns)) - set(["Year", "Month", "Day", "Hour"]))
#     type_transformer = FunctionTransformer(to_numeric)
#     outlier_transformer = FunctionTransformer(remove_outlier)
#     rolling_transformer = FunctionTransformer(rolling)
#     num_pl = Pipeline(
#         steps=[
#             ("numeric_transform", type_transformer),
#             # ('roll_transform', rolling_transformer),
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#         ],
#     )
#     scaler = MinMaxScaler()
#     preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
#     res = preprocessor.fit_transform(df)
#     res = np.array(res)
#     n_ = res.shape[1]
#     res = np.reshape(res, (-1, 1))
#     res = np.where(res <= threshold, res, threshold)
#     res = scaler.fit_transform(res)
#     res = np.reshape(res, (-1, n_))
#     trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
#     trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
#     return trans_df, scaler

def preprocess_pipeline(df, threshold=50):
    # type_transformer = FunctionTransformer(to_numeric)
    # outlier_transformer = FunctionTransformer(remove_outlier)
    # rolling_transformer = FunctionTransformer(rolling)
    # num_pl = Pipeline(
    #     steps=[
    #         ("numeric_transform", type_transformer),
    #         # ('roll_transform', rolling_transformer),
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #     ],
    # )
    scaler = MinMaxScaler()
    # preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    # res = preprocessor.fit_transform(df)
    # res = np.array(res)
    n_ = df.shape[1]
    res = np.reshape(df, (-1, 7))
    res = np.where(res <= threshold, res, threshold)
    res = scaler.fit_transform(res)
    res = np.reshape(res, (-1, n_,7))
    # trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    # trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
    return res, scaler

def get_list_file(folder_path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return onlyfiles

def comb_df(file_path,pm_df,res):
  list_file = get_list_file(file_path)
  list_file.remove("PM2.5.csv")
  list_file.remove("location.csv")
  column = [res[i] for i in list(pm_df.columns)[1:]]
  comb_arr = pm_df.iloc[:,1:].to_numpy()
  comb_arr = np.expand_dims(comb_arr,-1)
  for file_name in list_file:
    df = pd.read_csv(file_path + file_name)
    # preprocess()
    df = df.fillna(5)
    df = df[column]
    arr = df.to_numpy()
    arr = np.expand_dims(arr,-1)
    comb_arr = np.concatenate((comb_arr,arr),-1)
  return comb_arr,column

from torch.utils.data import Dataset

def location_arr(file_path, res):
    location_df = pd.read_csv(file_path +"location.csv")
    list_location = []
    for i in res.keys():
        loc = location_df[location_df['location'] == res[i]].to_numpy()[0,1:]
        list_location.append([loc[1],loc[0]])
    return np.array(list_location)
class AQDataSet(Dataset):
    def __init__(
        self,
        data_df,
        location_df,
        list_train_station,
        input_dim,
        test_station=None,
        test=False,
        transform=None,
        top_k=10,
    ) -> None:
        super().__init__()
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (
            test_station in list_train_station
        ), "tram test khong trong tram train"
        # self.list_cols_train = ["Station_{}".format(i) for i in list_train_station]
        self.list_cols_train_int = list_train_station
        self.transform = transform
        self.input_len = input_dim
        self.test = test
        self.data_df = data_df
        self.location = location_df
        self.top_k = top_k
        # test data
        if self.test:
            test_station = int(test_station)
            lst_cols_input_test_int = list(
                set(self.list_cols_train_int)
                - set([random.choice(self.list_cols_train_int)])
            )
            lst_cols_input_test = "Station_{}".format(
                lst_cols_input_test_int
            )  # trong 28 tram, bo random 1 tram de lam input cho test
            self.X_test = data_df[:, lst_cols_input_test]
            self.l_test = self.get_distance_matrix(
                lst_cols_input_test_int, test_station
            )
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int)
            self.Y_test = data_df[:,test_station]


    def get_distance(self, coords_1, coords_2):
        import geopy.distance

        return geopy.distance.geodesic(coords_1, coords_2).km

    def get_distance_matrix(self, list_col_train_int, target_station):
        # import pdb; pdb.set_trace()
        matrix = []
        for i in list_col_train_int:
            matrix.append(
                self.get_distance(self.location[i], self.location[target_station])
            )
        res = np.array(matrix)
        return res

    def get_reverse_distance_matrix(self, list_col_train_int, target_station):
        distance_matrix = self.get_distance_matrix(list_col_train_int, target_station)
        reverse_matrix = 1 / distance_matrix
        return reverse_matrix / reverse_matrix.sum()

    def get_adjacency_matrix(self, list_col_train_int):
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
        if self.test:
            x = self.X_test[index : index + self.input_len, :]
            y = self.Y_test[index + self.input_len + self.output_len - 1]
            G = self.G_test
            l = self.l_test
        else:
            # chon 1 tram ngau  nhien trong 28 tram lam target tai moi sample
            # import pdb; pdb.set_trace()
            picked_target_station_int = random.choice(self.list_cols_train_int)
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            # picked_target_station = "Station_{}".format(picked_target_station_int)
            # lst_col_train = ["Station_{}".format(i) for i in lst_col_train_int]
            x = self.data_df[
                index : index + self.input_len , lst_col_train_int,:
            ]
            # x = np.expand_dims(x, -1)
            y = self.data_df[
                index + self.input_len - 1, picked_target_station_int,0
            ]
            G = self.get_adjacency_matrix(lst_col_train_int)
            l = self.get_reverse_distance_matrix(
                lst_col_train_int, picked_target_station_int
            )
        sample = {"X": x, "Y": y, "G": np.array(G), "l": np.array(l)}
        return sample

    def __len__(self) -> int:
        return self.data_df.shape[0] - (self.input_len)
