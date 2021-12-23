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
    fl = file_path + "PM.csv"

    df = pd.read_csv(fl)
    cols = df.columns.to_list()

    res, res_rev = {}, {}

    for i, col in enumerate(cols):

        if i == 0:
            res.update({"Year": col})
            res_rev.update({col: "Year"})
        elif i == 1:
            res.update({"Month": col})
            res_rev.update({col: "Month"})
        elif i == 2:
            res.update({"Day": col})
            res_rev.update({col: "Day"})
        elif i == 3:
            res.update({"Hour": col})
            res_rev.update({col: "Hour"})
        else:
            i -= 4
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


def preprocess_pipeline(df, threshold=50):
    sta_ = ["Station_{}".format(i) for i in range(65)]
    lst_cols = list(set(list(df.columns)) - set(["Year", "Month", "Day", "Hour"]))
    type_transformer = FunctionTransformer(to_numeric)
    outlier_transformer = FunctionTransformer(remove_outlier)
    rolling_transformer = FunctionTransformer(rolling)
    num_pl = Pipeline(
        steps=[
            ("numeric_transform", type_transformer),
            # ('roll_transform', rolling_transformer),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ],
    )
    scaler = MinMaxScaler()
    preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    res = preprocessor.fit_transform(df)
    res = np.array(res)
    n_ = res.shape[1]
    res = np.reshape(res, (-1, 1))
    res = np.where(res <= threshold, res, threshold)
    res = scaler.fit_transform(res)
    res = np.reshape(res, (-1, n_))
    trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
    return trans_df, scaler


from torch.utils.data import Dataset


class AQDataSet(Dataset):
    def __init__(
        self,
        data_df,
        location_df,
        list_train_station,
        input_dim,
        output_dim=1,
        test_station=None,
        test=False,
        transform=None,
        top_k=10,
        interpolate=False
    ) -> None:
        super().__init__()
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (
            test_station in list_train_station
        ), "tram test khong trong tram train"
        self.list_cols_train = ["Station_{}".format(i) for i in list_train_station]
        self.list_cols_train_int = list_train_station
        self.transform = transform
        self.input_len = input_dim
        self.output_len = output_dim
        self.test = test
        self.data_df = data_df
        self.location = location_df
        self.top_k = top_k
        self.interpolate = interpolate # neu interpolate thi tinh adj_matrix cho ca 28 tram roi swap row cua tram target xuong duoi cung
        # test data
        if self.test:
            test_station = int(test_station)
            lst_cols_input_test_int = list(
                set(self.list_cols_train_int)
                - set([random.choice(self.list_cols_train_int)])
            )
            lst_cols_input_test = "Station_{}".format(
                lst_cols_input_test_int
            )  # trong 28 tram, bo random 1 tram 
            self.X_test = data_df.loc[:, lst_cols_input_test].to_numpy()
            self.l_test = self.get_distance_matrix(
                lst_cols_input_test_int, test_station
            )
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int, test_station)
            self.Y_test = data_df.loc[:, "Station_{}".format(test_station)].to_numpy()

    def get_knn_correlation_adjacency_matrix(self, list_col_train, top_k):
        """
          Build adjacency matrix based on correlation of nearest neighbour based on correlation
        """
        lst_arr = []
        for col in list_col_train:
            data = np.array(self.data_df.loc[:, col])
            lst_arr.append(data)
        lst_arr_np = np.array(lst_arr)
        corr_matrix = np.corrcoef(lst_arr_np)
        # get ind of highest correlated idx
        top_max = np.argpartition(corr_matrix, -top_k, axis=1)[:, -top_k:]
        # get adj matrix
        final_adj_matrix = []
        for arr in top_max:
            lst = []
            for num in range(len(list_col_train)):
                if num in arr:
                    lst.append(1)
                else:
                    lst.append(0)
            final_adj_matrix.append(lst)
        adj_matrix = np.array(final_adj_matrix)

        final_corr_matrix = np.multiply(
            corr_matrix, adj_matrix
        )  # correlation matrix * adj_matrix

        res = np.expand_dims(final_corr_matrix, 0)
        res = np.repeat(res, self.input_len, 0)
        return res, final_corr_matrix

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
        if self.test:
            x = self.X_test[index : index + self.input_len, :]
            y = self.Y_test[index + self.input_len + self.output_len - 1]
            G = self.G_test
            l = self.l_test
        else:
            # chon 1 tram ngau  nhien trong 28 tram lam target tai moi sample
            # import pdb; pdb.set_trace()
            picked_target_station_int = random.choice(self.list_cols_train_int)
            # print(picked_target_station_int)
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            picked_target_station = "Station_{}".format(picked_target_station_int)
            lst_col_train = ["Station_{}".format(i) for i in lst_col_train_int]
            x = self.data_df.loc[
                index : index + self.input_len - 1, lst_col_train
            ].to_numpy()
            x = np.expand_dims(x, -1)
            y = self.data_df.loc[
                index + self.input_len + self.output_len - 1, picked_target_station
            ]
            # import pdb;
            # pdb.set_trace()
            if not self.interpolate:
                G = self.get_adjacency_matrix(lst_col_train_int, picked_target_station_int)
            else:
                new_list_col_train_int = lst_col_train_int.copy()
                new_list_col_train_int.append(picked_target_station_int)
                G = self.get_adjacency_matrix(new_list_col_train_int, picked_target_station_int)
            l = self.get_reverse_distance_matrix(
                lst_col_train_int, picked_target_station_int
            )
            # corr_matrix, final_corr_matrix = self.get_knn_correlation_adjacency_matrix(
            #     lst_col_train, self.top_k
            # )
        sample = {"X": x, "Y": y, "G": np.array(G), "l": np.array(l)}
        return sample

    def __len__(self) -> int:
        return self.data_df.shape[0] - (self.input_len)

from utils.ultilities import  config_seed
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
        output_dim=1,
        interpolate=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    for v in train_dataloader:
        print('X: ')
        print(v['X'].size())
        print('Y: ')
        print(v['Y'].size())
        print('G: ')
        print(v['G'].size())
        break