from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
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
from torch_sparse import SparseTensor
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
    # H_ssa_L20 = SSA(res, 20)
    # res = H_ssa_L20.get_lst_sigma()
    # print(res.shape)
    res = np.where(res <= threshold, res, threshold)

    # res = scaler.fit_transform(res)
    res = np.reshape(res, (-1, n_))

    trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
    return trans_df, scaler


class AQDataset2(Dataset):
    def __init__(
        self,
        data_df,
        location,
        list_train_station,
        input_dim,
        output_dim=1,
        test_station=None,
        test=False,
        transform=None,
        top_k=10,
        topology_construction="knn_adj_matrix",
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
        self.location = location
        self.top_k = top_k
        self.topology_construction = topology_construction

        # test data
        if self.test:
            test_station = int(test_station)
            self.lst_cols_input_test_int = list(
                set(self.list_cols_train_int)
                - set([random.choice(self.list_cols_train_int)])
            )
            lst_cols_input_test = [
                "Station_{}".format(i) for i in self.lst_cols_input_test_int
            ]  # trong 28 tram, bo random 1 tram de lam input cho test
            self.X_test = data_df.loc[:, lst_cols_input_test].to_numpy()
            self.l_test = self.get_reverse_distance_matrix(
                self.lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df.loc[:, "Station_{}".format(test_station)].to_numpy()

    def get_distance(self, coords_1, coords_2):
        import geopy.distance

        return geopy.distance.vincenty(coords_1, coords_2).km

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
        return adjacency_matrix

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
        res = np.multiply(corr_matrix, adj_matrix)  # correlation matrix * adj_matrix
        return res

    def get_edge_index_weight(
        self, adj_matrix=None, corr_matrix=None, topology_construction="knn_adj_matrix"
    ):
        """
        Get edge index and edge weight in format of Pytorch Temporal
        """
        edge_lst = []
        edge_weight = []
        if topology_construction == "knn_adj_matrix":
            for i, lst_corr in enumerate(corr_matrix):
                for j, corr in enumerate(lst_corr):
                    if corr > 0:
                        edge_lst.append((i, j))
                        edge_weight.append(corr)

        elif topology_construction == "distance":
            #   print(adj_matrix)
            for i, lst_adj in enumerate(adj_matrix):
                for j, dist in enumerate(lst_adj):
                    edge_lst.append((i, j))
                    edge_weight.append(dist)
        np_edge_idx = np.transpose(np.array(edge_lst))
        np_edge_weight = np.array(edge_weight)
        return (
            torch.from_numpy(np_edge_idx).type(torch.LongTensor),
            torch.from_numpy(np_edge_weight).type(torch.FloatTensor),
        )

    def __getitem__(self, index: int):
        """
        Return in format of Pytorch Geometric Data
          x: [node, num_feature]
          y: [target value]
          edge_index: adjacency list
          edge_attr: value of edge in order of edge_idx list
          Exp: Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])
        """

        if self.test:
            x = self.X_test[index : index + self.input_len, :]
            x = torch.from_numpy(np.transpose(x)).type(torch.FloatTensor)
            l = self.l_test
            y = np.array(self.Y_test[index + self.input_len + self.output_len - 1])
            y = torch.from_numpy(y).type(torch.FloatTensor)
            final_adj_matrix = self.get_adjacency_matrix(self.lst_cols_input_test_int)
            # final_corr_matrix  = self.get_knn_correlation_adjacency_matrix(lst_col_train, self.top_k)
            if self.topology_construction == "knn_adj_matrix":
                edge_idx, edge_weight = self.get_edge_index_weight(
                    corr_matrix=final_corr_matrix,
                    topology_construction=self.topology_construction,
                )
            else:
                edge_idx, edge_weight = self.get_edge_index_weight(
                    adj_matrix=final_adj_matrix,
                    topology_construction=self.topology_construction,
                )

        else:
            picked_target_station_int = random.choice(self.list_cols_train_int)
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            picked_target_station = "Station_{}".format(picked_target_station_int)
            lst_col_train = ["Station_{}".format(i) for i in lst_col_train_int]

            final_adj_matrix = self.get_adjacency_matrix(lst_col_train_int)
            final_corr_matrix = self.get_knn_correlation_adjacency_matrix(
                lst_col_train, self.top_k
            )

            if self.topology_construction == "knn_adj_matrix":
                edge_idx, edge_weight = self.get_edge_index_weight(
                    corr_matrix=final_corr_matrix,
                    topology_construction=self.topology_construction,
                )
            else:
                edge_idx, edge_weight = self.get_edge_index_weight(
                    adj_matrix=final_adj_matrix,
                    topology_construction=self.topology_construction,
                )
            x = np.transpose(
                self.data_df.loc[
                    index : index + self.input_len - 1, lst_col_train
                ].to_numpy()
            )
            x = torch.from_numpy(x).type(torch.FloatTensor)
            l = self.get_reverse_distance_matrix(
                lst_col_train_int, picked_target_station_int
            )
            y = np.array(
                self.data_df.loc[
                    index + self.input_len + self.output_len - 1, picked_target_station
                ]
            )
            y = torch.from_numpy(y).type(torch.FloatTensor)
        ans = Data(edge_index=edge_idx, x=x, y=y, edge_attr=edge_weight, pos=l)
        return ans

    def __len__(self) -> int:
        return self.data_df.shape[0] - (self.input_len)
