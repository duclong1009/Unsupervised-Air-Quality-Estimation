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
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


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
            res_rev.update({col: stat_name})

    pm_df = df.rename(columns=res_rev)
    return res, res_rev, pm_df


def to_numeric(x):
    x_1 = x.apply(pd.to_numeric, errors="coerce")
    res = x_1.clip(lower=0)
    return res


def remove_outlier(x):
    # remove 97th->100th percentile
    pass


import numpy as np


def rolling(x):
    res = []
    for col in list(x.columns):
        ans = x[col].rolling(2, min_periods=1)
        res.append(ans)
    # pdb.set_trace()
    ans = np.array(res)
    return ans  # rolling va lay mean trong 3 timeframe gan nhat


def preprocess_pipeline(df):
    # loc null va replace = most frequent
    lst_cols = list(set(list(df.columns)) - set(["Year", "Month", "Day", "Hour"]))

    type_transformer = FunctionTransformer(to_numeric)
    outlier_transformer = FunctionTransformer(remove_outlier)
    rolling_transformer = FunctionTransformer(rolling)
    num_pl = Pipeline(
        steps=[
            ("numeric_transform", type_transformer),
            # ('roll_transform', rolling_transformer),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ('scaler', MinMaxScaler())
        ],
    )
    preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    res = preprocessor.fit_transform(df)
    trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
    return trans_df
