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
import numpy as np
from SSA import reconstruct_long_arr


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

def clipping(x):
    max_clip = np.percentile(x, 97)
    return np.clip(x, a_min=0.1, a_max=max(0.1,max_clip))

def preprocess_pipeline(df):
    lst_cols = list(set(list(df.columns)) - set(["Year", "Month", "Day", "Hour"]))
    type_transformer = FunctionTransformer(to_numeric)
    clipping_transformer = FunctionTransformer(clipping)
    num_pl = Pipeline(
        steps=[
            ("numeric_transform", type_transformer),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("clipping", clipping_transformer)
        ],
    )
    scaler = MinMaxScaler()
    preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    res = preprocessor.fit_transform(df)
    res = np.transpose(np.array(res))

    for i in range(len(res)):
        noise  = np.abs(np.divide(np.random.normal(0, 1, len(res[i])), np.array([100 for i in range(len(res[i]))  ] )) )
        res[i] += noise
        inp = res[i].tolist()
        print( "Input: "+ str(len(inp)))
        res[i] = reconstruct_long_arr(inp, len_component=8000, window_length=20, lst_reconstruct_idx=[i for i in range(0,13)])

    n_ = res.shape[1]
    res = np.reshape(res, (-1, 1))
    # res = np.where(res <= threshold, res, threshold)
    res = scaler.fit_transform(res)
    res = np.reshape(res, (-1, n_))


    trans_df = pd.DataFrame(res, columns=lst_cols, index=df.index)
    trans_df[["Year", "Month", "Day", "Hour"]] = df[["Year", "Month", "Day", "Hour"]]
    return trans_df, scaler


if __name__=="__main__":
    data_file = "/home/aiotlab/projects/hungvv/stdgi-hung/stdgi/data/Station_5.csv"
    df = pd.read_csv(data_file)
    df_preprocess = preprocess_pipeline(df)
    print(df_preprocess.head())