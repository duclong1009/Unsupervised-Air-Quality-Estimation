from utils.loader import comb_df
from utils.loader import get_columns,AQDataSet,location_arr

if __name__ == "__main__":
    file_path = "./data/Beijing/"
    res,res_rev,df = get_columns(file_path)
    a,b = comb_df(file_path,df,res)
    print(a.shape)
    print([res_rev[i] for i in b])
    location_ = location_arr(file_path,res)
    dataset = AQDataSet(a,location_,[i for i in range(20)],12)
    for i in dataset:
        print(i["X"].shape)
        break
    # print(location_.shape)