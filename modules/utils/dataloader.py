from torch.utils.data import Dataset, DataLoader
import random 
import numpy as np 

class AQDataSet(Dataset):
    def __init__(self, data_df,location_df, list_train_station, input_dim, output_dim=1, test_station = None, test= False, transform= None, top_k=10) -> None:
        super().__init__()
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (test_station in list_train_station), "tram test khong trong tram train"
        self.list_cols_train = ['Station_{}'.format(i) for i in list_train_station]
        self.list_cols_train_int = list_train_station
        self.transform = transform
        self.input_len = input_dim
        self.output_len = output_dim
        self.test = test
        self.data_df = data_df
        self.location = location_df
        self.top_k = top_k

        #test data
        if self.test:
            test_station = int(test_station)
            lst_cols_input_test_int = list(set(self.list_cols_train_int) - set([random.choice(self.list_cols_train_int)]))
            lst_cols_input_test = "Station_{}".format(lst_cols_input_test_int) # trong 28 tram, bo random 1 tram de lam input cho test  
            self.X_test = data_df.loc[:, lst_cols_input_test].to_numpy()
            self.l_test = self.get_distance_matrix(lst_cols_input_test_int,test_station)
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int)
            self.Y_test = data_df.loc[:,"Station_{}".format(test_station)].to_numpy()

    def get_knn_correlation_adjacency_matrix(self, list_col_train, top_k):
        '''
          Build adjacency matrix based on correlation of nearest neighbour based on correlation
        '''
        lst_arr =[]
        for col in list_col_train:
          data = np.array(self.data_df.loc[:, col])
          lst_arr.append(data)
        lst_arr_np = np.array(lst_arr)
        corr_matrix = np.corrcoef(lst_arr_np)
        # get ind of highest correlated idx 
        top_max = np.argpartition(b, -top_k, axis=1)[:, -top_k:]
        final_adj_matrix =  []
        for arr in top_max:
          lst = []
          for num in range(len(list_col_train)):
            if num in arr:
              lst.append(1)
            else:
              lst.append(0)
          final_adj_matrix.append(lst)
        adj_matrix = np.array(final_adj_matrix)

        final_corr_matrix = np.multiply(corr_matrix, adj_matrix ) # correlation matrix * adj_matrix 
        res = np.expand_dims(final_corr_matrix,0)
        res = np.repeat(res,self.input_len,0)
        return res, final_corr_matrix

    def get_edge_index(self, list_col_train_int, corr_matrix):
        res = []
        for idx, col_int in enumerate(list_col_train_int):
            src_stat = col_int
            idx_target_stat  = [ i for i,v in enumerate(corr_matrix[idx]) if v > 0]
            target_stat = [(src_stat, list_col_train_int[idx]) for idx in idx_target_stat ]
            res += target_stat
        np_edge_idx = np.transpose(np.array(res))
        return np_edge_idx

    def get_distance(self,coords_1,coords_2):
        import geopy.distance
        return geopy.distance.vincenty(coords_1, coords_2).km

    def get_distance_matrix(self,list_col_train,target_station):
        matrix = []
        for i in list_col_train:
            matrix.append(self.get_distance(self.location[i],self.location[target_station]))
        res = np.array(matrix)
        res =  res/res.sum()
        return res


    def get_adjacency_matrix(self, list_col_train):
        adjacency_matrix = []
        for i in list_col_train:
          # print(i)
          adjacency_matrix.append(self.get_distance_matrix(list_col_train,i))

        res =  np.array(adjacency_matrix) * 0.7 + np.eye(len(list_col_train)) * 0.3
        res = np.expand_dims(res,0)
        res = np.repeat(res,self.input_len,0)
        return res  
    
    def __getitem__(self, index: int):
        if self.test:
            x = self.X_test[index:index+self.input_len,:]
            y = self.Y_test[index + self.input_len + self.output_len -1]
            G = self.G_test
            l = self.l_test
        else:
            #chon 1 tram ngau  nhien trong 28 tram lam target tai moi sample
            # import pdb; pdb.set_trace()
            picked_target_station_int = random.choice(self.list_cols_train_int)
            lst_col_train_int = list(set(self.list_cols_train_int) - set([picked_target_station_int]))
            picked_target_station = 'Station_{}'.format(picked_target_station_int)
            lst_col_train = ['Station_{}'.format(i) for i in lst_col_train_int]
            x = self.data_df.loc[index:index+self.input_len-1,lst_col_train].to_numpy()
            x = np.expand_dims(x,-1)
            y = self.data_df.loc[index+self.input_len+self.output_len-1,picked_target_station]
            G = self.get_adjacency_matrix(lst_col_train_int)
            l = self.get_distance_matrix(lst_col_train_int,picked_target_station_int)
            corr_matrix, final_corr_matrix  = self.get_knn_correlation_adjacency_matrix(lst_col_train, self.top_k)
            edge_index = self.get_edge_index(lst_col_train_int, res, final_corr_matrix)

        sample = {"X":x,"Y" : y, "G" : G, "l" : l, "idx": index, "knn_G": corr_matrix, "edge_index": edge_index}
        return sample

    def __len__(self) -> int:
        return self.data_df.shape[0] - (self.input_len)
