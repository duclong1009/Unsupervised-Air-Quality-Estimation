import argparse
from multiprocessing import reduction
import numpy as np 
import random 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 
import yaml
import math
from .modules.utils.utilities import *
from .modules.utils.loader import *
from .modules.utils.early_stop import * 
from .models.idw_knn import *
import torch.nn.functional  as F 
from .modules.utils.utilities import mape_loss
from .models.fcnn import * 
from .modules.train.fcnn import * 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        type=int,
                        default=0)
    parser.add_argument('--test',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')   
    parser.add_argument('--target_station',
                        default=14,
                        type=int)
    parser.add_argument('--test_station',
                        default=[29,30,31,32,33,34,35,36,37,38,39,40],
                        type=list)                                             
    parser.add_argument('--model_type',
                        default='knn_idw',
                        type=str)
    return parser.parse_args()

class AQDataset(Dataset):
    def __init__(self, aq, meteo, gt):
        super(AQDataset, self).__init__()
        self.aq = aq 
        self.meteo = meteo 
        self.gt = gt  
    
    def __len__(self):
        return self.aq.shape[0]

    def __getitem__(self, index):
        aq = self.aq[index, :].astype(np.float32)
        meteo = self.meteo[index, :].astype(np.float32)
        y =  np.array(self.gt[index], dtype=np.float32)
        return {
            'x': torch.from_numpy(aq),
            'meteo': torch.from_numpy(meteo), 
            'y': torch.from_numpy(y)
        }

def get_dataset_train(pm_df, temp_df, wind_df, valid_stat_knn, config):
    """
    Return train dataloader
    --------------------
    Input:
        data: pm2.5, temp, wind
        valid_stat_knn
        k: num neigh 
        config: config of model
    --------------------
    Output:
        X: (N, k)
        meteo: (N, k, f)
        y: (N, 1)
    """
    train_pct, test_pct = config['train_pct'], config['test_pct']
    len_train = int(pm_df.shape[0] * train_pct)

    xs, meteos, ys  = [], [], []

    for stat in valid_stat_knn.keys():
        idx_target = stat 
        idx_train = valid_stat_knn[stat]

        x =  pm_df.iloc[:len_train,idx_train].to_numpy()

        temp = temp_df.iloc[:len_train,idx_target].to_numpy()
        temp = np.expand_dims(temp, axis=1)
        wind = wind_df.iloc[:len_train,idx_target].to_numpy()
        wind = np.expand_dims(wind, axis=1)
    
        meteo = np.concatenate([temp,wind], axis=1)
        y = pm_df.iloc[:len_train, idx_target].to_numpy()

        x = np.expand_dims(x, axis=1)
        meteo = np.expand_dims(meteo, axis=1)
        y = np.expand_dims(y, axis=1)

        xs.append(x)
        meteos.append(meteo)
        ys.append(y)

    xs = np.concatenate(xs, axis=1)
    ys = np.concatenate(ys, axis=1)
    meteos = np.concatenate(meteos, axis=1)

    x_cat = xs.reshape(-1,xs.shape[2])
    
    meteo_cat = meteos.reshape(-1,meteos.shape[2])
    y_cat = ys.reshape(-1)

    dataset = AQDataset(x_cat, meteo_cat, y_cat) 
    return dataset

def get_dataset_test(pm_df, temp_df, wind_df, test_stat_knn, stat,config):
    """
    Return train dataloader
    --------------------
    Input:
        data: pm2.5, temp, wind
        stat: target_stat_idx
        k: num neigh 
        config: config of model
    --------------------
    Output:
        X: (N, k, 1)
        meteo: (1, M)
        y: 1 
    """
    train_pct, test_pct = config['train_pct'], config['test_pct']
    
    len_train = int(pm_df.shape[0] * train_pct)

    xs, meteos, ys  = [], [], []

    idx_target = stat 
    idx_train = test_stat_knn[stat]
    x =  pm_df.iloc[len_train:,idx_train].to_numpy()
    temp = temp_df.iloc[len_train:,idx_target].to_numpy()
    temp = np.expand_dims(temp, axis=1)
    wind = wind_df.iloc[len_train:,idx_target].to_numpy()
    wind = np.expand_dims(wind, axis=1)
    meteo = np.concatenate([temp,wind], axis=1)
    y = pm_df.iloc[len_train:, idx_target].to_numpy()
    x = np.expand_dims(x, axis=1)
    meteo = np.expand_dims(meteo, axis=1)
    y = np.expand_dims(y, axis=1)

    xs.append(x)
    ys.append(y)
    meteos.append(meteo)

    xs = np.concatenate(xs, axis=1)
    ys = np.concatenate(ys, axis=1)
    meteos = np.concatenate(meteos, axis=1)

    x_cat = xs.reshape(-1,xs.shape[2]) 
    meteo_cat = meteos.reshape(-1,meteos.shape[2])
    y_cat = ys.reshape(-1)
    dataset = AQDataset(x_cat, meteo_cat, y_cat) 
    return dataset 

def data_processing(file_path, loc_path):
    pm_scaler = MinMaxScaler()
    temp_scaler = MinMaxScaler()
    wind_scaler = MinMaxScaler() 

    pm_df = pd.read_csv(file_path + 'PM2_5.csv', index_col=0)
    pm_df = pm_df.iloc[:,3:]
    pm_df = pm_df.fillna(10)    

    temp_df = pd.read_csv(file_path + '2m_temperature.csv', index_col=0)
    temp_df = temp_df.iloc[:,3:]
    temp_df = temp_df.fillna(10)

    wind_df = pd.read_csv(file_path + 'wind_speed.csv', index_col=0)
    wind_df = wind_df.iloc[:,3:]
    wind_df = wind_df.fillna(10)

    shape_pm = pm_df.values.shape[1]
    shape_temp = temp_df.values.shape[1]
    shape_wind = wind_df.values.shape[1]

    pm_np = pm_df.values.reshape(-1,1) # (937, 141) -> (141 * 937, 1)
    temp_np = temp_df.values.reshape(-1,1)
    wind_np = wind_df.values.reshape(-1,1)

    pm_scaler.fit(pm_np)
    temp_scaler.fit(temp_np)
    wind_scaler.fit(wind_np)
    
    pm_np_ = pm_scaler.transform(pm_np).reshape(-1, shape_pm)
    temp_np_ = temp_scaler.transform(temp_np).reshape(-1, shape_temp)
    wind_np_ = wind_scaler.transform(wind_np).reshape(-1, shape_wind)

    pm_df_ = pd.DataFrame(data=pm_np_, columns=pm_df.columns)
    temp_df_ = pd.DataFrame(data=temp_np_, columns=temp_df.columns)
    wind_df_ = pd.DataFrame(data=wind_np_, columns=wind_df.columns)

    df_loc = pd.read_csv(loc_path, index_col=False)
    return pm_df_, temp_df_, wind_df_, df_loc, pm_scaler, temp_scaler, wind_scaler
    # return pm_df, temp_df, wind_df, df_loc, pm_scaler, temp_scaler, wind_scaler

if __name__ == '__main__':
    args = parse_args()
    seed(args)

    dataset = 'beijing'
    if dataset == 'uk':
        file_path = "data/AQ_UK_ft/"
        test = {
            'train': [15, 17, 19, 21, 48, 73, 96, 114, 131],
            'valid': [20, 34, 56, 85],
            'test': [98,99, 135, 136]
        }
    elif dataset =='beijing':
        file_path = "data/AQ_Beijing_ft/"
        test = {
            'train': [18, 11, 3, 15, 8, 1, 9],
            'valid': [12, 7, 2, 10, 13],
            'test': [1, 4, 5, 6]
        }

    loc_path  = file_path + 'location.csv'
    config_path = './config/'

    device ='cpu'
    model_type = 'fcnn' # att_fcnn 

    pm_df, temp_df, wind_df, df_loc, pm_scaler, temp_scaler, wind_scaler = data_processing(file_path, loc_path)

    config_file = config_path + model_type + '.yml'

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = './result/' + model_type +'/' + dataset +'/'
    ckpt_path = './checkpoint/' + model_type +'/' + dataset +'/'
    ckpt_dir =  ckpt_path + 'checkpoint.pt' 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    es = EarlyStopping(
        patience=config['patience'],
        verbose=True,
        delta=0.0,
        path=ckpt_dir
    )

    train_idx = test['train']
    val_idx = test['valid'] 
    test_idx = test['test']

    lst_stats_train = df_loc.iloc[train_idx, 0].tolist() #n,2
    lst_stats_test = df_loc.iloc[test_idx, 0].tolist() #m,2
    lst_stats_val = df_loc.iloc[val_idx, 0].tolist() #m,2
    
    train_loc = df_loc.iloc[train_idx, [2,1]].to_numpy()
    val_loc = df_loc.iloc[val_idx, [2,1]].to_numpy()
    test_loc = df_loc.iloc[test_idx, [2,1]].to_numpy()

    # get nearest neighbour 
    idw_tree = tree(train_loc)

    # idx of train stat nearneigh
    _, lst_val_idx = idw_tree.nearest_neighbor(val_loc, k=config['num_neigh'])
    _, lst_test_idx = idw_tree.nearest_neighbor(test_loc, k=config['num_neigh'])

    valid_stat_knn, test_stat_knn = {}, {}

    for idx, val_stat in enumerate(val_idx):
        valid_stat_knn.update({val_stat: [train_idx[ix] for ix in lst_val_idx[idx]]})

    for idx, test_stat in enumerate(test_idx): 
        test_stat_knn.update({test_stat: [train_idx[ix] for ix in lst_test_idx[idx]]})

    train_dataset = get_dataset_train(pm_df, temp_df, wind_df, valid_stat_knn, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)   

    model = FCNN(config, device).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['learning_rate_decay'], patience=5,min_lr=1.0e-6)
    
    train_loss = {'epoch': [], 'train_loss': []}
    for epoch in range(config['epochs']):
        if not es.early_stop:
            epoch_loss = train_fcnn(model, train_dataloader, loss_fn, optimizer, device, scheduler)
            train_loss['epoch'].append(epoch) #
            train_loss['train_loss'].append(epoch_loss)
            print("Epoch loss: {}".format(epoch_loss))
            es(epoch_loss, model) 
    load_model(model, ckpt_dir)

    results = []
    for stat in test_stat_knn.keys():
        test_dataset = get_dataset_test(pm_df, temp_df, wind_df, test_stat_knn, stat, config)
        test_dataloader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)  
        result = test_fcnn(model, test_dataloader, device, pm_scaler, output_dir, stat, config)
        print(result)
        results.append(result)

    df_res = pd.DataFrame(data=results)
    df_res.to_csv(output_dir +f"result_bs_{config['batch_size']}.csv")
