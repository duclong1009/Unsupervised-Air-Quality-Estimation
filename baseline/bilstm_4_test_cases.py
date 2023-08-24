import argparse
from statistics import mode
import torch
import torch.nn as nn
import numpy as np
import random
# from tqdm.notebook import tqdm
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import yaml

from .layers.predictor import Predictor
from .modules.utils.utilities import *
from .modules.utils.loader import *
from .modules.utils.interpolate import *
from .modules.utils.early_stop import *

from .models.tcgru import TCGRU
from .models.idwknn import *
from .models.bilstm_idw import *

from .modules.train.tcgru import *
from .modules.train.bilstm_idw import *
# from modules.train.knn_idw import *

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
                        default=[29, 30, 31, 32, 33,
                                 34, 35, 36, 37, 38, 39, 40],
                        type=list)
    parser.add_argument('--num_station',
                        default=30,
                        type=int)
    parser.add_argument('--spatial_res',
                        default=0.05,
                        type=float)
    parser.add_argument('--temporal_res',
                        default=5,
                        type=int)
    parser.add_argument('--neighbours',
                        default=1,
                        type=int)
    parser.add_argument('--num_epochs',
                        default=50,
                        type=int)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--patience',
                        default=10,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float)
    parser.add_argument('--model_type',
                        default='bilstm_idw',
                        type=str)
    parser.add_argument('--checkpoint_file',
                        default='./checkpoint/bilstm-idw/',
                        type=str)
    parser.add_argument('--visualize_dir',
                        default='./output/visualize/bilstm-idw/',
                        type=str)
    parser.add_argument('--dataset', 
                        type=str, 
                        choices=['beijing', 'uk'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'uk':
        file_path = "./data/AQ_UK/"
        data_path = file_path + 'pm2_5.csv'
        loc_path  = file_path + 'location.csv'
    elif args.dataset == 'beijing':
        file_path = "./data/AQ_Beijing24/"
        data_path = file_path + 'pm2_5.csv'
        loc_path  = file_path + 'location.csv'

    config_path = './config/'
    model_type = 'bilstm_idw'
    config_file = config_path + model_type + '.yml'
    weight_path = './checkpoint/' + model_type + '/checkpoint.pt'

    if args.dataset == "beijing":
        train_station = [18, 11, 3, 15, 8, 1, 9]
        valid_station = [12, 7, 2, 10, 13]
        test_station = [1, 4, 5, 6]
    elif args.dataset == "uk":
        train_station = [15, 17, 19, 21, 48, 73, 96, 114, 131]
        valid_station = [20, 34, 56, 85]
        test_station = [98, 99, 135, 136]

    test_case_1 = {
        "train": train_station,
        "valid": valid_station,
        "test" : test_station
    }

    # test_case_1 = {
    #     'train': [0, 8, 12, 21, 23, 24, 20, 29, 1, 2, 5, 14, 15, 28],
    #     'valid': [4, 6, 7, 19, 22, 16, 13, 9, 11, 27],
    #     'test': [24, 25, 26, 27, 28, 18]
    # }
    # test_case_1 = {
    #     'train':[ 9,  8,  4, 22, 14,  5,  2, 27,  6, 13, 20, 15, 24, 12],
    #     'valid': [0, 1, 7, 11, 16, 19, 21, 23, 28, 29],
    #     'test': [17,25,26,3,10,18]
    # }
    # test_case_2 = {
    #     'train': [8, 12,21, 26, 25, 15,28,10, 18, 11,14, 20,29],
    #     'valid': [6, 24, 19, 17, 23, 1, 5,2, 13, 16, 27],
    #     'test': [0, 7, 4, 22, 3,9]
    # }
    # test_case_3 = {
    #     'train': [4, 7, 12, 17, 22, 25,16, 3,1, 2, 5, 11, 14, 18],
    #     'valid': [0, 6, 8, 19, 23,20, 29,10, 27, 15],
    #     'test': [24, 26, 21,13,9, 28]
    # }
    # test_case_3 = {
    #     'train': [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19],
    #     'valid': [27,28,29,30],
    #     'test': [11,12,13,14,20,21,22,23,24,25,26,31,32,33,34]
    # }
    # test_case_4 = {
    #     'train': [1,2,3,16,17,18,19, 28,29,30,22,23, 11,12,13,14],
    #     'valid': [27,24,25,26],
    #     'test': [20,21, 31,32,33,34, 4,5,6,7,8,9,10,15, 0]
    # }
    # tests = [test_case_1,test_case_2,test_case_3]
    tests = [test_case_1]

    for idx, test in enumerate(tests):
        es = EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=0.0,
            path=weight_path
        )
        output_dir = './result/' + model_type + \
            '/'+  args.dataset + '/'+ 'test_{}'.format(idx+1) + '/'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # target_idx = test['valid']
        target_idx = [random.choice(test['valid'])]
        test_idx = test['test']
        train_idx = test['train']
        
        window = config['data']['window']
        batch_size = config['train']['batch_size']

        lstm_input_size = config['model']['lstm_input_size']
        lstm_hidden_size = config['model']['lstm_hidden_size']
        lstm_num_layers = config['model']['lstm_num_layers']
        linear_hidden_size = config['model']['linear_hidden_size']
        idw_hidden_size = config['model']['idw_hidden_size']

        # training hyper-params
        batch_size = config['train']['batch_size']
        learning_rate = config['train']['learning_rate']
        learning_rate_decay = config['train']['learning_rate_decay']
        epochs = config['train']['epochs']
        train_pct = config['data']['train_pct']
        valid_pct = config['data']['valid_pct']
        # get data
        total_res = []
        # for cur_test in test['test']:
        #     cur_test_idx = [cur_test]
        train_df, valid_df, test_df = split_dataset_bilstm_idw(
            data_path, train_pct, valid_pct)
        train_dataset = PMDatasetBiLSTMIDW(
            train_df, train_idx, target_idx, test_idx, window, training=True)
        valid_dataset = PMDatasetBiLSTMIDW(
            valid_df, train_idx, target_idx, test_idx, window, training=True)
        test_dataset = PMDatasetBiLSTMIDW(
            test_df, train_idx, target_idx, test_idx, window, training=False)

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=False)

        # create model
        idweight = comp_idweight(
            loc_path, train_idx, target_idx).to(device)
        # import pdb; pdb.set_trace()
        model = IDW_BiLSTM_Model(
            lstm_input_size,
            lstm_hidden_size, lstm_num_layers,
            linear_hidden_size, idw_hidden_size,
            idweight, device
        )
        model = model.to(device)

        train_losses = []
        val_losses = []
        # create optimizer
        idw_matrix = comp_idweight(
            loc_path, train_idx, test_idx).to(device)
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=learning_rate_decay)

        train_time = 0.0 
        for epoch in range(epochs):
            if not es.early_stop:
                print("Epoch {}/{}: \n".format(epoch, epochs))
                
                torch.cuda.synchronize()
                train_it_start = int(round(time.time()*1000))
                train_losses.append(
                    train_loop_bilstm_idw(
                        model, train_dataloader, loss_fn, optimizer, device
                    )
                )
                val_loss = eval_loop_bilstm_idw(
                    model, valid_dataloader, loss_fn, device, scheduler
                )
                torch.cuda.synchronize()
                time_elapsed = int(round(time.time()*1000)) - train_it_start
                train_time += time_elapsed

                val_losses.append(val_loss)
                es(val_loss, model)
        print("Testing")
        result = test_loop_bilstm_idw(
            model, test_dataloader, idw_matrix, device, output_dir, test_idx, train_time
        )
        df_res = pd.DataFrame(data=result)
        df_res.to_csv(output_dir + 'result.csv')
