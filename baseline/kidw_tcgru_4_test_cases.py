import argparse
from statistics import mode
import torch 
import torch.nn as nn 
import numpy as np 
import random 
# from tqdm import tqdm
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
                        default=100,
                        type=int,
                        help='Seed')   
    parser.add_argument('--target_station',
                        default=14,
                        type=int)
    parser.add_argument('--test_station',
                        default=[29,30,31,32,33,34,35,36,37,38,39,40],
                        type=list)
    parser.add_argument('--num_station',
                        default=30,
                        type=int)
    parser.add_argument('--spatial_res',
                        default=0.5, #0.05 - beijing - 0.5- uk
                        type=float)
    parser.add_argument('--temporal_res',
                        default=5,
                        type=int)
    parser.add_argument('--neighbours',
                        default=10,
                        type=int)
    parser.add_argument('--num_epochs',
                        default=1000, #1000
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--patience',
                        default=30,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.005,
                        type=float)                                                
    parser.add_argument('--model_type',
                        default='kidw_tcgru',
                        type=str)
    parser.add_argument('--checkpoint_file',
                        default='./checkpoint/',
                        type=str)
    parser.add_argument('--visualize_dir', 
                        default='./output/kidw_tcgru/visualize/',
                        type=str)
    parser.add_argument('--dataset', 
                        type=str, 
                        choices=['beijing', 'uk'])
    return parser.parse_args()

if __name__=="__main__":
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
    model_type = 'kidw_tcgru'
    config_file = config_path + model_type + '.yml'
    weight_path = './checkpoint/' + model_type + '/'

    #4 test cases, 3 model 
    if args.dataset == "beijing":
        train_station = [18, 11, 3, 15, 8, 1, 9]
        valid_station = [12, 7, 2, 10, 13]
        test_station = [1, 4, 5, 6]
    elif args.dataset == "uk":
        train_station = [15, 17, 19, 21, 48, 73, 96, 114, 131]
        valid_station = [20, 34, 56, 85]
        test_station = [98,99, 135, 136]

    test_case_1 = {
        "train": train_station,
        "valid": valid_station,
        "test" : test_station
    }
    # test_case_1 = {
    #     'train': [0, 8, 12, 21, 23, 24, 20, 29, 1, 2, 5, 14, 15, 28],
    #     'valid': [4, 6, 7, 19, 22, 16, 13, 9, 11, 27],
    #     'test': [17, 25, 26, 3, 10, 18]
    # }
    # test_case_2 = {
    #     'train': [9, 11,13, 16,0, 1, 2, 4, 5, 6, 7, 8, 12,29],
    #     'valid': [14,27,20, 15, 19, 21, 22, 23, 24, 28],
    #     'test': [17,25,26,3,10,18]
    # }
    # test_case_3 = {
    #     'train': [4, 7, 12, 17, 22, 25,16, 3,1, 2, 5, 11, 14, 18],
    #     'valid': [0, 6, 8, 19, 23,20, 29,10, 27, 15],
    #     'test': [24, 26, 21,13,9, 28] 
    # }
    # test_case_2 = {
    #     'train': [4, 7, 12, 17, 22, 25,16, 3,1, 2, 5, 11, 14, 18],
    #     'valid': [0, 6, 8, 19, 23,20, 29,10, 27, 15],
    #     'test': [24, 26, 21,13,9, 28] 
    # }
    # test_case_2 = {
    #     'train': [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    #     'valid': [31,32,33,34],
    #     'test': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] 
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
    # tests = [test_case_1, test_case_2, test_case_3, test_case_4]
    tests = [test_case_1]
    for idx, test in enumerate(tests):
        output_dir = './result/' + model_type +'/' + args.dataset +'/'  + 'test_{}'.format(idx+1) +'/'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        ckpt_path = weight_path +'/checkpoint.pt'
        es = EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=0.0,
            path=ckpt_path
        )
            
        # target_idx = random.choice()
        # test_idx = 
        n_station = len(test['train'])
        args.target_station = [test['valid'][-1] ]
        args.train_station = test['train']
        # writer = SummaryWriter()

        # train_dataloader, valid_dataloader, test_dataloader, loc_df, params, net_loc, test_idx
        total_pred = []
        # for cur_test in test['test']:
        #     cur_test_idx = [cur_test]
        args.test_station = test['test']
        
        train_dataloader, test_dataloader, loc_df, params, net_loc, target_idx, test_idx, scaler = create_required_data(file_path, args, config)

        model = TCGRU(params).to(device)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['train']['learning_rate_decay'], patience=8,min_lr=1.0e-6)
        # breakpoint()
        predictor_train = Predictor(loc_df, target_idx, net_loc, params, device, train=True)
        predictor_test = Predictor(loc_df, test_idx, net_loc, params, device)
        train_loss = {'epoch': [], 'train_loss': []}
        val_loss = {'epoch': [], 'val_loss': []}
        
        saved_epoch = 0 

        train_time = 0.0

        for epoch in range(args.num_epochs):
        # for epoch in range(1):
            if not es.early_stop:
                torch.cuda.synchronize()
                train_it_start = int(round(time.time()*1000))

                epoch_loss = train_loop_tcgru(
                    model, train_dataloader, predictor_train, loss_fn, optimizer, device, scheduler)
                # train_loss.append({"epoch": epoch, "train_loss": epoch_loss})

                torch.cuda.synchronize()
                time_elapsed = int(round(time.time()*1000)) - train_it_start
                train_time += time_elapsed

                train_loss['epoch'].append(epoch)
                train_loss['train_loss'].append(epoch_loss)
                print("Epoch loss: {}".format(epoch_loss))
                es(epoch_loss, model)

            # writer.add_scalar("Loss", epoch_loss, epoch)

        df_loss = pd.DataFrame(data= train_loss)
        df_valloss = pd.DataFrame(data = val_loss)
        df_loss.to_csv("./output/kidw_tcgru/train_loss_{}.csv".format(idx))
        df_valloss.to_csv("./output/kidw_tcgru/val_loss_{}.csv".format(idx))
        visualize(train_loss, val_loss, args, idx)
        load_model(model, weight_path  + "checkpoint.pt")
        result = test_loop(model, test_dataloader, predictor_test, device, args, output_dir, scaler, train_time)

        df_res = pd.DataFrame(data=result)
        df_res.to_csv(output_dir + 'result.csv')
        del model
