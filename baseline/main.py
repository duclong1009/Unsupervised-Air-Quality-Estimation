import argparse
from statistics import mode
import torch 
import torch.nn as nn 
import numpy as np 
import random 
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
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

def  parse_args():
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
    parser.add_argument('--num_station',
                        default=39,
                        type=int)
    parser.add_argument('--spatial_res',
                        default=0.05,
                        type=float)
    parser.add_argument('--temporal_res',
                        default=5,
                        type=int)
    parser.add_argument('--neighbours',
                        default=10,
                        type=int)
    parser.add_argument('--num_epochs',
                        default=100,
                        type=int)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--patience',
                        default=10,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=5e-3,
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
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "./data/BeijingSSA/"
    data_path = file_path + 'pm2_5.csv'
    loc_path  = file_path + 'location.csv'
    config_path = './config/'
    model_type = args.model_type

    config_file = config_path + model_type + '.yml'
    weight_path = './checkpoint/' + model_type + '/checkpoint.pt'
    output_dir = './result/' + model_type +'/'
    es = EarlyStopping(
            patience=10,
            verbose=True,
            delta=0.0,
            path=weight_path
        )

    if model_type == 'tcgru':
        writer = SummaryWriter()

        train_dataloader, valid_dataloader, test_dataloader, loc_df, params, net_loc, test_idx, scaler  = create_required_data(file_path, args)

        model = TCGRU(params).to(device)
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # earlyStopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=True)
        # model.set_callbacks([earlyStopping])

        predictor = Predictor(loc_df, test_idx, net_loc, params, device)
        train_loss = {'epoch': [], 'train_loss': []}
        val_loss = {'epoch': [], 'val_loss': []}
        
        saved_epoch = 0 

        for epoch in tqdm(range(args.num_epochs)):
            epoch_loss = train_loop_tcgru(
                model, train_dataloader, loss_fn, optimizer, device)
            # train_loss.append({"epoch": epoch, "train_loss": epoch_loss})
            train_loss['epoch'].append(epoch)
            train_loss['train_loss'].append(epoch_loss)
        
            if epoch % 3 == 0:
                rmse, _, _ = eval_loop_tcgru(model, valid_dataloader, predictor, device)

                val_loss['epoch'].append(epoch)
                val_loss['val_loss'].append(rmse)

                save_checkpoint(model, optimizer, args.checkpoint_file + "tcgru_{}".format(epoch))
                saved_epoch = epoch
                writer.add_scalar("Validation loss", rmse, epoch)
            writer.add_scalar("Loss", epoch_loss, epoch)

        df_loss = pd.DataFrame(data= train_loss)
        df_valloss = pd.DataFrame(data = val_loss)
        df_loss.to_csv("./output/log/train_loss.csv")
        df_valloss.to_csv("./output/log/val_loss.csv")
        visualize(train_loss, val_loss, args)
        load_model(model, args.checkpoint_file + "tcgru_{}".format(saved_epoch))
        test_loop(model, test_dataloader, predictor, device, args)
        writer.flush()
        writer.close()
    
    elif model_type == 'bilstm_idw':
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # load config 
        target_idx = config['data']['target_idx']
        test_idx = config['data']['test_idx']
        train_idx = list(set(range(35)) - set(test_idx) - set([target_idx]))

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
        train_df, valid_df, test_df = split_dataset_bilstm_idw(data_path,train_pct, valid_pct)
        train_dataset = PMDatasetBiLSTMIDW(
            train_df, target_idx, test_idx, window, training=True)
        valid_dataset = PMDatasetBiLSTMIDW(
            valid_df, target_idx, test_idx, window, training=True)
        test_dataset = PMDatasetBiLSTMIDW(
            test_df, target_idx, test_idx, window, training=False)

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=True)

        # create model
        idweight = comp_idweight(
            loc_path, train_idx, [target_idx]).to(device)
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
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=learning_rate_decay)

        for epoch in tqdm(range(epochs)):
            if not es.early_stop:
                print("Epoch {}/{}: \n".format(epoch, epochs))
                train_losses.append(
                    train_loop_bilstm_idw(
                        model, train_dataloader, loss_fn, optimizer, device 
                    )
                )
                val_loss = eval_loop_bilstm_idw(
                    model, valid_dataloader, loss_fn, device, scheduler
                )
                val_losses.append(val_loss)
                es(val_loss, model)

        print("Testing")
        test_loop_bilstm_idw(
            model, test_dataloader, idw_matrix, device, output_dir
        )

    elif model_type == 'idwknn':
        pred_pm, test_pm = interpolator(file_path, args)
    else:
        print("Wrong model name: {}".format(model_type))
        