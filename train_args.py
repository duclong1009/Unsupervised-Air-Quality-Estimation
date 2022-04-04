import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from src.modules.train.test import cal_acc, test_atten_decoder_fn
from utils.ultilities import config_seed, load_model, save_checkpoint, EarlyStopping
from utils.loader import get_data_array, preprocess_pipeline, AQDataSet
from torch.utils.data import DataLoader
from src.models.stdgi import Attention_STDGI, InterpolateAttention_STDGI
from src.models.decoder import Decoder, InterpolateAttentionDecoder, InterpolateDecoder
from src.modules.train.train import train_atten_decoder_fn
from src.modules.train.train import train_atten_stdgi
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument(
        "--train_station",
        default=[i for i in range(20)],
        type=list,
    )
    parser.add_argument(
        "--test_station",
        default=[i for i in range(20, 35, 1)],
        type=list,
    )
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--sequence_length", default=12, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs_decoder", default=100, type=int)
    parser.add_argument("--lr_decoder", default=5e-3, type=float)
    parser.add_argument("--cnn_hid_dim", default=128, type=int)
    parser.add_argument("--fc_hid_dim", default=64, type=int)
    parser.add_argument("--rnn_type", default="LSTM", type=str)
    parser.add_argument("--n_layers_rnn", default=1, type=int)
    parser.add_argument("--climate_features", default=['lrad', 'shum', 'pres', 'temp', 'wind', 'srad'], type=list)
    parser.add_argument("--lr_decay_ratio",type= float)
    return parser.parse_args()


from utils.loader import comb_df
from utils.loader import get_columns, AQDataSet, location_arr
import logging
import wandb
import json

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    args = parse_args()

    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))

    config_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "./data/Beijing2/"
    comb_arr, location_, station, features_name = get_data_array(file_path,config['climate_features'])
    trans_df,climate_df, scaler = preprocess_pipeline(comb_arr)
    config["features"] = features_name
    config['loss'] = "mse"
    # breakpoint()
    decoder_name = f"{config['cnn_hid_dim']}_{config['fc_hid_dim']}_{config['n_layers_rnn']}_{config['rnn_type']}"
    wandb.init(
        entity="aiotlab",
        project="Sweep-stdgi",
        group="Train_1000ts_PM2.5",
        name=decoder_name,
        config=config,
    )

    train_dataset = AQDataSet(
        data_df=trans_df[:1000],
        climate_df = climate_df[:1000],
        location_df=location_,
        list_train_station=config['train_station'],
        input_dim=config['sequence_length'],
        interpolate=False,
    )
    # breakpoint()
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    # config["loss"] = 'mse'
    
    # Model Stdgi
    stdgi = Attention_STDGI(
            in_ft=9,
            out_ft=60,
            en_hid1=200,
            en_hid2=400,
            dis_hid=6,
        ).to(device)
    l2_coef = 0.0
    mse_loss = nn.MSELoss()
    load_model(stdgi,"./out/checkpoint/" + "stdgi2" + ".pt")
    decoder = Decoder(
            69,
            1,
            n_layers_rnn=config['n_layers_rnn'],
            rnn=config['rnn_type'],
            cnn_hid_dim=config['cnn_hid_dim'],
            fc_hid_dim=config['fc_hid_dim'],
            n_features=len(config['climate_features'])
        ).to(device)
    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=config['lr_decoder'], weight_decay=l2_coef
    )

    early_stopping_decoder = EarlyStopping(
        patience=config['patience'],
        verbose=True,
        delta=0,
        path="./out/checkpoint/" + decoder_name + ".pt",
    )
    scheduler = ReduceLROnPlateau(optimizer_decoder, 'min', factor=config['lr_decay_ratio'])

    for i in range(config['num_epochs_decoder']):
        if not early_stopping_decoder.early_stop:
            epoch_loss = train_atten_decoder_fn(
                stdgi,
                decoder,
                train_dataloader,
                mse_loss,
                optimizer_decoder,
                device,
                interpolate=False,
            )
            early_stopping_decoder(epoch_loss, decoder)
            scheduler.step(epoch_loss)
            print("Epochs/Loss: {}/ {}".format(i, epoch_loss))
    train_loss = early_stopping_decoder.best_score
    wandb.log({"train_loss": train_loss})
    load_model(decoder, "./out/checkpoint/" + decoder_name + ".pt")
    list_acc = []
    predict = {}
    mae_mean = 0
    mape_mean = 0
    mse_mean = 0
    for test_station in config['test_station']:
        test_dataset = AQDataSet(
            data_df=trans_df[:1000],
            climate_df = climate_df[:1000],
            location_df=location_,
            list_train_station=[i for i in range(20)],
            test_station=test_station,
            test=True,
            input_dim=config['sequence_length'],
            interpolate=False,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False
        )
        # breakpoint()
        list_prd, list_grt = test_atten_decoder_fn(
            stdgi, decoder, test_dataloader, device, False, scaler
        )
        mae, mse, mape, rmse, r2, corr = cal_acc(list_prd, list_grt)
        # breakpoint()
        mae_mean += mae
        mape_mean += mape
        mse_mean += mse
        list_acc.append([test_station, mae, mse, mape, rmse, r2, corr])
        predict[test_station] = {"grt": list_grt, "prd": list_prd}
        print("Test Accuracy: {}".format(mae, mse, corr))
    wandb.log({"acc/mae": mae_mean,"acc/mape":mape_mean,"acc/mse":mse_mean})
    df = pd.DataFrame(
        np.array(list_acc),
        columns=["STATION", "MAE", "MSE", "MAPE", "RMSE", "R2", "CORR"],
    )
    wandb.log({"test_acc": df})
    for test_station in config['test_station']:
        prd = predict[test_station]["prd"]
        grt = predict[test_station]["grt"]
        x = 800
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(np.arange(x), grt[:x], label="grt")
        ax.plot(np.arange(x), prd[:x], label="prd")
        ax.legend()
        ax.set_title(f"Tram_{test_station}")
        wandb.log({"Tram_{}".format(test_station): wandb.Image(fig)})
