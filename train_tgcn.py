import argparse
from distutils.util import copydir_run_2to3
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from src.modules.train.test import cal_acc, test_atten_decoder_fn
from utils.ultilities import config_seed, load_model, save_checkpoint, EarlyStopping
from utils.loader import get_data_array, preprocess_pipeline, AQDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.stdgi import EGCN_STDGI, Attention_STDGI
from src.models.decoder import (
    Decoder,
    Global_Decoder,
    Local_Decoder,
    Local_Global_Decoder,

    WoCli_Decoder,
)
from src.modules.train.train import (
    train_atten_decoder_fn,
    train_egcn,
    train_egcn_decoder_fn,
)
from src.modules.train.train import train_atten_stdgi
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument(
        "--train_station",
        default=[i for i in range(8)],
        type=list,
    )
    parser.add_argument(
        "--test_station",
        default=[i for i in range(8, 12, 1)],
        type=list,
    )
    # Config STDGI
    parser.add_argument("--input_dim", default=9, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--sequence_length", default=12, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--lr_stdgi", default=0.001, type=float)
    parser.add_argument("--num_epochs_stdgi", default=100, type=int)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--en_hid1", default=200, type=int)
    parser.add_argument("--en_hid2", default=400, type=int)
    parser.add_argument("--dis_hid", default=6, type=int)
    parser.add_argument("--act_fn", default="relu", type=str)
    parser.add_argument("--delta_stdgi", default=0, type=float)
    parser.add_argument("--stdgi_noise_min", default=0.4, type=float)
    parser.add_argument("--stdgi_noise_max", default=0.7, type=float)
    # Config Decoder
    parser.add_argument("--num_epochs_decoder", default=100, type=int)
    parser.add_argument("--lr_decoder", default=0.001, type=float)
    parser.add_argument("--delta_decoder", default=0, type=float)
    parser.add_argument("--cnn_hid_dim", default=64, type=int)
    parser.add_argument("--fc_hid_dim", default=128, type=int)
    parser.add_argument("--rnn_type", default="LSTM", type=str)
    parser.add_argument("--n_layers_rnn", default=1, type=int)
    
    # Config Data
    parser.add_argument("--decoder_type", default="default", type=str,choices=["default","local_attention","global_attention","localglobal_attention","wo_cli"])
    parser.add_argument("--train_pct", default=0.6, type=float)
    parser.add_argument("--valid_pct", default=0.25, type=float)
    parser.add_argument("--test_pct", default=0.15, type=float)
    parser.add_argument("--wo_climate", action="store_true")
    parser.add_argument("--use_wind", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_wandb", action="store_false")
    parser.add_argument(
        "--climate_features",
        default=[
            "2m_temperature",
            "surface_pressure",
            "evaporation",
            "total_precipitation",
        ],
        type=list,
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gede", "wogcn", "wornnencoder"]
    )
    parser.add_argument("--group_name",type=str,default="",required=True)
    parser.add_argument("--dataset", type=str, choices=["beijing", "uk"])
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
    # device = torch.device("cpu")

    if args.dataset == "uk":
        file_path = "./data/uk_AQ/"
    elif args.dataset == "beijing":
        file_path = "./data/beijing_AQ/"
    if args.use_wind:
        args.climate_features = [
            "2m_temperature",
            "surface_pressure",
            "evaporation",
            "total_precipitation",
            "wind_speed",
            "wind_angle",
        ]

    comb_arr, location_, station, features_name, corr = get_data_array(args, file_path)
    args.input_dim = len(features_name)
    # print(station)
    trans_df, climate_df, scaler = preprocess_pipeline(comb_arr, args)
    config["features"] = features_name
    test_name = "test1"
    if args.dataset == "beijing":
        args.train_station = [18, 11, 3, 15, 8, 1, 9]
        args.valid_station = [12, 7, 2, 10, 13]
        args.test_station = [0, 4, 5, 6]
    elif args.dataset == "uk":
        args.train_station = [15, 17, 19, 21, 48, 73, 96, 114, 131]
        args.valid_station = [20, 34, 56, 85]
        args.test_station = [97, 98, 134, 137]
    corr = None
    args.num_input_station = len(args.train_station) -1 
    train_dataset = AQDataSet(
        data_df=trans_df[:],
        climate_df=climate_df,
        location_df=location_,
        list_train_station=args.train_station,
        input_dim=args.sequence_length,
        corr=corr,
        args=args,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # config["loss"] = 'mse'
    args.name = f"{args.model_type}_{args.decoder_type}_{args.dataset}_{args.seed}_{args.name}"
    if args.log_wandb:
        wandb.init(
            entity="aiotlab",
            project="Spatial_PM2.5",
            group="merged_AQ_{}".format(args.dataset),
            name=f"{args.name}",
            config=config,
        )
    
    # Model Stdgi
    stdgi = Attention_STDGI(
        in_ft=args.input_dim,
        out_ft=args.output_stdgi,
        en_hid1=args.en_hid1,
        en_hid2=args.en_hid2,
        dis_hid=args.dis_hid,
        stdgi_noise_min=args.stdgi_noise_min,
        stdgi_noise_max=args.stdgi_noise_max,
        model_type=args.model_type,
        num_input_station=args.num_input_station
    ).to(device)
    l2_coef = 0.0
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    stdgi_optimizer_e = torch.optim.Adam(
        stdgi.encoder.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    stdgi_optimizer_d = torch.optim.Adam(
        stdgi.disc.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    if not os.path.exists(f"output/{args.group_name}/checkpoint/"):
        print(f"Make dir output/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"output/{args.group_name}/checkpoint/")
    early_stopping_stdgi = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_stdgi,
        path=f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt",
    )
    # scheduler_stdgi = ReduceLROnPlateau(stdgi_optimizer, "min", factor=0.5, patience=3)

    logging.info(
        f"Training stdgi || attention decoder || epochs {args.num_epochs_stdgi} || lr {args.lr_stdgi}"
    )
    train_stdgi_loss = []
    for i in range(args.num_epochs_stdgi):
        if not early_stopping_stdgi.early_stop:
            loss = train_atten_stdgi(
                stdgi, train_dataloader, stdgi_optimizer_e,stdgi_optimizer_d, bce_loss, device,n_steps=2
            )
            early_stopping_stdgi(loss, stdgi)
            # scheduler_stdgi.step(loss)
            if args.log_wandb:
                wandb.log({"loss/stdgi_loss": loss})
            logging.info("Epochs/Loss: {}/ {}".format(i, loss))
    if args.log_wandb:
        wandb.run.summary["best_loss_stdgi"] = early_stopping_stdgi.best_score
    load_model(stdgi, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")

    if args.wo_climate:  # khong dung climate embedding
        decoder = WoCli_Decoder(
            args.input_dim + args.output_stdgi,
            args.output_dim,
            n_layers_rnn=args.n_layers_rnn,
            rnn=args.rnn_type,
            cnn_hid_dim=args.cnn_hid_dim,
            fc_hid_dim=args.fc_hid_dim,
            n_features=len(args.climate_features),
            num_input_stat=len(args.train_station),
        ).to(device)
    else:
        if args.decoder_type == "default":
            print("Using default decoder")
            decoder = Decoder(
                args.input_dim + args.output_stdgi,
                args.output_dim,
                n_layers_rnn=args.n_layers_rnn,
                rnn=args.rnn_type,
                cnn_hid_dim=args.cnn_hid_dim,
                fc_hid_dim=args.fc_hid_dim,
                n_features=len(args.climate_features),
                num_input_stat=len(args.train_station),
            ).to(device)
        elif args.decoder_type == "local_attention":
            print("Using local attention decoder")
            decoder = Local_Decoder(
                args.input_dim + args.output_stdgi,
                args.output_dim,
                n_layers_rnn=args.n_layers_rnn,
                rnn=args.rnn_type,
                cnn_hid_dim=args.cnn_hid_dim,
                fc_hid_dim=args.fc_hid_dim,
                n_features=len(args.climate_features),
                num_input_stat=len(args.train_station),
            ).to(device)
        elif args.decoder_type == "global_attention":
            print("Using global attention decoder")
            decoder = Global_Decoder(
                args.input_dim + args.output_stdgi,
                args.output_dim,
                n_layers_rnn=args.n_layers_rnn,
                rnn=args.rnn_type,
                cnn_hid_dim=args.cnn_hid_dim,
                fc_hid_dim=args.fc_hid_dim,
                n_features=len(args.climate_features),
                num_input_stat=len(args.train_station),
            ).to(device)
        elif args.decoder_type == "localglobal_attention":
            print("Using local and global attention decoder")
            decoder = Local_Global_Decoder(
                args.input_dim + args.output_stdgi,
                args.output_dim,
                n_layers_rnn=args.n_layers_rnn,
                rnn=args.rnn_type,
                cnn_hid_dim=args.cnn_hid_dim,
                fc_hid_dim=args.fc_hid_dim,
                n_features=len(args.climate_features),
                num_input_stat=len(args.train_station),
            ).to(device)
    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=args.lr_decoder, weight_decay=l2_coef
    )

    early_stopping_decoder = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_decoder,
        path=f"output/{args.group_name}/checkpoint/decoder_{args.name}.pt",
    )
    # scheduler_encoder = ReduceLROnPlateau(
    #     stdgi_optimizer, "min", factor=0.5, patience=3
    # )

    for i in range(args.num_epochs_decoder):
        if not early_stopping_decoder.early_stop:
            # uncomment
            train_loss = train_atten_decoder_fn(
                stdgi,
                decoder,
                train_dataloader,
                mse_loss,
                optimizer_decoder,
                device,
            )
            # scheduler_encoder.step(train_loss)
            valid_loss = 0
            for valid_station in args.valid_station:
                valid_dataset = AQDataSet(
                    data_df=trans_df[:],
                    climate_df=climate_df,
                    location_df=location_,
                    list_train_station=args.train_station,
                    test_station=valid_station,
                    valid=True,
                    input_dim=args.sequence_length,
                    # output_dim=args.output_dim,
                    corr=corr,
                    args=args,
                )
                valid_dataloader = DataLoader(
                    valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                valid_loss_ = test_atten_decoder_fn(
                    stdgi,
                    decoder,
                    valid_dataloader,
                    device,
                    mse_loss,
                    test=False,
                    args=args,
                )
                valid_loss += valid_loss_
            valid_loss = valid_loss / len(args.valid_station)
            early_stopping_decoder(valid_loss, decoder)
            print(
                "Epochs/Loss: {}/Train loss: {} / Valid loss: {}".format(
                    i, train_loss, valid_loss
                )
            )
            if args.log_wandb:
                wandb.log({"loss/decoder_loss": train_loss})
    load_model(decoder, f"output/{args.group_name}/checkpoint/decoder_{args.name}.pt")

    if args.log_wandb:
        wandb.run.summary["best_loss_decoder"] = early_stopping_decoder.best_score

    # test
    list_acc = []
    predict = {}
    for test_station in args.test_station:
        test_dataset = AQDataSet(
            data_df=trans_df,
            climate_df=climate_df[:],
            location_df=location_,
            list_train_station=args.train_station,
            test_station=test_station,
            test=True,
            input_dim=args.sequence_length,
            # output_dim=args.output_dim,
            corr=corr,
            args=args,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        list_prd, list_grt, _ = test_atten_decoder_fn(
            stdgi,
            decoder,
            test_dataloader,
            device,
            mse_loss,
            scaler,
            args=args,
        )
        output_arr = np.concatenate(
            (np.array(list_grt).reshape(-1, 1), np.array(list_prd).reshape(-1, 1)),
            axis=1,
        )
        out_df = pd.DataFrame(output_arr, columns=["ground_truth", "prediction"])
        out_dir = "output/{}/{}/".format(args.dataset, args.model_type)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_df.to_csv(out_dir + f"Station_{test_station}.csv")
        mae, mse, mape, rmse, r2, corr_, mdape = cal_acc(list_prd, list_grt)
        list_acc.append([test_station, mae, mse, mape, mdape, rmse, r2, corr_])
        predict[test_station] = {"grt": list_grt, "prd": list_prd}
        print("Test Accuracy: {}".format(mae, mse, corr))

    for test_station in args.test_station:
        df = pd.DataFrame(data=predict[test_station], columns=["grt", "prd"])
        if args.log_wandb:
            wandb.log({f"Station_{test_station}": df})
    tmp = np.array(list_acc).mean(0)
    list_acc.append(tmp)
    df = pd.DataFrame(
        np.array(list_acc),
        columns=["STATION", "MAE", "MSE", "MAPE", "MDAPE", "RMSE", "R2", "CORR"],
    )
    print(df)
    if args.log_wandb:
        wandb.log({"test_acc": df})
    for test_station in args.test_station:
        prd = predict[test_station]["prd"]
        grt = predict[test_station]["grt"]
        x = len(grt)
        fig, ax = plt.subplots(figsize=(40, 8))
        # ax.figure(figsize=(20,8))
        ax.plot(np.arange(x), grt[:x], label="grt")
        ax.plot(np.arange(x), prd[:x], label="prd")
        ax.legend()
        ax.set_title(f"Tram_{test_station}")
        if args.log_wandb:
            wandb.log({"Tram_{}".format(test_station): wandb.Image(fig)})
    if args.log_wandb:
        wandb.finish()
