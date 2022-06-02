import argparse
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
from src.models.stdgi import Attention_STDGI, InterpolateAttention_STDGI
from src.models.decoder import (
    Decoder,
    InterpolateAttentionDecoder,
    InterpolateDecoder,
    WoCli_Decoder,
)
from src.modules.train.train import train_atten_decoder_fn
from src.modules.train.train import train_atten_stdgi


def print_mem():
    # import psutil
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    # print('memory % used:', psutil.virtual_memory()[2])
    import os, psutil

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes


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
    parser.add_argument("--input_dim", default=9, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--sequence_length", default=12, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--lr_stdgi", default=5e-3, type=float)
    parser.add_argument("--num_epochs_stdgi", default=30, type=int)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--checkpoint_stdgi", default="stdgi", type=str)
    parser.add_argument("--output_path", default="./out/", type=str)
    parser.add_argument("--en_hid1", default=200, type=int)
    parser.add_argument("--en_hid2", default=400, type=int)
    parser.add_argument("--dis_hid", default=6, type=int)
    parser.add_argument("--act_fn", default="relu", type=str)
    parser.add_argument("--delta_stdgi", default=0, type=float)
    parser.add_argument("--num_epochs_decoder", default=30, type=int)
    parser.add_argument("--lr_decoder", default=5e-3, type=float)
    parser.add_argument("--checkpoint_decoder", default="decoder", type=str)
    parser.add_argument("--delta_decoder", default=0, type=float)
    parser.add_argument("--cnn_hid_dim", default=64, type=int)
    parser.add_argument("--fc_hid_dim", default=128, type=int)
    parser.add_argument("--rnn_type", default="LSTM", type=str)
    parser.add_argument("--n_layers_rnn", default=1, type=int)
    parser.add_argument("--interpolate", default=False, type=bool)
    parser.add_argument("--attention_decoder", default=False, type=bool)
    # parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--stdgi_noise_min", default=0.4, type=float)
    parser.add_argument("--stdgi_noise_max", default=0.7, type=float)
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_wandb", action="store_false")
    parser.add_argument(
        "--climate_features",
        default=["2m_temperature", "surface_pressure", "evaporation", "wind_speed"],
        type=list,
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gede", "wogcn", "wornnencoder", "woclimate"]
    )
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
    file_path = "./data/merged_AQ/"
    comb_arr, location_, station, features_name,corr = get_data_array(
        file_path, args.climate_features
    )
    print(station)
    trans_df, climate_df, scaler = preprocess_pipeline(comb_arr)
    config["features"] = features_name
    test_name = "test1"
    # args.train_station = [ 92,  18,  38,  37,  16,  76,  27, 131,  35,  22,  81,  80,  30,
    #     82, 129,  49, 101, 102, 130, 107,  99]
    # args.valid_station = [122, 100,  42,  26,  36, 113,  74, 126, 132, 116,  72, 117, 104,
    #     68,   0]
    # args.test_station = [69, 6, 135, 71, 137, 41, 73, 28, 29, 127]
    args.train_station = [15, 17, 19, 21, 48, 73, 96, 114, 131, 137]
    args.valid_station = [20, 34, 56, 85]
    args.test_station = [97 ,98,134 ]
    train_dataset = AQDataSet(
        data_df=trans_df[:],
        climate_df=climate_df,
        location_df=location_,
        list_train_station=args.train_station,
        input_dim=args.sequence_length,
        interpolate=args.interpolate,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # config["loss"] = 'mse'
    if args.log_wandb:
        wandb.init(
            entity="aiotlab",
            project="Spatial_PM2.5",
            group="merged_AQ",
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
    early_stopping_stdgi = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_stdgi,
        path="./out/checkpoint/" + args.checkpoint_stdgi + ".pt",
    )
    logging.info(
        f"Training stdgi ||  interpolate {args.interpolate} || attention decoder {args.attention_decoder} || epochs {args.num_epochs_stdgi} || lr {args.lr_stdgi}"
    )
    train_stdgi_loss = []
    print_mem()
    for i in range(args.num_epochs_stdgi):
        if not early_stopping_stdgi.early_stop:
            loss = train_atten_stdgi(
                stdgi,
                train_dataloader,
                stdgi_optimizer_e,
                stdgi_optimizer_d,
                bce_loss,
                device,
                interpolate=args.interpolate,
            )
            early_stopping_stdgi(loss, stdgi)
            if args.log_wandb:
                wandb.log({"loss/stdgi_loss": loss})
            logging.info("Epochs/Loss: {}/ {}".format(i, loss))
            print(f"Epoch_{i} -------------------------------")
            print_mem()
    if args.log_wandb:
        wandb.run.summary["best_loss_stdgi"] = early_stopping_stdgi.best_score
    load_model(stdgi, "./out/checkpoint/" + args.checkpoint_stdgi + ".pt")
    if args.model_type == "woclimate":
        decoder = WoCli_Decoder(
            args.input_dim + args.output_stdgi,
            args.output_dim,
            n_layers_rnn=args.n_layers_rnn,
            rnn=args.rnn_type,
            cnn_hid_dim=args.cnn_hid_dim,
            fc_hid_dim=args.fc_hid_dim,
            n_features=len(args.climate_features),
        ).to(device)

    else:
        decoder = Decoder(
            args.input_dim + args.output_stdgi,
            args.output_dim,
            n_layers_rnn=args.n_layers_rnn,
            rnn=args.rnn_type,
            cnn_hid_dim=args.cnn_hid_dim,
            fc_hid_dim=args.fc_hid_dim,
            n_features=len(args.climate_features),
        ).to(device)

    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=args.lr_decoder, weight_decay=l2_coef
    )

    early_stopping_decoder = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_decoder,
        path="./out/checkpoint/" + args.checkpoint_decoder + ".pt",
    )

    for i in range(args.num_epochs_decoder):
        if not early_stopping_decoder.early_stop:
            train_loss = train_atten_decoder_fn(
                stdgi,
                decoder,
                train_dataloader,
                mse_loss,
                optimizer_decoder,
                device,
                interpolate=args.interpolate,
            )
            valid_loss = 0
            for valid_station in args.valid_station:
                valid_dataset = AQDataSet(
                    data_df=trans_df,
                    climate_df=climate_df,
                    location_df=location_,
                    list_train_station=args.train_station,
                    test_station=valid_station,
                    test=True,
                    input_dim=args.sequence_length,
                    # output_dim=args.output_dim,
                    interpolate=args.interpolate,
                )
                valid_dataloader = DataLoader(
                    valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                valid_loss_ = test_atten_decoder_fn(
                    stdgi, decoder, valid_dataloader, device, mse_loss, test=False
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
    load_model(decoder, "./out/checkpoint/" + args.checkpoint_decoder + ".pt")
    # for name, param in decoder.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # breakpoint()
    if args.log_wandb:
        wandb.run.summary["best_loss_decoder"] = early_stopping_decoder.best_score

    # test
    list_acc = []
    predict = {}
    # print(args.test_station)
    # print(args.train_station)
    for test_station in args.test_station:
        test_dataset = AQDataSet(
            data_df=trans_df,
            climate_df=climate_df,
            location_df=location_,
            list_train_station=args.train_station,
            test_station=test_station,
            test=True,
            input_dim=args.sequence_length,
            # output_dim=args.output_dim,
            interpolate=args.interpolate,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        list_prd, list_grt, _ = test_atten_decoder_fn(
            stdgi, decoder, test_dataloader, device, mse_loss, args.interpolate, scaler
        )
        output_arr = np.concatenate(
            (np.array(list_grt).reshape(-1, 1), np.array(list_prd).reshape(-1, 1)),
            axis=1,
        )
        out_df = pd.DataFrame(output_arr, columns=["ground_truth", "prediction"])
        out_df.to_csv(f"output/Station_{test_station}.csv")
        mae, mse, mape, rmse, r2, corr,mdape = cal_acc(list_prd, list_grt)
        list_acc.append([test_station, mae, mse, mape,mdape, rmse, r2, corr])
        predict[test_station] = {"grt": list_grt, "prd": list_prd}
        print("Test Accuracy: {}".format(mae, mse, corr))
        if args.log_wandb:
            wandb.log({f"Station_{test_station}": list_prd})

    for test_station in args.test_station:
        df = pd.DataFrame(data=predict[test_station], columns=["grt", "prd"])
        df.to_csv(
            "./result/{}/Station_{}.csv".format(test_name, test_station), index=False
        )
    tmp = np.array(list_acc).mean(0)
    list_acc.append(tmp)
    df = pd.DataFrame(
        np.array(list_acc),
        columns=["STATION", "MAE", "MSE", "MAPE","MDAPE", "RMSE", "R2", "CORR"],
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
