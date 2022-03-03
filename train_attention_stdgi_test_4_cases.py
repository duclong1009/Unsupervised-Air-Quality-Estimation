import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm.notebook import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument(
        "--train_station",
        type=list,
    )
    parser.add_argument(
        "--test_station",
        type=list,
    )
    parser.add_argument("--input_dim", default=9, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--sequence_length", default=12, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--patience", default=5, type=int)

    parser.add_argument("--lr_stdgi", default=5e-3, type=float)
    parser.add_argument("--num_epochs_stdgi", default=100, type=int)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--checkpoint_stdgi", default="stdgi", type=str)
    parser.add_argument("--output_path", default="./out/", type=str)
    parser.add_argument("--en_hid1", default=200, type=int)
    parser.add_argument("--en_hid2", default=400, type=int)
    parser.add_argument("--dis_hid", default=6, type=int)
    parser.add_argument("--act_fn", default="relu", type=str)
    parser.add_argument("--delta_stdgi", default=0, type=float)

    parser.add_argument("--num_epochs_decoder", default=100, type=int)
    parser.add_argument("--lr_decoder", default=5e-3, type=float)
    parser.add_argument("--checkpoint_decoder", default="decoder", type=str)
    parser.add_argument("--delta_decoder", default=0, type=float)
    parser.add_argument("--cnn_hid_dim", default=64, type=int)
    parser.add_argument("--fc_hid_dim", default=128, type=int)
    parser.add_argument("--rnn_type", default="LSTM", type=str)
    parser.add_argument("--n_layers_rnn", default=3, type=int)
    parser.add_argument("--interpolate", default=False, type=bool)
    parser.add_argument("--attention_decoder", default=False, type=bool)
    # parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--name", type=str)
    parser.add_argument("--climate_features", default=['lrad', 'shum', 'pres', 'temp', 'wind', 'srad'], type=list)
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
    file_path = "./data/Beijing2/"
    comb_arr, location_, station, features_name = get_data_array(file_path,args.climate_features)
    trans_df,climate_df, scaler = preprocess_pipeline(comb_arr)
    config["features"] = features_name

    # 4 thi nghiem
    tests = [
        {   'name': 'test_1',
            'train': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            'valid': [16,17,18,19],
            'test': [20,21, 22,23,24,25,26,27,28,29,30,31,32,33,34] 
        }, 
        {   'name': 'test_2',
            'train': [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            'valid': [31,32,33,34],
            'test': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] 
        }, 
        {   'name': 'test_3',
            'train': [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19],
            'valid': [27,28,29,30],
            'test': [11,12,13,14,20,21,22,23,24,25,26,31,32,33,34] 
        },
        {   'name': 'test_4',
            'train': [1,2,3,16,17,18,19, 28,29,30,22,23, 11,12,13,14],
            'valid': [27,24,25,26],
            'test': [20,21, 31,32,33,34, 4,5,6,7,8,9,10,15, 0] 
        }
    ]
    for test in tests:
        args.name = test['name']
        args.train_station = test['train']
        args.test_station = test['test']

        train_dataset = AQDataSet(
            data_df=trans_df,
            climate_df = climate_df,
            location_df=location_,
            list_train_station=args.train_station,
            input_dim=args.sequence_length,
            interpolate=args.interpolate,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        # config["loss"] = 'mse'
        wandb.init(
            entity="aiotlab",
            project="Spatial_PM2.5",
            group="Ablation_Study",
            name=f"{args.name}",
            config=config,
        )
        # Model Stdgi
        if not args.interpolate:
            stdgi = Attention_STDGI(
                in_ft=args.input_dim,
                out_ft=args.output_stdgi,
                en_hid1=args.en_hid1,
                en_hid2=args.en_hid2,
                dis_hid=args.dis_hid,
            ).to(device)
        else:
            stdgi = InterpolateAttention_STDGI(
                in_ft=args.input_dim,
                out_ft=args.output_stdgi,
                en_hid1=args.en_hid1,
                en_hid2=args.en_hid2,
                dis_hid=args.dis_hid,
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
        # stdgi.apply(init_weights)
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
                wandb.log({"loss/stdgi_loss": loss})
                logging.info("Epochs/Loss: {}/ {}".format(i, loss))
        wandb.run.summary["best_loss_stdgi"] = early_stopping_stdgi.best_score
        load_model(stdgi,"./out/checkpoint/" + args.checkpoint_stdgi + ".pt")

        if not args.interpolate:
            decoder = Decoder(
                args.input_dim + args.output_stdgi,
                args.output_dim,
                n_layers_rnn=args.n_layers_rnn,
                rnn=args.rnn_type,
                cnn_hid_dim=args.cnn_hid_dim,
                fc_hid_dim=args.fc_hid_dim,
                n_features=len(args.climate_features)
            ).to(device)

        else:
            if not args.attention_decoder:
                decoder = InterpolateDecoder(
                    args.input_dim + args.output_stdgi,
                    args.output_dim,
                    n_layers_rnn=args.n_layers_rnn,
                    rnn=args.rnn_type,
                    cnn_hid_dim=args.cnn_hid_dim,
                    fc_hid_dim=args.fc_hid_dim,
                ).to(device)
            else:
                decoder = InterpolateAttentionDecoder(
                    args.input_dim + args.output_stdgi,
                    args.output_dim,
                    num_stat=len(args.train_station),
                    stdgi_out_dim=args.output_stdgi,
                    n_layers_rnn=args.n_layers_rnn,
                    rnn=args.rnn_type,
                    cnn_hid_dim=args.cnn_hid_dim,
                    fc_hid_dim=args.fc_hid_dim,
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

                epoch_loss = train_atten_decoder_fn(
                    stdgi,
                    decoder,
                    train_dataloader,
                    mse_loss,
                    optimizer_decoder,
                    device,
                    interpolate=args.interpolate,
                )
                early_stopping_decoder(epoch_loss, decoder)
                print("Epochs/Loss: {}/ {}".format(i, epoch_loss))
                wandb.log({"loss/decoder_loss": epoch_loss})
        load_model(decoder, "./out/checkpoint/" + args.checkpoint_decoder + ".pt")
        wandb.run.summary["best_loss_decoder"] = early_stopping_decoder.best_score
        # test
        list_acc = []
        predict = {}
        for test_station in args.test_station:
            test_dataset = AQDataSet(
                data_df=trans_df,
                climate_df = climate_df,
                location_df=location_,
                list_train_station=[i for i in range(20)],
                test_station=test_station,
                test=True,
                input_dim=args.sequence_length,
                # output_dim=args.output_dim,
                interpolate=args.interpolate,
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False
            )
            # breakpoint()
            list_prd, list_grt = test_atten_decoder_fn(
                stdgi, decoder, test_dataloader, device, args.interpolate, scaler
            )
            mae, mse, mape, rmse, r2, corr = cal_acc(list_prd, list_grt)
            # breakpoint()
            list_acc.append([test_station, mae, mse, mape, rmse, r2, corr])
            predict[test_station] = {"grt": list_grt, "prd": list_prd}
            print("Test Accuracy: {}".format(mae, mse, corr))
            wandb.log({f"Station_{test_station}": list_prd})
        df = pd.DataFrame(
            np.array(list_acc),
            columns=["STATION", "MAE", "MSE", "MAPE", "RMSE", "R2", "CORR"],
        )
        wandb.log({"test_acc": df})
        df.to_csv(args.output_path + "test/acc_{}.csv".format(args.name), index=False)
        for test_station in args.test_station:
            prd = predict[test_station]["prd"]
            grt = predict[test_station]["grt"]
            x = 800
            fig, ax = plt.subplots(figsize=(20, 8))
            # ax.figure(figsize=(20,8))
            ax.plot(np.arange(x), grt[:x], label="grt")
            ax.plot(np.arange(x), prd[:x], label="prd")
            ax.legend()
            ax.set_title(f"Tram_{test_station}")
            wandb.log({"Tram_{}".format(test_station): wandb.Image(fig)})

        del stdgi
        del decoder 

