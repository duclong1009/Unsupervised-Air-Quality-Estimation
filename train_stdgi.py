import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument("--target_station", default=14, type=int)
    parser.add_argument(
        "--train_station",
        default=[i for i in range(28)],
        type=list,
    )
    parser.add_argument("--input_dim", default=12, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--model_type", default="gcn", type=str)
    parser.add_argument("--load_model", default=False)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--checkpoint_file", default="./checkpoint/stdgi/", type=str)
    parser.add_argument("--visualize_dir", default="./output/visualize/", type=str)
    parser.add_argument("--topology_construction", default="distance", type=str)
    parser.add_argument("--path_model", default="", type=str)
    return parser.parse_args()


import pandas as pd
from modules.utils.ultilities import seed, save_checkpoint, load_model
from modules.utils.dataloader import get_columns, preprocess_pipeline, AQDataset2
from models.stdgi import STDGI
from modules.train import train_stdgi_fn

if __name__ == "__main__":
    args = parse_args()
    seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "./data/"
    location = pd.read_csv(file_path + "locations.csv").to_numpy()
    location = location[:, 1:]
    res, res_rev, pm_df = get_columns(file_path)
    trans_df, scaler = preprocess_pipeline(pm_df)
    train_dataset = AQDataset2(
        data_df=trans_df,
        location=location,
        list_train_station=args.train_station,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        topology_construction=args.topology_construction,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    stdgi = STDGI(args.input_dim, args.output_stdgi, args.model_type).to(device)
    l2_coef = 0.0
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(stdgi.parameters(), lr=args.lr, weight_decay=l2_coef)
    if args.load_model:
        load_model(stdgi, args.path_model)
    train_stdgi_loss = []
    for i in range(100):
        loss = train_stdgi_fn(
            stdgi, train_dataloader, optimizer, bce_loss, device, gconv="gcn"
        )
        save_checkpoint(stdgi, optimizer, f"../checkpoint/stdgi/stdgi_epoch_{i}")
        print("Epoch: {}/ Loss: {}".format(i, loss))
        train_stdgi_loss.append(loss)
