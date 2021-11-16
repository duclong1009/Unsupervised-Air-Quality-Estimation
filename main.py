
import argparse
import torch 
import numpy as np
import random 
import wandb
import time
import sys
import os
from modules.train.model import STDGISequential


parser = argparse.ArgumentParser(description='STDGI')
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--output', type=str, default='output/results/')
parser.add_argument("--seed", type=int, default=99)
parser.add_argument('--top_k', type=int, default=10)

parser.add_argument('--gconv', type=str, default='gcn')
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lrate', type=float, default=None)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--early_stop', type=int, default=20)



args = parser.parse_args()
print(args)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic=True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = STDGISequential(args.seed)

if __name__ == '__main__':
    model = get_model()
    model.set_args(args)
    model.init()
    model.run()
    model.test()
    model.log()