# Overview
Project for unsupervised prediction in spatial


##  Install package
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html  
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html  
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html  
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html  
pip install torch-geometric 

## Run model
conda activate gcn
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --gconv=gcn --ngpus=3 --rnn_type=lstm