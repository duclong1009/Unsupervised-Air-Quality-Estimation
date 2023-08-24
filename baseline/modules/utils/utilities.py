import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

def mape_loss(pred, target, device, reduction='mean'):
    """
    input, output: tensor of same shape
    """
    target = torch.where(
        target == 0, 
        torch.tensor(1e-6, device=device), 
        target
    )
    diff = (pred - target) / target
    if reduction == 'mean':
        mape = diff.abs().mean()
    elif reduction == 'sum':
        mape = diff.abs().sum()
    else:
        mape = diff
    return mape

def get_params(
    loc_df, spatial_res, 
    temporal_res, num_neighbors
):    
    """
    Get parameters of the model.
    
    h, w: height, width of the network
    """
    
    x0 = loc_df.iloc[:, 0].min()  
    y0 = loc_df.iloc[:, 1].min()
    x_max = loc_df.iloc[:, 0].max()  
    y_max = loc_df.iloc[:, 1].max()                           
    h = int((y_max - y0) // spatial_res + 2)
    w = int((x_max - x0) // spatial_res + 2)    
    return spatial_res, temporal_res, num_neighbors, h, w

def create_network(params, loc_df):   # verified
    """
      Construct a rectangular virtual station network.
      net_loc: lat & lon, shape (num_stations, 2)
    """
    x0 = loc_df.iloc[:, 0].min()  
    y0 = loc_df.iloc[:, 1].min()       
    spatial_res, _, _, h, w = params
    lat = np.arange(x0, x0 + spatial_res * (w - 0.9), spatial_res)
    lon = np.arange(y0, y0 + spatial_res * (h - 0.9), spatial_res)
    net_loc = np.vstack(
        [np.tile(lat, len(lon)), np.repeat(lon, len(lat))]).transpose()
    return net_loc

def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def save_checkpoint(model,optimizer,path):
    checkpoints = {"model_dict": model.state_dict(),
                   "optimizer_dict": optimizer.state_dict()}
    torch.save(checkpoints,path)

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["model_dict"])

def visualize(train_loss, val_loss, args, idx):
    plt.plot(train_loss['epoch'], train_loss['train_loss'], label = "Training loss", linestyle="-.")
    plt.plot(val_loss['epoch'], val_loss['val_loss'], label = "Validation loss", linestyle=":")
    plt.legend()
    plt.title("Training vs validation loss")
    plt.savefig(args.visualize_dir + "viz{}.png".format(idx))

def comp_distance(X1, X2):
    """
    Compute distance of locations in X1 wrt. locations in X2. 
    Use earth radius R = 6,371 km.
    
    Args:
    X1: lat and lon, shape (m, 2)
    X2: lat and lon, shape (m, 2)
    
    Returns:
    output: distances, shape (m, n) where each row is a location 
    in X1, each column is a location in X2.
    """
    R = 6371.
    X1 = X1 * np.pi/180.
    X2 = X2 * np.pi/180.
    
    A = torch.cos(X1[:,[1]]) @ torch.cos(X2[:,[1]]).T
    A = A * torch.cos(X1[:,[0]] - X2[:,[0]].T)
    A += torch.sin(X1[:,[1]]) @ torch.sin(X2[:,[1]]).T
    A = torch.where(
        A > 1., torch.tensor(1., dtype=torch.float64), A
    )     # remove warning
    
    return R * torch.arccos(A) 

def comp_idweight(filepath, train_idx, test_idx):
    """
    Compute Inverse Distance Weights of target station
    wrt. other stations. Use float64 for better accuracy.

    output shape: (num_test_stations, num_train_stations)
    """
    loc_df = pd.read_csv(filepath, header=0, usecols=['latitude', 'longitude'])
    loc_df = loc_df[['latitude', 'longitude']]
    # import pdb; pdb.set_trace()
    train_loc = torch.tensor(
        loc_df.iloc[train_idx].to_numpy(),
        dtype=torch.float64
    )
    test_loc = torch.tensor(
        loc_df.iloc[test_idx].to_numpy(),
        dtype=torch.float64
    )
    inv_distance = 1. / comp_distance(test_loc, train_loc)
    idw_matrix = inv_distance / inv_distance.sum(axis=1, keepdims=True) 
    return idw_matrix.type(torch.float32)
