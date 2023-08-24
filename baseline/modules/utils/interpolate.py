import numpy as np 
from models.idw_knn import *

def haverside_dist(X1, X2):
    """
    Compute haverside (great circle) distance of locations in X1 and X2, use 
    earth radius R = 6,371 km.
    
    Args:
    X1: lat and lon, shape (m1, 2)
    X2: lat and lon, shape (m2, 2)
    
    Returns:
    ouput: distance matrix, shape (m1, m2)
    """
    R = 6371.
    X1 = X1 * np.pi/180. # conver to rad 
    X2 = X2 * np.pi/180.
    
    A = np.cos(X1[:,[1]]) @ np.cos(X2[:,[1]]).T # cos(ya) * cos(yb)
    A = A * np.cos(X1[:,[0]] - X2[:,[0]].T) # cos(ya) * cos(yb) * cos(xa - xb)
    A += np.sin(X1[:,[1]]) @ np.sin(X2[:,[1]]).T # sin(ya) * sin(yb)
    A = np.where(A > 1., 1., A)  # for stability 
    return R * np.arccos(A)

def interpolate_pm(
    unknown_loc, known_loc, known_pm, params, reshape=True
):
    """
      Interpolate the PM2.5 values of unknown locations, using k nearest known stations. 
      known_pm: array (L, num_known)
      unkown_pm: array (L, 1, H, W)
    """
    # import pdb; pdb.set_trace()
    _, _, num_neighbors, h, w = params
    

    unknown_lst = []
    for idx, row in enumerate(known_pm):
        known_data = np.transpose(row)
        idw_tree = tree(known_loc, known_data)
        unknown_data = idw_tree(unknown_loc, k = 5)
        unknown_lst.append(unknown_data.tolist())
    unknown_pm = np.array(unknown_lst)
    # distance = haverside_dist(unknown_loc, known_loc)
    # distance = np.where(distance < 1e-6, 1e-6, distance)      
    # bound = np.partition(
    #     distance, num_neighbors - 1, axis=1
    # )[:, [num_neighbors - 1]] # lay distance cua tram thu k 
    # neighbor_mask = np.where(distance <= bound, 1., np.nan) # lay cac tram co distance < bound 
    
    # neighbor_dist = distance * neighbor_mask 
    # R = 1 / neighbor_dist
    # weight = R / np.nansum(R, axis=1, keepdims=True)
    # weight = np.nan_to_num(weight, nan=0.)
    # unknown_pm = known_pm @ weight.T
    if reshape:
        unknown_pm = unknown_pm.reshape([-1, 1, h, w])
    return unknown_pm
