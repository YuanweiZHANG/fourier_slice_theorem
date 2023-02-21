import csv
import numpy as np
import torch

def save_object(object, filepath):
    """
    Save 3D object into a CSV file.

    Parameters
    ----------
    object : tensor
        3D object. Should has dimension as D x H x W.
    filepath : str
        CSV file path. Should end up with '.csv'.
    """
    if object.ndim != 3:
        raise ValueError('The input object must be C x H x W')

    with open(filepath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['x', 'y', 'z', 'density'])

        D, H, W = object.shape
        for z in range(D):
            for x in range(H):
                for y in range(W):
                    csvwriter.writerow(x, y, z, object[z,x,y])
    return

def read_object(filepath, center=True):
    """
    Read object from a CSV file.

    Parameters
    ----------
    filepath : str
        CSV file path. Should end up with '.csv'. 
        This CSV file should has 'x', 'y', 'z', 'density' as the row title.
    center : bool
        Whether to put object at the center and delete.
    
    Returns
    -------
    object : tensor
        3D object.
    """
    # z, x, y = np.loadtxt(filepath, dtype=np.int32, delimiter=',', skiprows=1, \
    #                      usecols=(0,1,2), unpack=True)
    x, y, z = np.loadtxt(filepath, dtype=np.int32, delimiter=',', skiprows=1, \
                         usecols=(0,1,2), unpack=True)
    density = np.loadtxt(filepath, dtype=np.float32, delimiter=',', skiprows=1,\
                         usecols=3, unpack=True)

    D_min, H_min, W_min = z.min(), x.min(), y.min()
    D_max, H_max, W_max = z.max(), x.max(), y.max()
    if center:
        DHW_max = max(max(D_max-D_min, H_max-H_min), W_max-W_min)
    else:
        DHW_max = max(max(D_max, H_max), W_max)
    object = np.zeros((DHW_max+1, DHW_max+1, DHW_max+1))
    for i in range(x.shape[0]):
        if center:
            new_x = x[i] + (DHW_max - H_max - H_min) // 2
            new_y = y[i] + (DHW_max - W_max - W_min) // 2
            new_z = z[i] + (DHW_max - D_max - D_min) // 2
        else:
            new_x = x[i]
            new_y = y[i]
            new_z = z[i]
        object[DHW_max-new_z, new_x, new_y] = density[i]
    
    return torch.from_numpy(object)
    