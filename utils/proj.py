import torch
import numpy as np

from .interp import interp_linear_2D, interp_lanczos_2D, interp_lanczos_speedup_2D
from .interp import interp_linear_3D, interp_lanczos_3D

def make_base_grid_4D(N, C, H, W):
    """
    Make a base grid based on 4D input.

    Returns
    -------
    base_grid : tensor
        Base grid for slicing 2D image. Dimenstion is N x H x 2.
        y axis value is 0.
    """
    xpr = torch.linspace(0, H, H) - H / 2
    ypr = torch.zeros(xpr.shape)
    base_grid = torch.stack((xpr, ypr), dim=1)
    base_grid = base_grid.repeat(N, 1, 1, 1)
    return base_grid

def make_base_grid_5D(N, C, D, H, W):
    """
    Make a base grid based on 5D input.

    Returns
    -------
    base_grid : tensor
        Base grid for slicing 3D object. Dimenstion is N x H x W x 3.
        z axis value is 0.
    """
    xpr, ypr = torch.meshgrid(torch.arange(H) - H / 2, 
                                 torch.arange(W) - W / 2, 
                                 indexing='ij')
    zpr = torch.zeros(xpr.shape)
    base_grid = torch.stack((zpr, xpr, ypr), dim=2)
    base_grid = base_grid.repeat(N, 1, 1, 1, 1)
    return base_grid

def base_grid_generator3d_xy(size):
    """Compute grid for the center slice
       This slice's coordinates are [x, y, 0]
    """
    N, C, H, W, D = size
    x = np.linspace(-H / 2, H / 2 - 1, H)
    y = np.linspace(-H / 2, H / 2 - 1, H)
    base_grid = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T
    base_grid = np.hstack([base_grid, np.zeros((H * W, 1))])
    base_grid = np.expand_dims(base_grid.reshape(H, W, 1, 3), 0)
    base_grid = base_grid.repeat(N, 0)

    return torch.Tensor(base_grid)


def slice_2D(image, rotation, mode='linear'):
    """
    Slicing a 1D signal in 2D image.

    Parameters
    ----------
    image : tensor
        2D image in frequency domain. Should has dimension (N) x 2 x H x W.
        C = 2 representing real and imaginary parts. H must be equal to W.
    rotation : tensor
        Rotation matrix representing the slicing direction.
        Show has dimension as N x 2 x 2.
    mode : str
        Interpolation method used in slicing. Methods available: 
        'linear', 'lanczos', 'fast-lanczos'.

    Returns
    -------
    projection : tensor
        The slicing result in the frequency domain. Equal to projection in the \
        spatial domain. Dimension is N x C x H
    """
    if rotation.ndim != 3 or rotation.shape[1] != 2 or rotation.shape[2] != 2:
        raise ValueError('The input rotation matrix must be N x 2 x 2')
    
    device = rotation.device
    N = rotation.shape[0]

    if image.ndim == 3:
        image = image.unsqueeze(0).repeat(N, 1, 1, 1)
    elif image.ndim == 4:
        if image.shape[0] != N:
            raise ValueError('The input 2D image batch size should be equal to \
                              the rotation batch size')
    else:
        raise ValueError('The input 2D image must be (N) x 2 x H x W')
    if image.shape[1] != 2:
        raise ValueError('The input 2D image must be (N) x 2 x H x W')
    
    C, H, W = image.shape[1:]
    if H != W:
        raise ValueError('The input 2D image must has the same H and W')
    
    base_grid = make_base_grid_4D(N, C, H, W).to(device)
    grid = torch.bmm(base_grid.view(N, H, 2), rotation).view(N, H, 2)
    grid[:,:,0] += H / 2
    grid[:,:,1] += W / 2

    if mode == 'linear':
        projection = interp_linear_2D(image, grid)
    elif mode == 'lanczos':
        projection = interp_lanczos_2D(image, grid)
    elif mode == 'fast-lanczos':
        projection = interp_lanczos_speedup_2D(image, grid)
    else:
        raise ValueError('Unknown interpolation: {}'.format(mode))
    
    return projection

def slice_3D(object, rotation, mode='linear'):
    """
    Slicing a 2D plane in 3D object.

    Parameters
    ----------
    object : tensor
        3D object in frequency domain. Should has dimension (N) x 2 x D x H x W.
        C = 2 representing real and imaginary parts. D = H = W.
    rotation : tensor
        Rotation matrix representing the slicing direction. 
        Show has dimension as N x 3 x 3.
    mode : str
        Interpolation method used in slicing. Methods available: 
        'linear', 'lanczos', 'fast-lanczos'.
    
    Returns
    -------
    projection : tensor
        The slicing result in the frequency domain. Equal to projection in the \
        spatial domain. Dimension is N x C x H x W
    """
    if rotation.ndim != 3 or rotation.shape[1] != 3 or rotation.shape[2] != 3:
        raise ValueError('The input rotation matrix must be N x 3 x 3')
    
    device = rotation.device
    N = rotation.shape[0]

    if object.ndim == 4:
        object = object.unsqueeze(0).repeat(N, 1, 1, 1, 1)
    elif object.ndim == 5:
        if object.shape[0] != N:
            raise ValueError('The input 3D object batch size should be equal to \
                              the rotation batch size')
    else:
        raise ValueError('The input 3D object must be (N) x 2 x D x H x W')
    if object.shape[1] != 2:
        raise ValueError('The input 3D object must be (N) x 2 x D x H x W')

    C, D, H, W = object.shape[1:]
    if D != H or D != W or H != W:
        raise ValueError('The input 3D object must has the same D, H and W')
    
    base_grid = make_base_grid_5D(N, C, D, H, W).to(device)
    # base_grid = base_grid_generator3d_xy((N,C,D,H,W))
    grid = torch.bmm(base_grid.view(N, H*W, 3), rotation).view(N, H, W, 3)
    grid[:,:,:,0] += D / 2
    grid[:,:,:,1] += H / 2
    grid[:,:,:,2] += W / 2

    if mode == 'linear':
        projection = interp_linear_3D(object, grid.view(N, H*W, 3))
    elif mode == 'lanczos':
        projection = interp_lanczos_3D(object, grid.view(N, H*W, 3))
    else:
        raise ValueError('Unknown interpolation: {}'.format(mode))
    
    return projection.view(N, C, H, W)
