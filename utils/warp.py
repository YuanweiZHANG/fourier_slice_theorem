import torch

from .interp import interp_linear_2D, interp_lanczos_2D, interp_lanczos_speedup_2D

def make_base_grid_4D(N, C, H, W):
    """
    Make a base grid based on 4D input (typically a 2D image).

    Returns
    -------
    base_grid : tensor
        Base grid for 2D image. Dimension is N x H x W x 2.
    """
    xpr, ypr = torch.meshgrid(torch.arange(H) - H / 2, 
                              torch.arange(W) - W / 2, 
                              indexing='ij')
    base_grid = torch.stack((xpr, ypr), dim=2)
    base_grid = base_grid.repeat(N, 1, 1, 1)
    return base_grid

def make_base_grid_5D(N, C, D, H, W):
    """
    Make a base grid based on 5D input (typically a 3D object).
    
    Returns
    -------
    base_grid : tensor
        Base grid for 3D object. Dimension is N x D x H x W x 3.
    """
    zpr, xpr, ypr = torch.meshgrid(torch.arange(D) - D / 2,
                                   torch.arange(H) - H / 2, 
                                   torch.arange(W) - W / 2, 
                                   indexing='ij')
    base_grid = torch.stack((zpr, xpr, ypr), dim=3)
    base_grid = base_grid.repeat(N, 1, 1, 1, 1)
    return base_grid

def warp_2D(image, rotation, mode='linear'):
    """
    Considering that most of the library such as scimage and pillow 
    do not implement Lanczos interpolation for rotation, therefore,
    we implement rotation (warp) by ourselves.

    Parameters
    ----------
    image : tensor
        Input image. Should has dimension as (N) x (C) x H x W.
    rotation : tensor
        Warp rotation matrix. Show has dimension as N x 2 x 2.
    mode : str
        Interpolation method used in warp. Methods available: 
        'linear', 'lanczos', 'fast-lanczos'.
    
    Returns
    -------
    warped_image : tensor
        Warped image. Dimension is N x C x H x W.
    """

    if rotation.ndim != 3 or rotation.shape[1] != 2 or rotation.shape[2] != 2:
        raise ValueError('The input rotation matrix must be N x 2 x 2')
    
    device = rotation.device
    N = rotation.shape[0]

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1)
    elif image.ndim == 3:
        image = image.unsqueeze(0).repeat(N, 1, 1, 1)
    elif image.ndim == 4:
        if image.shape[0] != N:
            raise ValueError('The input 2D image batch size should be equal to \
                              the rotation batch size')
    else:
        raise ValueError('The input 2D image must be (N) x (C) x H x W')

    C, H, W = image.shape[1:]
    
    base_grid = make_base_grid_4D(N, C, H, W).to(device)
    grid = torch.bmm(base_grid.view(N, H*W, 2), rotation).view(N, H, W, 2)

    grid[:,:,:,0] += H / 2
    grid[:,:,:,1] += W / 2

    if mode == 'linear':
        warped_image = interp_linear_2D(image, grid.view(N, H*W, 2))
    elif mode == 'lanczos':
        warped_image = interp_lanczos_2D(image, grid.view(N, H*W, 2))
    elif mode == 'fast-lanczos':
        warped_image = interp_lanczos_speedup_2D(image, grid.view(N, H*W, 2))
    else:
        raise ValueError('Unknown interpolation: {}'.format(mode))
    
    return warped_image.view(N, C, H, W)

