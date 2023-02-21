import torch

def get_value_2D(image, index, padding_mode='zeros'):
    """
    Get 2D image value given an index.

    Parameters
    ----------
    image : tensor
        Input image. Should has dimension as N x C x H x W.
    index : tensor (long)
        (x, y). Should has dimension as N_idx x C x M x 4. 
        4 represents image.ndim. 
        M might be H*W or H according to different tasks.
    padding_mode : str
        Control values when index is out of boundary. Mode available: 
        'zeros', 'border'.

    Returns
    -------
    value : tensor
        Dimension is N_idx x C x M.
    """
    if image.ndim != 4:
        raise ValueError('The input image must be N x C x H x W')
    if index.ndim != 4 or index.shape[-1] != 4:
        raise ValueError('The input index must be N_idx x C x M x 4')

    N, C, H, W = image.shape
    N_idx = index.shape[0]
    M = index.shape[2]

    index = index.long()

    if padding_mode == 'zeros':
        index_should_be_zeros  = index[:,:,:,2] < 0
        index_should_be_zeros |= index[:,:,:,2] > H - 1
        index_should_be_zeros |= index[:,:,:,3] < 0
        index_should_be_zeros |= index[:,:,:,3] > W - 1
        index[:,:,:,2:4][index_should_be_zeros] = 0
    elif padding_mode == 'border':
        index[:,:,:,2] = torch.clamp(index[:,:,:,2], 0, H - 1)
        index[:,:,:,3] = torch.clamp(index[:,:,:,3], 0, W - 1)
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))
    
    value = image[torch.unbind(index.view(N_idx * C * M, 4), dim=-1)] \
            .view(N_idx, C, M)
    
    if padding_mode == 'zeros':
        value[index_should_be_zeros] = 0

    return value

def get_value_3D(object, index, padding_mode='zeros'):
    """
    Get 3D object value given an index.

    Parameters
    ----------
    object : tensor
        Input object. Should has dimension as N x C x D x H x W.
    index : tensor (long)
        (z, x, y). Should has dimension as N_idx x C x M x 5.
        5 represents object.ndim.
        M might be D*H*W or H*W according to different tasks.
    padding_mode : str
        Control values when index is out of boundary. Mode available:
        'zeros', 'border'.

    Returns
    -------
    value : tensor
        Dimension is N_idx x C x D x H x W.
    """
    if object.ndim != 5:
        raise ValueError('The input object must be N x C x D x H x W')
    if index.ndim != 4 or index.shape[-1] != 5:
        raise ValueError('The input index must be N_idx x C x M x 5')
    
    N, C, D, H, W = object.shape
    N_idx = index.shape[0]
    M = index.shape[2]

    index = index.long()

    if padding_mode == 'zeros':
        index_should_be_zeros  = index[:,:,:,2] < 0
        index_should_be_zeros |= index[:,:,:,2] > D - 1
        index_should_be_zeros |= index[:,:,:,3] < 0
        index_should_be_zeros |= index[:,:,:,3] > H - 1
        index_should_be_zeros |= index[:,:,:,4] < 0
        index_should_be_zeros |= index[:,:,:,4] > W - 1
        index[:,:,:,2:5][index_should_be_zeros] = 0
    elif padding_mode == 'border':
        index[:,:,:,2] = torch.clamp(index[:,:,:,2], 0, D - 1)
        index[:,:,:,2] = torch.clamp(index[:,:,:,3], 0, H - 1)
        index[:,:,:,2] = torch.clamp(index[:,:,:,4], 0, W - 1)
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))
    
    value = object[torch.unbind(index.view(N_idx * C * M, 5), dim=-1)] \
            .view(N_idx, C, M)
    
    if padding_mode == 'zeros':
        value[index_should_be_zeros] = 0
    
    return value

def interp_linear_2D(image, grid):
    """
    2D linear interpolation (bilinear)

    Parameters
    ----------
    image : tensor
        Input image. Should has dimension as N x C x H x W.
    grid : tensor
        Resampled point positions. Should has dimension as N x M x 2.
    
    Returns
    -------
    resampled : tensor
        Resampled result. Dimension is N x C x M.
    """
    if image.ndim != 4:
        raise ValueError('The input image must be N x C x H x W')
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError('The input grid must be N x M x 2')

    device = grid.device
    N, C, H, W = image.shape
    M = grid.shape[1]

    N_index = torch.arange(N).repeat(C,M,1).permute(2,0,1).to(device)
    C_index = torch.arange(C).repeat(N,M,1).permute(0,2,1).to(device)
    
    ix = grid[:,:,0].unsqueeze(1).repeat(1,C,1)
    iy = grid[:,:,1].unsqueeze(1).repeat(1,C,1)

    # north-east-south-west
    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw
    iy_ne = iy_nw + 1
    ix_sw = ix_nw + 1
    iy_sw = iy_nw
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    tn = ix - ix_nw
    tw = iy - iy_nw

    v_nw = get_value_2D(image, torch.stack((N_index,C_index,ix_nw,iy_nw), dim=-1))
    v_ne = get_value_2D(image, torch.stack((N_index,C_index,ix_ne,iy_ne), dim=-1))
    v_sw = get_value_2D(image, torch.stack((N_index,C_index,ix_sw,iy_sw), dim=-1))
    v_se = get_value_2D(image, torch.stack((N_index,C_index,ix_se,iy_se), dim=-1))

    v_n = v_nw * (1 - tw) + v_ne * tw
    v_s = v_sw * (1 - tw) + v_se * tw

    resampled = v_n * (1 - tn) + v_s * tn

    return resampled

def interp_linear_3D(object, grid):
    """
    3D linear interpolation (trilinear).

    Parameters
    ----------
    object : tensor
        Input object. Should has dimension as N x C x D x H x W.
    grid : tensor
        Resampled point positions. Should has dimension as N x M x 3.
    
    Returns
    -------
    resampled : tensor
        Resampled result. Dimension is N x C x M.
    """
    if object.ndim != 5:
        raise ValueError('The input object must be N x C x D x H x W')
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError('The input grid must be N x M x 3')

    device = grid.device
    N, C, D, H, W = object.shape
    M = grid.shape[1]
    
    N_index = torch.arange(N).repeat(C,M,1).permute(2,0,1).to(device)
    C_index = torch.arange(C).repeat(N,M,1).permute(0,2,1).to(device)

    iz = grid[:,:,0].unsqueeze(1).repeat(1,C,1)
    ix = grid[:,:,1].unsqueeze(1).repeat(1,C,1)
    iy = grid[:,:,2].unsqueeze(1).repeat(1,C,1)

    # top-bottom-north-east-south-west
    iz_tnw = torch.floor(iz)
    ix_tnw = torch.floor(ix)
    iy_tnw = torch.floor(iy)
    iz_tne = iz_tnw
    ix_tne = ix_tnw
    iy_tne = iy_tnw + 1
    iz_tsw = iz_tnw
    ix_tsw = ix_tnw + 1
    iy_tsw = iy_tnw
    iz_tse = iz_tnw
    ix_tse = ix_tnw + 1
    iy_tse = iy_tnw + 1
    iz_bnw = iz_tnw + 1
    ix_bnw = ix_tnw
    iy_bnw = iy_tnw
    iz_bne = iz_tnw + 1
    ix_bne = ix_tnw
    iy_bne = iy_tnw + 1
    iz_bsw = iz_tnw + 1
    ix_bsw = ix_tnw + 1
    iy_bsw = iy_tnw
    iz_bse = iz_tnw + 1
    ix_bse = ix_tnw + 1
    iy_bse = iy_tnw + 1

    tt = iz - iz_tnw
    tn = ix - ix_tnw
    tw = iy - iy_tnw

    v_tnw = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_tnw,ix_tnw,iy_tnw), dim=-1))
    v_tne = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_tne,ix_tne,iy_tne), dim=-1))
    v_tsw = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_tsw,ix_tsw,iy_tsw), dim=-1))
    v_tse = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_tse,ix_tse,iy_tse), dim=-1))
    v_bnw = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_bnw,ix_bnw,iy_bnw), dim=-1))
    v_bne = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_bne,ix_bne,iy_bne), dim=-1))
    v_bsw = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_bsw,ix_bsw,iy_bsw), dim=-1))
    v_bse = get_value_3D(object, 
                torch.stack((N_index,C_index,iz_bse,ix_bse,iy_bse), dim=-1))
    
    v_tn = v_tnw * (1 - tw) + v_tne * tw
    v_ts = v_tsw * (1 - tw) + v_tse * tw
    v_bn = v_bnw * (1 - tw) + v_bne * tw
    v_bs = v_bsw * (1 - tw) + v_bse * tw

    v_t = v_tn * (1 - tn) + v_ts * tn
    v_b = v_bn * (1 - tn) + v_bs * tn

    resampled = v_t * (1 - tt) + v_b * tt

    return resampled

def lanczos_kernel(x, a):
    """
    Compute Lanczos kernel from a sequence x given a.
    
    Parameters
    ----------
    x : tensor
        positions in Lanczos kernel.
    a : int
        Lanczos kernel's hyperparameter
    
    Returns
    -------
    lanczos : tensor
        The same dimension as input x. (Refer to Wikipedia.)
        lanczos(x) = sinc(pi*x)sinc(pi*x/a) = a*sin(pi*x)sin(pi*x/a)/(pi^2x^2).
        sinc(x) = sin(x)/x (unnormalized).
        sinc(x) = sin(pi*x)/(pi*x) (normalized).
    """
    lanczos = torch.sinc(x) * torch.sinc(x / a)
    lanczos[x <= -a] = lanczos[x >= a] = 0
    return lanczos

def interp_lanczos_2D(image, grid, a = 3):
    """
    2D Lanczos interpolation.

    Parameters
    ----------
    image : tensor
        Input image. Should has dimension as N x C x H x W.
    grid : tensor
        Resampled point positions. Should has dimension as N x M x 2.
    a : int
        Lanczos kernel's hyperparameter.
    
    Returns
    -------
    resampled : tensor
        Resampled result. Dimension is N x C x M.
    """
    if image.ndim != 4:
        raise ValueError('The input image must be N x C x H x W')
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError('The input grid must be N x M x 2')

    device = grid.device
    N, C, H, W = image.shape
    M = grid.shape[1]
    
    N_index = torch.arange(N).repeat(C,M,1).permute(2,0,1).to(device)
    C_index = torch.arange(C).repeat(N,M,1).permute(0,2,1).to(device)
    
    ix = grid[:,:,0].unsqueeze(1).repeat(1,C,1)
    iy = grid[:,:,1].unsqueeze(1).repeat(1,C,1)

    # north-east-south-west
    ix_nw = torch.floor(ix) - a + 1
    iy_nw = torch.floor(iy) - a + 1

    size = a * 2 - 1
    tempx = torch.zeros((N, C, M, size), dtype=image.dtype).to(device)

    m = 0
    for j in range(size):
        jj = iy_nw + j
        for i in range(size):
            ii = ix_nw + i
            tempx[:,:,:,m] += get_value_2D(image, \
                                torch.stack((N_index,C_index,ii,jj), dim=-1)) \
                                * lanczos_kernel(ix - ii, a)
        m += 1

    resampled = torch.zeros((N, C, M), dtype=image.dtype).to(device)

    m = 0
    for j in range(size):
        jj = iy_nw + j
        resampled += tempx[:,:,:,m] * lanczos_kernel(iy - jj, a)
        m += 1

    return resampled

def interp_lanczos_3D(object, grid, a = 3):
    """
    3D Lanczos interpolation.
    
    Parameters
    ----------
    object : tensor
        Input object. Should has dimension as N x C x D x H x W.
    grid : tensor
        Resampled point positions. Should has dimension as N x M x 3.
    a : int
        Lanczos kernel's hyperparamter.

    Returns
    -------
    resampled : tensor
        Resampled result. Dimension is N x C x M.
    """
    if object.ndim != 5:
        raise ValueError('The input object must be N x C x D x H x W')
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError('The input grid must be N x M x 3')
    
    device = grid.device
    N, C, D, H, W = object.shape
    M = grid.shape[1]

    N_index = torch.arange(N).repeat(C,M,1).permute(2,0,1).to(device)
    C_index = torch.arange(C).repeat(N,M,1).permute(0,2,1).to(device)

    iz = grid[:,:,0].unsqueeze(1).repeat(1,C,1)
    ix = grid[:,:,1].unsqueeze(1).repeat(1,C,1)
    iy = grid[:,:,2].unsqueeze(1).repeat(1,C,1)

    # top-bottom-north-east-south-west
    iz_tnw = torch.floor(iz)
    ix_tnw = torch.floor(ix)
    iy_tnw = torch.floor(iy)

    size = a * 2 - 1
    tempx = torch.zeros((N, C, M, size, size), dtype=object.dtype).to(device)

    m = 0
    for k in range(size):
        kk = iz_tnw + k
        n = 0
        for j in range(size):
            jj = iy_tnw + j
            for i in range(size):
                ii = ix_tnw + i
                tempx[:,:,:,m,n] += get_value_3D(object, \
                    torch.stack((N_index,C_index,kk,ii,jj), dim=-1)) \
                    * lanczos_kernel(ix - ii, a)
            n += 1
        m += 1
    
    tempy = torch.zeros((N, C, M, size), dtype=object.dtype).to(device)
    
    m = 0
    for k in range(size):
        kk = iz_tnw + k
        n = 0
        for j in range(size):
            jj = iy_tnw + j
            tempy[:,:,:,m] += tempx[:,:,:,m,n] * lanczos_kernel(iy-jj, a)
            n += 1
        m += 1
    
    resampled = torch.zeros((N, C, M), dtype=object.dtype).to(device)

    m = 0
    for k in range(size):
        kk = iz_tnw + k
        resampled += tempy[:,:,:,m] * lanczos_kernel(iz - kk, a)
        m += 1

    return resampled
 
def interp_lanczos_speedup_2D(image, grid, a = 3):
    """
    2D Lanczos interpolation (fast version).

    Parameters
    ----------
    image : tensor
        Input image. Should has dimension as N x C x H x W.
    grid : tensor
        Resampled point positions. Should has dimension as N x M x 2.
    a : int
        Lanczos kernel's hyperparameter.
    
    Returns
    -------
    resampled : tensor
        Resampled result. Dimension is N x C x M.
    """
    if image.ndim != 4:
        raise ValueError('The input image must be N x C x H x W')
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError('The input grid must be N x M x 2')

    device = grid.device
    N, C, H, W = image.shape
    M = grid.shape[1]
    
    N_index = torch.arange(N).repeat(C,M,1).permute(2,0,1).to(device)
    C_index = torch.arange(C).repeat(N,M,1).permute(0,2,1).to(device)
    
    ix = grid[:,:,0].unsqueeze(1).repeat(1,C,1)
    iy = grid[:,:,1].unsqueeze(1).repeat(1,C,1)

    # north-east-south-west
    ix_nw = torch.floor(ix) - a + 1
    iy_nw = torch.floor(iy) - a + 1

    size = a * 2 - 1

    f_idx_x = (ix - ix_nw).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,size,size) \
                - torch.arange(size).to(device)
    f_idx_y = (iy - iy_nw).unsqueeze(-1).repeat(1,1,1,size) \
                - torch.arange(size).to(device)
    lanczos_fx = lanczos_kernel(f_idx_x, a)
    lanczos_fy = lanczos_kernel(f_idx_y, a)

    size_j, size_i = torch.meshgrid(torch.arange(size), torch.arange(size), \
                                    indexing='ij')
    size_i, size_j = size_i.to(device), size_j.to(device)
    x_offset = ((ix_nw.unsqueeze(-1).unsqueeze(-1)).repeat(1,1,1,size,size) \
                + size_i).permute(3,4,0,1,2).view(size*size*N,C,M)
    y_offset = ((iy_nw.unsqueeze(-1).unsqueeze(-1)).repeat(1,1,1,size,size) \
                + size_j).permute(3,4,0,1,2).view(size*size*N,C,M)

    tempx = (get_value_2D(image, torch.stack((
             N_index.unsqueeze(0).repeat(size*size,1,1,1).view(size*size*N,C,M), \
             C_index.unsqueeze(0).repeat(size*size,1,1,1).view(size*size*N,C,M), \
             x_offset, y_offset), dim=-1)).view(size,size,N,C,M).permute(2,3,4,0,1) \
            * lanczos_fx).sum(-1)
    resampled = (tempx * lanczos_fy).sum(-1)

    return resampled
