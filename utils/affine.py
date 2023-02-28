import torch

def rot_2D(thetas):
    """
    Given angles, generate 2D rotation matrices.

    Parameters
    ----------
    thetas : tensor
        A sequence of angles in radius ([0, 2pi)]). The dimension is N.

    Returns
    -------
    R : tensor
        Rotation matrices in 2D. The dimension is N x 2 x 2.
    """
    if thetas.ndim != 1:
        raise ValueError('The input thetas must be N')
    N = thetas.shape[0]
    cos_a, sin_a = torch.cos(thetas).unsqueeze(1), torch.sin(thetas).unsqueeze(1)
    R = torch.cat((cos_a, -sin_a, sin_a, cos_a), 1).resize(N, 2, 2)
    return R

class Rotation2Ds_Z(torch.autograd.Function):
    """
    This autograd function is used for optimizing 2D rotation matrix.
    """
    @staticmethod
    def forward(ctx, theta):
        cos_a, sin_a = torch.cos(theta).unsqueeze(1), torch.sin(theta).unsqueeze(1)
        R = torch.cat((cos_a, -sin_a, sin_a, cos_a), 1).resize(theta.shape[0], 2, 2)
        ctx.save_for_backward(theta)
        return R
    
    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_input = -torch.sin(theta)*grad_output[:,0,0] + \
                     -torch.cos(theta)*grad_output[:,0,1] + \
                      torch.cos(theta)*grad_output[:,1,0] + \
                     -torch.sin(theta)*grad_output[:,1,1]
        return grad_input
