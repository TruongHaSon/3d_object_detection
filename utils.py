import time
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from collections import namedtuple, defaultdict, Counter

def rotate(vector, angle):
    """
    Rotate a vector around the y-axis
    """
    sinA, cosA = torch.sin(angle), torch.cos(angle)
    xvals =  cosA * vector[..., 0] + sinA * vector[..., 2]
    yvals = vector[..., 1]
    zvals = -sinA * vector[..., 0] + cosA * vector[..., 2]
    return torch.stack([xvals, yvals, zvals], dim=-1)


def perspective(matrix, vector):
    """Projext a 3D vector to 2D image using transformation matrix.

    Args:
        matrix (torch.tensor): transformation matrix with the shape (3, 4)
        vector: 3D vector with the shape of (3)
    Return:
        2D vector in 2D space (u/z, v/z)
    """
    vector = vector.unsqueeze(-1) #[3, 1]
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def make_grid(grid_size, grid_offset, grid_res):
    """
    Args:
        grid_size: size of grid.
        grid_offset (int): grid offset.
        grid_res (float): resolution of grid.
    Returns:
        grid: a grid where y is fixed, only has x_coords
            and z_coords. and the shape is (160, 160, 3)
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0., width, grid_res) + xoff
    zcoords = torch.arange(0., depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def bbox_corners(obj):
    """Get the corners of 3D bounding box

    Args:
        obj (namedtuple): an 3D object in label.
    Return:
        corners (torch.tensor): the corners of 3D bb with shape (8,3)
    """

    # Get corners of bounding box in object space
    offsets = torch.tensor([
        [-.5,  0., -.5],    # Back-left lower
        [ .5,  0., -.5],    # Front-left lower
        [-.5,  0.,  .5],    # Back-right lower
        [ .5,  0.,  .5],    # Front-right lower
        [-.5, -1., -.5],    # Back-left upper
        [ .5, -1., -.5],    # Front-left upper
        [-.5, -1.,  .5],    # Back-right upper
        [ .5, -1.,  .5],    # Front-right upper
    ])
    corners = offsets * torch.tensor(obj.dimensions)
    # corners = corners[:, [2, 0, 1]]

    # Apply y-axis rotation
    corners = rotate(corners, torch.tensor(obj.angle))

    # Apply translation
    corners = corners + torch.tensor(obj.position)
    return corners

def corners3d(obj, calib, color='b'):
    """Get corners of 3D bounding boxes and project to 2D image
    
    Args: 
        obj (namedtuple): object in image
        calib (torch.tensor): Intrinsic matrix with the shape of (3, 4).
        color (str): color
    Return:
        img_corners (np.array): Corners of 3D bounding box in 2D image space
    """
    # Get corners of 3D bounding box
    corners = bbox_corners(obj)

    # Project into image coordinates
    img_corners = perspective(calib.cpu(), corners).numpy()

    return img_corners

