import torch

def make_grid(grid_size, grid_offset, grid_res):
    """
    Args:
        grid_size (tuple): (x, z) size of grid.
        grid_offset (int): grid offset (x, y, z).
        grid_res (float): grid resolution.
    Returns:
        grid: a grid where y is fixed, only has x_coords
            and z_coords. and the shape is (x/res, z/res, 3)
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0., width, grid_res) + xoff
    zcoords = torch.arange(0., depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)