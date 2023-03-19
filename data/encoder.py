import math
import torch
import torch.nn.functional as F
from .. import utils

class ObjectEncoder(object):

    def __init__(self, classnames=['Car'], pos_std=[.5, .36, .5], 
                 log_dim_mean=[[0.42, 0.48, 1.35]], 
                 log_dim_std=[[.085, .067, .115]], sigma=1., nms_thresh=0.05):
        
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std)
        self.log_dim_mean = torch.tensor(log_dim_mean)
        self.log_dim_std = torch.tensor(log_dim_std)

        self.sigma = sigma
        self.nms_thresh = nms_thresh
        
    
    def encode_batch(self, objects, grids):
        '''
        Encode data in each batch base on objects and grid
        Args:
          objects: tuple of all object in image
          grids: ground grid in one batch with y = 1.74 torch.Size([1, 120, 120, 3])
        Return:
          heatmaps: torch.Size([1, 1, 119, 119])
          pos_offsets: torch.Size([1, 1, 3, 119, 119])
          dim_offsets: torch.Size([1, 1, 3, 119, 119])
          ang_offsets: torch.Size([1, 1, 2, 119, 119])
          mask: torch.Size([1, 1, 119, 119])        
        '''

        # Encode batch element by element
        batch_encoded = [self.encode(objs, grid) for objs, grid 
                         in zip(objects, grids)]
        
        # Transpose batch
        return [torch.stack(t) for t in zip(*batch_encoded)]


    def encode(self, objects, grid):
        '''
        Encode data base on objects and grid
        Args:
          objects: tuple of all object in image
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          heatmaps: torch.Size(1, 119, 119])
          pos_offsets: torch.Size([1, 3, 119, 119])
          dim_offsets: torch.Size([1, 3, 119, 119])
          ang_offsets: torch.Size([1, 2, 119, 119])
          mask: torch.Size([1, 1, 119, 119])        
        '''
        # Filter objects by class name
        objects = [obj for obj in objects if obj.classname in self.classnames]


        # Skip empty examples
        if len(objects) == 0:
            return self._encode_empty(grid)
        
        # Construct tensor representation of objects
        classids = torch.tensor([self.classnames.index(obj.classname) 
                                for obj in objects], device=grid.device)
        positions = grid.new([obj.position for obj in objects])
        dimensions = grid.new([obj.dimensions for obj in objects])
        angles = grid.new([obj.angle for obj in objects])

        # Assign objects to locations on the grid
        mask, indices = self._assign_to_grid(
            classids, positions, dimensions, angles, grid)

        # Encode object heatmaps
        heatmaps = self._encode_heatmaps(classids, positions, grid)
        
        # Encode positions, dimensions and angles
        pos_offsets = self._encode_positions(positions, indices, grid)
        dim_offsets = self._encode_dimensions(classids, dimensions, indices)
        ang_offsets = self._encode_angles(angles, indices)

        return heatmaps, pos_offsets, dim_offsets, ang_offsets, mask   
    

    def _assign_to_grid(self, classids, positions, dimensions, angles, grid):
        
        '''
        Return positive locations and the id of the corresponding instance (True/False) and (1/0)
        Args:
          classids : torch.Size([number of car]) with each elements = 0
          positions: 3D object location in camera coordinates [-pi..pi] torch.Size([number of car, 3])
          dimensions: 3D object dimensions: height, width, length (in meters) torch.Size([number of car, 3])
          angles: Observation angle of object raning [-pi..pi] with torch.Size([number of car])
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          labels: mask with element = True if grid cells which lie within each object torch.Size([1, 119, 119])
          indices mask with element = 1 if grid cells which lie within each object torch.Size([1, 119, 119])
        '''
        # Compute grid centers
        centers = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.

        # Transform grid into object coordinate systems
        local_grid = rotate(centers - positions.view(-1, 1, 1, 3), 
            -angles.view(-1, 1, 1)) / dimensions.view(-1, 1, 1, 3)
        
        # Find all grid cells which lie within each object
        inside = (local_grid[..., [0, 2]].abs() <= 0.5).all(dim=-1)
        
        # Expand the mask in the class dimension NxDxW * NxC => NxCxDxW
        class_mask = classids.view(-1, 1) == torch.arange(
            len(self.classnames)).type_as(classids)
        class_inside = inside.unsqueeze(1) & class_mask[:, :, None, None]

        # Return positive locations and the id of the corresponding instance 
        labels, indices = torch.max(class_inside, dim=0)
        return labels, indices
    

    def _encode_heatmaps(self, classids, positions, grid):
        '''
        Return the confidence map S(x, z)  which indicates the probability that there exists an object with a bounding box centred on location (x, y0, z).
        Args:
          classids : torch.Size([number of car]) with each elements = 0
          positions: 3D objects location in camera coordinates (in meters) with shape torch.Size([number of car, 3]) 
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          heatmaps with shape torch.Size([1, 119, 119])
        '''
        centers = (grid[1:, 1:, [0, 2]] + grid[:-1, :-1, [0, 2]]) / 2.
        positions = positions.view(-1, 1, 1, 3)[..., [0, 2]]

        # Compute per-object heatmaps
        sqr_dists = (positions - centers).pow(2).sum(dim=-1) 
        obj_heatmaps = torch.exp(-0.5 * sqr_dists / self.sigma ** 2)

        heatmaps = obj_heatmaps.new_zeros(self.nclass, *obj_heatmaps.size()[1:])
        for i in range(self.nclass):
            mask = classids == i
            if mask.any():
                heatmaps[i] = torch.max(obj_heatmaps[mask], dim=0)[0]
        return heatmaps

    def _encode_positions(self, positions, indices, grid):
        '''   
        Predicts the relative offset ∆pos from grid cell locations on the ground plane (x, y0, z) to the center of
            the corresponding ground truth object pi
        Args:
          positions: 3D objects location in camera coordinates (in meters) with shape torch.Size([number of car, 3]) 
          indices mask with element = 1 if grid cells which lie within each object torch.Size([1, 119, 119])
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          pos_offsets: torch.Size([1, 3, 119, 119])
        '''
        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.

        # Encode positions into the grid
        C, D, W = indices.size()
        positions = positions.index_select(0, indices.view(-1)).view(C, D, W, 3)

        # Compute relative offsets and normalize
        pos_offsets = (positions - centers) / self.pos_std.to(positions)
        return pos_offsets.permute(0, 3, 1, 2)
    
    def _encode_dimensions(self, classids, dimensions, indices):
        ''' 
        Predicts the logarithmic scale offset ∆dim between the assigned ground truth object i 
          with dimensions di and the mean dimensions over all objects of the given class
        Args:
          classids : torch.Size([number of car]) with each elements = 0 
          dimensions: 3D object dimensions: height, width, length (in meters) of object torch.Size([number of car, 3])
          indices mask with element = 1 if grid cells which lie within each object torch.Size([1, 119, 119])
        Return:
          pos_offsets: torch.Size([1, 3, 119, 119])
        '''
        # Convert mean and std to tensors
        log_dim_mean = self.log_dim_mean.to(dimensions)[classids]
        log_dim_std = self.log_dim_std.to(dimensions)[classids]

        # Compute normalized log scale offset
        dim_offsets = (torch.log(dimensions) - log_dim_mean) / log_dim_std

        # Encode dimensions to grid
        C, D, W = indices.size()
        dim_offsets = dim_offsets.index_select(0, indices.view(-1))
        return dim_offsets.view(C, D, W, 3).permute(0, 3, 1, 2)
    

    def _encode_angles(self, angles, indices):
        '''
        Predicts the sine and cosine of the objects orientation θi about the y-axis.
        Args:
          angles: Observation angle of object raning [-pi..pi] with torch.Size([number of car])
          indices mask with element = 1 if grid cells which lie within each object torch.Size([1, 119, 119])
        Return:
          objects orientation: torch.Size([1, 2, 119, 119])
        '''

        # Compute rotation vector
        sin = torch.sin(angles)[indices]
        cos = torch.cos(angles)[indices]
        return torch.stack([cos, sin], dim=1)
    

    def _encode_empty(self, grid):
        depth, width, _ = grid.size()
        '''        
        If there is no object in image, return empty tensors
        Args:
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          heatmaps: torch.Size([1, 119, 119])
          pos_offsets: torch.Size([1, 3, 119, 119])
          dim_offsets: torch.Size([1, 3, 119, 119])
          ang_offsets: torch.Size([1, 2, 119, 119])
          mask: torch.Size([1, 119, 119])
        '''
        # Generate empty tensors
        heatmaps = grid.new_zeros((self.nclass, depth-1, width-1))
        pos_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        dim_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        ang_offsets = grid.new_zeros((self.nclass, 2, depth-1, width-1))
        mask = grid.new_zeros((self.nclass, depth-1, width-1)).bool()
        return heatmaps, pos_offsets, dim_offsets, ang_offsets, mask
    

    def decode(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        '''
        decode data --> objects base of given data
        Args:
          heatmaps: torch.Size([1, 159, 159])
          pos_offsets: torch.Size([1, 3, 159, 159])
          dim_offsets: torch.Size([1, 3, 159, 159])
          ang_offsets: torch.Size([1, 2, 159, 159])
          grid: ground grid with y = 1.74 torch.Size([160, 160, 3]) 
        Return:
          objects: list of objects with each of object has ['classname', 'position', 'dimensions', 'angle', 'score']
        '''
        # Apply NMS to find positive heatmap locations
        peaks, scores, classids = self._decode_heatmaps(heatmaps)

        # Decode positions, dimensions and rotations
        positions = self._decode_positions(pos_offsets, peaks, grid)
        dimensions = self._decode_dimensions(dim_offsets, peaks)
        angles = self._decode_angles(ang_offsets, peaks)

        objects = list()
        for score, cid, pos, dim, ang in zip(scores, classids, positions, 
                                             dimensions, angles):
            objects.append(ObjectData(
                self.classnames[cid], pos, dim, ang, score))
        
        return objects
    

    def decode_batch(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, 
                     grids):
        
        boxes = list()
        for hmap, pos_off, dim_off, ang_off, grid in zip(heatmaps, pos_offsets, 
                                                         dim_offsets, 
                                                         ang_offsets, grids):
            boxes.append(self.decode(hmap, pos_off, dim_off, ang_off, grid))
        
        return boxes

    def _decode_heatmaps(self, heatmaps):
        '''
        Args:
          heatmaps: the confidence map S(x, z) torch.Size([1, 159, 159])
        Return:
          peaks: the boolean mask of peak locations. torch.Size([1, 159, 159])
          scores: Score of each object torch.Size([number of objects])
          classids: idx of each object --> class torch.Size([number of objects])     
        '''
        peaks = non_maximum_suppression(heatmaps, self.sigma)
        scores = heatmaps[peaks]
        classids = torch.nonzero(peaks)[:, 0]
        return peaks, scores.cpu(), classids.cpu()


    def _decode_positions(self, pos_offsets, peaks, grid):
        '''     
        Args: 
          pos_offsets: torch.Size([1, 3, 159, 159])
          peaks: the boolean mask of peak locations torch.Size([1, 159, 159])
          grid: ground grid with y = 1.74 torch.Size([120, 120, 3])
        Return:
          positions[peaks]: 3D object location in camera coordinates [-pi..pi] of each object torch.Size([number of objects, 3])
        '''
        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.

        # Un-normalize grid offsets
        positions = pos_offsets.permute(0, 2, 3, 1) * self.pos_std.to(grid) \
            + centers
        return positions[peaks].cpu()
    
    def _decode_dimensions(self, dim_offsets, peaks):
        '''  
        Args    
          dim_offsets: torch.Size([1, 159, 159, 3])
          peaks: the boolean mask of peak locations torch.Size([1, 159, 159])
        Return:
          dimensions[peaks]:3D object dimensions: height, width, length (in meters) torch.Size([number of objects, 3])
        '''
        dim_offsets = dim_offsets.permute(0, 2, 3, 1)
        dimensions = torch.exp(
            dim_offsets * self.log_dim_std.to(dim_offsets) \
                + self.log_dim_mean.to(dim_offsets))
        return dimensions[peaks].cpu()
    
    def _decode_angles(self, angle_offsets, peaks):
        cos, sin = torch.unbind(angle_offsets, 1)
        '''
        Args:
          angle_offsets: torch.Size([1, 2, 159, 159])
          peaks: the boolean mask of peak locations torch.Size([1, 159, 159])
        Return
          angles: Observation angle of object raning [-pi..pi] with torch.Size([number of car])
        '''
        return torch.atan2(sin, cos)[peaks].cpu()


def non_maximum_suppression(heatmaps, sigma=1.0, thresh=0.6, max_peaks= 10):
    '''
    The peaks tensor is a boolean mask that is True for pixels that are both a maximum and above the threshold.
    Args: 
      heatmaps: torch.Size([1, 119, 119])
      sigma: the standard deviation of the Gaussian kernel used for smoothing 
      thresh: the threshold for peak detection 
      max_peaks: the maximum number of peaks to keep.
    Return: 
      peaks: the boolean mask of peak locations. torch.Size([1, 119, 119])
    '''
    # Smooth with a Gaussian kernel
    num_class = heatmaps.size(0)
    kernel = gaussian_kernel(sigma).to(heatmaps)
    kernel = kernel.expand(num_class, num_class, -1, -1)
    smoothed = F.conv2d(
        heatmaps[None], kernel, padding=int((kernel.size(2)-1)/2))

    # Max pool over the heatmaps
    max_inds = F.max_pool2d(smoothed, 3, stride=1, padding=1, 
                               return_indices=True)[1].squeeze(0)

    # Find the pixels which correspond to the maximum indices
    _, height, width = heatmaps.size()
    flat_inds = torch.arange(height*width).type_as(max_inds).view(height, width)
    peaks = (flat_inds == max_inds) & (heatmaps > thresh)
    
    # Keep only the top N peaks
    if peaks.long().sum() > max_peaks:
        scores = heatmaps[peaks]
        scores, _ = torch.sort(scores, descending=True)
        peaks = peaks & (heatmaps > scores[max_peaks-1])
    return peaks