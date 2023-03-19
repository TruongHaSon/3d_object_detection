import cv2
from utils import corners3d
import numpy as np
import torch.nn.functional as F
from matplotlib.lines import Line2D

def draw_projected_box3d(image, corners3d, color, thickness=2):
    """Draw 3d bounding box in image
    
    Args:
        image (np.array): RGB image
        corners3d (np.array): (8,3) array of vertices (in image plane) for the 3d box in following order:
            4 -------- 6
           /|         /|
          5 -------- 7 .
          | |        | |
          . 0 -------- 2
          |/         |/
          1 -------- 3
        color (tupple): color of the bounding box
        thickness (int): thickness of the box
    Returns:
        image (np.array): output image
    """

    corners3d = corners3d.astype(np.int32)
    # create a rectangle with light blue color
    rect = np.copy(image)
    cv2.rectangle(rect, (corners3d[5, 0], corners3d[5, 1]), (corners3d[3, 0], corners3d[3, 1]), color, -1)

    # merge the rectangle with the original image
    image = cv2.addWeighted(image, 0.5, rect, 0.5, 0)

    for k in range(0, 4, 2):
        i, j = k, (k + 1) % 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)      
    for k in range(0, 2, 1):
        i, j = k, (k + 2) % 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 2) % 4 + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)        
    for k in range(0, 4):
        i, j = k, k + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
    return image

def choose_color(name):
    if name == 'Car':
        color = (0, 0, 255)  # Màu xanh dương
    elif name == 'Van':
        color = (255, 165, 0)  # Màu cam
    elif name == 'Truck':
        color = (255, 255, 0)  # Màu vàng
    elif name == 'Pedestrian':
        color = (128, 0, 128)  # Màu tím
    elif name == 'Cyclist':
        color = (0, 255, 0)  # Màu xanh lá
    else:
        color = (255, 0, 0)  # Màu đỏ
    return color

def draw_3d_boxes(img, objects, calib):
    """Draw 3D bounding box with each object in image
    Args:
        image (np.array): RGB image
        objects (list of nametupled): list of object in image
        calib (torch.tensor): intrinsic matrix with shape (3, 4)
    Returns:
        image (np.array): output image with 3D bounding box
    """
    
    for object in objects:
        if object.classname in ['Car', 'Van', 'Truck', 'Pedestrian','Cyclist']: 
            name = object.classname
            color = choose_color(name)
            corners_3d = corners3d(object, calib)

            # Draw 3d bounding box
            img = draw_projected_box3d(img, corners_3d, color)

            # Find location for label
            points = [(int(corners_3d[0, 0]), int(corners_3d[0, 1])), (int(corners_3d[1, 0]), int(corners_3d[1, 1])), (int(corners_3d[2, 0]), int(corners_3d[2, 1])), (int(corners_3d[3, 0]), int(corners_3d[3, 1]))]
            min_x = min(point[0] for point in points)
            max_y = max(point[1] for point in points)
          
            # Show label
            label_size, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (min_x, max_y+10), (min_x+label_size[0], max_y-label_size[1]+10), (0,0,0), cv2.FILLED)
            img = cv2.putText(img, name, (min_x, max_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            #center = np.mean(corners_3d, axis = 0).astype(np.int32)
            #img = cv2.circle(img, (center[0], center[1]), 5, color, -1)
    return img


def vis_score(image, calib, objects, grid, cmap='binary', ax=None):
    score = torch.randn(1, 1, 119, 119)  # adjust size to match X and Y dimensions of grid
    score = score[0, 0] * 0

    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        _, ax = plt.subplots()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Plot true objects
    for i, obj in enumerate(objects):
        if obj.classname == 'Car':
            color = 'tab:blue'  # Màu xanh dương
        elif obj.classname == 'Van':
            color = 'tab:orange'  # Màu cam
        elif obj.classname == 'Truck':
            color = 'tab:olive'  # Màu xanh lá cây nhạt
        elif obj.classname == 'Pedestrian':
            color = 'tab:purple'  # Màu tím
        elif obj.classname == 'Cyclist':
            color = 'tab:green'  # Màu xanh lá cây đậm
        else:
            color = 'tab:red'  # Màu đỏ

        # Get corners of 3D bounding box
        corners = bbox_corners(obj)
        corners = corners[:, [0, 2]]
        ax.add_line(Line2D(*corners[[0, 1]].T, c=color))
        ax.add_line(Line2D(*corners[[1, 3]].T, c=color))
        ax.add_line(Line2D(*corners[[3, 2]].T, c=color))
        ax.add_line(Line2D(*corners[[2, 0]].T, c=color))

    ax.set_aspect('equal')
    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    # Return the modified axis object
    return ax
    
