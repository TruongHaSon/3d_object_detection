import cv2
from utils import corners3d
import numpy as np

def draw_projected_box3d(image, corners3d, color, thickness=1):
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

def draw_3d_boxes(img, objects, calib):
    """Draw 3D bounding box with each object in image

    Args:
        image (np.array): RGB image
        objects (list of nametupled): list of object in image
        calib (torch.tensor): intrinsic matrix with shape (3, 4)
    Returns:
        image (np.array): output image with 3D bounding box
    """
    
    occ_to_color = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 255, 255)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for object in objects:
        if object.classname in ['Pedestrian', 'Car', 'Cyclist']: 
            name = object.classname
            color = occ_to_color[int(object.occlusion)]
            corners_3d = corners3d(object, calib)
            # Draw 3d bounding box
            img = draw_projected_box3d(img, corners_3d, color)
            img = cv2.putText(img, name, (int(corners_3d[5, 0]), int(corners_3d[5, 1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            center = np.mean(corners_3d, axis = 0).astype(np.int32)
            img = cv2.circle(img, (center[0], center[1]), 5, color, -1)

    return img
    