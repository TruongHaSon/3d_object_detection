import os
import cv2
import numpy as np
from PIL import Image
from data.kitti import read_kitti_objects
import torch
from .. import utils
from .kitti import read_kitti_objects, read_kitti_calib



def draw_projected_box3d(image, corners3d, color, thickness=1):
    ''' Draw 3d bounding box in image
    input:
        image: RGB image
        corners3d: (8,3) array of vertices (in image plane) for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    corners3d = corners3d.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k, k + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)


def draw_3d_boxes(img, objects, calib):
    occ_to_color = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 255, 255)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for object in objects:
        if object.classname in ['Pedestrian', 'Car', 'Cyclist']: 
            
            name = object.classname
            color = occ_to_color[int(object.occlusion)]
            
            corners_3d = corners3d(object, calib)
            draw_projected_box3d(img, corners_3d, color)
            img = cv2.putText(img, name, (int(corners_3d[5, 0]), int(corners_3d[5, 1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            center = np.array([(corners_3d[0, 0] + corners_3d[1, 0] + corners_3d[2, 0] + corners_3d[3, 0] + corners_3d[4, 0] + corners_3d[5, 0] + corners_3d[6, 0] + corners_3d[7, 0]) / 8
            , (corners_3d[0, 1] + corners_3d[1, 1] + corners_3d[2, 1] + corners_3d[3, 1] + corners_3d[4, 1] + corners_3d[5, 1] + corners_3d[6, 1] + corners_3d[7, 1]) / 8], dtype=np.float32)
            center_int = center.astype(np.int32)
            img = cv2.circle(img, (center_int[0], center_int[1]), 5, color, -1)  # draw the center box
    cv2.imwrite("3d_boxes.jpg", img)


if __name__ == '__main__':
    
    idx = 10
    # idx = 34
    kitti_root = "/home/son/Documents/Project/KITTI/object/training"
    img_file = os.path.join(kitti_root, 'image_2/{:06d}.png'.format(idx))
    
    image = np.array(Image.open(img_file))
     
    # Load annotations
    label_file = os.path.join(kitti_root, 'label_2/{:06d}.txt'.format(idx))
    
    objects = read_kitti_objects(label_file)

    # Load calibration matrix
    calib = os.path.join(kitti_root, 'calib/{:06d}.txt'.format(idx))
    
    calib = read_kitti_calib(calib)

    # Draw 3d bounding boxes
    draw_3d_boxes(image, objects, calib)
    
    
