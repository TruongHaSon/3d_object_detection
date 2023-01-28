import os
import cv2
import numpy as np
from PIL import Image
from data.kitti import read_kitti_objects
import torch
from .. import utils


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

    # Draw 2d and 3d bounding boxes
    draw_2d_boxes(image, objects)
    draw_3d_boxes(image, objects, calib)
