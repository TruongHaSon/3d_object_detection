import torch
from data.kitti import KittiObjectDataset
from data.augmentation import AugmentedObjectDataset
from visualization.visualization import draw_3d_boxes
import numpy as np
import cv2

if __name__ == '__main__':
    # Dataset unit-test
    kitti_root = "../KITTI"
    dataset = KittiObjectDataset(kitti_root)
    idx, image, calib, objects, grid = dataset[0]
    print("Done")
    # Try some augmentation
    train_image_size = (1080, 360)
    train_grid_size = (120, 120)
    aug_dataset = AugmentedObjectDataset(
        dataset,
        train_image_size,
        train_grid_size
    )
    idx, image, calib, objects, grid = aug_dataset[20]

    img = draw_3d_boxes(np.array(image), objects, calib)
    cv2.imwrite("3d_boxes.jpg", img)
