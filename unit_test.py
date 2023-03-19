import torch
from data.kitti import KittiObjectDataset
from data.augmentation import AugmentedObjectDataset
from visualization.visualization import draw_3d_boxes
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Dataset unit-test
    kitti_root = "/content/drive/MyDrive/3d_object_detection/KITTI"
    dataset = KittiObjectDataset(kitti_root)
    idx, image, calib, objects, grid = dataset[0]
    # Try some augmentation
    train_image_size = (1080, 360)
    train_grid_size = (120, 120)
    aug_dataset = AugmentedObjectDataset(
        dataset,
        train_image_size,
        train_grid_size
    )
    idx, image, calib, objects, grid = aug_dataset[1743]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    #Visualize bounding box
    img = draw_3d_boxes(np.array(image), objects, calib)
    cv2.imwrite("3d_boxes.jpg", img)
    ax1.imshow(img)
    ax1.set_title('3D bounding box')
    #Visualize score
    vis_score(image, calib, objects, grid, ax=ax2)
    ax2.set_title('Score')
    plt.show()
