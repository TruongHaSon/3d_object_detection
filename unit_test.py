import torch
from data.kitti import KittiObjectDataset

if __name__ == '__main__':
    kitti_root = "../KITTI"
    dataset = KittiObjectDataset(kitti_root)
    idx, image, calib, objects, grid = dataset[0]
    print("Done")