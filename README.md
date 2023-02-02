![ảnh](https://user-images.githubusercontent.com/85555398/215302814-99827e41-951e-44c7-9c63-649ead810f38.png)
This is a PyTorch implementation of the OFTNet network from the paper Orthographic Feature Transform for Monocular 3D Object Detection. The code currently supports training the network from scratch on the KITTI dataset - intermediate results can be visualised using Tensorboard. The current version of the code is intended primarily as a reference, and for now does not support decoding the network outputs into bounding boxes via non-maximum suppression. This will be added in a future update. Note also that there are some slight implementation differences from the original code used in the paper.


![image](https://user-images.githubusercontent.com/85555398/216355298-89990653-f61f-45e2-bb8f-60c0c3fbb377.png)
Data augmentation Since our method relies on a fixed mapping from the image plane to the ground plane, we
found that extensive data augmentation was essential for the network to learn robustly. We adopt three types of widelyused augmentations: random cropping, scaling and horizontal flipping, adjusting the camera calibration parameters f and (cu, cv) accordingly to reflect these perturbations
