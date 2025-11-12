# Monocular Visual Odometry with OpenCV

This repo is a demo on using OpenCV to perform monocular visual odometry from a series of video images from a moving vehicle. It takes a series of images from a camera and outputs a video of the camera with visual tracking features overlayed, as well as a bird's eye view map of the path taken by the vehicle.



# Requirements

1. Install OpenCV for python
```pip install opencv-python```

2. Download the KITTI odometry dataset from this website: https://www.cvlibs.net/datasets/kitti/eval_odometry.php. Download the grayscale dataset (22 GB) and the ground truth poses (4MB). The KITTI dataset contains several "sequences", which are different routes that a car has driven, each with their own set of images from two separate cameras on the car. Each sequence has an ```image_0``` and ```image_1``` folder containing images from each of the two cameras, but this sript uses only the ```image_0``` image set. The scripts expects the ```image_0``` folder to be inside a folder called ```data```. The ground truth data for all sequences is combined into one .txt file, and this script expects that file to be called ```ground_truth.txt```.

# Contents

```opencv.py```: A script that you can run on the KITTI odometry dataset that outputs a live image of the path taken along with a video of the camera feed with visual tracking features overlayed.

