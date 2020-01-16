# FRC 2019 Hatch vision based on PyTorch implementation of a YOLO v3 Object Detector

This is a computer vision model based on yolov3 in python. The model was trained by manually drawing bounding boxes around the hatch for 1000+ pictures and left to train overnight using Supervisely. After there was detection, I added code to distinguish the hatches relative distance by calculating for the area of the bounding boxes and determining that the biggest bounding box will be the around the closest object. This information will then be used by the robot to drive to it and pick it up.

## Features Added
1. Added the ability for the detector to select closet object if it detects object more then once. This will be useful for robotics locking onto the closest object.
2. Trained model using [Supervisely](https://supervise.ly/)

## Weights
[Link to weights for Hatch](https://drive.google.com/file/d/1jDIXOPzHHXc8evMTW-0w_kXeUvOXYbqh/view?usp=sharing)
