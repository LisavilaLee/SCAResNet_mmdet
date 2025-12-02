# SCAResNet_mmdet

Code for SCAResNet. MMDetection based. The paper, SCAResNet: A ResNet Variant Optimized for Tiny Object Detection in Transmission and Distribution Towers, published in [IEEE Geoscience and Remote Sensing Letters](https://ieeexplore.ieee.org/document/10251830).

The innovation of SCAResNet is reflected in the resnet.py file, which given now.

Since MMDetection updates so frequently, I suggest first learning how to run the current version of [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main), and then adapting the file [mmdet/models/backbones/resnet.py](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/resnet.py) accordingly.

The code implementing the innovations of the paper is in this project’s resnet.py, specifically in lines 21–256 and 896–899. The rest of the code remains unchanged from the original MMDetection. You can compare with the latest version of MMDetection to apply the modifications in resnet.py.
