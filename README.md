# README - Object detection and segmentation using PennFudanPed/ dataset

This folder contains data and various code samples related to using object detectors and object segmentation. The
original code was adapted from [Pytorch - TorchVision Object Detection Finetuning Tutorial](http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and [David Macêdo](https://github.com/dlmacedo) Github. The intent of this code is to cover all stages in the object
detection and segmentation pipeline as a programming practice. It is true that not all aspects can be covered.
It uses pre-trained models from [Pytorch](https://pytorch.org/) and the Penn-Fudan Database from [here](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)

## Models used
* [Mask R-CNN](https://arxiv.org/abs/1703.06870)
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* [Models and pre-trained weights](https://pytorch.org/vision/stable/models.html#models-and-pre-trained-weights)


# Links to tutorials, useful information
* [David Macêdo](https://github.com/dlmacedo)
* [Deep Learning courses](https://dlmacedo.com/courses/deeplearning/)
* [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
* [Pytorch - TorchVision Object Detection Finetuning Tutorial](http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [TorchVision Instance Segmentation Finetuning Tutorial](https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/torchvision_finetuning_instance_segmentation.ipynb)

## Pytorch visualization utils
* [torchvision](https://github.com/pytorch/vision)
* [Pytorch visualization utils](https://pytorch.org/vision/stable/utils.html)
* [Example gallery](https://pytorch.org/vision/stable/auto_examples/index.html)
* [An Introduction to PyTorch Visualization Utilities](https://debuggercafe.com/an-introduction-to-pytorch-visualization-utilities/)
* [Visualization utilities](https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html)
* [Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html)
* [torchvision - read_image()](https://pytorch.org/vision/main/generated/torchvision.io.read_image.html)
* [REPURPOSING MASKS INTO BOUNDING BOXES](https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html#sphx-glr-auto-examples-plot-repurposing-annotations-py)

## Pytorch tensors
* With video [Introduction to PyTorch Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
* [TORCH.TENSOR](https://pytorch.org/docs/stable/tensors.html)  
* [PyTorch PIL to Tensor and vice versa](https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312)
* [Pytorch Converting tensors to images](https://discuss.pytorch.org/t/converting-tensors-to-images/99482)
* Good tutorial about Numpy. [Introduction to NumPy and OpenCV](http://vision.deis.unibo.it/~smatt/DIDATTICA/Sistemi_Digitali_M/PDF/Introduction_to_NumPy_and_OpenCV.pdf)
* [Data transfer to and from PyTorch](https://www.simonwenkel.com/notes/software_libraries/opencv/opencv-cuda-integration.html#accessing-gpumat-as-pytorch-tensor)


### Conversions
* PIL.Image to Tensor. [Converting an image to a Torch Tensor in Python](https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/)
* Numpy to PIL. [Convert a NumPy array to an image](https://www.geeksforgeeks.org/convert-a-numpy-array-to-an-image/)
* [Plot `torch.Tensor` using OpenCV](https://discuss.pytorch.org/t/plot-torch-tensor-using-opencv/20059)
* [How do I display a single image in PyTorch?](https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch)


## Installing tools
* [How to install the NVIDIA drivers on Ubuntu 20.04 Focal Fossa Linux](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
* [CUDA Toolkit 12.1 Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)


# Project files

## Training code

| Folders                    | Description            |
|---------------------------|-------------------------|
| main_free_gpu_cache.py | Tool for clean GPU memory |
| main_training_code.py | Code to train people detector using PennFudanPed/ dataset. This script produces a file with weights in format .pth |
| tv-training-code_corrected.py | Original code to train people detector using PennFudanPed/ dataset. This script produces a file with weights in format .pth |

## Torchvision examples
Using Pytorch library to show images and masks.

| Folders                    | Description            |
|---------------------------|-------------------------|
| torchvision_01.py | From PennFudanPed it uses torchvision library to read a .PNG image, makes transformations using GPU/CPU and show it on the screen. |
| torchvision_02.py | Takes instance segmentation mask images, transforms from Tensor to Pillow image, after it merges the masks in one image. |

## Tensors examples
![transform_examples](https://github.com/juancarlosmiranda/object_detector_tutorial/blob/main/docs/img/MIND_MAP_COMPONENTS_PYTORCH_TENSORS.png?raw=true)

Basic examples using image transforms offered by torchvision.transforms.functional.
Two ways to call the same function.
```
import torchvision.transforms.functional as F
p_img_01 = F.to_pil_image(tensor_img)
p_img_01.show()
```
```
import torchvision.transforms as T
transform = T.ToPILImage()
transforms.append(T.ToTensor())
p_img_01 = transform(tensor_img.to(device))
```

| Folders                    | Description            |
|---------------------------|-------------------------|
| tensor_conversion_pytorch.py | Read images using read_image() conversion, basic pipeline. |
| tensor_conversion_pil.py | Read images using PIL.Image.open() conversion, basic pipeline. |
| tensor_conversion_opencv.py | Read images using OpenCV cv2.imread() conversion, basic pipeline. |

Connecting tensor conversion with deep learning models. Examples using MASK R-CNN (from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn(pretrained=True)).
The result is a binary mask converted.

| Folders                    | Description            |
|---------------------------|-------------------------|
| tensor_conversion_01.py | Read images using read_image() conversion. |
| tensor_conversion_02.py | Read images using PIL.Image.open() conversion. |
| tensor_conversion_03.py | Read images using cv2.imread() conversion. |



# Model pipelines for bounding box (BBOX) and mask segmentation (MASK)

Testing bounding box models(BBOX) and mask segmentation models (MASK) sequence in PennFudanPed/ 

| Folders                    | Description            |
|---------------------------|-------------------------|
| main_pennfudanpen_bbox_01.py | Detecting people using PennFudanPed/ dataset with from torchvision.models.detection.fasterrcnn_resnet50_fpn pretrained model |
| main_pennfudanpen_mask_01.py | Detecting apples using PennFudanPed/ dataset with from from torchvision.models.detection import maskrcnn_resnet50_fpn pretrained model |

Testing bounding box models(BBOX) and mask segmentation models (MASK) sequence in a normal image.

| Folders                    | Description            |
|---------------------------|-------------------------|
| main_story_rgb_bbox_01.py | Detecting people using story_rgb/ dataset with from torchvision.models.detection.fasterrcnn_resnet50_fpn pretrained model |
| main_story_rgb_mask_01.py | Detecting apples using story_rgb/ dataset with from from torchvision.models.detection import maskrcnn_resnet50_fpn pretrained model |
| main_story_rgb_mask_02.py | Detecting apples using story_rgb/ dataset with from from torchvision.models.detection import maskrcnn_resnet50_fpn pretrained model  saving data in an output/ folder|


Checking the trained weight in a .pth file with a MASK R-CNN model.

| Folders                    | Description            |
|---------------------------|-------------------------|
| main_evaluate_pennfudanpen_code.py | Detecting people using random images from PennFudanPed/ dataset, with torchvision.models.detection import maskrcnn_resnet50_fpn pretrained model and load trained weights from a file .pth |
| main_evaluate_people_code.py | Detecting people using test images torchvision.models.detection import maskrcnn_resnet50_fpn pretrained model and load trained weights from a file .pth |

## Webcam examples RGB camera

| Folders                    | Description            |
|---------------------------|-------------------------|
| webcam_basic_loop_01.py | Basic loop to extract frames from webcam. |
| webcam_obj_detect_01.py | . |

# Requirements

## Hardware and software stack used

* Ubuntu 20.04.3 LTS 64 bits.
* Intel® Core™ i7-8750H CPU @ 2.20GHz × 12.
* GeForce GTX 1050 Ti Mobile.
* Windows 10
* Python 3.8.10

## Edition tools


## Python stack environment

### Create de environment

```
python3 -m pip install python-venv
pip3 install python-venv
python -m venv ./object_detector_tutorial_venv
source ./venv/bin/activate
python --version
pip install --upgrade pip
```

### Installing libraries

```
pip install requirements_windows.txt
```

## Installing in Windows 10

```
pip install opencv-python
```

## Installing Ubuntu 20.04 LTS

Install Python tools

```
sudo apt install python3-pip
sudo apt install python3.8-venv
```

# Installing CUDA toolkit Linux notes

## Deleting any nvidia data

```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
sudo rm -rf /usr/local/cuda*
sudo apt-get purge nvidia*
sudo apt-get update
sudo apt-get autoremove
sudo apt-get autoclean
```

## Install nvidia-cuda-toolkit

Download the current toolkit available from
NVIDIA [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

## Installing driver

```
sudo apt-get update
sudo ubuntu-drivers autoinstall
nvidia-driver-470
```

## Checking CUDA version installed

```
nvcc --version
nvidia-smi
```
