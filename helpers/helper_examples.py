"""
Project:
Author:
Date:
Description:
...

Use:
"""
import torch
import numpy as np

# Managing images formats
from torchvision.io import read_image
from PIL import Image
import references.detection.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms as transforms


# GLOBAL
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def show_image_list(imgs_list):
    """
    Show images from tensor data
    """
    if not isinstance(imgs_list, list):
        imgs = [imgs_list]
    for i, img in enumerate(imgs):
        img = img.detach()  # TODO: ???
        p_img_01 = F.to_pil_image(img)  # TODO: ???
        p_img_01.show()


def show_one_image(t_image):
    """
    Show images from tensor data
    """
    p_img_01 = F.to_pil_image(t_image)  # TODO: ???
    p_img_01.show()


def merge_masks(masks):
    """
    Return a Tensor with merged masks
    """
    merged_mask = masks[0]  # assign the first mask
    for mask in masks:
        merged_mask = mask + merged_mask

    return merged_mask


def merge_masks_02(masks):
    """
    Return a Tensor with merged masks
    """
    print('masks.size()->', masks.size())
    print('masks.size(dim=0)->', masks.size(dim=0))
    print('masks.size(dim=1)->', masks.size(dim=1))
    if masks.size(dim=0) == 0:
        merged_mask = [masks]
        pass
    else:
        merged_mask = masks[0]  # assign the first mask

    for mask in masks:
        merged_mask = mask + merged_mask

    return merged_mask


def get_transform(train):
    """

    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def read_transform_return(image: Image):
    """
    Receives as input an Pillow.Image and returns
    Mixing numoy arrays and torch
    """
    transform = transforms.Compose([transforms.ToTensor(), ])
    image = np.array(image)
    image_transposed = np.transpose(image, [2, 0, 1])
    # Convert to uint8 tensor.
    int_input = torch.tensor(image_transposed)
    # Convert to float32 tensor.
    tensor_input = transform(image)
    tensor_input = torch.unsqueeze(tensor_input, 0)  # ??
    # F.to_tensor(np.transpose(image, [2, 0, 1]))
    return int_input, tensor_input