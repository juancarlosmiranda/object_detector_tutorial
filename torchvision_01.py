"""
Project:
Author:
Date:
Description:
...

Use:
"""
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import read_image


def torchvision_01():
    """
    From PennFundanPed it uses torchvision library to read a .PNG image, makes transformations using GPU/CPU and show it
    on the screen.
    """
    print('------------------------------------')
    print('Reading images from PennFundanPed')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # define path to datasets
    dataset_folder = os.path.join('dataset', 'PennFudanPed')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'PNGImages'
    path_masks_folder = 'PedMasks'
    path_dataset_images = os.path.join(path_dataset, path_images_folder)
    path_dataset_masks = os.path.join(path_dataset, path_masks_folder)

    # image names here
    image_01_name = 'FudanPed00001.png'
    image_01_mask_name = 'FudanPed00001_mask.png'

    # device settings and transformations
    device_selected = 'cuda' if torch.cuda.is_available() else 'cpu'

    # reading images using torchvision
    path_image_01 = os.path.join(path_dataset_images, image_01_name)
    path_image_01_mask = os.path.join(path_dataset_masks, image_01_mask_name)
    image_01 = read_image(path_image_01)
    image_01_mask = read_image(path_image_01_mask)

    # transformation here
    p_img_01 = F.to_pil_image(image_01_mask.to(device_selected))  # image_01 is a {Tensor:3}, p_img_01 is a {Image}
    p_img_01.show()


    p_img_01_mask = F.to_pil_image(image_01.to(device_selected))  # image_01 is a {Tensor:3}, p_img_01 is a {Image}
    p_img_01_mask.show()


if __name__ == '__main__':
    torchvision_01()
