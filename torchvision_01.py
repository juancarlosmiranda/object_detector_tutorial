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
from torchvision.io import read_image


def show_images(imgs_list):
    """
    From Tensor to PIL Image, it receives a list of images and show them on the screen
    list of tensors {Tensor: 3}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(imgs_list, list):
        imgs_list = [imgs_list]  # check if this is a list or not

    for i, img in enumerate(imgs_list):
        transform = T.ToPILImage()  # conversion in PIL data
        p_img_01 = transform(img.to(device))  # make transform of tensor in device
        p_img_01.show()


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
    transform = T.ToPILImage()  # conversion in PIL data

    # reading images using torchvision
    path_image_01 = os.path.join(path_dataset_images, image_01_name)
    path_image_01_mask = os.path.join(path_dataset_masks, image_01_mask_name)
    image_01 = read_image(path_image_01)
    image_01_mask = read_image(path_image_01_mask)

    # transformation here
    p_img_01 = transform(image_01.to(device_selected))  # image_01 is a {Tensor:3}, p_img_01 is a {Image}
    p_img_01.show()

    p_img_01_mask = transform(image_01_mask.to(device_selected))  # make transform of tensor in device
    p_img_01_mask.show()


if __name__ == '__main__':
    print('torchvision_01() -> reading images from disk')
    torchvision_01()
