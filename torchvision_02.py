"""
Project:
Author:
Date:
Description:
From PennFundanPed, it uses the torchvision library to read a .PNG image, opens the images from the dataset and displays
 the original image followed by a slicer mask, this merges everything into one.

Source:

Use:
"""
import os
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

def merge_masks(masks):
    """
    Return a Tensor with merged masks
    masks is {Tensor: 2}
    """
    merged_mask = masks[0]  # assign the first mask
    for mask in masks:
        merged_mask = mask + merged_mask

    return merged_mask


def torchvision_02():
    """
    Takes instance segmentation mask images, transforms from Tensor to Pillow image, after it merges the masks in ç
    one image.
    """
    print('------------------------------------')
    print('Reading images from PennFundanPed and showing binary masks')
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
    image_01_name = 'PennPed00096.png'
    image_01_mask_name = 'PennPed00096_mask.png'

    # device settings and transformations
    device_selected = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = T.ToPILImage()  # conversion in PIL data

    # reading images using torchvision
    path_image_01 = os.path.join(path_dataset_images, image_01_name)
    path_mask_01 = os.path.join(path_dataset_masks, image_01_mask_name)

    image_01 = read_image(path_image_01)  # Get Tensor data
    image_mask_01 = read_image(path_mask_01)  # torchvision.io.read_image() get Tensor data

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(image_mask_01)  # special function to see how many mask has

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = image_mask_01 == obj_ids[:, None, None]

    [number_of_masks, rows, cols] = masks.size()  # it get the number of masks instances
    print('------------------------------------')
    print('Loading....')
    print(f'path_image_01={path_image_01}')
    print(f'image_mask_01={path_mask_01}')
    print(f'masks.size()={masks.size()}')
    print(f'number_of_masks={number_of_masks}')
    print('------------------------------------')

    #----------------------------------
    # merged binary masks example from predictions
    merged_masks = merge_masks(masks)
    merged_binary_img = Image.fromarray(merged_masks.mul(255).byte().cpu().numpy())
    merged_binary_img.show('binary mask to show')
    #----------------------------------


if __name__ == '__main__':
    print('torchvision_02() -> reading instance mask images from disk and merge into binary mask')
    torchvision_02()
