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
from PIL import Image


def show_images(imgs_list):
    """
    From Tensor to PIL Image
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(imgs_list, list):
        imgs_list = [imgs_list]  # check if this is a list or not

    for i, img in enumerate(imgs_list):
        transform = T.ToPILImage()  # conversion in PIL data
        p_img_01 = transform(img.to(device))  # make transform of tensor in device
        p_img_01.show()



def main_masks_loop():
    print('------------------------------------')
    print('MAIN OBJECT DETECTION EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')
    dataset_folder = os.path.join('dataset', 'PennFudanPed')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'PNGImages'
    path_masks_folder = 'PedMasks'

    path_dataset_images = os.path.join(path_dataset, path_images_folder)
    path_dataset_masks = os.path.join(path_dataset, path_masks_folder)

    print('------------------------------------')
    print('------------------------------------')
    image_01_name = 'FudanPed00001.png'
    image_01_mask_name = 'FudanPed00001_mask.png'

    path_image_01 = os.path.join(path_dataset_images, image_01_name)
    path_image_01_mask = os.path.join(path_dataset_masks, image_01_mask_name)

    image_01 = read_image(path_image_01)
    image_01_mask = read_image(path_image_01_mask)

    images_list = [image_01, image_01_mask]  # create image lists

    show_images(images_list)

    #m1 = Image.fromarray(image_01_mask.mul(255).byte().cpu().numpy())
    #m1.show('binary mask to show')




if __name__=='__main__':
    print('Testing mask 01')
    main_masks_loop()



    pass