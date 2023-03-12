"""
Project: Object detector and segmentation tutorial https://github.com/juancarlosmiranda/object_detector_tutorial
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2021
Description:
Conversion to torch.Tensor from image.

Use:
"""
import os
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as F

def tensor_conversion_pytorch():
    print('------------------------------------')
    print('Tensor conversion to torch.Tensor')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')
    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------
    # Datasets
    # -------------------------------------------
    dataset_folder = os.path.join('dataset', 'testing_performance')
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'images'
    path_dataset_images = os.path.join(path_dataset, path_images_folder)

    # -------------------------------------------
    # Open image with torchvision.io.read_image()
    # -------------------------------------------
    img_to_eval_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
    path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)

    # image reading
    img_to_eval_int = read_image(path_img_to_eval)  # {Tensor:3} Loads image in tensor format, get Tensor data

    # conversion to tensor
    img_to_eval_float32 = F.convert_image_dtype(img_to_eval_int, torch.float32)  # {Tensor with values between 0..1}
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]  # to model directly

    print('------------------------------------')
    print(f'Main parameters')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    print(f'img_to_eval_list={img_to_eval_list}')



if __name__ == '__main__':
    tensor_conversion_pytorch()
    pass
