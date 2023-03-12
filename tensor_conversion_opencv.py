"""
Project: Object detector and segmentation tutorial https://github.com/juancarlosmiranda/object_detector_tutorial
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2021
Description:
Conversion to torch.Tensor from OpenCV image.

Use:
"""
import os
import torch
import cv2
import torchvision.transforms.functional as F


def tensor_conversion_opencv():
    print('------------------------------------')
    print('Tensor conversion to OpenCV image')
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
    # Open image with OpenCV cv2.imread
    # -------------------------------------------
    img_to_eval_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
    path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)

    # image reading
    cv_img_to_eval = cv2.imread(path_img_to_eval)  # ndarray:(H,W, 3)

    # conversion to tensor
    img_to_eval_float32 = F.to_tensor(cv_img_to_eval)
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]

    # convert again to OpenCV
    cv_new_img = img_to_eval_float32.mul(255).permute(1, 2, 0).detach().cpu().byte().numpy()
    cv2.imshow('showing with cv2', cv_new_img)
    cv2.waitKey()

    print('------------------------------------')
    print(f'Main parameters')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    print(f'img_to_eval_list={img_to_eval_list}')


if __name__ == "__main__":
    tensor_conversion_opencv()
