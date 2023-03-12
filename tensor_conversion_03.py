"""
Project: Object detector and segmentation tutorial https://github.com/juancarlosmiranda/object_detector_tutorial
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2021
Description:

Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial. http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Explained by David Macedo.
https://github.com/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/finetuning_torchvision_models_tutorial.ipynb

Adapted by Juan Carlos Miranda as a programming practice, February 2021.

Use:
"""

import os
import time
import torch
import torchvision

# Managing images formats
import cv2
import torchvision.transforms.functional as F
from PIL import Image
# Deep learning models
# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#instance-seg-output
from torchvision.models.detection import maskrcnn_resnet50_fpn


def merge_masks(masks):
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


def tensor_conversion_03():
    print('------------------------------------')
    print('MAIN STORY OBJECT DETECTION EVALUATION')
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
    # Open image with Pillow.Image.open() and torchvision.io.read_image()
    # -------------------------------------------
    img_to_eval_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
    #img_to_eval_name = '20210523_red_cross.png'
    img_to_eval_name = 'PATT_01_MIRANDA.png'
    path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)
    cv_img_to_eval = cv2.imread(path_img_to_eval)  # ndarray:(H,W, 3)
    img_to_eval_float32 = F.to_tensor(cv_img_to_eval)
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]
    cv_new_img = img_to_eval_float32.mul(255).permute(1, 2, 0).detach().cpu().byte().numpy()
    #cv2.imshow('showing with cv2', cv_new_img)
    #cv2.waitKey()
    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    score_threshold = 0.7
    start_time_model_load = time.time()
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()
    # ----------------------------

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    with torch.no_grad():
        predictions_model = model(img_to_eval_list)  # todo:? why []?
    end_time_eval = time.time()

    # -------------------------------------
    # Managing prediction, making something here (filtering, extracting)
    # -------------------------------------
    pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
    pred_masks = predictions_model[0]['masks']

    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    masks_filtered = pred_masks[pred_scores >= score_threshold]
    final_masks = masks_filtered > 0.5  # to clean bad pixels
    final_masks = final_masks.squeeze(1)  # ?

    # -------------------------------------
    # It displays the results on the screen according to the colours.
    # -------------------------------------
    # Display with PIL.Image
    merged_masks = merge_masks(final_masks)
    merged_binary_img = Image.fromarray(merged_masks.mul(255).byte().cpu().numpy())
    merged_binary_img.show('binary mask to show')

    # Display with OpenCV
    cv_merged_binary_img = merged_masks.mul(255).byte().cpu().numpy()
    cv2.imshow('binary mask to show cv2', cv_merged_binary_img)
    cv2.waitKey()
    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    process_time_eval = total_time_model_load + total_time_eval
    #w, h = p_img_to_eval.size
    print('------------------------------------')
    print(f'Main parameters')
    print('------------------------------------')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    #print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')
    print(f'process_time_eval={process_time_eval}')


if __name__ == "__main__":
    tensor_conversion_03()
