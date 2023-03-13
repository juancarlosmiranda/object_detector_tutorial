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
import numpy as np

# Managing images formats
import cv2
from PIL import Image
import torchvision.transforms.functional as F

# Deep learning models
# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#instance-seg-output
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Drawing on the screen
from torchvision.utils import draw_bounding_boxes
from helpers.helper_examples import COCO_INSTANCE_CATEGORY_NAMES
from helpers.helper_examples import merge_masks


def tensor_conversion_04():
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
    #img_to_eval_name = 'PATT_01_MIRANDA.png'
    path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)

    # -------------------------------------------
    # Opening from file image
    # -------------------------------------------
    cv_img_to_eval = cv2.imread(path_img_to_eval)  # ndarray:(H,W, 3)
    img_to_eval_float32 = F.to_tensor(cv_img_to_eval)
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]

    # -------------------------------------------
    # Simulating an input as a frame
    # -------------------------------------------
    cv_img = cv2.cvtColor(cv_img_to_eval, cv2.COLOR_BGR2RGB)
    image_transposed = np.transpose(cv_img, [2, 0, 1])
    img_to_eval_uint8 = torch.tensor(image_transposed)  # used with torchvision.draw_bounding_boxes()
    img_to_eval_float32 = F.to_tensor(cv_img)
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    score_threshold = 0.6
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
    pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy()
    # pred_boxes_2 = predictions_model[0]['boxes'].detach().cpu().numpy().astype(int)
    pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
    pred_masks = predictions_model[0]['masks']
    pred_labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions_model[0]['labels'].cpu().numpy()]
    # ---------------------------------
    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    boxes_filtered = pred_boxes[pred_scores >= score_threshold].astype(np.int32)
    labels_filtered = pred_labels[:len(boxes_filtered)]
    # -------------------------------------

    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    masks_filtered = pred_masks[pred_scores >= score_threshold]
    final_masks = masks_filtered > 0.5  # to clean bad pixels
    final_masks = final_masks.squeeze(1)  # ?

    # -------------------------------------
    # Drawing bounding boxes with Pytorch
    # -------------------------------------
    colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))
    colours_to_draw = [tuple(color) for color in colours]
    result_with_boxes = draw_bounding_boxes(
        image=img_to_eval_uint8,
        boxes=torch.tensor(boxes_filtered), width=1,
        colors=colours_to_draw,
        labels=labels_filtered,
        fill=True  # this complete fill in bounding box
    )
    # ------------------------------------
    # Conversion from Tensor a PIL.Image
    # ------------------------------------
    p_result_with_boxes = F.to_pil_image(result_with_boxes)
    image_drawed_numpy = np.array(p_result_with_boxes)
    image_drawed = cv2.cvtColor(image_drawed_numpy, cv2.COLOR_RGB2BGR)
    cv2.imshow('showing with cv2', image_drawed)
    cv2.waitKey()

    # -------------------------------------
    # Display mask with PIL
    # -------------------------------------
    # Display with PIL.Image
    merged_masks = merge_masks(final_masks)
    merged_binary_img = Image.fromarray(merged_masks.mul(255).byte().cpu().numpy())
    merged_binary_img.show('binary mask to show with PIL')

    # -------------------------------------
    # Display mask with OpenCV
    # -------------------------------------
    cv_merged_binary_img = merged_masks.mul(255).byte().cpu().numpy()
    cv2.imshow('binary mask to show cv2', cv_merged_binary_img)
    cv2.waitKey()

    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    process_time_eval = total_time_model_load + total_time_eval
    w, h, channel = cv_img_to_eval.shape

    print('------------------------------------')
    print(f'Main parameters')
    print('------------------------------------')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')
    print(f'process_time_eval={process_time_eval}')


if __name__ == "__main__":
    tensor_conversion_04()
