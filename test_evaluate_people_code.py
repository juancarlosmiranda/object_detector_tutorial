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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import references.detection.transforms as T
import torchvision.transforms.functional as transform

from PIL import Image


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    print('get_model_instance_segmentation(num_classes):')
    print('in_features -->', in_features)
    print('in_features_mask -->', in_features_mask)
    print('hidden_layer -->', hidden_layer)
    print('num_classes -->', num_classes)

    return model


def get_transform(train):
    """

    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def merge_masks(masks):
    """
    Return a Tensor with merged masks
    """
    merged_mask = masks[0]  # assign the first mask
    for mask in masks:
        merged_mask = mask + merged_mask

    return merged_mask


def main_evaluate_people_loop():
    print('------------------------------------')
    print('MAIN STORY OBJECT DETECTION EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # -------------------------------------------
    # Datasets
    # -------------------------------------------
    dataset_folder = os.path.join('dataset', 'people')
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'images'
    path_dataset_images = os.path.join(path_dataset, path_images_folder)

    # -------------------------------------------
    # Open image with Pillow.Image.open() and torchvision.io.read_image()
    # -------------------------------------------
    image_to_eval_name = 'PATT_01_MIRANDA.png'
    image_to_eval_name = 'kayaking_20211120.png'
    image_to_eval_name = '20210523_red_cross.png'
    path_image_to_eval = os.path.join(path_dataset_images, image_to_eval_name)
    p_img_to_evaluate = Image.open(path_image_to_eval)  # {PngImageFile}
    t_img_to_eval = transform.to_tensor(p_img_to_evaluate)

    # -------------------------------------------
    # Trained parameters for models
    # -------------------------------------------
    trained_model_folder = 'trained_model'
    trained_model_path = os.path.join(main_path_project, trained_model_folder)
    file_name_model = 'MODEL_SAVED.pth'
    file_model_path = os.path.join(trained_model_path, file_name_model)
    # -------------------------------------------

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    score_threshold = 0.5
    num_classes = 2
    start_time_model_load = time.time()
    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(file_model_path))  # equivalent to maskrcnn_resnet50_fpn(), but with file loading
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()
    # ----------------------------

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    # Data type int_input {Tensor:3}, tensor_input {Tensor:1}
    # t_img_to_evaluate transformed to Tensor
    with torch.no_grad():
        predictions_model = model([t_img_to_eval.to(device_selected)])  # todo:? why []?
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
    merged_masks = merge_masks(final_masks)
    merged_binary_img = Image.fromarray(merged_masks.mul(255).byte().cpu().numpy())
    merged_binary_img.show('binary mask to show')

    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    w, h = p_img_to_evaluate.size
    print('------------------------------------')
    print(f'Main parameters')
    print('------------------------------------')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_image_to_eval}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'Trained model file_model_path={file_model_path}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')


if __name__ == "__main__":
    main_evaluate_people_loop()
