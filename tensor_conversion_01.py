"""
Project:
Author:
Date:
Description:

This is an example based on bounding box detection, in which the model predicts something and an image is showed.

* https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
* https://debuggercafe.com/an-introduction-to-pytorch-visualization-utilities/

Output
---------
Bounding boxes rectangles
Mask instance segmentation.
Merged binary mask

Use:
"""
import os
import time
import torch

# Managing images formats
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms.functional as F

from helpers.helper_examples import merge_masks

# Deep learning models
# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#instance-seg-output
from torchvision.models.detection import maskrcnn_resnet50_fpn


def tensor_conversion_01():
    print('------------------------------------')
    print('Using read_image() MASK R-CNN tensor conversion mask segmentation.')
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
    img_to_eval_int = read_image(path_img_to_eval)  # {Tensor:3} Loads image in tensor format, get Tensor data
    img_to_eval_float32 = F.convert_image_dtype(img_to_eval_int, torch.float32)  # {Tensor with values between 0..1}
    img_to_eval_list = [img_to_eval_float32.to(device_selected)]

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    # loading the trained model only once to reduce time
    score_threshold = 0.6
    start_time_model_load = time.time()  # time in seconds since the epoch as a floating number
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    with torch.no_grad():
        predictions_model = model(img_to_eval_list)
    end_time_eval = time.time()

    # -------------------------------------
    # Managing prediction, making something here (filtering, extracting)
    # -------------------------------------
    pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy()
    pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
    pred_masks = predictions_model[0]['masks']

    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    masks_filtered = pred_masks[pred_scores >= score_threshold]
    final_masks = masks_filtered > 0.5  # ?
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
    process_time_eval = total_time_model_load + total_time_eval
    print('------------------------------------')
    print(f'Main parameters')
    print('------------------------------------')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')
    print(f'process_time_eval={process_time_eval}')


if __name__ == '__main__':
    tensor_conversion_01()
    pass
