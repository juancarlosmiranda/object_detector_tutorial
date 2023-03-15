"""
Project:
Author:
Date:
Description:

This is an example based on bounding box detection, in which the model predicts something and an image is showed.

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
import numpy as np

# Managing images formats
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms.functional as F

# Deep learning models
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Drawing on the screen
from torchvision.utils import draw_segmentation_masks

from helpers.helper_examples import COCO_INSTANCE_CATEGORY_NAMES
from helpers.helper_examples import show_one_image
from helpers.helper_examples import merge_masks
from helpers.helper_examples import read_transform_return


def main_masks_story_rgb_02():
    print('------------------------------------')
    print('MAIN MASK EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # -------------------------------------------
    # Datasets
    # -------------------------------------------
    dataset_folder = os.path.join('dataset', 'story_rgb')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'images'
    path_dataset_images = os.path.join(path_dataset, path_images_folder)

    # -------------------------------------------
    # Output results
    # -------------------------------------------
    output_folder = 'output'
    path_output = os.path.join(main_path_project, output_folder)
    img_result_rgb = 'result_rgb_.png'
    img_result_mask = 'result_mask_.png'

    path_image_01_result_rgb = os.path.join(path_output, img_result_rgb)
    path_image_01_result_mask = os.path.join(path_output, img_result_mask)

    # -------------------------------------------
    # Open image with Pillow.Image.open() and torchvision.io.read_image()
    # -------------------------------------------
    img_to_eval_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
    path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)
    p_img_to_eval = Image.open(path_img_to_eval)  # {PngImageFile}
    # used to draw masks
    t_img_to_eval = read_image(path_img_to_eval)  # Get Tensor data

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    # loading the trained model only once to reduce time
    score_threshold = 0.8
    start_time_model_load = time.time()
    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device_selected = torch.device('cpu')
    model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    # Data type int_input {Tensor:3}, tensor_input {Tensor:1}
    int_input, tensor_input = read_transform_return(p_img_to_eval)
    with torch.no_grad():
        predictions_model = model(tensor_input.to(device_selected))
    end_time_eval = time.time()

    # -------------------------------------
    # Managing prediction, making something here (filtering, extracting)
    # -------------------------------------
    pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy()
    pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
    pred_labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions_model[0]['labels'].cpu().numpy()]
    pred_masks = predictions_model[0]['masks']

    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    boxes_filtered = pred_boxes[pred_scores >= score_threshold].astype(np.int32)
    labels_filtered = pred_labels[:len(boxes_filtered)]
    masks_filtered = pred_masks[pred_scores >= score_threshold]

    final_masks = masks_filtered > 0.5  # ?
    final_masks = final_masks.squeeze(1)  # ?

    # -------------------------------------
    # It displays the results on the screen according to the colours.
    # -------------------------------------
    colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))  # random colours
    colours_to_draw = [tuple(color) for color in colours]
    # save masks detected
    mask_seg_result = draw_segmentation_masks(
        image=t_img_to_eval,
        masks=final_masks,
        colors=colours_to_draw,
        alpha=0.8
    )
    # Manage with Pillow
    p_mask_img = F.to_pil_image(mask_seg_result)
    p_mask_img.save(path_image_01_result_rgb)

    # TODO: add as result rgb export from Pillow, and conversion from tensor
    # save binary image detected by model
    merged_masks = merge_masks(final_masks)
    merged_binary_img = Image.fromarray(merged_masks.mul(255).byte().cpu().numpy())
    # merged_binary_img.show('binary mask to show')

    # -------
    # mask_seg_result.save(path_image_01_result_rgb)
    merged_binary_img.save(path_image_01_result_mask)

    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    w, h = p_img_to_eval.size
    print('------------------------------------')
    print(f'Main parameters')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_img_to_eval}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')


# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
# https://debuggercafe.com/an-introduction-to-pytorch-visualization-utilities/


if __name__ == '__main__':
    print('main_masks_story_rgb_02()')
    main_masks_story_rgb_02()
    pass
