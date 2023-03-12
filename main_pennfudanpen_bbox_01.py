"""
Project: Object detector and segmentation tutorial https://github.com/juancarlosmiranda/object_detector_tutorial
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2021
Description:

Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial. http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Explained by David Macedo.
https://github.com/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/finetuning_torchvision_models_tutorial.ipynb

Adapted by Juan Carlos Miranda as a programming practice, February 2021.

From one image saved in .png, the script opens that image and evaluates it using an object detection model.
Example based on bounding box detection, in which the model predicts something and an image is showed.
With label prediction in rectangle.
1) Load a RGB image
2) Define an object detector model fasterrcnn_resnet50_fpn
3) Evaluate RGB image.


Codes based on:
* https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
* https://debuggercafe.com/an-introduction-to-pytorch-visualization-utilities/

Use:
"""
import os
import time
import torch
import numpy as np

# Managing images formats
import torchvision.transforms.functional as F
from torchvision.io import read_image
from PIL import Image

# deep learning models
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# Drawing on the screen
from torchvision.utils import draw_bounding_boxes

from helpers.helper_examples import COCO_INSTANCE_CATEGORY_NAMES
from helpers.helper_examples import show_one_image
from helpers.helper_examples import merge_masks
from helpers.helper_examples import read_transform_return

def main_bbox_pennfundanped():
    print('------------------------------------')
    print('Example BBOX with PennFundanPed dataset')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # -------------------------------------------
    # Datasets
    # -------------------------------------------
    dataset_folder = os.path.join('dataset', 'PennFudanPed')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)
    path_images_folder = 'PNGImages'
    path_dataset_images = os.path.join(path_dataset, path_images_folder)

    # -------------------------------------------
    # Open image with Pillow.Image.open()
    # -------------------------------------------
    # data about image to evaluate here, open with Pillow
    image_to_eval_name = 'FudanPed00005.png'
    path_image_to_eval = os.path.join(path_dataset_images, image_to_eval_name)
    p_img_to_evaluate = Image.open(path_image_to_eval)  # {PngImageFile}

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    # loading the trained model only once to reduce time
    score_threshold = 0.8
    start_time_model_load = time.time()
    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False) #(pretrained=True, min_size=800)  # todo: put parameters in variable
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    # evaluation inside the object detector model, iterative task
    start_time_eval = time.time()  # this is the evaluation
    int_input, tensor_input = read_transform_return(p_img_to_evaluate)
    with torch.no_grad():
        predictions_model = model(tensor_input.to(device_selected))
    end_time_eval = time.time()

    # TODO: add intermediate layer here
    # -------------------------------------
    # Managing prediction, making something here (filtering, extracting)
    # -------------------------------------
    pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy()
    pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
    pred_labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions_model[0]['labels'].cpu().numpy()]

    # -------------------------------------
    # Filtering predictions according to rules
    # -------------------------------------
    boxes_filtered = pred_boxes[pred_scores >= score_threshold].astype(np.int32)
    labels_filtered = pred_labels[:len(boxes_filtered)]
    # -------------------------------------

    # -------------------------------------
    # It displays the results on the screen according to the colours.
    # -------------------------------------
    colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))
    colours_to_draw = [tuple(color) for color in colours]
    result_with_boxes = draw_bounding_boxes(
        image=int_input,
        boxes=torch.tensor(boxes_filtered), width=1,
        colors=colours_to_draw,
        labels=labels_filtered,
        fill=True  # this complete fill in bounding box
    )
    # show_one_image(result_with_boxes) # optional if there are other transformations
    p_result_with_boxes = F.to_pil_image(result_with_boxes)
    p_result_with_boxes.show()

    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    h, w = p_img_to_evaluate.size
    print('------------------------------------')
    print(f'Main parameters')
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_image_to_eval}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')


if __name__ == '__main__':
    print('Testing BBOX object detector')
    main_bbox_pennfundanped()
    pass



    # -------------------------------------------
    # Visualizing bounding boxes
    # -------------------------------------------
    # boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
    # colors = ["blue", "yellow"]
    # img_bbox = draw_bounding_boxes(image_01, boxes, colors=colors, width=5)
    # show(img_bbox)

    # boxes = torch.tensor([
    #    [135, 50, 210, 365],
    #    [210, 59, 280, 370],
    #    [300, 240, 375, 380]
    # ])
    # colors = ['red', 'red', 'green']
    # result = draw_bounding_boxes(
    #    image=image_01,
    #    boxes=boxes,
    #    colors=colors,
    #    width=3
    # )
    # show(result)
    ####################################