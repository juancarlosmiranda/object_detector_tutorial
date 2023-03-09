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
import numpy as np
import torchvision.transforms.functional as F

# Managing images formats
from torchvision.io import read_image
from PIL import Image

# Deep learning models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as transforms

# Drawing on the screen
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

# GLOBAL
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = transforms.Compose([transforms.ToTensor(), ])


def read_transform_return(image: Image):
    """
    Receives as input an Pillow.Image and returns
    """
    image = np.array(image)
    image_transposed = np.transpose(image, [2, 0, 1])
    # Convert to uint8 tensor.
    int_input = torch.tensor(image_transposed)
    # Convert to float32 tensor.
    tensor_input = transform(image)
    tensor_input = torch.unsqueeze(tensor_input, 0)
    return int_input, tensor_input


def show_image_list(imgs_list):
    """
    Show images from tensor data
    """
    if not isinstance(imgs_list, list):
        imgs = [imgs_list]
    for i, img in enumerate(imgs):
        img = img.detach()  # TODO: ???
        p_img_01 = F.to_pil_image(img)  # TODO: ???
        p_img_01.show()


def show_one_image(t_image):
    """
    Show images from tensor data
    """
    p_img_01 = F.to_pil_image(t_image)  # TODO: ???
    p_img_01.show()


def merge_masks(masks):
    """
    Return a Tensor with merged masks
    """
    merged_mask = masks[0]  # assign the first mask
    for mask in masks:
        merged_mask = mask + merged_mask

    return merged_mask


def main_mask_pennfundanped():
    print('------------------------------------')
    print('Example MASK with PennFundanPed dataset')
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
    # Open image with Pillow.Image.open() and torchvision.io.read_image()
    # -------------------------------------------
    image_to_eval_name = 'FudanPed00005.png'
    path_image_to_eval = os.path.join(path_dataset_images, image_to_eval_name)
    p_img_to_evaluate = Image.open(path_image_to_eval)  # {PngImageFile}
    # used to draw masks
    t_img_to_evaluate = read_image(path_image_to_eval)  # {Tensor:3} Loads image in tensor format, get Tensor data

    # ------------------------------------------
    # Model initialization for object prediction
    # -------------------------------------------
    # loading the trained model only once to reduce time
    score_threshold = 0.8
    start_time_model_load = time.time()
    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device_selected = torch.device('cpu')
    model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)  # (pretrained=True, min_size=800)
    model.to(device_selected)
    model.eval()  # enabling evaluation mode
    end_time_model_load = time.time()

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    # Data type int_input {Tensor:3}, tensor_input {Tensor:1}
    int_input, tensor_input = read_transform_return(p_img_to_evaluate)  # Used to draw images on screen
    with torch.no_grad():
        predictions_model = model(tensor_input.to(device_selected))

    end_time_eval = time.time()

    # -------------------------------------
    # Managing prediction, making something here (filtering, extracting)
    # -------------------------------------
    # output = predictions_model[0]
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

    # -------------------------------------
    # It displays the results on the screen according to the colours.
    # -------------------------------------
    colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))  # random colours
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

    final_masks = masks_filtered > 0.5  # ?
    final_masks = final_masks.squeeze(1)  # ?

    # save masks detected
    mask_seg_result = draw_segmentation_masks(
        image=t_img_to_evaluate,
        masks=final_masks,
        colors=colours_to_draw,
        alpha=0.8
    )
    show_one_image(mask_seg_result)

    # save image with bounding boxes
    # image: torch.Tensor,
    # masks: torch.Tensor,
    # alpha: float = 0.8,

    # to draw bounding boxes and mask at the same time
    bbox_mask_result = draw_segmentation_masks(
        image=result_with_boxes,
        masks=final_masks,
        colors=colours_to_draw,
        alpha=0.8
    )
    show_one_image(bbox_mask_result)

    # ----------------------------------
    # merged binary masks example from predictions, save binary image detected by model
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
    print(f'path_dataset_images={path_dataset_images}')
    print(f'path_img_to_evaluate_01={path_image_to_eval}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')


if __name__ == '__main__':
    print('Testing mask 01')
    main_mask_pennfundanped()
    pass
