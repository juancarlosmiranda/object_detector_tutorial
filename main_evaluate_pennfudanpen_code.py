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

from pennfudanpenn_data.penn_fundan_dataset import PennFudanDataset
from PIL import Image
from helpers.helper_examples import get_transform

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

    print('in_features -->', in_features)
    print('in_features_mask -->', in_features_mask)
    print('hidden_layer -->', hidden_layer)
    print('num_classes -->', num_classes)

    return model


def main_evaluate_pennfundanpen_loop():
    print('------------------------------------')
    print('MAIN OBJECT DETECTION EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # -------------------------------------------
    # Datasets
    # -------------------------------------------
    dataset_folder = os.path.join('dataset', 'PennFudanPed_01')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)

    # -------------------------------------------
    # Trained parameters for models
    # -------------------------------------------
    trained_model_folder = 'trained_model'
    trained_model_path = os.path.join(main_path_project, trained_model_folder)
    file_name_model = 'model_maskrcnn_20230329_173739.pth'
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

    # ----------------------------
    # use our dataset and defined transformations
    dataset = PennFudanDataset(path_dataset, get_transform(train=True))
    dataset_test = PennFudanDataset(path_dataset, get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # ----------------------------

    # -------------------------------------
    # Image evaluation with model
    # -------------------------------------
    start_time_eval = time.time()  # this is the evaluation
    # Data type int_input {Tensor:3}, tensor_input {Tensor:1}
    # t_img_to_evale transformed to Tensor. Pick one image from the test set. {Tensor:3} Loads image in tensor format, get Tensor data
    t_img_to_eval, _ = dataset_test[0]
    model.eval()  # put the model in evaluation mode
    with torch.no_grad():
        predictions_models = model([t_img_to_eval.to(device_selected)])
    end_time_eval = time.time()

    pred_mask_t = predictions_models[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
    p_img = Image.fromarray(t_img_to_eval.mul(255).permute(1, 2, 0).byte().numpy())
    p_pred_mask = Image.fromarray(pred_mask_t)
    p_img.show('Pillow img to show')
    p_pred_mask.show('mask to show')

    # -------------------------------------
    # Display data on screen
    # -------------------------------------
    total_time_model_load = end_time_model_load - start_time_model_load
    total_time_eval = end_time_eval - start_time_eval
    w, h = p_img.size
    print('------------------------------------')
    print(f'Main parameters')
    print('------------------------------------')
    print(f'path_dataset={path_dataset}')
    print(f'Image size width={w} height={h}')
    print(f'device_selected={device_selected}')
    print(f'score_threshold={score_threshold}')
    print(f'Trained model file_model_path={file_model_path}')
    print(f'model={type(model).__name__}')
    print(f'total_time_model_load={total_time_model_load}')
    print(f'total_time_eval={total_time_eval}')


if __name__ == "__main__":
    main_evaluate_pennfundanpen_loop()
