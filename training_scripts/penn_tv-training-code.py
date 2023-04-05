# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
from datetime import datetime

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pennfudanpenn_utility.engine import train_one_epoch, evaluate

import pennfudanpenn_utility.utils as utils
import pennfudanpenn_utility.transforms as T

from penn_fundan_dataset import PennFudanDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: 512"


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_maskrcnn_model_instance(num_classes):
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
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def main_penn_loop_training():
    main_path_project = os.path.abspath('..')
    dataset_folder = os.path.join('dataset', 'PennFudanPed_02')
    path_dataset = os.path.join(main_path_project, dataset_folder)
    # train on the GPU or on the CPU, if a GPU is not available
    #device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_selected = torch.device('cuda')
    #-------------------------------
    # config for files models
    #-------------------------------
    trained_model_folder = 'trained_model'  # put here YOUR_FOLDER
    trained_model_path = os.path.join(main_path_project, trained_model_folder)
    # --------------------------------
    # PREPARING FILE NAME
    # --------------------------------
    now = datetime.now()
    datetime_experiment = now.strftime("%Y%m%d_%H%M%S")
    filename_model = 'model_maskrcnn_{}.pth'.format(datetime_experiment)
    filename_model_path = os.path.join(trained_model_path, filename_model)
    #-------------------------------

    # --------------------------------------------
    # parameters
    # --------------------------------------------
    device_selected = torch.device('cpu')
    num_epochs = 1  # let's train it for 10 epochs
    batch_size = 2  # 64  # it increments the amount of memory to allocate
    num_workers = 4
    num_classes = 2  # our dataset has two classes only - background and person
    # ---
    # optimizer
    # ---
    lr = 0.02
    momentum = 0.9
    weight_decay = 1e-4 # 0.0005 #0.0005
    # ---
    # lr_scheduler
    step_size = 3
    gamma = 0.1
    # ---
    print_freq = 10
    # --------------------------------------------
    # use our dataset and defined transformations
    dataset = PennFudanDataset(path_dataset, get_transform(train=True))
    dataset_test = PennFudanDataset(path_dataset, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_maskrcnn_model_instance(num_classes)
    model.to(device_selected)  # move model to the right device

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("Start training")
    start_time_training = time.time()
    # -------------------------------------------
    for epoch in range(num_epochs):
        # train for one epoch, printing every print_freq iterations
        train_one_epoch(model, optimizer, data_loader, device_selected, epoch, print_freq=print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate after every epoch
        print(f'Evaluating epoch {epoch}')
        evaluate(model, data_loader_test, device=device_selected)
    # -------------------------------------------
    end_time_training = time.time()
    total_time_training = end_time_training - start_time_training
    # -------------------------------------------
    print('Finished training')
    print(f'total_time_training={total_time_training}')
    print('Training time:', time.strftime("%H:%M:%S", time.gmtime(total_time_training)))
    print(f'device_selected ->{device_selected}')
    print(f'model={type(model).__name__}')
    print('Saving model in file ->', filename_model_path)
    torch.save(model.state_dict(), filename_model_path)

if __name__ == "__main__":
    main_penn_loop_training()
