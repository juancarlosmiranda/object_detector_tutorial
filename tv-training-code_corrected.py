# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from references.detection.engine import train_one_epoch, evaluate
import utils
import references.detection.transforms as T
import gc


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

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

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    print('------------------------------------')
    print('MAIN OBJECT DETECTION')
    print('------------------------------------')
    # home_user = os.path.join('C:', '\\', 'Users', 'Usuari')
    # main_path_project = os.path.join(home_user, 'PycharmProjects')
    main_path_project = os.path.abspath('.')
    # path_dataset = os.path.join('C:', '\\', 'Users', 'Usuari', 'PycharmProjects', 'object_detector', 'dataset', 'PennFudanPed')
    dataset_folder = os.path.join('dataset', 'PennFudanPed')
    path_dataset = os.path.join(main_path_project, dataset_folder)

    use_cuda = torch.cuda.is_available()
    #device_selected = torch.device("cuda:0" if use_cuda else "cpu")
    #device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_selected = torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 4}
    # ----------------------
    torch.cuda.empty_cache()

    print('device->', device_selected)
    print('max_size->', torch.backends.cuda.cufft_plan_cache.max_size)
    print('cache.size->', torch.backends.cuda.cufft_plan_cache.size)
    print('summary->', torch.cuda.memory_summary(device=None, abbreviated=False))
    a = torch.cuda.memory_allocated(0)
    print('memory_allocated', a)


    # ----------------------
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset(path_dataset, get_transform(train=True))
    dataset_test = PennFudanDataset(path_dataset, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # ------------------------------
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'],
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, params['batch_size'], shuffle=False, num_workers=params['num_workers'],
        collate_fn=utils.collate_fn)

    #del dataset, dataset_test,
    #del data_loader, data_loader_test,
    gc.collect()

    # ------------------------------

    # define training and validation data loaders
    #data_loader = torch.utils.data.DataLoader(dataset, **params)
    #data_loader_test = torch.utils.data.DataLoader(dataset_test, **params)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device_selected)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device_selected, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device_selected)

    print("That's it!")
    
if __name__ == "__main__":
    main()
