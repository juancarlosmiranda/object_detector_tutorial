import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


#from torchvision import transforms
#import matplotlib.pyplot as plt
from device_data_loader import *
from penn_fundan_dataset import PennFudanDataset

if __name__ == '__main__':
    print('------------------------------------')
    print('MAIN FINETUNNING')
    print('------------------------------------')
    home_user = os.path.join('C:', '\\', 'Users', 'Usuari')
    main_path_project = os.path.join(home_user, 'PycharmProjects', 'object_detector')
    path_dataset = os.path.join('C:', '\\', 'Users', 'Usuari', 'Downloads', 'datasets_downloaded', 'PennFudanPed')
    #################################################
    # General options to show at predict time
    # get default device
    device = get_default_device()
    print('device selected -->', device)
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
