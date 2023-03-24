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
import time
import os
import torch
import references.detection.utils as utils
import gc



from penn_fundan_dataset import PennFudanDataset
from references.detection.engine import train_one_epoch, evaluate
from detector.model_helper import get_model_instance_segmentation, get_transform


from GPUtil import showUtilization as gpu_usage
from numba import cuda


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def main_loop_training():
    print('MAIN OBJECT DETECTION TRAINING')
    free_gpu_cache()
    main_path_project = os.path.abspath('..')
    dataset_folder = os.path.join('dataset', 'PennFudanPed')
    path_dataset = os.path.join(main_path_project, dataset_folder)
    #-------------------------------
    # config for files models
    #-------------------------------
    trained_model_folder = 'trained_model'  # put here YOUR_FOLDER
    trained_model_path = os.path.join(main_path_project, trained_model_folder)
    file_name_model = 'MODEL_SAVED.pth'  # put here YOUR_FILE_NAME
    file_model_path = os.path.join(trained_model_path, file_name_model)
    #-------------------------------
    num_epochs = 1  # let's train it for 10 epochs
    num_classes = 2  # our dataset has two classes only - background and person

    device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    """
    ERROR WITH CUDA: Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 72.00 MiB (GPU 0; 2.00 GiB total capacity; 1.60 GiB already allocated; 0 bytes free; 1.68 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
    """
    device_selected = torch.device('cpu')
    #device_selected = torch.device('cuda')

    # use our dataset and defined transformations
    dataset = PennFudanDataset(path_dataset, get_transform(train=True))
    dataset_test = PennFudanDataset(path_dataset, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    # todo: check these parameters
    batch_size = 2 #8  # 2   -  1
    num_workers = 4 #16  # 4
    print_freq = 10

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.to(device_selected)  # move model to the right device

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    print(' Training the network -->')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device_selected, epoch, print_freq=print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device_selected)
    print('Finished training')


    print(f'num_epochs to train ->{num_epochs}')
    print(f'num_classes train ->{num_classes}')
    print(f'device_selected ->{device_selected}')
    print(f'Configuring our model ->{model}')
    print(f'Saving model in file ->{file_model_path}')
    torch.save(model.state_dict(), file_model_path)


if __name__ == "__main__":
    main_loop_training()
