import os
import torchvision
from PIL import Image


def main_loop_general_object_detection():
    print('------------------------------------')
    print('MAIN OBJECT DETECTION EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')
    dataset_root_folder = os.path.join('dataset')
    dataset_folder = os.path.join(dataset_root_folder, 'dataset_apples')  # YOUR_DATASET HERE
    path_dataset = os.path.join(main_path_project, dataset_folder)
    test_image_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # set it to evaluation mode, as the model behaves differently
    # during training_utils and during evaluation
    model.eval()
    image = Image.open(os.path.join(path_dataset, test_image_name))

    image_tensor = torchvision.transforms.functional.to_tensor(image)
    # pass a list of (potentially different sized) tensors
    # to the model, in 0-1 range. The model will take care of
    # batching them together and normalizing
    output = model([image_tensor])
    # output is a list of dict, containing the postprocessed predictions
    print('output-->',output)



if __name__ == '__main__':
    main_loop_general_object_detection()
