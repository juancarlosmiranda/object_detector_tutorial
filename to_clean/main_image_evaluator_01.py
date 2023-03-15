"""
Project: TorchVision 0.3 Object Detection Finetuning Tutorial
Author: Juan Carlos Miranda
Date: December 2021
Description:
 Adapted from https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
...

Use:
 python -m unittest ./tests/test_object_detector
 test_object_detector.TestStringMethods.test_object_detection_api

"""

import os
import time
import cv2
from detector.obj_detector_frame import ObjectDetectorFrame


def test_image_evaluator():
    print('------------------------------------')
    print('MAIN OBJECT DETECTION EVALUATION')
    print('------------------------------------')
    main_path_project = os.path.abspath('.')

    # DATASET ---------------
    dataset_root_folder = os.path.join('dataset')
    dataset_folder = os.path.join(dataset_root_folder, 'dataset_apples')  # YOUR_DATASET HERE

    #dataset_folder = os.path.join('dataset', 'PennFudanPed')  # YOUR_DATASET HERE

    #path_dataset = os.path.join(main_path_project, dataset_folder, 'PNGImages')
    path_dataset = os.path.join(main_path_project, dataset_folder)
    test_image_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
    #test_image_name = 'FudanPed00001.png'
    ################################

    # TRAINED MODEL ---------------
    trained_model_folder = 'trained_model'
    trained_model_path = os.path.join(main_path_project, trained_model_folder)
    file_name_model = 'MODEL_SAVED.pth'
    file_model_path = os.path.join(trained_model_path, file_name_model)
    ################################


    # open image with OpenCV
    test_image_path = os.path.join(path_dataset, test_image_name)
    test_image = cv2.imread(test_image_path)
    # -----------------------------
    #cv2.imshow('webcam feed', test_image)
    # ----------------------------
    #cv2.waitKey()

    ################################
    time_1 = time.time()
    obj_detector = ObjectDetectorFrame(file_model_path)
    time_2 = time.time()
    time_total = time_2 - time_1
    print('load ObjectDetectorFrame time_total-->', time_total)
    a_threshold = 0.7
    # -----------------------------

    # ----------------------------
    # make something with frame
    # ----------------------------
    image_analysed = obj_detector.object_detection_in_frame(test_image, a_threshold)
    # ----------------------------
    cv2.imshow('Analysed', image_analysed)
    # ----------------------------
    cv2.waitKey()


if __name__ == "__main__":
    test_image_evaluator()
