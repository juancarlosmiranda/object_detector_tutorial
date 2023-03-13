"""
Project: TorchVision 0.3 Object Detection Finetuning Tutorial
Author: Juan Carlos Miranda
Date: December 2021
Description:
 Examples based in https://docs.python.org/3/library/unittest.html
...

Use:
 python -m unittest ./tests/test_object_detector
 test_object_detector.TestStringMethods.test_object_detection_api

"""
import unittest
import os
import time
import cv2
from detector.obj_detector_frame_02 import ObjectDetectorFrame02


class TestStringMethods(unittest.TestCase):

    def setUp(self) -> None:
        print("setUp(self)")
        self.BASE_DIR = os.path.abspath('.')
        ################################
        # config for files models
        main_path_project = os.path.abspath('.')
        trained_model_folder = 'trained_model'
        trained_model_path = os.path.join(main_path_project, trained_model_folder)
        file_name_model = 'MODEL_SAVED.pth'
        self.file_model_path = os.path.join(trained_model_path, file_name_model)
        ################################
        time_1 = time.time()
        self.obj_detector = ObjectDetectorFrame02(self.file_model_path)
        time_2 = time.time()
        time_total = time_2 - time_1
        print('load ObjectDetectorFrame time_total-->', time_total)

    def test_object_detection_in_frame(self):
        main_path_project = os.path.abspath('.')
        # -------------------------------------------
        # Datasets
        # -------------------------------------------
        dataset_folder = os.path.join('assets')  # YOUR_DATASET HERE
        path_dataset = os.path.join(main_path_project, dataset_folder)
        path_images_folder = 'images'
        path_dataset_images = os.path.join(path_dataset, path_images_folder)

        # -------------------------------------------
        # Output results
        # -------------------------------------------
        output_folder = 'output'
        path_output = os.path.join(main_path_project, output_folder)
        image_01_result_rgb = 'result_rgb_.png'
        image_01_result_mask = 'result_mask_.png'
        path_image_01_result_rgb = os.path.join(path_output, image_01_result_rgb)

        # -------------------------------------------
        # Open image with OpenCV cv2.imread
        # -------------------------------------------
        img_to_eval_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'
        # img_to_eval_name = 'FudanPed00020.png'
        img_to_eval_name = '20210523_red_cross.png'

        path_img_to_eval = os.path.join(path_dataset_images, img_to_eval_name)

        # image reading
        cv_img_to_eval = cv2.imread(path_img_to_eval)  # ndarray:(H,W, 3)
        # cv2.imshow('NOTHING->', cv_img_to_eval)
        # cv2.waitKey()

        score_threshold = 0.8
        # -----------------------------------
        time_1 = time.time()
        image_analised = self.obj_detector.object_detection_in_frame(cv_img_to_eval, score_threshold)
        time_2 = time.time()
        time_total = time_2 - time_1
        print('time_total-->', time_total)
        # -----------------------------------

        # ----------------------------
        # resized
        SCREEN_SCALE_FX = 0.5
        SCREEN_SCALE_FY = 0.5
        sized_frame = cv2.resize(image_analised, (0, 0), fx=SCREEN_SCALE_FX, fy=SCREEN_SCALE_FY)
        # cv2.imshow("Resized image", sized_frame)
        cv2.imshow("Resized image", image_analised)
        cv2.waitKey()
        cv2.imwrite(path_image_01_result_rgb, image_analised)
        # ----------------------------

        # ----------------------------


if __name__ == '__main__':
    unittest.main()
