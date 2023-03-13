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
from detector.obj_detector_frame_01 import ObjectDetectorFrame01


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
        self.obj_detector = ObjectDetectorFrame01(self.file_model_path)
        time_2 = time.time()
        time_total = time_2 - time_1
        print('load ObjectDetectorFrame time_total-->', time_total)

    def test_object_detection_api(self):
        print("test_object_detection_api(self) -->")
        # img_name = 'FudanPed00020.png'
        # img_name = 'banner-diverse-group-of-people-2.jpg'
        img_name = '20210927_114012_k_r2_e_000_150_138_2_0_C.png'

        an_image_path = os.path.join(self.BASE_DIR, 'assets', 'images', img_name)
        an_image_analised_path = os.path.join(self.BASE_DIR, 'output', 'result_' + img_name)
        a_threshold = 0.8
        # -----------------------------------
        time_1 = time.time()
        image_analised = self.obj_detector.object_detection_api(an_image_path, a_threshold)
        time_2 = time.time()
        time_total = time_2 - time_1
        print('time_total-->', time_total)
        # -----------------------------------

        # ----------------------------
        scale_percent = 25  # percent of original size
        width = int(image_analised.shape[1] * scale_percent / 100)
        height = int(image_analised.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(image_analised, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Resized image", resized)
        #cv2.imwrite(an_image_analised_path, image_analised)
        # ----------------------------
        cv2.waitKey()

    def test_object_detection_in_frame(self):
        print("test_object_detection_in_frame(self) -->")
        img_name = 'FudanPed00020.png'
        an_image_path = os.path.join(self.BASE_DIR, 'assets', img_name)
        an_image_analised_path = os.path.join(self.BASE_DIR, 'output', 'result_' + img_name)
        a_threshold = 0.5
        # -----------------------------------
        time_1 = time.time()
        frame = cv2.imread(an_image_analised_path)  # ndarray type
        image_analised = self.obj_detector.object_detection_in_frame(frame, a_threshold)
        time_2 = time.time()
        time_total = time_2 - time_1
        print('time_total-->', time_total)
        # -----------------------------------

        # ----------------------------
        scale_percent = 25  # percent of original size
        width = int(image_analised.shape[1] * scale_percent / 100)
        height = int(image_analised.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(image_analised, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Resized image", resized)
        #cv2.imwrite(an_image_analised_path, image_analised)
        # ----------------------------
        cv2.waitKey()


if __name__ == '__main__':
    unittest.main()
