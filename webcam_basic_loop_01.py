"""
Project: TorchVision 0.3 Object Detection Finetuning Tutorial
Author: Juan Carlos Miranda
Date: February 2021
Description:
Adapted from https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
Basic loop to extract frames from webcam


Use:


"""

import os
import cv2


def making_something(img_to_draw):
    """
    To simulate frame processing here
    :param img_to_draw:
    :return:
    """
    print(f'making_something(): ->')
    print(f'{img_to_draw.shape}')
    msg_any_key = 'Press any key'
    cv2.putText(img_to_draw, msg_any_key, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=3)
    return img_to_draw

def main_loop_webcam():
    main_path_project = os.path.abspath('.')
    # -------------------------------------------
    # Parameters for cameras
    # -------------------------------------------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # -------------------------------------------
    print('Basic loop to extract frames -->')
    # -----------------------------
    while True:
        ret, frame = cap.read()  # read in real time from camera
        # ----------------------------
        # make something with frame
        # ----------------------------
        image_analysed = making_something(frame)
        cv2.imshow('webcam feed', image_analysed)
        # ----------------------------
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # ----------------------------
    # close camera
    # ----------------------------
    cap.release()
    cv2.destroyAllWindows()
    # ----------------------------

if __name__ == "__main__":
    main_loop_webcam()
