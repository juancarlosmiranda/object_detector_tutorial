import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

# Drawing on the screen
# from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from helpers.helper_examples import COCO_INSTANCE_CATEGORY_NAMES


class ObjectDetectorPreTrainedMASK:
    """
    It implements draw_bounding_boxes() to draw in the screen. For this it is necessary to make conversion
    of data structures such as OpenCV to Tensor_float32 and tensor_uint8

    """
    device_selected = None
    num_classes = 2
    model = None

    def __init__(self, file_path_trained_model=None):
        # -----------------------------------------------
        self.device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        score_threshold = 0.7
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_threshold)
        # -----------------------------------------------
        if file_path_trained_model is not None:  # os.path.exists(file_path_trained_model):
            self.model.load_state_dict(torch.load(file_path_trained_model))
        # -----------------------------------------------
        self.model.to(self.device_selected)
        self.model.eval()
        # -----------------------------------------------

    def set_default(self):
        print('set_default')
        self.device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        score_threshold = 0.7
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_threshold)
        self.model.to(self.device_selected)
        self.model.eval()
        pass

    def object_detection_in_frame(self, a_frame_image, score_threshold=0.8):
        image_drawn = None
        # conversion from BGR to RGB
        cv_img = cv2.cvtColor(a_frame_image, cv2.COLOR_BGR2RGB)
        image_transposed = np.transpose(cv_img, [2, 0, 1])
        img_to_eval_uint8 = torch.tensor(image_transposed)  # used with torchvision.draw_bounding_boxes()
        img_to_eval_float32 = F.to_tensor(a_frame_image)  # used with detection model
        img_to_eval_list = [img_to_eval_float32.to(self.device_selected)]

        # ---------------------------------
        # Get prediction with model here
        # -------------------------------------
        with torch.no_grad():
            predictions_model = self.model(img_to_eval_list)
        # -------------------------------------
        # Managing prediction, making something here (filtering, extracting)
        # -------------------------------------
        pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy()
        pred_scores = predictions_model[0]['scores'].detach().cpu().numpy()
        pred_masks = predictions_model[0]['masks']
        pred_labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions_model[0]['labels'].cpu().numpy()]
        # ---------------------------------

        # -------------------------------------
        # Filtering predictions according to rules
        # -------------------------------------
        boxes_filtered = pred_boxes[pred_scores >= score_threshold].astype(np.int32)
        labels_filtered = pred_labels[:len(boxes_filtered)]
        masks_filtered = pred_masks[pred_scores >= score_threshold]

        final_masks = masks_filtered > 0.5  # ?
        final_masks = final_masks.squeeze(1)  # ?

        # -------------------------------------
        # Drawing bounding boxes with Pytorch
        # -------------------------------------
        colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))
        colours_to_draw = [tuple(color) for color in colours]
        #result_with_boxes = draw_bounding_boxes(
        #    image=img_to_eval_uint8,
        #    boxes=torch.tensor(boxes_filtered), width=1,
        #    colors=colours_to_draw,
        #    labels=labels_filtered,
        #    fill=False  # this complete fill in bounding box
        #)

        mask_seg_result = draw_segmentation_masks(
            image=img_to_eval_uint8,
            masks=final_masks,
            colors=colours_to_draw,
            alpha=0.7
        )

        # ------------------------------------
        # conversion from Tensor a PIL.Image and OpenCV
        # ------------------------------------
        p_result_with_boxes = F.to_pil_image(mask_seg_result)  # result_with_boxes)
        image_drawn_numpy = np.array(p_result_with_boxes)
        image_drawn = cv2.cvtColor(image_drawn_numpy, cv2.COLOR_RGB2BGR)

        # ---------------------------------
        return image_drawn
