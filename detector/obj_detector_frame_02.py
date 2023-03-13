import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from detector.model_helper import get_model_instance_segmentation

# Drawing on the screen
from torchvision.utils import draw_bounding_boxes
from helpers.helper_examples import COCO_INSTANCE_CATEGORY_NAMES


class ObjectDetectorFrame02:
    """
    It implements draw_bounding_boxes() to draw in the screen. For this it is necessary to make conversion
    of data structures such as OpenCV to Tensor_float32 and tensor_uint8

    """
    device_selected = None
    num_classes = 2
    model = None

    def __init__(self, file_path_trained_model):
        # -----------------------------------------------
        if file_path_trained_model is None:
            self.set_default()
        else:
            # IS NOT BY DEFAULT, loading model pre-trained
            # todo: add if file exists
            self.device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model = get_model_instance_segmentation(self.num_classes)
            self.model.load_state_dict(torch.load(file_path_trained_model))
            self.model.to(self.device_selected)
            self.model.eval()
        # -----------------------------------------------

    def set_default(self):
        print('set_default')
        self.device_selected.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pass

    def object_detection_in_frame(self, a_frame_image, score_threshold=0.8):
        # convert to PIL image because in internal libraries is the default format used
        image_drawn = None
        cv_img = cv2.cvtColor(a_frame_image, cv2.COLOR_BGR2RGB)
        image_transposed = np.transpose(cv_img, [2, 0, 1])
        img_to_eval_uint8 = torch.tensor(image_transposed)  # used with torchvision.draw_bounding_boxes()
        img_to_eval_float32 = F.to_tensor(a_frame_image)
        img_to_eval_list = [img_to_eval_float32.to(self.device_selected)]
        # ---------------------------------
        # Get prediction here
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
        # -------------------------------------

        # -------------------------------------
        # Drawing bounding boxes with Pytorch
        # -------------------------------------
        colours = np.random.randint(0, 255, size=(len(boxes_filtered), 3))
        colours_to_draw = [tuple(color) for color in colours]
        result_with_boxes = draw_bounding_boxes(
            image=img_to_eval_uint8,
            boxes=torch.tensor(boxes_filtered), width=1,
            colors=colours_to_draw,
            labels=labels_filtered,
            fill=False  # this complete fill in bounding box
        )
        # ------------------------------------
        # conversion from Tensor a PIL.Image
        # ------------------------------------
        p_result_with_boxes = F.to_pil_image(result_with_boxes)
        image_drawn_numpy = np.array(p_result_with_boxes)
        image_drawn = cv2.cvtColor(image_drawn_numpy, cv2.COLOR_RGB2BGR)

        # ---------------------------------
        return image_drawn
