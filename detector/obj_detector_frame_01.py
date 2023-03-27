import torch
import cv2
import training_utils.transforms as T
from PIL import Image
from detector.model_helper import get_model_instance_segmentation


class ObjectDetectorFrame01:
    device_selected = None
    num_classes = 2
    model = None

    def __init__(self, file_path_trained_model):
        if file_path_trained_model is None:
            self.set_default()
        else:
            # IS NOT BY DEFAULT, loading model pre-trained
            # todo: add if file exists
            #self.device_selected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.device_selected = torch.device('cpu')
            self.model = get_model_instance_segmentation(self.num_classes)
            self.model.load_state_dict(torch.load(file_path_trained_model))
            self.model.to(self.device_selected)
            self.model.eval()

    def set_default(self):
        print('set_default')
        self.device_selected.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pass

    def draw_bounding_boxes(self, img_path, predicted_boxes):
        # todo: check this variables, where put that
        # todo: this could be changed using torchvision libraries
        rect_th = 2
        text_size = 3
        text_th = 2
        CLASS_NAME = 'person'

        # for to draw rectangles
        img_bounding_boxes = cv2.imread(img_path)
        for i in range(len(predicted_boxes)):
            cv2.rectangle(img_bounding_boxes, predicted_boxes[i][0], predicted_boxes[i][1], color=(0, 255, 0),
                          thickness=rect_th)
            cv2.putText(img_bounding_boxes, CLASS_NAME, predicted_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size,
                        (0, 255, 0), thickness=text_th)
        pass
        return img_bounding_boxes

    def object_detection_api(self, an_image_path, threshold=0.8):
        img = Image.open(an_image_path).convert("RGB")
        trans1 = T.ToTensor()
        img_t = trans1(img)
        # ---------------------------------
        # Get prediction here
        with torch.no_grad():
            prediction = self.model([img_t[0].to(self.device_selected)])
        # ---------------------------------
        # format conversion to draw bounding boxes
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in
                      list(prediction[0]['boxes'].detach().numpy())]
        pred_score = list(prediction[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # only draw great than threshold
        pred_boxes = pred_boxes[:pred_t + 1]  # a processed list
        # ------------------
        img_drawed = self.draw_bounding_boxes(an_image_path, pred_boxes)
        return img_drawed

    def draw_bounding_boxes_frame(self, img_bounding_boxes, predicted_boxes):
        # todo: check this variables, where put that
        rect_th = 1
        text_size = 2
        text_th = 1
        CLASS_NAME = 'person'

        for i in range(len(predicted_boxes)):
            cv2.rectangle(img_bounding_boxes, predicted_boxes[i][0], predicted_boxes[i][1], color=(0, 255, 0),
                          thickness=rect_th)
            cv2.putText(img_bounding_boxes, CLASS_NAME, predicted_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size,
                        (0, 255, 0), thickness=text_th)
        pass
        return img_bounding_boxes

    def object_detection_in_frame(self, a_frame_image, threshold=0.8):
        # convert to PIL image because in internal libraries is the default format used
        image_drawed = None
        img = Image.fromarray(a_frame_image).convert("RGB")
        trans1 = T.ToTensor()
        img_t = trans1(img)
        # ---------------------------------
        # Get prediction here
        with torch.no_grad():
            prediction = self.model([img_t[0].to(self.device_selected)])
        # ---------------------------------
        # format conversion to draw bounding boxes
        # it is slow, a better solution is
        # pred_boxes = predictions_model[0]['boxes'].detach().cpu().numpy().astype(np.int32)
        # but it needs comprehension of array to organize data to draw
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in
                      list(prediction[0]['boxes'].detach().numpy())]
        pred_score = list(prediction[0]['scores'].detach().numpy())

        if pred_score[0] > threshold:
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # only draw great than threshold
            pred_boxes = pred_boxes[:pred_t + 1]  # a processed list
            # todo: put here pytorch utilities
            image_drawed = self.draw_bounding_boxes_frame(a_frame_image, pred_boxes)
        else:
            print('Values under threshold->')
            print('pred_boxes->', pred_boxes)
            print('pred_score->', pred_score)
        return image_drawed
