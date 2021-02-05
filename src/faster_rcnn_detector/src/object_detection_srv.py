#!/usr/bin/env python3
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")

import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../../'))

import rospy
import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2 import modeling
modeling.roi_heads.roi_heads
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from invigorate_msgs.srv import ObjectDetection, ObjectDetectionResponse
from config.config import *

# --------- SETTINGS ------------
VISUALIZE = False

class ObjectDetector(DefaultPredictor):

    def __call__(self, original_image, rois=None):

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class ObjectDetectionService():
    def __init__(self):
        # init Detectron2
        self._cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # self._cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
        self._cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self._cfg.MODEL.WEIGHTS = os.path.join(NN_MODEL_PATH, 'model_final_r50_fpn.pth')
        self._predictor = DefaultPredictor(self._cfg)

        self._class_names = [{'supercategory': 'sports', 'id': 1, 'name': 'sports ball'},
                             {'supercategory': 'kitchen', 'id': 2, 'name': 'bottle'},
                             {'supercategory': 'kitchen', 'id': 3, 'name': 'cup'},
                             {'supercategory': 'kitchen', 'id': 4, 'name': 'knife'},
                             {'supercategory': 'food', 'id': 5, 'name': 'banana'},
                             {'supercategory': 'food', 'id': 6, 'name': 'apple'},
                             {'supercategory': 'food', 'id': 7, 'name': 'carrot'},
                             {'supercategory': 'electronic', 'id': 8, 'name': 'mouse'},
                             {'supercategory': 'electronic', 'id': 9, 'name': 'remote'},
                             {'supercategory': 'electronic', 'id': 10, 'name': 'cell phone'},
                             {'supercategory': 'indoor', 'id': 11, 'name': 'book'},
                             {'supercategory': 'indoor', 'id': 12, 'name': 'scissors'},
                             {'supercategory': 'indoor', 'id': 13, 'name': 'teddy bear'},
                             {'supercategory': 'indoor', 'id': 14, 'name': 'toothbrush'},
                             {'supercategory': 'indoor', 'id': 15, 'name': 'box'}]

        # init ros service
        self._service = rospy.Service('object_detection_srv', ObjectDetection, self._detect_objects)
        rospy.loginfo("object_detection_srv inited")

    def _imgmsg_to_cv2(self, img_msg):
        dtype, n_channels = np.uint8, 3 # hard code
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                        dtype=dtype, buffer=img_msg.data)
        return im

    def _call_back(self, req):
        if req.img.encoding != '8UC3':
            rospy.logerr("object_detection_srv, image encoding not supported!!")
            res = ObjectDetectionResponse()
            return res

        img_cv2 = self._imgmsg_to_cv2(req.image)
        outputs = self._predictor([img_cv2, detected_instances=blahblahs])
        img_cv2 = self._imgmsg_to_cv2(req.img)
        rois = np.array(req.rois).view(-1, 4)
        num_box, pred_bboxes, pred_classes, cls_scores = self._detect_objects(img_cv2, rois)

        pred_classes = pred_classes.cpu().numpy().tolist()
        pred_bboxes = pred_bboxes.tensor.cpu().numpy().reshape(-1).tolist()
        cls_scores = cls_scores.cpu().numpy().tolist()

        res = ObjectDetectionResponse()
        res.num_box = num_box
        res.bbox = pred_bboxes
        res.cls = pred_classes
        res.cls_scores = cls_scores
        res.box_feats = json.dumps("")


    def _detect_objects(self, img_cv2, rois=None):

        outputs = self._predictor(img_cv2)
        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        # self._visualize(img_cv2, outputs)

        return len(outputs["instances"].pred_classes), outputs["instances"].pred_boxes, \
               outputs["instances"].pred_classes, outputs["instances"].scores

    def _visualize(self, im, outputs):
        if not VISUALIZE:
            return

        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("img", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('object_detection_service')
    object_detection_service = ObjectDetectionService()
    img_path = "../../images/1.png"
    object_detection_service._detect_objects(cv2.imread(img_path), rois=None)
    rospy.spin()

