#!/usr/bin/env python3
import sys 
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")

import rospy

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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from invigorate_msgs.srv import *

# --------- SETTINGS ------------
VISUALIZE = False

class ObjectDetectionService():
    def __init__(self):
        # init Detectron2
        self._cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self._predictor = DefaultPredictor(self._cfg)

        self._class_names = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

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

    def _detect_objects(self, req):
        if req.img.encoding != '8UC3':
            rospy.logerr("object_detection_srv, image encoding not supported!!")
            res = ObjectDetectionResponse()
            return res

        img_cv2 = self._imgmsg_to_cv2(req.img)
        outputs = self._predictor(img_cv2)
        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        self._visualize(img_cv2, outputs)

        pred_classes = outputs["instances"].pred_classes.cpu().numpy().tolist()
        num_box = len(pred_classes)
        pred_bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().reshape(-1).tolist()
        cls_scores = outputs["instances"].scores.cpu().numpy().tolist()
        # pred_bboxes =  pred_bboxes.cpu().numpy().tolist()

        res = ObjectDetectionResponse()
        res.num_box = num_box
        res.bbox = pred_bboxes
        res.cls = pred_classes
        res.cls_scores = cls_scores
        res.box_feats = json.dumps("")

        return res

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
    rospy.spin()

