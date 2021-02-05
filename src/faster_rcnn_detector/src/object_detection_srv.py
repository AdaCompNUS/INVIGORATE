#!/usr/bin/env python3
import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
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
import torch
from torch import nn
from torch.nn import functional as F
# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2 import modeling
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from invigorate_msgs.srv import ObjectDetection, ObjectDetectionResponse
from config.config import *
import pdb
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
        self._cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
        # self._cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # self._cfg.MODEL.WEIGHTS = os.path.join(NN_MODEL_PATH, 'model_final_r50_fpn.pth')
        self._cfg.MODEL.WEIGHTS = os.path.join(NN_MODEL_PATH, 'model_final_cascade.pth')
        self._predictor = DefaultPredictor(self._cfg)

        # init ros service
        self._service = rospy.Service('object_detection_srv', ObjectDetection, self._call_back)
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

        img_cv2 = self._imgmsg_to_cv2(req.img)
        if len(req.rois) == 0:
            rois = None
        else:
            rois = np.array([req.rois]).reshape(-1, 4)
        num_box, pred_bboxes, pred_classes, cls_scores = self._detect_objects(img_cv2, rois)

        pred_classes = pred_classes.cpu().numpy().astype(np.int32).tolist()
        pred_bboxes = pred_bboxes.cpu().numpy().astype(np.float64).reshape(-1).tolist()
        cls_scores = cls_scores.cpu().numpy().astype(np.float64).reshape(-1).tolist()
        print(num_box)
        print(pred_bboxes)
        print(pred_classes)
        print(cls_scores)

        res = ObjectDetectionResponse()
        res.num_box = num_box
        res.bbox = pred_bboxes
        res.cls = pred_classes
        res.cls_scores = cls_scores
        res.box_feats = json.dumps("")
        return res

    def _detect_objects(self, img_cv2, rois=None):

        outputs = self._predictor(img_cv2, rois)

        if rois is None:
            # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
            # self._visualize(img_cv2, outputs)
            scores = outputs["instances"].original_scores
            bg_scores = torch.clamp(1 - scores.sum(dim=1), min=0.)
            scores = torch.cat([bg_scores.unsqueeze(1), scores], dim = 1)
            return len(outputs["instances"].pred_classes), outputs["instances"].pred_boxes.tensor, \
                   outputs["instances"].pred_classes + 1, scores
        else:

            scores = F.softmax(outputs, dim=1)
            scores[:, 0], scores[:, -1] = scores[:, -1].clone(), scores[:, 0].clone()
            cls = scores.argmax(dim=1)
            return len(cls), torch.as_tensor(rois), cls, scores

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
    rospy.init_node('detectron2_server')
    object_detection_service = ObjectDetectionService()
    rospy.spin()

