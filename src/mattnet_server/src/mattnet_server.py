from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
import cv2
from cv_bridge import CvBridge
br = CvBridge()

import argparse
import os
import os.path as osp
import numpy as np
import torch  # put this before scipy import
from scipy.misc import imread, imresize
import sys

from vmrn_msgs.srv import MAttNetGrounding, MAttNetGroundingResponse

from MAttNet.tools.mattnet import MattNet

# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

class mattnet_server(object):
    def __init__(self):
        self._classes = ['__background__',  # always index 0
                         'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                         'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                         'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                         'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                         'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

        # arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='refcoco',
                            help='dataset name: refclef, refcoco, refcoco+, refcocog')
        parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
        parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
        parser.add_argument('--id', type=int, default='1', help='gpuid')
        args = parser.parse_args('')

        # init mattnet
        self.mattnet = MattNet(args)

        s = rospy.Service('mattnet_server', MAttNetGrounding, self.mattnet_serv_callback)
        print("Ready to ground object.")

    def mattnet_serv_callback(self, req):
        img_msg = req.img
        img = br.imgmsg_to_cv2(img_msg)
        bboxes = req.bbox
        cls = req.cls
        expr = req.expr
        cls_names = [self._classes[i] for i in cls]

        # forward image
        bboxes = np.array(bboxes).reshape(-1, 4)
        cls = np.array(cls).reshape(-1,1)

        bboxes = np.concatenate([bboxes, cls], -1)
        with torch.no_grad():
            img_data = self.mattnet.forward_image_with_bbox(img, bboxes=bboxes, classes=cls_names)
            entry, score = self.mattnet.comprehend(img_data, expr)
        torch.cuda.empty_cache()

        res = MAttNetGroundingResponse()
        res.ground_prob = score
        return res


if __name__=="__main__":
    rospy.init_node('mattnet_server')
    mattnet_server()
    rospy.spin()