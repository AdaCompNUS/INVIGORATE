from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
import cv2
from cv_bridge import CvBridge
import argparse
import os
import os.path as osp
import numpy as np
import torch  # put this before scipy import
from scipy.misc import imread, imresize
import sys
import json

from mattnet_v2 import MattNetV2
from invigorate_msgs.srv import MAttNetGroundingV2, MAttNetGroundingV2Response

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
        parser.add_argument('--dataset', type=str, default='refcoco_small',
                            help='dataset name: refclef, refcoco, refcoco+, refcocog')
        parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
        parser.add_argument('--model_id', type=str, default='rcnn_cmr_with_st_from_pretrained', help='model id name')
        parser.add_argument('--id', type=int, default='1', help='gpuid')
        args = parser.parse_args('')

        # init mattnet
        self.mattnet = MattNetV2(args)

        s = rospy.Service('mattnet_server_v2', MAttNetGroundingV2, self.mattnet_serv_callback)
        print("Ready to ground object.")

    def mattnet_serv_callback(self, req):
        expr = req.expr
        img_data = json.loads(req.img_data)

        with torch.no_grad():
            entry = self.mattnet.comprehend(img_data, expr)
        torch.cuda.empty_cache()

        res = MAttNetGroundingV2Response()
        print(entry['overall_scores'])
        res.ground_scores = entry['overall_scores']
        return res


if __name__=="__main__":
    rospy.init_node('mattnet_server')
    mattnet_server()
    rospy.spin()