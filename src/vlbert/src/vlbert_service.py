import rospy
import cv2
from cv_bridge import CvBridge
import argparse
import os
import os.path as osp
import numpy as np
import sys
import json

from vmrn_msgs.srv import *

class vlbert_server(object):
    def __init__(self):
        # init vlbert
        # TODO

        s = rospy.Service('vlbert_service', VLBert, self.vlbert_callback)
        print("Ready to run vlbert!.")

    def vlbert_callback(self, req):
        expr = req.expr
        img_msg = req.img_msg
        bboxes = req.bboxes

        # run inference
        # TODO
        # grounding_scores, rel_score_mat, rel_mat, grasps 

        resp = VLBertResponse()
        resp.grounding_scores = grounding_scores
        resp.rel_score_mat = rel_score_mat
        resp.rel_mat = rel_mat

        return resp

if __name__=="__main__":
    rospy.init_node('mattnet_server')
    vlbert_server()
    rospy.spin()