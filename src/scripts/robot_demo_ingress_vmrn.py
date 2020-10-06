#!/usr/bin/env python
import warnings
try:
    NO_ROS = False
    from libraries.ros_utils.baxter_api import *
    from libraries.ros_utils.calibrate import calibrate_kinect
    from libraries.ros_utils.kinect_subscriber import kinect_reader
except:
    NO_ROS = True
    warnings.warn("Baxter interface not available")

import sys
sys.path.append("..")
import rospy
from libraries.data_viewer.data_viewer import DataViewer
from libraries.data_viewer.paper_fig_generator import gen_paper_fig
from invigorate.invigorate import Invigorate

import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
import os

import time
import datetime
# from stanfordcorenlp import StanfordCoreNLP

br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')
from invigorate.invigorate import Invigorate
from config.config import CLASSES

YCROP = (250, 650)
XCROP = (650, 1050)

def execute_action(a):
    raw_input("Excuted?")

def init_robot():
    ###### init robot
    print("Enabling robot... ")
    rs = init_baxter_robot()
    move_limb_to_neutral("left")
    move_limb_to_neutral("right")
    return rs

def init_kinect():
    ###### init kinect
    camera_cfg = {'imgtype': 'hd'}
    kinect1 = kinect_reader(camera_cfg)
    rospy.sleep(2)
    is_cali = 0
    if is_cali:
        robot_coordinate = np.array([0.445995351324, 0.125117819286, -0.137774866033,
                                     0.456399916939,-0.0797581793528, -0.135652335083,
                                     0.644892941006, 0.135176302844, -0.13299852714,
                                     0.658655809167, -0.0656790015993, 0.0628109954456])
        calibrate_kinect(robot_coordinate, kinect1)
        transmat = np.loadtxt('rosapi/trans_mat.txt')
    else:
        transmat = np.loadtxt('rosapi/trans_mat.txt')
    return kinect1, transmat

def read_img(running_mode, kinect=None):

    def read_realtime_img(kinect):
        img, depth = kinect.get_image()
        depth = depth % 4096
        img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
        depth = depth[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
        return img, depth

    # read image
    if running_mode == "demo":
        img = cv2.imread("images/17.png")
        depth = None
    elif running_mode == "kinect":
        assert kinect is not None
        img, depth = read_realtime_img(kinect)
    else:
        raise RuntimeError
    return img, depth

def main(s_ing_client):
    running_mode = "demo"

    kinect1 = None
    if not NO_ROS:
        rs = init_robot()
        kinect1, transmat = init_kinect()
    else:
        warnings.warn("No ros packages are available, demo mode is on.")
        running_mode = "demo"
    expr = raw_input("Please tell me what you want: ")
    related_classes = [cls for cls in CLASSES if cls in expr or expr in cls]

    # outer-loop planning: in each step, grasp the leaf-descendant node.
    while (True):
        img, depth = read_img(running_mode, kinect1)
        a, is_end = s_ing_client.decision_making_heuristic(img, expr, related_classes)
        # execute grasping action
        execute_action(a)
        # TODO: Determine whether the grasp is successful and then assigh this "removed" flag
        # s_ing_client.object_pool[ind_match[a % num_box]]["removed"] = True
        if is_end:
            break

    gen_paper_fig(expr, s_ing_client.result_container)

if __name__ == '__main__':
    s_ing_client = Invigorate()
    main(s_ing_client)

