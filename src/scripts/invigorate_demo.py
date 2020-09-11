#!/usr/bin/env python

import _init_path
import warnings
import sys
import rospy
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
import os
import time
import datetime
from PIL import Image
# from stanfordcorenlp import StanfordCoreNLP

from invigorate import *
from libraries.data_viewer.data_viewer import DataViewer
import libraries.caption_generator import caption_generator
import libraries.robots as robots

# -------- Settings --------
ROBOT = 'Dummy'

GENERATE_CAPTIONS = True

# -------- Constants --------
YCROP = (250, 650)
XCROP = (650, 1050)

# ------- Statics -----------

br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

def init_robot(robot):
    if robot == 'Dummy':
        robot = robots.dummy_robot.DummyRobot()
    elif robot == 'Fetch':
        robot = robots.fetch_robot.FetchRobot()
    else:
        raise Exception('robot not recognized!!')

    return robot

def main():
    rospy.init_node('INTEGRASE', anonymous=True)

    invigorate_client = Invigorate()
    data_viewer = DataViewer(CLASSES)
    robot = init_robot()

    expr = robot.listen()

    all_results = []
    to_grasp = False
    action = -1
    grasps = None
    quesetion_str = None
    answer = None
    while (True):
        if to_grasp:
            # after grasping, perceive new images
            img, _ = robot.read_imgs()

            # perception
            observations = invigorate_client.perceive_img(img, expr)
            num_box = observations['bboxes'].shape[0]
            vis_rel_score_mat = data_viewer.relscores_to_visscores()

            # state_estimation
            invigorate_client.estimate_state_with_observation(observations)
        else:
            # get user answer
            answer = robot.listen()
            invigorate_client.estimate_state_with_user_clue(action, answer)

        action = invigorate_client.plan() # action_idx.
        action_type = invigorate_client.get_action_type(action)
        to_grasp = False
        if action_type == 'GRASP_AND_END':
            grasp_target_idx = action
            print("Grasping object " + str(grasp_target_idx) + " and ending the program")
            to_grasp = True
        elif action_type == 'GRASP_AND_CONTINUE': # if it is a grasp action
            grasp_target_idx = action - num_box
            print("Grasping object " + str(grasp_target_idx) + " and continuing")
            to_grasp = True
        elif action_type == 'Q1':
            target_idx = action_type - 2 * num_box
            if GENERATE_CAPTIONS:
                # generate caption
                bboxes = observations['bboxes']
                caption = caption_generator.generate_caption(img, bboxes, target_idx)
                question_str = Q1["type1"].format(caption)
            else:
                question_str = Q1["type1"].format(str(target_idx) + "th object")
        else: # action type is Q2
            if target_prob[-1] == 1:
                question_str = Q2["type2"]
            elif (target_prob[:-1] > 0.02).sum() == 1:
                target_idx = np.argmax(target_prob[:-1])
                if GENERATE_CAPTIONS:
                    # generate caption
                    caption = caption_generator.generate_caption(img, bboxes, target_idx)
                    question_str = Q2["type3"].format(caption)
                else:
                    question_str = Q2["type3"].format(target_idx + "th object")
            else:
                question_str = Q2["type1"]

        if to_grasp:
            # display grasp
            im = data_viewer.display_obj_to_grasp(img.copy(), bboxes, grasp_target_idx)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im)
            im_pil.show()

            # execute grasping action
            grasps = observations['grasps']
            grasp = grasps[action % num_box][:8] + np.tile([XCROP[0], YCROP[0]], 4)
            robot.grasp(grasp)
            # TODO: Determine whether the grasp is successful and then assign this "removed" flag
            invigorate_client.transit_state(action)
        else:
            robot.say(question_str)

        # generate debug images
        img = observations['img']
        bboxes = observations['bboxes']
        rel_mat = observations['rel_mat']
        rel_score_mat = observations['rel_score_mat']
        target_prob = invigorate_client.belief['target_prob']
        data_viewer.save_visualization_imgs(img, bboxes, rel_mat, rel_score_mat, expr, target_prob, action, grasps.copy(), question_str, answer)

        to_cont = raw_input('To_continue?')
        if to_cont != 'y':
            break

if __name__ == '__main__':
    main()