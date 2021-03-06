#!/usr/bin/env python

'''
TODO
P1:
*. compare kinect
*. Try higher resolution realsense [Done, it does not work]
*. Manually adjust position and check it reports so many collision, knife, toothbrush

P2
*. question answering for where is it? do not constrain clue to "it is under sth"
*. Fix collision checking for case 2 -> route to the center of the body
*. collision checking against other objects.
*. Only genenrate captions against relavant context object
*. The target probability does not persist to the next iteration*
*. object persistency issue. what if it does not get detected in one iteration. <To test>
*. name filter
    *. soft?
    *. can't filter out if det_score is high
    *. name filter before mattnet grounding
*. MATTnet groudning score:
    *. mattnet can't handle the case where true white box is not detected but false black box is detected.
    *. only one object in scene. Mattnet always not sure.
* Handling of user answer "no"??
* Can't handle "it is under the apple on the right"
* Can't handle "yellow cup under the mouse"
*. Mattnet can't really differentiate between context object and target object
*. handle "remote controller under the blue box"
'''

import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import warnings
import rospy
import cv2
from cv_bridge import CvBridge
# import torch
import numpy as np
import os
import time
import datetime
from PIL import Image
# from stanfordcorenlp import StanfordCoreNLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import logging

from config.config import *
from libraries.data_viewer.data_viewer import DataViewer
from invigorate.invigorate import Invigorate
# from invigorate.baseline import Baseline
from invigorate.greedy import Greedy
from invigorate.heuristic import Heuristic
from invigorate.no_interaction import NoInteraction
from invigorate.no_multistep import NoMultistep, NoMultistepAll
from libraries.robots.dummy_robot import DummyRobot
from libraries.utils.log import LOGGER_NAME

# -------- Settings --------
ROBOT = 'Fetch'
# ROBOT = 'Dummy'ap
GENERATE_CAPTIONS = False
DISPLAY_DEBUG_IMG = True

if GENERATE_CAPTIONS:
    from libraries.caption_generator import caption_generator

if ROBOT == 'Fetch':
    from libraries.robots.fetch_robot import FetchRobot

# -------- Constants --------
EXEC_GRASP = 0
EXEC_ASK = 1
EXEC_DUMMY_ASK = 2
# DISPLAY_DEBUG_IMG = "matplotlib"
DISPLAY_DEBUG_IMG = 'pil'
DEBUG = True

# ------- Statics -----------
logger = logging.getLogger(LOGGER_NAME)
br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

def init_robot(robot):
    if robot == 'Dummy':
        robot = DummyRobot()
    elif robot == 'Fetch':
        robot = FetchRobot()
    else:
        raise Exception('robot not recognized!!')

    return robot

def main():
    rospy.init_node('INVIGORATE', anonymous=True)

    if EXP_SETTING == "invigorate":
        invigorate_client = Invigorate()
    elif EXP_SETTING == 'greedy':
        invigorate_client = Greedy()
    elif EXP_SETTING == 'heuristic':
        invigorate_client = Heuristic()
    elif EXP_SETTING == "no_interaction":
        invigorate_client = NoInteraction()
    elif EXP_SETTING == "no_multistep":
        invigorate_client = NoMultistep()
    elif EXP_SETTING == "no_multistep_all":
        invigorate_client = NoMultistepAll()

    logger.info("SETTING: {}".format(EXP_SETTING))

    data_viewer = DataViewer(CLASSES)
    robot = init_robot(ROBOT)

    # get user command
    expr = robot.listen()

    all_results = []
    exec_type = EXEC_GRASP
    action = -1
    grasps = None
    question_str = None
    answer = None
    dummy_question_answer = None
    to_end = False
    first_time = True
    while not to_end:
        logger.info("----------------------------------------------------------------------------------------")
        logger.info("Start of iteration")

        if exec_type == EXEC_GRASP:
            # after grasping, perceive new images
            img, _ = robot.read_imgs()

            # NOTE: only applicable for EXPERIMENT mode. Ensure the first picture is the same for all baselines!
            if first_time and MODE == EXPERIMENT:
                origin_img_path = osp.join(EXP_RES_DIR, "../origin.png")
                # if EXP_SETTING == "greedy":
                #     cv2.imwrite(origin_img_path, img) # if greedy, write image
                # else:
                img = cv2.imread(origin_img_path) # for others, read image
                first_time = False

            # state_estimation
            res = invigorate_client.estimate_state_with_img(img, expr)
            if not res:
                logger.info("exit!!")
                return
        elif exec_type == EXEC_ASK:
            # get user answer
            answer = robot.listen()

            # state_estimation
            invigorate_client.estimate_state_with_user_answer(action, answer)
        elif exec_type == EXEC_DUMMY_ASK:
            # get user answer
            answer = dummy_question_answer

            # state_estimation
            invigorate_client.estimate_state_with_user_answer(action, answer)
        else:
            raise RuntimeError('Invalid exec_type')

        # debug
        img = invigorate_client.img
        bboxes = invigorate_client.belief['bboxes']
        num_obj = bboxes.shape[0]
        classes = invigorate_client.belief['classes']
        # rel_mat = observations['rel_mat']
        rel_score_mat = invigorate_client.belief['rel_prob']
        rel_mat, _ = invigorate_client.rel_score_process(rel_score_mat)
        target_prob = invigorate_client.belief['target_prob']
        # imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, save=False)
        # if DISPLAY_DEBUG_IMG:
        #     data_viewer.display_img(imgs['final_img'], mode=DISPLAY_DEBUG_IMG)
        # cv2.imwrite("outputs/final.png", imgs['final_img'])

        # plan for optimal actions
        action = invigorate_client.plan_action() # action_idx.
        action_type, target_idx = invigorate_client.parse_action(action, num_obj)

        to_grasp = False
        if action_type == 'GRASP_AND_END':
            logger.info("Grasping object " + str(target_idx) + " and ending the promgram")
            exec_type = EXEC_GRASP
            to_end = True
        elif action_type == 'GRASP_AND_CONTINUE':
            logger.info("Grasping object " + str(target_idx) + " and continuing")
            exec_type = EXEC_GRASP
        elif action_type == 'Q1':
            logger.info("Askig Q1 about " + str(target_idx) + " and continuing")
            if GENERATE_CAPTIONS:
                # generate caption
                subject = invigorate_client.subject[-1]
                caption = caption_generator.generate_caption(img, bboxes, classes, target_idx, subject)
                question_str = Q1["type1"].format(caption)
            else:
                question_str = Q1["type1"].format(str(target_idx) + "th object")
            exec_type = EXEC_ASK

        # debug
        if DISPLAY_DEBUG_IMG:
            imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, action=action,
                question_str=question_str, save=False)
            data_viewer.display_img(imgs['final_img'], mode=DISPLAY_DEBUG_IMG)

        # exec action
        if exec_type == EXEC_GRASP:
            grasps = invigorate_client.belief['grasps']
            logger.debug("grasps.shape {}".format(grasps.shape))
            object_name = CLASSES[classes[target_idx][0]]
            is_target = (action_type == 'GRASP_AND_END')

            # display grasp
            im = data_viewer.display_obj_to_grasp(img.copy(), bboxes, grasps, target_idx)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite("outputs/grasp.png", im)
            if DISPLAY_DEBUG_IMG is not None:
                if DISPLAY_DEBUG_IMG == "matplotlib":
                    plt.figure(DISPLAY_FIGURE_ID)
                    plt.axis('off')
                    plt.imshow(im)
                    plt.show()
                elif DISPLAY_DEBUG_IMG == "pil":
                    im_pil = Image.fromarray(im)
                    im_pil.show()

            # say
            # TODO generate caption??
            if not is_target:
                robot.say("I will have to grasp the {} first".format(object_name))
            else:
                robot.say("now I can grasp the {}".format(object_name))

            # execute grasping action
            grasp = grasps[action % num_obj][:8]
            res = robot.grasp(grasp, is_target=is_target)
            if not res:
                logger.error('grasp failed!!!')
                if not is_target:
                    robot.say("sorry I can't grasp the {}, could you help me remove it?".format(object_name))
                else:
                    robot.say("sorry I can't grasp the {}, but it is for you".format(object_name))
                # rospy.sleep(5)
            if res and is_target:
                robot.say("this is for you")
        elif exec_type == EXEC_ASK:
            robot.say(question_str)
            # exec_type = EXEC_GRASP # TEST

        # transit state
        invigorate_client.transit_state(action)

        # generate debug images
        data_viewer.gen_final_paper_fig(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, action, grasps, question_str, answer)

        if DEBUG:
            to_cont = raw_input('To_continue?')
            if to_cont == 'q':
                break

    print("exit!")
    # rospy.sleep(10) # wait 10 second
    robot.move_arm_to_home()

if __name__ == '__main__':
    main()