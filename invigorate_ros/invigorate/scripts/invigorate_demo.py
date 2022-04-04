#!/usr/bin/env python

'''
TODO
P1:
*. compare kinect [Done, it works better]
*. Try higher resolution RealSence [Done, it does not work]
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
from invigorate_models.invigorate_rss import Invigorate
# from invigorate.baseline import Baseline
from invigorate_models.greedy import Greedy
from invigorate_models.heuristic import Heuristic
from invigorate_models.no_interaction import NoInteraction
from invigorate_models.no_multistep import NoMultistep, NoMultistepAll
from invigorate_models.invigorate_ijrr import InvigorateIJRR
from invigorate_models.invigorate_ijrr_v2 import InvigorateIJRRV2
from invigorate_models.invigorate_ijrr_v3 import InvigorateIJRRV3
from invigorate_models.invigorate_ijrr_v4 import InvigorateIJRRV4
from invigorate_models.invigorate_ijrr_v5 import InvigorateIJRRV5
from invigorate_models.invigorate_ijrr_v6 import InvigorateIJRRV6
from invigorate_models.invigorate_ijrr_no_point import InvigorateIJRRNoPoint
from invigorate_models.invigorate_ijrr_point import InvigorateIJRRPoint
from invigorate_models.invigorate_ijrr_point_old_caption import InvigorateIJRRPointOldCaption
from libraries.robots.dummy_robot import DummyRobot
from libraries.utils.log import LOGGER_NAME

# -------- Settings --------
ROBOT = 'Dummy'
# ROBOT = 'Dummy'
GENERATE_CAPTIONS = True
DISPLAY_DEBUG_IMG = True

if GENERATE_CAPTIONS:
    from libraries.caption_generator import caption_generator

if ROBOT == 'Fetch':
    from libraries.robots.fetch_robot import FetchRobot

# -------- Constants --------

# DISPLAY_DEBUG_IMG = "matplotlib"
DISPLAY_DEBUG_IMG = 'pil'
DEBUG = True

# ------- Statics -----------
logger = logging.getLogger(LOGGER_NAME)
br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

def init_robot(robot):
    if robot == 'Dummy':
        robot = DummyRobot(camera_topic='/kinect2/qhd/image_color')
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
    elif EXP_SETTING == "invigorate_ijrr":
        invigorate_client = InvigorateIJRR()
    elif EXP_SETTING == "invigorate_ijrr_v2":
        invigorate_client = InvigorateIJRRV2()
    elif EXP_SETTING == "invigorate_ijrr_v3":
        invigorate_client = InvigorateIJRRV3()
    elif EXP_SETTING == "invigorate_ijrr_v4":
        invigorate_client = InvigorateIJRRV4()
    elif EXP_SETTING == "invigorate_ijrr_v5":
        invigorate_client = InvigorateIJRRV5()
    elif EXP_SETTING == "invigorate_ijrr_v6":
        invigorate_client = InvigorateIJRRV6()
    elif EXP_SETTING == "invigorate_ijrr_no_pointing":
        invigorate_client = InvigorateIJRRPoint()
    elif EXP_SETTING == "invigorate_ijrr_old_caption":
        invigorate_client = InvigorateIJRRPointOldCaption()
    else:
        raise "exp setting not recognized!!"

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
    grasp_num = 0
    step_num = 0
    load_img = False
    while not to_end:
        logger.info("----------------------------------------------------------------------------------------")
        logger.info("Start of iteration")

        if exec_type == EXEC_GRASP:
            # after grasping, perceive new images
            load_img = False
            if MODE == EXPERIMENT and grasp_num >= 0:
                tmp = raw_input("load original img?")
                if tmp == 'y':
                    load_img = True
                    if EXP_SETTING == 'invigorate_ijrr_no_pointing' or EXP_SETTING == 'invigorate_ijrr_old_caption':
                        img_dir = osp.join(EXP_DATA_DIR, "../../result/{}/8".format((PARTICIPANT_NUM-1) * 10 + SCENE_NUM))
                    else:
                        img_dir = osp.join(EXP_DATA_DIR, "../../experiment/participant {}/{}/4".format(PARTICIPANT_NUM, SCENE_NUM))

                    # logger.info("read from: {}".format(img_dir))
                    img_list = os.listdir(img_dir)
                    img_list = [i for i in img_list if "origin" in i]
                    img_list = sorted(img_list)

            if load_img:
                img = cv2.imread(osp.join(img_dir, img_list[step_num]))
            else:
                img, _ = robot.read_imgs()

            # NOTE: only applicable for EXPERIMENT mode. Ensure the first picture is the same for all baselines!
            # if first_time and MODE == EXPERIMENT:
            #     # origin_img_path = osp.join(img_dir, "origin.png")
            #     origin_img_path = osp.join(EXP_DATA_DIR, "origin.png")
            #     # if EXP_SETTING == "greedy":
            #     #     cv2.imwrite(origin_img_path, img) # if greedy, write image
            #     # else:
            #     img = cv2.imread(origin_img_path) # for others, read image
            #     first_time = False

            #     # origin_img_path = osp.join(EXP_DATA_DIR, "origin.png")

            grasp_num += 1

            # state_estimation
            res = invigorate_client.estimate_state_with_img(img, expr)
            if not res:
                logger.info("exit!!")
                return
        elif exec_type in {EXEC_ASK_WITH_POINTING, EXEC_ASK_WITHOUT_POINT}:
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
            logger.info("Asking about and Pointing to {:s}th Object".format(str(target_idx)) + " and continuing")
            if GENERATE_CAPTIONS:
                # generate caption
                subject = invigorate_client.subject[-1]
                caption = caption_generator.generate_caption(
                    img, bboxes, classes, target_idx, subject)
                question_str = Q1["type1"].format(caption)
            else:
                question_str = Q1["type1"].format(str(target_idx) + "th object")
            logger.info("Generated Question: {:s}".format(question_str))
            exec_type = EXEC_ASK_WITH_POINTING
        elif action_type == 'Q_IJRR':
            caption = invigorate_client.belief["questions"][target_idx]
            if hasattr(invigorate_client, 'search_answer'):
                answer = invigorate_client.search_answer(caption)
                if answer is not None:
                    exec_type = EXEC_DUMMY_ASK
                    dummy_question_answer = answer
                else:
                    exec_type = EXEC_ASK_WITHOUT_POINT
            else:
                exec_type = EXEC_ASK_WITHOUT_POINT

            # HACK: form the object-agnostic question with 'a' instead of 'the'
            caption = caption.replace('the ', 'a ', 1)
            # HACK: to make the caption more natural
            caption = caption.replace('the right', 'right side')
            caption = caption.replace('the left', 'left side')
            caption = caption.replace('the top', 'far')
            caption = caption.replace('the bottom', 'bottom')
            question_str = Q1["type1"].format(caption)
            logger.info("Only Askig Question: {:s}".format(question_str))

        elif action_type == 'Q_IJRR_WITH_POINTING':
            print(target_idx, invigorate_client.belief["questions"])

            caption = invigorate_client.belief["questions"][target_idx]
            # HACK: We want the robot ask the question like `Do you mean this red apple?' when pointing to some object.
            caption = caption.replace('the ', 'this ', 1)
            question_str = Q1["type1"].format(caption)
            logger.info("Asking about and Pointing to {:s}th Object".format(str(target_idx)) + " and continuing")
            logger.info("Generated Question: {:s}".format(question_str))
            exec_type = EXEC_ASK_WITH_POINTING

        # debug
        if hasattr(invigorate_client, 'pos_expr'):
            expr = invigorate_client.pos_expr
        else:
            expr = invigorate_client.expr[0]
        imgs = data_viewer.generate_visualization_imgs(
            img, bboxes, classes, rel_mat, rel_score_mat,
            expr, target_prob,
            action=action, action_type=action_type,
            exec_type=exec_type, target_idx=target_idx,
            question_str=question_str, save=True)
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
        elif exec_type == EXEC_ASK_WITH_POINTING:
            # TODO: IMPLEMENT POINTING POSTURES
            robot.say(question_str)
        elif exec_type == EXEC_ASK_WITHOUT_POINT:
            robot.say(question_str)
            # exec_type = EXEC_GRASP # TEST

        # transit state
        invigorate_client.transit_state(action)

        # generate debug images
        data_viewer.gen_final_paper_fig(imgs, expr)

        if EXP_SETTING in {"invigorate_ijrr_v2", "invigorate_ijrr_v3"}:
            for k, v in invigorate_client.timers.items():
                print(k, sum(v) / len(v))

        if DEBUG:
            to_cont = raw_input('To_continue?')
            if to_cont == 'q':
                break

        step_num += 1

    print("exit!")
    # rospy.sleep(10) # wait 10 second
    robot.move_arm_to_home()

if __name__ == '__main__':
    main()