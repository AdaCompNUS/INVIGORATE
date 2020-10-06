#!/usr/bin/env python

'''
TODO
*. question answering for where is it?
*. grasp random objects when background has high prob?
*. User answer confirmed object bug <To test>
*. Fix collision checking for case 2 -> route to the center of the body
*. accelerate grasp collision check by
    *. running on GPU
    *. Further segment pc <resolved>
    *. greedy algo <resolved>
*. mattnet can't handle the case where true white box is not detected but false black box is detected.
*. grasp sequence bug
*. object persistency
'''

import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import warnings
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
import matplotlib
matplotlib.use('Agg')
import nltk

from config.config import *
from libraries.data_viewer.data_viewer import DataViewer
from invigorate.invigorate import Invigorate
from libraries.caption_generator import caption_generator
from libraries.robots.dummy_robot import DummyRobot

# -------- Settings --------
ROBOT = 'Fetch'
GENERATE_CAPTIONS = True
DISPLAY_DEBUG_IMG = True

if ROBOT == 'Fetch':
    from libraries.robots.fetch_robot import FetchRobot

# -------- Constants --------
EXEC_GRASP = 0
EXEC_ASK = 1
EXEC_DUMMY_ASK = 2

# ------- Statics -----------

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

def process_user_command(command, nlp_server="nltk"):
    if nlp_server == "nltk":
        text = nltk.word_tokenize(command)
        pos_tags = nltk.pos_tag(text)
    else:
        doc = self.stanford_nlp_server(command)
        pos_tags = [(d.text, d.xpos) for d in doc.sentences[0].words]

    # the object lies after the verb
    verb_ind = -1
    for i, (token, postag) in enumerate(pos_tags):
        if postag.startswith("VB"):
            verb_ind = i

    particle_ind = -1
    for i, (token, postag) in enumerate(pos_tags):
        if postag in {"RP"}:
            particle_ind = i

    ind = max(verb_ind, particle_ind)
    clue_tokens = [token for (token, _) in pos_tags[ind+1:]]
    clue = ' '.join(clue_tokens)
    print("Processed clue: {:s}".format(clue if clue != '' else "None"))

    return clue

def main():
    rospy.init_node('INVIGORATE', anonymous=True)
    invigorate_client = Invigorate()
    data_viewer = DataViewer(CLASSES)
    robot = init_robot(ROBOT)

    # get user command
    expr = robot.listen()
    expr = process_user_command(expr)

    all_results = []
    exec_type = EXEC_GRASP
    action = -1
    grasps = None
    question_str = None
    answer = None
    dummy_question_answer = None
    to_end = False
    while not to_end:
        print("------------------------")
        print("Start of iteration")

        if exec_type == EXEC_GRASP:
            # after grasping, perceive new images
            img, _ = robot.read_imgs()

            # perception
            observations = invigorate_client.perceive_img(img, expr)
            if observations is None:
                print("WARNING: nothing is detected, abort!!!")
                break
            num_box = observations['bboxes'].shape[0]

            # state_estimation
            invigorate_client.estimate_state_with_observation(observations)
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
        img = observations['img']
        bboxes = observations['bboxes']
        classes = observations['classes']
        rel_mat = observations['rel_mat']
        rel_score_mat = observations['rel_score_mat']
        target_prob = invigorate_client.belief['target_prob']
        imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, save=False)
        if DISPLAY_DEBUG_IMG:
            data_viewer.display_img(imgs['final_img'])
        cv2.imwrite("outputs/final.png", imgs['final_img'])

        # plan for optimal actions
        action = invigorate_client.decision_making_heuristic() # action_idx.
        action_type = invigorate_client.get_action_type(action)

        to_grasp = False
        if action_type == 'GRASP_AND_END':
            grasp_target_idx = action
            print("Grasping object " + str(grasp_target_idx) + " and ending the program")
            exec_type = EXEC_GRASP
            to_end = True
        elif action_type == 'GRASP_AND_CONTINUE': # if it is a grasp action
            grasp_target_idx = action - num_box
            print("Grasping object " + str(grasp_target_idx) + " and continuing")
            exec_type = EXEC_GRASP
        elif action_type == 'Q1':
            target_idx = action - 2 * num_box
            if GENERATE_CAPTIONS:
                # generate caption
                caption = caption_generator.generate_caption(img, bboxes, classes, target_idx)
                question_str = Q1["type1"].format(caption)
            else:
                question_str = Q1["type1"].format(str(target_idx) + "th object")
            exec_type = EXEC_ASK
        else: # action type is Q2
            if invigorate_client.clue is not None:
                # special case.
                dummy_question_answer = invigorate_client.clue
                question_str = ''
                exec_type = EXEC_DUMMY_ASK
            elif target_prob[-1] == 1:
                question_str = Q2["type2"]
                exec_type = EXEC_ASK
            elif (target_prob[:-1] > 0.02).sum() == 1:
                target_idx = np.argmax(target_prob[:-1])
                if GENERATE_CAPTIONS:
                    # generate caption
                    caption = caption_generator.generate_caption(img, bboxes, classes, target_idx)
                    question_str = Q2["type3"].format(caption)
                else:
                    question_str = Q2["type3"].format(target_idx + "th object")
                exec_type = EXEC_ASK
            else:
                question_str = Q2["type1"]
                exec_type = EXEC_ASK

        # exec action
        if exec_type == EXEC_GRASP:
            grasps = observations['grasps']
            print("grasps.shape {}".format(grasps.shape))
            object_name = CLASSES[classes[grasp_target_idx][0]]
            is_target = (action_type == 'GRASP_AND_END')

            # display grasp
            im = data_viewer.display_obj_to_grasp(img.copy(), bboxes, grasps, grasp_target_idx)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite("outputs/grasp.png", im)
            if DISPLAY_DEBUG_IMG:
                im_pil = Image.fromarray(im)
                im_pil.show()

            # say
            # TODO generate caption??
            if not is_target:
                robot.say("I will have to grasp the {} first".format(object_name))
            else:
                robot.say("now I can grasp the {}".format(object_name))

            # execute grasping action
            grasp = grasps[action % num_box][:8]
            res = robot.grasp(grasp, is_target=is_target)
            if not res:
                print('grasp failed!!!')
                if not is_target:
                    robot.say("sorry I can't grasp the {}, could you help me remove it?".format(object_name))
                else:
                    robot.say("sorry I can't grasp the {}, but it is for you".format(object_name))
                rospy.sleep(5)
            if res and is_target:
                robot.say("this is for you")
        elif exec_type == EXEC_ASK:
            robot.say(question_str)

        # transit state
        invigorate_client.transit_state(action)

        # generate debug images
        img = observations['img']
        bboxes = observations['bboxes']
        classes = observations['classes']
        rel_mat = observations['rel_mat']
        rel_score_mat = observations['rel_score_mat']
        target_prob = invigorate_client.belief['target_prob']
        data_viewer.gen_final_paper_fig(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, action, grasps, question_str, answer)

        # to_cont = raw_input('To_continue?')
        # if to_cont != 'y':
        #     break

    print("exit!")
    rospy.sleep(10) # wait 10 second
    robot.move_arm_to_home()

if __name__ == '__main__':
    main()