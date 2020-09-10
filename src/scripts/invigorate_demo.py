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
    data_viewer = DataViewer(classes)
    robot = init_robot()

    expr = robot.listen()

    all_results = []

    while (True):
        if to_perceive:
            img, _ = robot.read_imgs()

            # perception
            observations = invigorate_client.perceive_img(img, expr)
            num_box = observations['bboxes'].shape[0]
            vis_rel_score_mat = relscores_to_visscores(observations['rel_score_mat'])

            # state_estimation
            invigorate_client.estimate_state_with_observation(observations)
        else:
            clue = 
            invigorate_client.estimate_state_with_user_clue(clue)

        best_action = inner_loop_planning(belief) # action_idx.
        if a < num_box:
            grasp_target_idx = a
            print("Grasping object " + str(grasp_target_idx) + " and ending the program")
        elif a < 2 * num_box: # if it is a grasp action
            grasp_target_idx = a - num_box
            print("Grasping object " + str(grasp_target_idx) + " and continuing")
        else:
            if a < 3 * num_box:
                target_idx = a - 2 * num_box
                if GENERATE_CAPTIONS:
                    # generate caption
                    caption = caption_generator.generate_caption(img, bboxes, target_idx)
                    question_str = Q1["type1"].format(caption)
                else:
                    question_str = Q1["type1"].format(str(target_idx) + "th object")
            else:
                if target_prob[-1] == 1:
                    question_str = Q2["type2"]
                elif (target_prob[:-1] > 0.02).sum() == 1:
                    target_idx = np.argmax(target_prob[:-1])
                    if GENERATE_CAPTIONS:
                        # generate caption
                        caption = caption_generator.generate_caption(img, bboxes, target_idx)
                        question_str = Q1["type1"].format(caption)
                    else:
                        question_str = Q2["type3"].format(target_idx + "th object")
                else:
                    question_str = Q2["type1"]
            
            robot.say(question_str)

            data = {"img": img,
                    "bbox": bboxes[:, :4].reshape(-1).tolist(),
                    "cls": bboxes[:, 4].reshape(-1).tolist(),
                    "mapping": ind_match}
            ans = robot.listen()

            if a < 3 * num_box:
                # we use binary variables to encode the answer of q1 questions.
                if ans in {"yes", "yeah", "yep", "sure"}:
                    for i, v in enumerate(s_ing_client.object_pool):
                        if i == ind_match[a - 2 * num_box]:
                            s_ing_client.object_pool[i]["is_target"] = True
                        else:
                            s_ing_client.object_pool[i]["is_target"] = False
                else:
                    obj_ind = ind_match[a - 2 * num_box]
                    s_ing_client.object_pool[obj_ind]["is_target"] = False
            else:
                if ans in {"yes", "yeah", "yep", "sure"}:
                    # s_ing_client.target_in_pool = True
                    # for i, v in enumerate(s_ing_client.object_pool):
                    #     if i not in ind_match.values():
                    #         s_ing_client.object_pool[i]["is_target"] = False
                    if (target_prob[:-1] > 0.02).sum() == 1:
                        obj_ind = ind_match[np.argmax(target_prob[:-1])]
                        s_ing_client.object_pool[obj_ind]["is_target"] = True
                else:
                    # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                    ans = ans[6:]
                    # for i in ind_match.values():
                    #     s_ing_client.object_pool[i]["is_target"] = False
                    s_ing_client.clue = ans
                    if (target_prob[:-1] > 0.02).sum() == 1:
                        obj_ind = ind_match[np.argmax(target_prob[:-1])]
                        s_ing_client.object_pool[obj_ind]["is_target"] = False
            belief = s_ing_client.update_belief(belief, a, ans, data)
            # TODO
            # inner_loop_results[-1]["answer"] = split_long_string("User's Answer: " + ans.upper())

        # display grasp
        im = data_viewer.display_obj_to_grasp(img.copy(), bboxes, grasp_target_idx)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im)
        im_pil.show()

        # execute grasping action
        grasp = grasps[a % num_box][:8] + np.tile([XCROP[0], YCROP[0]], 4)
        robot.grasp(grasp)
        # TODO: Determine whether the grasp is successful and then assign this "removed" flag
        s_ing_client.object_pool[ind_match[a % num_box]]["removed"] = True

        # if a < num_box:
        #     break

        # generate debug images
        data_viewer.save_visualization_imgs(img, bboxes, rel_mat, vis_rel_score_mat, expr, target_prob, a, data_viewer, grasps.copy(), question_str))

        to_cont = raw_input('To_continue?')
        if to_cont != 'y':
            break

if __name__ == '__main__':
    main()