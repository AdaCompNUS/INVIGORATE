#!/usr/bin/env python
'''
Action
0~N grasp and end
N+1 ~ 2N grasp and continue
2N+1 ~ 3N Ask do you mean
3N+1      ask where is
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

import vmrn._init_path
from vmrn.model.utils.net_utils import inner_loop_planning
from config.config import *
from invigorate.integrase import INTEGRASE
from libraries.data_viewer.data_viewer import DataViewer
from libraries.caption_generator import caption_generator
from libraries.robots.fetch_robot import FetchRobot
from libraries.robots.dummy_robot import DummyRobot

# -------- Settings --------
GENERATE_CAPTIONS = True

# ------- Statics -----------

br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

YCROP = (250, 650)
XCROP = (650, 1050)

def main():
    rospy.init_node('INTEGRASE', anonymous=True)
    s_ing_client = INTEGRASE()
    data_viewer = DataViewer(CLASSES)
    robot = FetchRobot()
    # robot = DummyRobot()

    expr = robot.listen()
    related_classes = [cls for cls in CLASSES if cls in expr or expr in cls]

    all_results = []
    # outer-loop planning: in each step, grasp the leaf-descendant node.
    while (True):
        inner_loop_results = []

        img, _ = robot.read_imgs()

        # display image for debug
        data_viewer.display_img(img)

        # perception
        bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, target_prob, ind_match, grasps = \
            s_ing_client.single_step_perception_new(img, expr, cls_filter=related_classes)
        bboxes = bboxes[:, :4]
        classes = bboxes[:, -1]
        num_box = bboxes.shape[0]
        question_str = None
        ans = None

        # outer-loop planning: in each step, grasp the leaf-descendant node.
        belief = {}
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob)
        belief["ground_prob"] = torch.from_numpy(target_prob)

        to_end = False
        # inner-loop planning, with a sequence of questions and a last grasping.
        while not to_end:
            a = inner_loop_planning(belief) # action_idx.
            if a < num_box:
                grasp_target_idx = a
                print("Grasping object " + str(grasp_target_idx) + " and ending the program")
                to_end = True
                break
            elif a < 2 * num_box: # if it is a grasp action
                grasp_target_idx = a - num_box
                print("Grasping object " + str(grasp_target_idx) + " and continuing")
                break
            else:
                if a < 3 * num_box:
                    target_idx = a - 2 * num_box
                    if GENERATE_CAPTIONS:
                        # generate caption
                        caption = caption_generator.generate_caption(img, bboxes, classes, target_idx)
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
                            caption = caption_generator.generate_caption(img, bboxes, classes, target_idx)
                            question_str = Q1["type1"].format(caption)
                        else:
                            question_str = Q2["type3"].format(target_idx + "th object")
                    else:
                        question_str = Q2["type1"]

                robot.say(question_str)

                data = {"img": img,
                        "bbox": bboxes.reshape(-1).tolist(),
                        "cls": bboxes.reshape(-1).tolist(),
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
        im = data_viewer.display_obj_to_grasp(img.copy(), bboxes, grasps, grasp_target_idx)
        data_viewer.display_img(im)

        # execute grasping action
        grasp = grasps[a % num_box][:8] # + np.tile([XCROP[0], YCROP[0]], 4)
        robot.grasp(grasp)
        # TODO: Determine whether the grasp is successful and then assign this "removed" flag
        s_ing_client.object_pool[ind_match[a % num_box]]["removed"] = True

        # if a < num_box:
        #     break

        # generate debug images
        data_viewer.gen_final_paper_fig(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, a, grasps.copy(), question_str, ans)

        to_cont = raw_input('To_continue?')
        if to_cont != 'y':
            break

if __name__ == '__main__':
    main()

'''
Legacy

def with_single_img(s_ing_client):
    im_id = raw_input("image ID: ")
    expr = raw_input("Please tell me what you want: ")
    related_classes = [cls for cls in classes if cls in expr or expr in cls]
    img = cv2.imread("images/" + im_id + ".png")

    data_viewer = DataViewer(classes)

    bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, qa_his = \
        s_ing_client.single_step_perception(img, expr, cls_filter=related_classes)
    num_box = bboxes.shape[0]

    # dummy action for initialization
    a = 3 * num_box + 1
    all_results = []
    # outer-loop planning: in each step, grasp the leaf-descendant node.
    vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
    belief = {}
    belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob) # [N+1, N+1] where [i, j] represents the probability of i being leaf and descendant of j.
                                                                # +1 because of bg

                                                                # grounding_belief, mrt_belief -> leaf & desc prob
                                                                # b = ()

    belief["ground_prob"] = torch.from_numpy(ground_result)

    # inner-loop planning, with a sequence of questions and a last grasping.
    a = inner_loop_planning(belief)

    current_date = datetime.datetime.now()
    image_id = "{}-{}-{}-{}".format(current_date.year, current_date.month, current_date.day,
                                    time.strftime("%H:%M:%S"))
    all_results.append(
        save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, data_viewer,
                           im_id=image_id))

    return ground_score, rel_score_mat
'''