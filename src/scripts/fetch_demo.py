#!/usr/bin/env python

'''
*.Leaf_desc_prob leaf prob size
*.a < 3 * num_box???
*.belief representation
*.add history ignored bounding boxes.
*.How to handle, 'its under the banana' -> leaf_desc_prob['banana'] increase?
*.overleaf
'''

'''
leaf_and_desc
   x1                 x2                vn
x1 p(x1=l)          p(x1=x2's l&d)    p(x1=vn's l&d)
x2 p(x2=x1's l&d)   p(x2=l)           p(x2=nv's l&d)
vn  N.A.              N.A.              N.A.

Assume p(x1) = 0, p(x2) = 1
'''

'''
Action
0~N grasp and end
N+1 ~ 2N grasp and continue
2N+1 ~ 3N Ask do you mean
3N+1      ask where is
'''

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
# from stanfordcorenlp import StanfordCoreNLP

import vmrn._init_path
# from vmrn.model.utils.data_viewer import dataViewer, gen_paper_fig # , paperFig
from vmrn.model.utils.net_utils import leaf_and_descendant_stats, inner_loop_planning, relscores_to_visscores

from fetch_robot import FetchRobot
from grasp_planner.integrase import *
from libraries.data_viewer.data_viewer import DataViewer, gen_paper_fig

# ------- Statics -----------

br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

YCROP = (250, 650)
XCROP = (650, 1050)

def vis_action(action_str, shape, draw_arrow = False):
    im = 255. * np.ones(shape)
    action_str = action_str.split("\n")

    mid_line = im.shape[0] / 2
    dy = 32
    y_b = mid_line - dy * len(action_str)
    for i, string in enumerate(action_str):
        cv2.putText(im, string, (0, y_b + i * dy),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), thickness=2)
    if draw_arrow:
        cv2.arrowedLine(im, (0, mid_line), (im.shape[1], mid_line), (0, 0, 0), thickness = 2, tipLength = 0.03)
    return im

def split_long_string(in_str, len_thresh = 30):
    in_str = in_str.split(" ")
    out_str = ""
    len_counter = 0
    for word in in_str:
        len_counter += len(word) + 1
        if len_counter > len_thresh:
            out_str += "\n" + word + " "
            len_counter = len(word) + 1
        else:
            out_str += word + " "
    return out_str

def save_visualization(img, bboxes, rel_mat, rel_score_mat, expr, ground_prob, a, data_viewer, grasps=None, im_id=None, tgt_size=500):
    if im_id is None:
        current_date = datetime.datetime.now()
        image_id = "{}-{}-{}-{}".format(current_date.year, current_date.month, current_date.day,
                                        time.strftime("%H:%M:%S"))
    ############ visualize
    # resize img for visualization
    scalar = float(tgt_size) / img.shape[0]
    img_show = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)

    vis_bboxes = bboxes * scalar
    vis_bboxes[:, -1] = bboxes[:, -1]
    grasps[:, :8] = grasps[:, :8] * scalar
    num_box = bboxes.shape[0]

    # object detection
    cls = bboxes[:, -1]
    if grasps is not None:
        object_det_img = data_viewer.draw_graspdet_with_owner(img_show.copy(), vis_bboxes, grasps)
    else:
        object_det_img = data_viewer.draw_objdet(img_show.copy(), vis_bboxes, list(range(cls.shape[0])))

    # relationship detection
    rel_det_img = data_viewer.draw_mrt(img_show.copy(), rel_mat, class_names= ground_prob.tolist()[:-1],
                                       rel_score=rel_score_mat, with_img=False, rel_img_size=500)
    rel_det_img = cv2.resize(rel_det_img, (img_show.shape[1], img_show.shape[0]))

    # grounding
    print("Grounding Probability: ")
    print(ground_prob.tolist())
    ground_img = data_viewer.draw_grounding_probs(img_show.copy(), expr, vis_bboxes, ground_prob[:-1])
    print("Grasping score: ")
    print(grasps[:, -1].tolist())

    question_type = None
    print("Optimal Action:")
    if a < num_box:
        action_str = "Grasping object " + str(a) + " and ending the program"
    elif a < 2 * num_box:
        action_str = "Grasping object " + str(a - num_box) + " and continuing"
    elif a < 3 * num_box:
        action_str = Q1["type1"].format(str(a - 2 * num_box) + "th object")
        question_type = "Q1_TYPE1"
    else:
        if ground_prob[-1] == 1:
            action_str = Q2["type2"]
            question_type = "Q2_TYPE2"
        elif (ground_prob[:-1] > 0.02).sum() == 1:
            action_str = Q2["type3"].format(str(np.argmax(ground_prob[:-1])) + "th object")
            question_type = "Q2_TYPE3"
        else:
            action_str = Q2["type1"]
            question_type = "Q2_TYPE1"
    print(action_str)

    action_img_shape = list(img_show.shape)
    action_img = vis_action(split_long_string(action_str), action_img_shape)
    final_img = np.concatenate([
        np.concatenate([object_det_img, rel_det_img], axis=1),
        np.concatenate([ground_img, action_img], axis=1),
    ], axis=0)

    # save result
    out_dir = "images/output"
    if im_id is None:
        im_id = str(datetime.datetime.now())
        origin_name = im_id + "_origin.png"
        save_name = im_id + "_result.png"
    else:
        origin_name = im_id.split(".")[0] + "_origin.png"
        save_name = im_id.split(".")[0] + "_result.png"
    origin_path = os.path.join(out_dir, origin_name)
    save_path = os.path.join(out_dir, save_name)
    i = 1
    while (os.path.exists(save_path)):
        i += 1
        save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
        save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(origin_path, img)
    cv2.imwrite(save_path, final_img)
    return {"origin_img": img_show,
            "od_img": object_det_img,
            "mrt_img": rel_det_img,
            "ground_img": ground_img,
            "action_str": split_long_string(action_str),
            "q_type": question_type}

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

def main():
    rospy.init_node('INTEGRASE', anonymous=True)
    s_ing_client = INTEGRASE()
    data_viewer = DataViewer(classes)
    robot = FetchRobot()

    # expr = raw_input("Please tell me what you want: ")
    expr = 'cup under banana'
    related_classes = [cls for cls in classes if cls in expr or expr in cls]
    # running_mode = "kinect"
    all_results = []

    # outer-loop planning: in each step, grasp the leaf-descendant node.
    while (True):
        img, _ = robot.read_imgs()

        # perception
        bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, ind_match, grasps = \
            s_ing_client.single_step_perception_new(img, expr, cls_filter=related_classes)
        num_box = bboxes.shape[0]

        # outer-loop planning: in each step, grasp the leaf-descendant node.
        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        belief = {}
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob)
        belief["ground_prob"] = torch.from_numpy(ground_result)

        # inner-loop planning, with a sequence of questions and a last grasping.
        while (True):
            a = inner_loop_planning(belief) # action_idx. 
            all_results.append(
                save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, data_viewer, grasps.copy()))
            if a < 2 * num_box: # if it is a grasp action
                break
            else:
                data = {"img": img,
                        "bbox": bboxes[:, :4].reshape(-1).tolist(),
                        "cls": bboxes[:, 4].reshape(-1).tolist(),
                        "mapping": ind_match}
                ans = raw_input("Your answer: ")
                all_results[-1]["answer"] = split_long_string("User's Answer: " + ans.upper())

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
                        if (ground_result[:-1] > 0.02).sum() == 1:
                            obj_ind = ind_match[np.argmax(ground_result[:-1])]
                            s_ing_client.object_pool[obj_ind]["is_target"] = True
                    else:
                        # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                        ans = ans[6:]
                        # for i in ind_match.values():
                        #     s_ing_client.object_pool[i]["is_target"] = False
                        s_ing_client.clue = ans
                        if (ground_result[:-1] > 0.02).sum() == 1:
                            obj_ind = ind_match[np.argmax(ground_result[:-1])]
                            s_ing_client.object_pool[obj_ind]["is_target"] = False
                belief = s_ing_client.update_belief(belief, a, ans, data)

        # execute grasping action
        grasp = grasps[a % num_box][:8] + np.tile([XCROP[0], YCROP[0]], 4)
        robot.grasp(grasp)
        # TODO: Determine whether the grasp is successful and then assigh this "removed" flag
        s_ing_client.object_pool[ind_match[a % num_box]]["removed"] = True

        if a < num_box:
            break

        # TODO test temp
        break

    gen_paper_fig(expr, all_results)

if __name__ == '__main__':
    main()

