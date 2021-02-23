#!/usr/bin/env python

import sys
import os.path as osp

from nltk import data
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
import math
from PIL import Image
# from stanfordcorenlp import StanfordCoreNLP
import matplotlib
matplotlib.use('Agg')
import nltk
import logging
import pickle

from config.config import *
from libraries.data_viewer.data_viewer import DataViewer
from invigorate.invigorate import *
from libraries.caption_generator import caption_generator
from libraries.robots.dummy_robot import DummyRobot
from libraries.utils.log import LOGGER_NAME
from invigorate_msgs.srv import *

# -------- Settings --------
ROBOT = 'Dummy'
GENERATE_CAPTIONS = False
DISPLAY_DEBUG_IMG = True

if ROBOT == 'Fetch':
    from libraries.robots.fetch_robot import FetchRobot

# -------- Constants --------
EXEC_GRASP = 0
EXEC_ASK = 1
EXEC_DUMMY_ASK = 2

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

def process_user_command(command, nlp_server="nltk"):
    if nlp_server == "nltk":
        text = nltk.word_tokenize(command)
        pos_tags = nltk.pos_tag(text)
    else:
        doc = stanford_nlp_server(command)
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
    logger.info("Processed clue: {:s}".format(clue if clue != '' else "None"))

    return clue

def generate_gt_tgt_prob(tgt_idx, num_box):
    gt_tgt_prob = np.zeros((num_box))
    for idx in tgt_idx:
        gt_tgt_prob[idx] = 1.0
    return gt_tgt_prob

def generate_gt_rel_mat(edges, num_box):
    gt_rel_mat = np.zeros((3, num_box, num_box))
    gt_rel_mat[2, :, :] = 1.0
    np.fill_diagonal(gt_rel_mat[2], 0.0)
    for u, v in edges:
        # u is descendant of v (u is on top of v)
        gt_rel_mat[0, v, u] = 1.0
        gt_rel_mat[2, u, v] = 0.0
        gt_rel_mat[1, u, v] = 1.0
        gt_rel_mat[2, v, u] = 0.0

    return gt_rel_mat

def cal_tgt_loss(gt_tgt_prob, pred_tgt_prob):
    loss = 0.0
    num_obj = len(gt_tgt_prob)
    for i in range(len(gt_tgt_prob)):
        loss += math.fabs(gt_tgt_prob[i] - pred_tgt_prob[i]) # L1 loss
    loss /= num_obj
    return loss

def cal_rel_loss(gt_rel_prob, pred_rel_prob):
    loss = 0.0
    gt_rel_prob = gt_rel_prob.reshape(-1)
    pred_rel_prob = pred_rel_prob.reshape(-1)
    for i in range(len(gt_rel_prob)):
        loss -= gt_rel_prob[i] * math.log(pred_rel_prob[i] + 1e-10) + (1 - gt_rel_prob[i]) * math.log(1 - pred_rel_prob[i] + 1e-10)
    return loss

# def cal_tgt_acc(gt_tgt_prob, pred_tgt_prob):
#     loss = 0.0
#     for i in range(len(gt_tgt_prob)):
#         loss += math.fabs(gt_tgt_prob[i] - pred_tgt_prob[i]) # L1 loss
#     return loss

def vilbert_grounding(img, bboxes, expr, num_box):
    # grounding
    rospy.wait_for_service('vilbert_grounding_service')
    try:
        vilbert_client = rospy.ServiceProxy('vilbert_grounding_service', Grounding)
        req = GroundingRequest()
        img_msg = br.cv2_to_imgmsg(img)
        req.img = img_msg
        req.bboxes = bboxes.flatten().tolist()
        req.expr = expr

        resp = vilbert_client(req)
        grounding_scores = np.array(list(resp.grounding_scores))
        # rel_score_mat = np.array(list(resp.rel_score_mat)).reshape(3, num_box, num_box)
        # rel_mat = np.argmax(rel_score_mat, axis=0) + 1
    except rospy.ServiceException as e:
        # print("Service call failed: %s"%e)
        return

    target_prob = np.exp(grounding_scores)
    # target_prob = np.append(target_prob, 0.0)
    target_prob = np.clip(target_prob, 0, 1)
    print(target_prob.sum())
    # grounding_scores_t = torch.from_numpy(grounding_scores)
    # target_prob = torch.nn.functional.softmax(grounding_scores_t).numpy()

    print('--------------------------------------------------------')
    logger.info('vilbert_grounding_scores: {}'.format(grounding_scores))
    logger.info('vilbert_target_prob: {}'.format(target_prob))
    print('--------------------------------------------------------')

    return grounding_scores, target_prob

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
    iter_num = 1
    invigorate_tgt_loss_list = []
    vilbert_tgt_loss_list = []
    while not to_end:
        logger.info("------------------------")
        logger.info("Start of iteration {}".format(iter_num))

        # perceive new images
        img, _ = robot.read_imgs()

        # state_estimation
        res = invigorate_client.estimate_state_with_img(img, expr)
        p_cand = invigorate_client.p_cand

        # debug
        img = invigorate_client.img
        bboxes = invigorate_client.belief['bboxes']
        num_obj = bboxes.shape[0]
        classes = invigorate_client.belief['classes']
        rel_score_mat = np.zeros((3, num_obj, num_obj)) # dummy
        rel_mat, _ = invigorate_client.rel_score_process(rel_score_mat)
        target_prob = invigorate_client.belief['target_prob']

        vis_bboxes = np.concatenate([bboxes, classes], axis=-1)
        objdet_img = data_viewer.draw_objdet(img.copy(), vis_bboxes, list(range(classes.shape[0])))
        data_viewer.display_img(objdet_img)
        imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, save=True)
        if DISPLAY_DEBUG_IMG:
            data_viewer.display_img(imgs['final_img'], mode="pil")
        # cv2.imwrite("outputs/final.png", imgs['final_img'])

        vilbert_grounding_scores, vilbert_target_prob = vilbert_grounding(img, bboxes, expr, num_obj)

        gt_target_str = raw_input("Enter idx for gt tgt:")
        if gt_target_str != "":
            gt_target_list = gt_target_str.split(" ")
            gt_targets = [int(i) for i in gt_target_list]
        else:
            gt_targets = []
        gt_tgt_prob = generate_gt_tgt_prob(gt_targets, num_obj)

        target_prob = p_cand

        logger.info("---------------tgt prob:")
        logger.info("gt: \n {}".format(gt_tgt_prob))
        logger.info("invigorate: \n {}".format(target_prob))
        logger.info("vlbert: \n {}".format(vilbert_target_prob))

        invigorate_tgt_loss = cal_tgt_loss(gt_tgt_prob, target_prob)
        vilbert_tgt_loss = cal_tgt_loss(gt_tgt_prob, vilbert_target_prob)

        logger.info("--------------- loss:")
        logger.info("invigorate: tgt: {}".format(invigorate_tgt_loss))
        logger.info("vilbert: tgt: {}".format(vilbert_tgt_loss))

        # invigorate_tgt_acc = cal_tgt_acc(gt_tgt_prob, target_prob)
        # vilbert_tgt_acc = cal_tgt_acc(gt_tgt_prob, vilbert_target_prob)

        result = {}
        result["gt_tgt_prob"] = gt_tgt_prob
        result["invigorate_tgt_prob"] = target_prob
        result["vilbert_tgt_prob"] = vilbert_target_prob
        result["vilbert_grounding_score"] = vilbert_grounding_scores
        result["invigorate_tgt_loss"] = invigorate_tgt_loss
        result["vilbert_tgt_loss"] = vilbert_tgt_loss

        f = open(osp.join(LOG_DIR, "result_{}.pkl".format(iter_num)), "wb")
        pickle.dump(result, f)

        invigorate_tgt_loss_list.append(invigorate_tgt_loss)
        vilbert_tgt_loss_list.append(vilbert_tgt_loss)

        to_cont = raw_input('To_continue?')
        if to_cont == 'q':
            break

        removed_obj = raw_input("removed obj number")
        removed_obj = int(removed_obj)
        action = removed_obj + num_obj
        invigorate_client.transit_state(action)

        iter_num += 1

    invigorate_tgt_loss_average = np.array(invigorate_tgt_loss_list).sum() / iter_num
    vilbert_tgt_loss_average = np.array(vilbert_tgt_loss_list).sum() / iter_num
    logger.info("--------------- average loss:")
    logger.info("invigorate: tgt: {}".format(invigorate_tgt_loss_average))
    logger.info("vilbert: tgt: {}".format(vilbert_tgt_loss_average))

    print("exit!")
    # rospy.sleep(10) # wait 10 second
    robot.move_arm_to_home()

if __name__ == '__main__':
    main()