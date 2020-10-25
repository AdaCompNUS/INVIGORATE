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

from config.config import *
from libraries.data_viewer.data_viewer import DataViewer
from invigorate.invigorate import *
from libraries.caption_generator import caption_generator
from libraries.robots.dummy_robot import DummyRobot
from libraries.utils.log import LOGGER_NAME

# -------- Settings --------
ROBOT = 'Fetch'
GENERATE_CAPTIONS = False
DISPLAY_DEBUG_IMG = False

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
    gt_tgt_prob = np.zeros((num_box + 1))
    gt_tgt_prob[tgt_idx] = 1.0
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
    for i in range(len(gt_tgt_prob)):
        loss -= gt_tgt_prob[i] * math.log(pred_tgt_prob[i]) + (1 - gt_tgt_prob[i]) * math.log(1 - pred_tgt_prob[i])
    return loss

def cal_rel_loss(gt_rel_prob, pred_rel_prob):
    loss = 0.0
    gt_rel_prob = gt_rel_prob.reshape(-1)
    pred_rel_prob = pred_rel_prob.reshape(-1)
    for i in range(len(gt_rel_prob)):
        loss -= gt_rel_prob[i] * math.log(pred_rel_prob[i] + 1e-5) + (1 - gt_rel_prob[i]) * math.log(1 - pred_rel_prob[i])
    return loss

def main():
    rospy.init_node('INVIGORATE', anonymous=True)

    invigorate_client = InvigorateMultiSingleStepComparison()

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
        logger.info("------------------------")
        logger.info("Start of iteration")

        # perceive new images
        img, _ = robot.read_imgs()

        # perception
        observations = invigorate_client.perceive_img(img, expr)
        if observations is None:
            logger.warning("nothing is detected, abort!!!")
            break
        num_box = observations['bboxes'].shape[0]

        # debug
        img = observations['img']
        bboxes = observations['bboxes']
        classes = observations['classes']

        vis_bboxes = np.concatenate([bboxes, classes], axis=-1)
        objdet_img = data_viewer.draw_objdet(img.copy(), vis_bboxes, list(range(classes.shape[0])))
        data_viewer.display_img(objdet_img)
        # imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, save=False)
        # if DISPLAY_DEBUG_IMG:
        #     data_viewer.display_img(imgs['final_img'])
        # cv2.imwrite("outputs/final.png", imgs['final_img'])

        # state_estimation
        ss_tgt_prob, ss_rel_mat = invigorate_client.estimate_state_with_observation_singlestep(observations)
        ms_tgt_prob, ms_rel_mat = invigorate_client.estimate_state_with_observation_multistep(observations)
        gt_target = raw_input("Enter idx for gt tgt:")
        gt_rel_str = raw_input("Enter edges for gt tgt:")
        gt_rel_list = gt_rel_str.split()
        gt_rel_list = [int(x) for x in gt_rel_list]
        gt_rel = np.array(gt_rel_list).reshape(-1, 2).tolist()
        gt_tgt_prob = generate_gt_tgt_prob(int(gt_target), num_box)
        gt_rel_mat = generate_gt_rel_mat(gt_rel, num_box)

        logger.info("---------------tgt prob:")
        logger.info("gt: \n {}".format(gt_tgt_prob))
        logger.info("single-step: \n {}".format(ss_tgt_prob))
        logger.info("multi-step: \n {}".format(ms_tgt_prob))

        logger.info("---------------rel:")
        logger.info("gt: \n {}".format(gt_rel_mat))
        logger.info("single-step: \n {}".format(ss_rel_mat))
        logger.info("multi-step: \n {}".format(ms_rel_mat))

        st_tgt_loss = cal_tgt_loss(gt_tgt_prob, ss_tgt_prob)
        st_rel_loss = cal_rel_loss(gt_rel_mat, ss_rel_mat)
        mt_tgt_loss = cal_tgt_loss(gt_tgt_prob, ms_tgt_prob)
        mt_rel_loss = cal_rel_loss(gt_rel_mat, ms_rel_mat)

        logger.info("--------------- loss:")
        logger.info("single-step: tgt: {}, rel: {}".format(st_tgt_loss, st_rel_loss))
        logger.info("multi-step: tgt: {}, rel: {}".format(mt_tgt_loss, mt_rel_loss))

        # generate debug images
        img = observations['img']
        bboxes = observations['bboxes']
        classes = observations['classes']
        rel_mat = observations['rel_mat']
        rel_score_mat = observations['rel_score_mat']
        target_prob = invigorate_client.belief['target_prob']
        data_viewer.gen_final_paper_fig(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, action, grasps, question_str, answer)

        to_cont = raw_input('To_continue?')
        if to_cont != 'y':
            break

    print("exit!")
    # rospy.sleep(10) # wait 10 second
    robot.move_arm_to_home()

if __name__ == '__main__':
    main()