#!/usr/bin/env python

'''
*. More structural target prob update, there should not be stages
'''

'''
Leaf and desc prob is stored as a NxN matrix.
M[a. b] represents the probability of a being the leaf and descendant of b
'''

'''
leaf_and_desc
   x1                 x2                vn
x1 p(x1=l)          p(x1=x2's l&d)    p(x1=vn's l&d)
x2 p(x2=x1's l&d)   p(x2=l)           p(x2=nv's l&d)
vn  N.A.              N.A.              N.A.

Assume p(x1) = 0, p(x2) = 1
'''

import warnings
import rospy
import cv2
from cv_bridge import CvBridge
import torch
from torch import t
import torch.nn.functional as f
import numpy as np
from scipy import optimize
import os
from torchvision.ops import nms
import time
import pickle as pkl
import os.path as osp
import copy
from sklearn.cluster import KMeans
from scipy import optimize
import logging
import matplotlib.pyplot as plt
from libraries.data_viewer.data_viewer import DataViewer
from libraries.density_estimator.density_estimator import object_belief, gaussian_kde, relation_belief
from invigorate_msgs.srv import ObjectDetection, VmrDetection, VLBert
# from libraries.ros_clients.detectron2_client import Detectron2Client
from libraries.ros_clients.detectron2_client import Detectron2Client
from libraries.ros_clients.vmrn_client import VMRNClient
from libraries.ros_clients.vilbert_client import VilbertClient
from libraries.ros_clients.mattnet_client import MAttNetClient
from config.config import *
from libraries.utils.log import LOGGER_NAME
from collections import OrderedDict
import nltk
import pdb

try:
    import stanza
except:
    warnings.warn("No NLP models are loaded.")

# -------- Settings ---------
DEBUG = False

# -------- Constants ---------
MAX_Q2_NUM = 1 # the robot can at most ask MAX_Q2_NUM Q2s.

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code ---------
class Invigorate(object):
    def __init__(self):
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        # self._grasp_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        # self._tpn_det = rospy.ServiceProxy('vlbert_server', VLBert)
        # self._vis_ground_client = VilbertClient()
        self._vis_ground_client = MAttNetClient()
        self._vmrn_client = VMRNClient()
        self._rel_det_client = self._vmrn_client
        self._grasp_det_client = self._vmrn_client

        self._br = CvBridge()

        self._policy_tree_max_depth = 3
        self._penalty_for_asking = -2
        self._penalty_for_fail = -10
        self.history_scores = []
        self.object_pool = []
        self.rel_pool = {}
        self.target_in_pool = None
        self._init_kde()
        self.belief = {}
        self.step_infos = {}
        self.q2_num_asked = 0
        self.expr = []
        self.clue = ""
        self.subject = []

        self.data_viewer = DataViewer(CLASSES)
        if NLP_SERVER == "stanza":
            try:
                self.stanford_nlp_server = stanza.Pipeline("en")
            except:
                warnings.warn("stanza needs python 3.6 or higher. "
                              "please update your python version and run 'pip install stanza'")

    def _merge_bboxes(self, bboxes, classes, scores, bboxes_his, classes_his, scores_his):
        curr_to_his = self._bbox_match(bboxes, bboxes_his, scores, scores_his)
        his_to_curr = {v: k for k, v in curr_to_his.items()}
        for i in range(bboxes_his.shape[0]):
            if i not in curr_to_his.values():
                bboxes = np.concatenate([bboxes, bboxes_his[i][None, :]], axis=0)
                classes = np.concatenate([classes, classes_his[i][None, :]], axis=0)
                scores = np.concatenate([scores, scores_his[i][None, :]], axis=0)
            else:
                scores[his_to_curr[i]] = (scores[his_to_curr[i]] + scores_his[i]) / 2
                classes[his_to_curr[i]] = scores[his_to_curr[i]][1:].argmax() + 1
        return bboxes, classes, scores

    def perceive_img(self, img, expr):
        '''
        @return bboxes,         [Nx5], xyxy
                scores,         [Nx1]
                rel_mat,        [NxN]
                rel_score_mat,  [3xNxN]
                leaf_desc_prob, [N+1xN+1]
                ground_score,   [N+1]
                ground_result,  [N+1]
                ind_match_dict,
                grasps,         [Nx5x8], 5 grasps for every object, every grasp is x1y1, x2y2, x3y3, x4y4
        '''

        tb = time.time()
        # get the subject of the expression
        if expr not in self.expr:
            self.expr.append(expr)
        self.subject = self._find_subject(expr)

        # object detection
        bboxes, classes, scores = self._object_detection(img)
        if bboxes is None:
            logger.warning("WARNING: nothing is detected")
            return None
        print('--------------------------------------------------------')
        logger.info('Perceive_img: _object_detection finished, all current detections: ')
        for i in range(bboxes.shape[0]):
            sc = scores[i].max()
            cls = CLASSES[int(classes[i].item())]
            logger.info("Class: {}, Score: {:.2f}, Location: {}".format(cls, sc, bboxes[i]))
        # double check the rois in our object pool
        rois = [o["bbox"][None, :] for o in self.object_pool if not o["removed"]]
        if len(rois) > 0:
            rois = np.concatenate(rois, axis=0)
            bboxes_his, classes_his, scores_his = self._object_detection(img, rois)
            logger.info('Perceive_img: _his_object_re-classification finished, all historic detections: ')
            for i in range(bboxes_his.shape[0]):
                sc = scores_his[i].max()
                cls = CLASSES[int(classes_his[i].item())]
                logger.info("Class: {}, Score: {:.2f}, Location: {}".format(cls, sc, bboxes_his[i]))
            bboxes, classes, scores = self._merge_bboxes(bboxes, classes, scores, bboxes_his, classes_his, scores_his)
            logger.info('Perceive_img: detection merging finished, '
                        'the final results that will be further merged into the object pool: ')
            for i in range(bboxes.shape[0]):
                sc = scores[i].max()
                cls = CLASSES[int(classes[i].item())]
                logger.info("Class: {}, Score: {:.2f}, Location: {}".format(cls, sc, bboxes[i]))

        self._bbox_post_process(bboxes, scores)
        logger.info('Perceive_img: Object pool updated, the remaining objects: ')
        for i, obj in enumerate(self.object_pool):
            if not obj["removed"]:
                sc = np.array(obj["cls_scores"]).mean(axis=0).max()
                cls = CLASSES[np.array(obj["cls_scores"]).mean(axis=0).argmax()]
                logger.info("Pool ind: {:d}, Class: {}, Score: {:.2f}, Location: {}".format(i, cls, sc, obj["bbox"]))
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates(renew=True)
        logger.info('Perceive_img: Object selected for latter process: {}'.format(pool_to_det.keys()))
        bboxes = np.asarray([self.object_pool[v]["bbox"].tolist() for k, v in det_to_pool.items()])
        classes = np.asarray([np.argmax(np.array(self.object_pool[v]["cls_scores"]).mean(axis=0)[1:]) + 1
                              for k, v in det_to_pool.items()]).reshape(-1, 1)

        # relationship
        rel_mat, rel_score_mat = self._rel_det_client.detect_obr(img, bboxes)
        logger.info('Perceive_img: mrt detection finished')

        # grounding
        grounding_scores = [self._vis_ground_client.ground(img, bboxes, e, classes) for e in self.expr]
        logger.info('Perceive_img: mattnet grounding finished')

        # relationship
        # rel_score_mat = self._vis_ground_client.ground(img, bboxes, expr)

        # grasp
        grasps = self._grasp_det_client.detect_grasps(img, bboxes)
        grasps = self._grasp_filter(bboxes, grasps)
        logger.info('Perceive_img: grasp detection finished')
        self._grasp_post_process(det_to_pool, grasps)

        # object and relationship detection post process
        rel_mat, rel_score_mat = self.rel_score_process(rel_score_mat)

        num_box = bboxes.shape[0]
        logger.info('Perceive_img: post process of object and mrt detection finished')
        print('--------------------------------------------------------')

        # combine into a dictionary
        observations = {}
        observations['img'] = img
        observations['expr'] = expr
        observations['num_box'] = num_box
        observations['bboxes'] = bboxes
        observations['classes'] = classes
        observations['det_scores'] = scores
        observations['rel_mat'] = rel_mat
        observations['rel_score_mat'] = rel_score_mat
        observations['grounding_scores'] = grounding_scores
        observations['grasps'] = grasps

        self.observations = observations
        return observations

    def _integrate_historic_answer(self, target_prob, det_to_pool):
        # incorporate QA history TODO
        target_prob_backup = target_prob.copy()
        for k, v in det_to_pool.items():
            if self.object_pool[v]["is_target"] == 1:
                target_prob[:] = 0.
                target_prob[k] = 1.
            elif self.object_pool[v]["is_target"] == 0:
                target_prob[k] = 0.
        # sanity check
        if target_prob.sum() > 0:
            target_prob /= target_prob.sum()
        else:
            # something wrong with the matching process. roll back
            target_prob = target_prob_backup
            for k, v in det_to_pool.items():
                target_prob[k] = 0
            target_prob /= target_prob.sum()
        return target_prob

    def estimate_state_with_observation(self, observations):
        img = observations['img']
        bboxes = observations['bboxes']
        classes = observations['classes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        grasps = observations['grasps']
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates()

        # Estimate rel_prob_mat and target_prob according to multi-step observations
        print('--------------------------------------------------------')
        logger.info('Step 1: raw grounding completed')
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        print('--------------------------------------------------------')

        # 1. multi-step visual grounding and obr detection
        rel_prob_mat = self._multi_step_mrt_estimation(rel_score_mat, det_to_pool)
        for g_score in grounding_scores:
            target_prob = self._multi_step_grounding(g_score, det_to_pool)

        print('--------------------------------------------------------')
        logger.info('Step 2: candidate prior probability from object detector incorporated')
        logger.debug('target_prob : {}'.format(target_prob))
        print('--------------------------------------------------------')

        print('--------------------------------------------------------')
        logger.info('After multistep, the results:')
        logger.info('after multistep, target_prob: {}'.format(target_prob))
        logger.info('after multistep, rel_score_mat: {}'.format(rel_prob_mat))
        print('--------------------------------------------------------')

        # 2. integrate historic answer
        target_prob = self._integrate_historic_answer(target_prob, det_to_pool)

        # update target_prob
        for k, v in det_to_pool.items():
            self.object_pool[v]["target_prob"] = target_prob[k]

        print('--------------------------------------------------------')
        logger.info('Step 3: incorporate QA history completed')
        logger.info('target_prob: {}'.format(target_prob))
        print('--------------------------------------------------------')

        self.step_infos["bboxes"] = bboxes
        self.step_infos["classes"] = classes
        self.step_infos["grasps"] = grasps
        self.belief['target_prob'] = target_prob
        self.belief['rel_prob'] = rel_prob_mat

    def estimate_state_with_user_answer(self, action, answer, observations):

        num_box = observations["num_box"]
        response, clue = self._process_user_answer(answer)
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates()

        action_type, _ = self.get_action_type(action)
        assert action_type == "Q1"

        print('--------------------------------------------------------')
        logger.info("Invigorate: handling answer for Q1")

        # Firstly, regrounding the target according to the answer if possible.
        if clue != "":
            # if there is a clue, regrounding the target with the clue and update the object belief
            img = observations['img']
            bboxes = observations['bboxes']
            classes = observations['classes']
            expr = observations['expr']
            regrounding_scores = self._vis_ground_client.ground(img, bboxes, expr, classes)
            target_prob = self._multi_step_grounding(regrounding_scores, det_to_pool)
            target_prob = self._integrate_historic_answer(target_prob, det_to_pool)
            self.expr.append(clue)
        else:
            target_prob = self.belief["target_prob"]

        target_idx = action - 2 * num_box
        if response is not None:
            if response is True:
                # set non-target
                target_prob[:] = 0
                for obj in self.object_pool:
                    obj["is_target"] = 0
                # set target
                target_prob[target_idx] = 1
                self.object_pool[det_to_pool[target_idx]]["is_target"] = 1
            else:
                target_prob[target_idx] = 0
                target_prob /= np.sum(target_prob)
                self.object_pool[det_to_pool[target_idx]]["is_target"] = 0

        logger.info("estimate_state_with_user_answer completed")
        print('--------------------------------------------------------')

        self.belief["target_prob"] = target_prob

    def plan_action(self):
        '''
        @return action, int, if 0 < action < num_obj, grasp obj with index action and end
                             if num_obj < action < 2*num_obj, grasp obj with index action and continue
                             if 2*num_obj < action < 3*num_obj, ask questions about obj with index action
        '''
        return self.decision_making_pomdp()

    def transit_state(self, action):
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates()

        action_type, _ = self.get_action_type(action)
        if action_type == 'Q1':
            # asking question does not change state
            return
        else:
            # mark object as being removed
            self.object_pool[det_to_pool[action % obj_num]]["removed"] = True
            return

    def get_action_type(self, action, num_obj=None):
        '''
        @num_obj, the number of objects in the state, bg excluded.
        @return action_type, a readable string indicating action type
                target_idx, the index of the target.
        '''

        if num_obj is None:
            _, _, num_obj = self._get_valid_obj_candidates()

        if action < num_obj:
            return 'GRASP_AND_END', action
        elif action < 2 * num_obj:
            return 'GRASP_AND_CONTINUE', action - num_obj
        elif action < 3 * num_obj:
            return 'Q1', action - 2 * num_obj

    def planning_with_macro(self, infos):
        """
        ALL ACTIONS INCLUDE:
        Do you mean ... ? (num_obj) + grasping macro (1)
        :param belief: including "leaf_desc_prob", "desc_num", and "target_prob"
        :param planning_depth:
        :return:
        """

        # initialize hyperparameters
        planning_depth = self._policy_tree_max_depth
        penalty_for_fail = self._penalty_for_fail
        penalty_for_asking = self._penalty_for_asking

        def gen_grasp_macro(belief):

            num_obj = belief["target_prob"].shape[0] - 1

            grasp_macros = {i: None for i in range(num_obj)}
            belief_infos = belief["infos"]

            cache_leaf_desc_prob = {}
            cache_leaf_prob = {}
            for i in range(num_obj + 1):
                grasp_macro = {"seq": [], "leaf_prob": []}
                grasp_macro["seq"].append(torch.argmax(belief_infos["leaf_desc_prob"][:, i]).item())
                grasp_macro["leaf_prob"].append(belief_infos["leaf_prob"][grasp_macro["seq"][0]].item())

                rel_mat = belief["rel_prob"].clone()
                while (grasp_macro["seq"][-1] != i):
                    removed = torch.tensor(grasp_macro["seq"]).type_as(rel_mat).long()
                    indice = ''.join([str(o) for o in np.sort(grasp_macro["seq"]).tolist()])
                    if indice in cache_leaf_desc_prob:
                        leaf_desc_prob = cache_leaf_desc_prob[indice]
                        leaf_prob = cache_leaf_prob[indice]
                    else:
                        rel_mat[0:2, removed, :] = 0.
                        rel_mat[0:2, :, removed] = 0.
                        rel_mat[2, removed, :] = 1.
                        rel_mat[2, :, removed] = 1.
                        triu_mask = torch.triu(torch.ones(rel_mat[0].shape), diagonal=1)
                        triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
                        rel_mat *= triu_mask

                        leaf_desc_prob, _, leaf_prob, _, _ = \
                            self._leaf_and_desc_estimate(rel_mat, removed=grasp_macro["seq"], with_virtual_node=True)

                        cache_leaf_desc_prob[indice] = leaf_desc_prob
                        cache_leaf_prob[indice] = leaf_prob

                    grasp_macro["seq"].append(torch.argmax(leaf_desc_prob[:, i]).item())
                    grasp_macro["leaf_prob"].append(leaf_prob[grasp_macro["seq"][-1]].item())

                grasp_macro["seq"] = torch.tensor(grasp_macro["seq"]).type_as(rel_mat).long()
                grasp_macro["leaf_prob"] = torch.tensor(grasp_macro["leaf_prob"]).type_as(rel_mat)
                grasp_macros[i] = grasp_macro

            return grasp_macros

        def grasp_reward_estimate(belief):
            # reward of grasping macro, equal to: desc_num * reward_of_each_grasp_step
            # POLICY: treat the object with the highest conf score as the target
            target_prob = belief["target_prob"]
            target = torch.argmax(target_prob).item()
            grasp_macros = belief["grasp_macros"][target]
            leaf_prob = grasp_macros["leaf_prob"]
            p_not_remove_non_leaf = torch.cumprod(leaf_prob, dim=0)[-1].item()
            p_fail = 1. - target_prob[target].item() * p_not_remove_non_leaf
            return penalty_for_fail * p_fail

        def belief_update(belief):
            '''
            @return updated_belief, [num_obj x 2 x num_obj]
                                    updated_belief[i] is the belief for asking q1 wrt to obj i.
                                    updated_belief[i][0] is the belief for getting answer no, updated_belief[i][1] is the belief for getting answer yes.
            '''
            I = torch.eye(belief["target_prob"].shape[0]).type_as(belief["target_prob"])
            updated_beliefs = []
            # Do you mean ... ?
            # Answer No
            beliefs_no = belief["target_prob"].unsqueeze(0).repeat(num_obj + 1, 1)
            beliefs_no *= (1. - I) # set belief wrt to obj being asked to 0
            beliefs_no /= torch.clamp(torch.sum(beliefs_no, dim=-1, keepdim=True), min=1e-10) # normalize
            # Answer Yes
            beliefs_yes = I.clone() # set belief wrt to obj being asked to 1 and set the rest to 0

            for i in range(beliefs_no.shape[0] - 1):
                updated_beliefs.append([beliefs_no[i], beliefs_yes[i]])

            return updated_beliefs

        def is_onehot(vec, epsilon=1e-2):
            return (torch.abs(vec - 1) < epsilon).sum().item() > 0

        def estimate_q_vec(belief, current_d):
            if current_d == planning_depth - 1:
                return torch.tensor([grasp_reward_estimate(belief)])
            else:
                # branches of grasping
                q_vec = [grasp_reward_estimate(belief)]

                # q-value for asking Q1
                target_prob = belief["target_prob"]
                new_beliefs = belief_update(belief) # calculate b' for asking q1 about different objects at once.
                new_belief_dict = copy.deepcopy(belief)
                for i, new_belief in enumerate(new_beliefs):
                    q = penalty_for_asking
                    for j, b in enumerate(new_belief):
                        new_belief_dict["target_prob"] = b

                        # # branches of asking questions
                        # if is_onehot(b):
                        #     t_q = penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max()
                        # else:
                        #     t_q = penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max()
                        # if j == 0:
                        #     # Answer is No
                        #     q += t_q * (1 - target_prob[i])
                        # else:
                        #     # Answer is Yes
                        #     q += t_q * target_prob[i]

                        # calculate value of new belief
                        if is_onehot(b):
                            # set the depth to maximum depth as there is no need to plan further
                            new_belief_val = estimate_q_vec(new_belief_dict, planning_depth - 1).max()
                        else:
                            new_belief_val = estimate_q_vec(new_belief_dict, current_d + 1).max()

                        if j == 0:
                            # Answer is No
                            q += (1 - target_prob[i]) * new_belief_val # observation prob * value of new belief
                        else:
                            # Answer is Yes
                            q += target_prob[i] * new_belief_val # observation prob * value of new belief
                    q_vec.append(q.item())

                return torch.Tensor(q_vec).type_as(belief["target_prob"])

        num_obj = self.belief["target_prob"].shape[0] - 1
        belief = copy.deepcopy(self.belief)
        for k in belief:
            if belief[k] is not None:
                belief[k] = torch.from_numpy(belief[k])
        belief["infos"] = infos
        for k in belief["infos"]:
            belief["infos"][k] = torch.from_numpy(belief["infos"][k])
        grasp_macros = belief["grasp_macros"] = gen_grasp_macro(belief)

        q_vec = estimate_q_vec(belief, 0)
        logger.info("Q Value for Each Action: ")
        logger.info("Grasping:{:.3f}".format(q_vec.tolist()[0]))
        logger.info("Asking Q1:{:s}".format(q_vec.tolist()[1:num_obj + 1]))
        # print("Asking Q2:{:.3f}".format(q_vec.tolist()[num_obj+1]))

        for k in grasp_macros:
            for kk in grasp_macros[k]:
                grasp_macros[k][kk] = grasp_macros[k][kk].numpy()
        return torch.argmax(q_vec).item(), grasp_macros

    def decision_making_pomdp(self):
        print('--------------------------------------------------------')
        target_prob = self.belief['target_prob']
        num_box = target_prob.shape[0] - 1
        rel_prob = self.belief['rel_prob']
        leaf_desc_prob,_, leaf_prob, _, _ = self._get_leaf_desc_prob_from_rel_mat(rel_prob, with_virtual_node=True)
        logger.info("decision_making_pomdp: ")
        infos = {
            "leaf_desc_prob": leaf_desc_prob,
            "leaf_prob": leaf_prob
        }

        a_macro, grasp_macros = self.planning_with_macro(infos)
        if a_macro == 0:
            # grasping
            tgt_id = np.argmax(target_prob)
            grasp_macro = grasp_macros[tgt_id]
            current_tgt = grasp_macro["seq"][0]

            if len(grasp_macro["seq"]) == 1:
                # grasp and end program
                action = current_tgt
            else:
                # grasp and continue
                action = current_tgt + num_box

        else:
            action = a_macro - 1 + 2 * num_box
        print('--------------------------------------------------------')
        return action

    def _init_kde(self):
        # with open(osp.join(KDE_MODEL_PATH, 'ground_density_estimation_vilbert.pkl')) as f:
        with open(osp.join(KDE_MODEL_PATH, 'ground_density_estimation_mattnet.pkl'), "rb") as f:
            if PYTHON_VERSION == "3":
                data = pkl.load(f, encoding='latin1')
            elif PYTHON_VERSION == "2":
                data = pkl.load(f)
            else:
                raise ValueError
        data = data["ground"]
        pos_data = []
        neg_data = []
        for d in data:
            for i, score in enumerate(d["scores"]):
                if str(i) in d["gt"]:
                    pos_data.append(score)
                else:
                    neg_data.append(score)
        pos_data = np.expand_dims(np.array(pos_data), axis=-1)
        pos_data = np.sort(pos_data, axis=0)[5:-5]
        neg_data = np.expand_dims(np.array(neg_data), axis=-1)
        neg_data = np.sort(neg_data, axis=0)[5:-5]
        kde_pos = gaussian_kde(pos_data, bandwidth=0.05)
        kde_neg = gaussian_kde(neg_data, bandwidth=0.05)
        self.obj_kdes = [kde_neg, kde_pos]

        with open(osp.join(KDE_MODEL_PATH, "relation_density_estimation.pkl"), "rb") as f:
            if PYTHON_VERSION == "3":
                rel_data = pkl.load(f, encoding='latin1')
            elif PYTHON_VERSION == "2":
                rel_data = pkl.load(f)
            else:
                raise ValueError
        parents = np.array([d["det_score"] for d in rel_data if d["gt"] == 1])
        children = np.array([d["det_score"] for d in rel_data if d["gt"] == 2])
        norel = np.array([d["det_score"] for d in rel_data if d["gt"] == 3])
        kde_parents = gaussian_kde(parents, bandwidth=0.2)
        kde_children = gaussian_kde(children, bandwidth=0.2)
        kde_norel = gaussian_kde(norel, bandwidth=0.2)
        self.rel_kdes = [kde_parents, kde_children, kde_norel]

    def _init_object(self, bbox, score):
        new_box = {}
        new_box["bbox"] = bbox
        new_box["cls_scores"] = [score.tolist()]
        new_box["cand_belief"] = object_belief()
        new_box["target_prob"] = 0.
        new_box["ground_scores_history"] = []
        new_box["clue"] = None
        new_box["clue_belief"] = object_belief()
        # if self.target_in_pool:
        #     new_box["confirmed"] = True
        # else:
        #     new_box["confirmed"] = False  # whether this box has been confirmed by user's answer
        new_box["is_target"] = -1 # -1 means unknown
        new_box["removed"] = False
        new_box["grasp"] = None
        new_box["num_det_failures"] = 0
        return new_box

    def _init_relation(self, rel_score):
        new_rel = {}
        new_rel["rel_score"] = rel_score
        new_rel["rel_score_history"] = []
        new_rel["rel_belief"] = relation_belief()
        return new_rel

    def _faster_rcnn_client(self, img, rois=None):
        img_msg = self._br.cv2_to_imgmsg(img)
        rois = [] if rois is None else rois.reshape(-1).tolist()
        res = self._obj_det(img_msg, False, rois)
        return res.num_box, res.bbox, res.cls, res.cls_scores

    def _grasp_client(self, img, bbox):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._grasp_det(img_msg, bbox)
        return res.grasps

    def _tpn_client(self, img, bbox, expr):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._tpn(img_msg, bbox, expr)
        return res.grounding_scores, res.rel_score_mat

    def _object_detection(self, img, rois=None):
        obj_result = self._faster_rcnn_client(img, rois)
        num_box = obj_result[0]
        if num_box == 0:
            return None, None, None

        bboxes = np.array(obj_result[1]).reshape(num_box, -1)
        bboxes = bboxes[:, :4]
        classes = np.array(obj_result[2]).reshape(num_box, 1)
        class_scores = np.array(obj_result[3]).reshape(num_box, -1)
        bboxes, classes, class_scores = self._bbox_filter(bboxes, classes, class_scores)

        class_names = [CLASSES[i[0]] for i in classes]
        logger.info('_object_detection: \n{}'.format(bboxes))
        logger.info('_object_detection classes: {}'.format(class_names))
        return bboxes, classes, class_scores

    def rel_score_process(self, rel_score_mat):
        '''
        rel_mat: np.array of size [num_box, num_box]
                 where rel_mat[i, j] is the relationship between obj i and obj j.
                 1 means i is the parent of j.
                 2 means i is the child of j.
                 3 means i has no relation to j.
        '''
        rel_mat = np.argmax(rel_score_mat, axis=0) + 1
        return rel_mat, rel_score_mat

    def _grasp_post_process(self, bbox_to_pool, grasps):
        for k, v in bbox_to_pool.items():
            self.object_pool[v]["grasps"] = grasps[k]

    def _bbox_post_process(self, bboxes, scores):
        not_removed = [i for i, o in enumerate(self.object_pool) if not o["removed"]]
        no_rmv_to_pool = OrderedDict(zip(range(len(not_removed)), not_removed))
        prev_boxes = np.array([b["bbox"] for b in self.object_pool if not b["removed"]])
        prev_scores = np.array([np.array(b["cls_scores"]).mean(axis=0).tolist() for b in self.object_pool if not b["removed"]])
        det_to_no_rmv = self._bbox_match(bboxes, prev_boxes, scores, prev_scores)
        det_to_pool = {i: no_rmv_to_pool[v] for i, v in det_to_no_rmv.items()}
        pool_to_det = {v: i for i, v in det_to_pool.items()}
        not_matched = set(range(bboxes.shape[0])) - set(det_to_pool.keys())
        # updating the information of matched bboxes
        for k, v in det_to_pool.items():
            self.object_pool[v]["bbox"] = bboxes[k]
            self.object_pool[v]["cls_scores"].append(scores[k].tolist())
        for i in range(len(self.object_pool)):
            if i not in det_to_pool.values():
                self.object_pool[i]["removed"] = True
        # initialize newly detected bboxes
        for i in not_matched:
            new_box = self._init_object(bboxes[i], scores[i])
            self.object_pool.append(new_box)
            det_to_pool[i] = len(self.object_pool) - 1
            pool_to_det[len(self.object_pool) - 1] = i
            for j in range(len(self.object_pool[:-1])):
                # initialize relationship belief
                new_rel = self._init_relation(np.array([0.33, 0.33, 0.34]))
                self.rel_pool[(j, det_to_pool[i])] = new_rel

    def _get_valid_obj_candidates(self, renew=False):
        if renew:
            obj_inds = [i for i, obj in enumerate(self.object_pool)
                        if not obj["removed"] and np.array(obj["cls_scores"]).mean(axis=0)[1:].max() > 0.5]
            self.pool_to_det = OrderedDict(zip(obj_inds, list(range(len(obj_inds)))))
            self.det_to_pool = OrderedDict(zip(list(range(len(obj_inds))), obj_inds))
            self.obj_num = len(self.pool_to_det)
        return self.pool_to_det, self.det_to_pool, self.obj_num

    def _initialize_cls_filter(self):
        subj_str = ''.join(self.subject)
        cls_filter = []
        for cls in CLASSES:
            cls_str = ''.join(cls.split(" "))
            if cls_str in subj_str or subj_str in cls_str:
                cls_filter.append(cls)
        assert len(cls_filter) <= 1
        return cls_filter

    def _multi_step_grounding(self, mattnet_score, det_to_pool):
        cls_filter = self._initialize_cls_filter()
        disable_cls_filter= False
        if len(cls_filter) == 0:
            # no filter is available
            disable_cls_filter = True
        cand_neg_prior = []
        cand_neg_llh = []
        cand_pos_prior = []
        cand_pos_llh = []
        for i, score in enumerate(mattnet_score):
            pool_ind = det_to_pool[i]
            self.object_pool[pool_ind]["cand_belief"].update(score, self.obj_kdes)
            self.object_pool[pool_ind]["ground_scores_history"].append(score)

            # likelihood
            cand_neg_llh.append(self.object_pool[pool_ind]["cand_belief"].belief[0])
            cand_pos_llh.append(self.object_pool[pool_ind]["cand_belief"].belief[1])

            if not disable_cls_filter:
                # prior prob
                p_pos_prior = 0
                p_neg_prior = 0
                cls_scores = np.array(self.object_pool[pool_ind]["cls_scores"]).mean(axis=0)
                for cls in CLASSES:
                    if cls in cls_filter:
                        p_pos_prior += cls_scores[CLASSES_TO_IND[cls]]
                    else:
                        p_neg_prior += cls_scores[CLASSES_TO_IND[cls]]
                cand_pos_prior.append(p_pos_prior)
                cand_neg_prior.append(p_neg_prior)
            else:
                cand_pos_prior.append(1.)
                cand_neg_prior.append(1.)

        p_cand_pos = np.array(cand_pos_prior) * np.array(cand_pos_llh)
        p_cand_neg = np.array(cand_neg_prior) * np.array(cand_neg_llh)

        # normalize the distribution
        p_cand = p_cand_pos / (p_cand_pos + p_cand_neg)

        ground_result = self._cal_target_prob_from_p_cand(p_cand)
        ground_result = np.append(ground_result, max(0.0, 1. - ground_result.sum()))
        return ground_result

    def _multi_step_mrt_estimation(self, rel_score_mat, det_to_pool):
        # multi-step observations for relationship probability estimation
        box_inds = det_to_pool.keys()
        # update the relation pool
        rel_prob_mat = np.zeros((3, len(box_inds), len(box_inds)))
        for i in range(len(box_inds)):
            box_ind_i = box_inds[i]
            for j in range(i + 1, len(box_inds)):
                box_ind_j = box_inds[j]
                pool_ind_i = det_to_pool[box_ind_i]
                pool_ind_j = det_to_pool[box_ind_j]
                if pool_ind_i < pool_ind_j:
                    rel_score = rel_score_mat[:, box_ind_i, box_ind_j]
                    self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].update(rel_score, self.rel_kdes)
                    rel_prob_mat[:, box_ind_i, box_ind_j] = self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief
                    rel_prob_mat[:, box_ind_j, box_ind_i] = [
                        self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[1],
                        self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[0],
                        self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[2], ]
                else:
                    rel_score = rel_score_mat[:, box_ind_j, box_ind_i]
                    self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].update(rel_score, self.rel_kdes)
                    rel_prob_mat[:, box_ind_j, box_ind_i] = self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief
                    rel_prob_mat[:, box_ind_i, box_ind_j] = [
                        self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief[1],
                        self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief[0],
                        self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief[2], ]
        return rel_prob_mat

    def _get_leaf_desc_prob_from_rel_mat(self, rel_prob_mat, sample_num = 1000, with_virtual_node=True):

        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(rel_prob_mat[0].shape), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            leaf_desc_prob, desc_prob, leaf_prob, desc_num, ance_num = \
                self._leaf_and_desc_estimate(torch.from_numpy(rel_prob_mat) * triu_mask, sample_num,
                                             with_virtual_node=with_virtual_node)

        leaf_desc_prob = leaf_desc_prob.numpy()
        desc_prob = desc_prob.numpy()
        leaf_prob = leaf_prob.numpy()
        desc_num = desc_num.numpy()
        ance_num = ance_num.numpy()

        logger.debug('leaf_desc_prob: \n{}'.format(leaf_desc_prob))
        logger.debug('leaf_prob: \n{}'.format(leaf_prob))

        return leaf_desc_prob, desc_prob, leaf_prob, desc_num, ance_num

    def _leaf_and_desc_estimate(self, rel_prob_mat, sample_num=1000, with_virtual_node=False, removed=None):
        # TODO: Numpy may support a faster implementation.
        def sample_trees(rel_prob, sample_num=1):
            return torch.multinomial(rel_prob, sample_num, replacement=True)

        cuda_data = False
        if rel_prob_mat.is_cuda:
            # this function runs much faster on CPU.
            cuda_data = True
            rel_prob_mat = rel_prob_mat.cpu()

        # add virtual node, with uniform relation priors
        num_obj = rel_prob_mat.shape[-1]
        if with_virtual_node:
            if removed is None:
                removed = []
            removed = torch.tensor(removed).long()
            v_row = torch.zeros((3, 1, num_obj + 1)).type_as(rel_prob_mat)
            v_column = torch.zeros((3, num_obj, 1)).type_as(rel_prob_mat)
            # no other objects can be the parent node of the virtual node,
            # i.e., we assume that the virtual node must be a root node
            # 1) if the virtual node is the target, its parents can be ignored
            # 2) if the virtual node is not the target, such a setting will
            # not affect the relationships among other nodes
            v_column[0] = 0
            v_column[1] = 1. / 3.
            v_column[2] = 2. / 3. # add the prob of "parent" onto the prob of "no rel"
            v_column[1, removed] = 0.
            v_column[2, removed] = 1.
            rel_prob_mat = torch.cat(
                [torch.cat([rel_prob_mat, v_column], dim=2),
                 v_row], dim=1)
        else:
            # initialize the virtual node to have no relationship with other objects
            v_row = torch.zeros((3, 1, num_obj + 1)).type_as(rel_prob_mat)
            v_column = torch.zeros((3, num_obj, 1)).type_as(rel_prob_mat)
            v_column[2, :, 0] = 1
            rel_prob_mat = torch.cat(
                [torch.cat([rel_prob_mat, v_column], dim=2),
                 v_row], dim=1)

        rel_prob_mat = rel_prob_mat.permute((1, 2, 0))
        mrt_shape = rel_prob_mat.shape[:2]
        rel_prob = rel_prob_mat.view(-1, 3)
        rel_valid_ind = rel_prob.sum(-1) > 0

        # sample mrts
        samples = sample_trees(rel_prob[rel_valid_ind], sample_num) + 1
        mrts = torch.zeros((sample_num,) + mrt_shape).type_as(samples)
        mrts = mrts.view(sample_num, -1)
        mrts[:, rel_valid_ind] = samples.permute((1, 0))
        mrts = mrts.view((sample_num,) + mrt_shape)
        p_mats = (mrts == 1)
        c_mats = (mrts == 2)
        adj_mats = p_mats + c_mats.transpose(1, 2)

        def v_node_is_leaf(adj_mat):
            return adj_mat[-1].sum() == 0

        def no_cycle(adj_mat):
            keep_ind = (adj_mat.sum(0) > 0)
            if keep_ind.sum() == 0:
                return True
            elif keep_ind.sum() == adj_mat.shape[0]:
                return False
            adj_mat = adj_mat[keep_ind][:, keep_ind]
            return no_cycle(adj_mat)

        def descendants(adj_mat):
            def find_children(node, adj_mat):
                return torch.nonzero(adj_mat[node]).view(-1).tolist()

            def find_descendant(node, adj_mat, visited, desc_mat):
                if node in visited:
                    return visited, desc_mat
                else:
                    desc_mat[node][node] = 1
                    for child in find_children(node, adj_mat):
                        visited, desc_mat = find_descendant(child, adj_mat, visited, desc_mat)
                        desc_mat[node] = (desc_mat[node] | desc_mat[child])
                    visited.append(node)
                return visited, desc_mat

            roots = torch.nonzero(adj_mat.sum(0) == 0).view(-1).tolist()
            visited = []
            desc_mat = torch.zeros(mrt_shape).type_as(adj_mat).long()
            for root in roots:
                visited, desc_list = find_descendant(root, adj_mat, visited, desc_mat)
            return desc_mat.transpose(0, 1)

        leaf_desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
        desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
        desc_num = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)
        ance_num = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)

        if with_virtual_node:
            v_desc_num_after_q2 = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)

        count = 0
        for adj_mat in adj_mats:
            if removed is None and with_virtual_node and v_node_is_leaf(adj_mat):
                continue
            if not no_cycle(adj_mat):
                continue
            else:
                desc_mat = descendants(adj_mat)
                desc_num += desc_mat.sum(0)
                ance_num += desc_mat.sum(1) - 1  # ancestors don't include the object itself
                leaf_desc_mat = desc_mat * (adj_mat.sum(1, keepdim=True) == 0)
                desc_prob += desc_mat
                leaf_desc_prob += leaf_desc_mat
                count += 1

        desc_num /= count
        ance_num /= count
        leaf_desc_prob /= count
        desc_prob /= count
        leaf_prob = leaf_desc_prob.diag()
        if cuda_data:
            leaf_desc_prob = leaf_desc_prob.cuda()
            leaf_prob = leaf_prob.cuda()
            desc_prob = desc_prob.cuda()
            ance_num = ance_num.cuda()
            desc_num = desc_num.cuda()

        return leaf_desc_prob, desc_prob, leaf_prob, desc_num, ance_num

    def _cal_target_prob_from_p_cand(self, pcand, sample_num=100000):
        pcand = torch.Tensor(pcand).reshape(1, -1)
        pcand = pcand.repeat(sample_num, 1)
        sampled = torch.bernoulli(pcand)
        sampled_sum = sampled.sum(-1)
        sampled[sampled_sum > 0] /= sampled_sum[sampled_sum > 0].unsqueeze(-1)
        sampled = np.clip(sampled.mean(0).cpu().numpy(), 0.01, 0.99)
        if sampled.sum() > 1:
            sampled /= sampled.sum()
        return sampled

    def _grasp_filter(self, boxes, grasps, mode="high score"):
        """
        mode: "high score" or "near center"
        TODO: support "collision free" mode
        """
        keep_g = []
        if mode == "near center":
            for i, b in enumerate(boxes):
                g = grasps[i]
                bcx = (b[0] + b[2]) / 2
                bcy = (b[1] + b[3]) / 2
                gcx = (g[:, 0] + g[:, 2] + g[:, 4] + g[:, 6]) / 4
                gcy = (g[:, 1] + g[:, 3] + g[:, 5] + g[:, 7]) / 4
                dis = np.power(gcx - bcx, 2) + np.power(gcy - bcy, 2)
                selected = np.argmin(dis)
                keep_g.append(g[selected])
        elif mode == "high score":
            for i, b in enumerate(boxes):
                g = grasps[i]
                selected = np.argmax(g[:, -1])
                keep_g.append(g[selected])
        return np.array(keep_g)

    def _bbox_filter(self, bbox, cls, cls_scores):
        # apply NMS
        bbox_scores = np.max(cls_scores, axis=1)
        keep = nms(torch.from_numpy(bbox), torch.from_numpy(bbox_scores), 0.7)
        keep = keep.view(-1).numpy().tolist()
        for i in range(bbox.shape[0]):
            if i not in keep and bbox_scores[i] > 0.9:
                keep.append(i)
        bbox = bbox[keep]
        cls = cls[keep]
        cls_scores = cls_scores[keep]
        return bbox, cls, cls_scores

    def _process_user_answer(self, answer):
        answer = answer.lower()

        subject = self.subject
        is_subject_informative = len(self._initialize_cls_filter()) > 0
        if is_subject_informative:
            subject = " ".join(subject)
            # replace the pronoun in the answer with the subject given by the user
            for pronoun in PRONOUNS:
                if pronoun in answer:
                    answer = answer.replace(pronoun, subject)

        answer = answer.replace(",", " ")  # delete all , in the answer
        answer = answer.replace(".", " ")  # delete all . in the answer
        answer = answer.replace("!", " ")  # delete all . in the answer
        answer = ' '.join(answer.split()).strip().split(' ')

        response = None
        for neg_ans in NEGATIVE_ANS:
            if neg_ans in answer:
                response = False
                answer.remove(neg_ans)

        for pos_ans in POSITIVE_ANS:
            if pos_ans in answer:
                assert response is None, "A positive answer should not appear with a negative answer"
                response = True
                answer.remove(pos_ans)

        answer = ' '.join(answer)

        return response, answer

    def _find_subject(self, expr, nlp_server="nltk"):
        if nlp_server == "nltk":
            text = nltk.word_tokenize(expr)
            pos_tags = nltk.pos_tag(text)
        else:
            doc = self.stanford_nlp_server(expr)
            pos_tags = [(d.text, d.xpos) for d in doc.sentences[0].words]

        subj_tokens = []
        for i, (token, postag) in enumerate(pos_tags):
            if postag in {"NN"}:
                subj_tokens.append(token)
                for j in range(i + 1, len(pos_tags)):
                    token, postag = pos_tags[j]
                    if postag in {"NN"}:
                        subj_tokens.append(token)
                    else:
                        break
                return subj_tokens

        return subj_tokens

    def _bbox_match(self, bbox, prev_bbox, scores=None, prev_scores=None, mode = "hungarian"):
        # match bboxes between two steps.
        def bbox_overlaps(anchors, gt_boxes):
            """
            anchors: (N, 4) ndarray of float
            gt_boxes: (K, 4) ndarray of float

            overlaps: (N, K) ndarray of overlap between boxes and query_boxes
            """
            N = anchors.size(0)
            K = gt_boxes.size(0)

            gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                        (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

            anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                        (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

            boxes = anchors.view(N, 1, 4).expand(N, K, 4)
            query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

            iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
                torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
                torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
            ih[ih < 0] = 0

            ua = anchors_area + gt_boxes_area - (iw * ih)
            overlaps = iw * ih / ua

            return overlaps

        if prev_bbox.size == 0:
            return {}
        ovs = bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
        ind_match_dict = {}
        if mode == "heuristic":
            # match bboxes between two steps.
            # TODO: THERE ARE BUGS HERE, WHICH ASSUMES THAT THE LAST DIM OF BBOX IS THE CLS
            cls_mask = np.zeros(ovs.shape, dtype=np.uint8)
            for i, cls in enumerate(bbox[:, -1]):
                cls_mask[i][prev_bbox[:, -1] == cls] = 1
            ovs_mask = (ovs > 0.8)
            ovs *= ((cls_mask + ovs_mask) > 0)
            mapping = np.argsort(ovs, axis=-1)[:, ::-1]
            ovs_sorted = np.sort(ovs, axis=-1)[:, ::-1]
            matched = (np.max(ovs, axis=-1) > 0.5)
            occupied = {i: False for i in range(mapping.shape[-1])}
            for i in range(mapping.shape[0]):
                if matched[i]:
                    for j in range(mapping.shape[-1]):
                        if not occupied[mapping[i][j]] and ovs_sorted[i][j] > 0.5:
                            ind_match_dict[i] = mapping[i][j]
                            occupied[mapping[i][j]] = True
                            break
                        elif ovs_sorted[i][j] <= 0.5:
                            break
        elif mode == "hungarian":
            ov_cost = 1. - ovs
            # normalize scores
            scores /= np.expand_dims(np.linalg.norm(scores, axis=-1), axis=-1)
            prev_scores /= np.expand_dims(np.linalg.norm(prev_scores, axis=-1), axis=-1)
            scores_cost = np.expand_dims(scores, 1) * np.expand_dims(prev_scores, 0)
            scores_cost = 1 - scores_cost.sum(-1)
            cost = 0.6 * ov_cost + 0.4 * scores_cost
            mapping = optimize.linear_sum_assignment(cost)

            thresh = 0.5
            for i in range(mapping[0].size):
                ind1 = mapping[0][i]
                ind2 = mapping[1][i]
                if cost[ind1][ind2] < thresh:
                    ind_match_dict[ind1] = ind2

        return ind_match_dict

class InvigorateMultiSingleStepComparison(Invigorate):
    def _cal_target_prob_from_ground_score(self, ground_scores):
        bg_score = 0.25
        ground_scores = np.append(ground_scores, bg_score)
        return f.softmax(torch.FloatTensor(ground_scores), dim=0).numpy()

    def estimate_state_with_observation_singlestep(self, observations):
        logger.info("Singlestep: estimate_state_with_observation")

        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']

        # Estimate leaf_and_desc_prob and target_prob according to multi-step observations
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        # NOTE: here no multi-step for both rel_prob_mat and target_prob
        rel_prob_mat = rel_score_mat
        leaf_desc_prob = self._get_leaf_desc_prob_from_rel_mat(rel_prob_mat)
        target_prob = self._cal_target_prob_from_ground_score(np.array(grounding_scores))
        logger.info('Step 1: raw grounding completed')
        logger.info('raw target_prob: {}'.format(target_prob))
        logger.info('raw leaf_desc_prob: \n{}'.format(leaf_desc_prob))

        return target_prob, rel_prob_mat

    def estimate_state_with_observation_multistep(self, observations):
        logger.info("Multistep: estimate_state_with_observation")

        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        ind_match_dict = observations['ind_match_dict']

        # Estimate leaf_and_desc_prob and target_prob according to multi-step observations
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        rel_prob_mat = self._multi_step_mrt_estimation(rel_score_mat, ind_match_dict)
        leaf_desc_prob = self._get_leaf_desc_prob_from_rel_mat(rel_prob_mat)
        target_prob = self._multi_step_grounding(grounding_scores, ind_match_dict)
        target_prob /= target_prob.sum()
        logger.info('Step 1: raw grounding completed')
        logger.debug('raw target_prob: {}'.format(target_prob))
        logger.debug('raw leaf_desc_prob: \n{}'.format(leaf_desc_prob))

        self.belief['leaf_desc_prob'] = leaf_desc_prob
        self.belief['target_prob'] = target_prob
        self.belief['clue_leaf_desc_prob'] = None

        return target_prob, rel_prob_mat