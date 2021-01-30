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
import nltk
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
        self.clue = None
        self.q2_num_asked = 0

        self.data_viewer = DataViewer(CLASSES)
        try:
            self.stanford_nlp_server = stanza.Pipeline("en")
        except:
            warnings.warn("stanza needs python 3.6 or higher. "
                          "please update your python version and run 'pip install stanza'")

    def _merge_bboxes(self, bboxes, classes, scores, bboxes_his, classes_his, scores_his):
        curr_to_his = self._bbox_match(bboxes, bboxes_his, scores, scores_his)
        for i in range(bboxes_his.shape[0]):
            if i not in curr_to_his.values():
                bboxes = np.concatenate([bboxes, bboxes_his[i][None, :]], axis=0)
                classes = np.concatenate([classes, classes_his[i][None, :]], axis=0)
                scores = np.concatenate([scores, scores_his[i][None, :]], axis=0)
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

        # object detection
        bboxes, classes, scores = self._object_detection(img)
        if bboxes is None:
            logger.warning("WARNING: nothing is detected")
            return None
        print('--------------------------------------------------------')
        logger.info('Perceive_img: _object_detection finished')
        # double check the rois in our object pool
        rois = [o["bbox"][None, :] for o in self.object_pool if not o["removed"]]
        if len(rois) > 0:
            rois = np.concatenate(rois, axis=0)
            bboxes_his, classes_his, scores_his = self._object_detection(img, rois)
            bboxes, classes, scores = self._merge_bboxes(bboxes, classes, scores, bboxes_his, classes_his, scores_his)

        im1 = img.copy()
        im1 = self.data_viewer.draw_objdet(im1, np.concatenate([bboxes, classes], axis=1))
        plt.axis('off')
        plt.imshow(im1)
        plt.show()

        if len(rois) > 0:
            im2 = img.copy()
            im2 = self.data_viewer.draw_objdet(im2, np.concatenate([bboxes_his, classes_his], axis=1))
            plt.axis('off')
            plt.imshow(im2)
            plt.show()

        # relationship
        rel_mat, rel_score_mat = self._rel_det_client.detect_obr(img, bboxes)
        logger.info('Perceive_img: mrt detection finished')

        # grounding
        grounding_scores = self._vis_ground_client.ground(img, bboxes, expr, classes)
        logger.info('Perceive_img: mattnet grounding finished')

        # relationship
        # rel_score_mat = self._vis_ground_client.ground(img, bboxes, expr)

        # grasp
        grasps = self._grasp_det_client.detect_grasps(img, bboxes)
        grasps = self._grasp_filter(bboxes, grasps)
        logger.info('Perceive_img: grasp detection finished')

        # object and relationship detection post process
        rel_mat, rel_score_mat = self.rel_score_process(rel_score_mat)
        ind_match_dict, not_matched = self._bbox_post_process(bboxes, scores, rel_score_mat, grasps)

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
        observations['ind_match_dict'] = ind_match_dict
        observations['not_matched'] = not_matched
        observations['det_scores'] = scores
        observations['rel_mat'] = rel_mat
        observations['rel_score_mat'] = rel_score_mat
        observations['grounding_scores'] = grounding_scores
        observations['grasps'] = grasps

        self.observations = observations
        return observations

    def estimate_state_with_observation(self, observations):
        img = observations['img']
        bboxes = observations['bboxes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        ind_match_dict = observations['ind_match_dict']
        obj_inds, obj_num = self._get_valid_obj_candidates(renew=True)

        # Estimate rel_prob_mat and target_prob according to multi-step observations
        print('--------------------------------------------------------')
        logger.info('Step 1: raw grounding completed')
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        print('--------------------------------------------------------')

        rel_prob_mat = self._multi_step_mrt_estimation(rel_score_mat, ind_match_dict)
        target_prob = self._multi_step_grounding(grounding_scores, ind_match_dict, expr)

        print('--------------------------------------------------------')
        logger.info('Step 2: candidate prior probability from object detector incorporated')
        logger.debug('target_prob : {}'.format(target_prob))
        print('--------------------------------------------------------')

        print('--------------------------------------------------------')
        logger.info('After multistep, the results:')
        logger.info('after multistep, target_prob: {}'.format(target_prob))
        logger.info('after multistep, rel_score_mat: {}'.format(rel_prob_mat))
        print('--------------------------------------------------------')

        # 2. incorporate QA history TODO
        target_prob_backup = target_prob.copy()
        for k, v in ind_match_dict.items():
            if self.object_pool[v]["is_target"] == 1:
                target_prob[:] = 0.
                target_prob[obj_inds[v]] = 1.
            elif self.object_pool[v]["is_target"] == 0:
                target_prob[obj_inds[v]] = 0.
        # sanity check
        if target_prob.sum() > 0:
            target_prob /= target_prob.sum()
        else:
            # something wrong with the matching process. roll back
            target_prob = target_prob_backup
            for k, v in ind_match_dict.items():
                target_prob[obj_inds[v]] = 0
            target_prob /= target_prob.sum()

        # update target_prob
        for k, v in ind_match_dict.items():
            self.object_pool[v]["target_prob"] = target_prob[k]

        print('--------------------------------------------------------')
        logger.info('Step 3: incorporate QA history completed')
        logger.info('target_prob: {}'.format(target_prob))
        print('--------------------------------------------------------')

        bbox_id_to_pool_id = {v:k for k,v in obj_inds.items()}
        self.step_infos["bboxes"] = np.asarray([self.object_pool[bbox_id_to_pool_id[i]]["bbox"].tolist() for i in range(obj_num)])
        self.step_infos["classes"] = np.asarray([np.argmax(self.object_pool[bbox_id_to_pool_id[i]]["cls_scores"]) for i in range(obj_num)]).reshape(-1, 1)
        self.step_infos["grasps"] = np.asarray([self.object_pool[bbox_id_to_pool_id[i]]["grasp"].tolist() for i in range(obj_num)])
        self.belief['target_prob'] = target_prob
        self.belief['rel_prob'] = rel_prob_mat

    def estimate_state_with_user_answer(self, action, answer):

        target_prob = self.belief["target_prob"]
        num_box = target_prob.shape[0] - 1
        ans = answer.lower()
        obj_inds, obj_num = self._get_valid_obj_candidates()
        bbox_id_to_pool_id = {v: k for k, v in obj_inds.items()}

        action_type, _ = self.get_action_type(action)
        assert action_type == "Q1"

        print('--------------------------------------------------------')
        logger.info("Invigorate: handling answer for Q1")
        target_idx = action - 2 * num_box
        if ans in {"yes", "yeah", "yep", "sure"}:
            # set non-target
            target_prob[:] = 0
            for obj in self.object_pool:
                obj["is_target"] = 0
            # set target
            target_prob[target_idx] = 1
            self.object_pool[bbox_id_to_pool_id[target_idx]]["is_target"] = 1
        elif ans in {"no", "nope", "nah"}:
            target_prob[target_idx] = 0
            target_prob /= np.sum(target_prob)
            self.object_pool[bbox_id_to_pool_id[target_idx]]["is_target"] = 0

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
        obj_inds, obj_num = self._get_valid_obj_candidates()
        bbox_id_to_pool_id = {v: k for k, v in obj_inds.items()}

        action_type, _ = self.get_action_type(action)
        if action_type == 'Q1':
            # asking question does not change state
            return
        else:
            # mark object as being removed
            # self.object_pool[bbox_id_to_pool_id[action % obj_num]]["removed"] = True
            return

    def get_action_type(self, action, num_obj=None):
        '''
        @num_obj, the number of objects in the state, bg excluded.
        @return action_type, a readable string indicating action type
                target_idx, the index of the target.
        '''

        if num_obj is None:
            _, num_obj = self._get_valid_obj_candidates()

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
        with open(osp.join(KDE_MODEL_PATH, 'ground_density_estimation_mattnet.pkl')) as f:
            data = pkl.load(f)
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
            rel_data = pkl.load(f)
        parents = np.array([d["det_score"] for d in rel_data if d["gt"] == 1])
        children = np.array([d["det_score"] for d in rel_data if d["gt"] == 2])
        norel = np.array([d["det_score"] for d in rel_data if d["gt"] == 3])
        kde_parents = gaussian_kde(parents, bandwidth=0.2)
        kde_children = gaussian_kde(children, bandwidth=0.2)
        kde_norel = gaussian_kde(norel, bandwidth=0.2)
        self.rel_kdes = [kde_parents, kde_children, kde_norel]

    def _init_object(self, bbox, score, grasp):
        new_box = {}
        new_box["bbox"] = bbox
        new_box["cls_scores"] = score
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
        new_box["grasp"] = grasp
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

    def _bbox_post_process(self, bboxes, scores, rel_scores, grasps):
        prev_boxes = np.array([b["bbox"] for b in self.object_pool])
        prev_scores = np.array([b["cls_scores"] for b in self.object_pool])
        det_to_pool = self._bbox_match(bboxes, prev_boxes, scores, prev_scores)
        det_to_pool = {k: v for k, v in det_to_pool.items() if self.object_pool[v]["removed"] == False}
        pool_to_det = {v: i for i, v in det_to_pool.items()}
        not_matched = set(range(bboxes.shape[0])) - set(det_to_pool.keys())
        # updating the information of matched bboxes
        for k, v in det_to_pool.items():
            self.object_pool[v]["bbox"] = bboxes[k]
            self.object_pool[v]["cls_scores"] = scores[k]
            self.object_pool[v]["grasps"] = grasps[k]

        for i in range(len(self.object_pool)):
            if i not in pool_to_det.keys():
            #     self.object_pool[i]["num_det_failures"] += 1
            # if self.object_pool[i]["num_det_failures"] >= 2:
                self.object_pool[i]["removed"] = True

        # initialize newly detected bboxes
        for i in not_matched:
            new_box = self._init_object(bboxes[i], scores[i], grasps[i])
            self.object_pool.append(new_box)
            det_to_pool[i] = len(self.object_pool) - 1
            pool_to_det[len(self.object_pool) - 1] = i
            for j in range(len(self.object_pool[:-1])):
                if j in pool_to_det.keys():
                    # det_to_pool[i] > all possible j.
                    new_rel = self._init_relation(rel_scores[:, pool_to_det[j], i])
                else:
                    new_rel = self._init_relation(np.array([0.33, 0.33, 0.34]))
                self.rel_pool[(j, det_to_pool[i])] = new_rel
        return det_to_pool, not_matched

    def _get_valid_obj_candidates(self, renew=False):
        if renew:
            obj_inds = [i for i, obj in enumerate(self.object_pool) if not obj["removed"]]
            self.obj_inds = dict(zip(obj_inds, list(range(len(obj_inds)))))
            self.obj_num = len(self.obj_inds)
        return self.obj_inds, self.obj_num

    def _multi_step_grounding(self, mattnet_score, ind_match_dict, expr):
        # num_box = len(mattnet_score)
        for i, score in enumerate(mattnet_score):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["cand_belief"].update(score, self.obj_kdes)
            self.object_pool[obj_ind]["ground_scores_history"].append(score)
        cand_posterior = [obj["cand_belief"].belief[1] for obj in self.object_pool if not obj['removed']]
        # incorporate prior from the object detector
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        cand_prior = []
        for obj in self.object_pool:
            if obj["removed"]:
                continue
            p_prior = 0
            for class_str in cls_filter:
                p_prior += obj["cls_scores"][CLASSES_TO_IND[class_str]]
            cand_prior.append(p_prior)
        p_cand = (np.array(cand_prior) * np.array(cand_posterior)).tolist()
        ground_result = self._cal_target_prob_from_p_cand(p_cand)
        ground_result = np.append(ground_result, max(0.0, 1. - ground_result.sum()))
        return ground_result

    def _multi_step_mrt_estimation(self, rel_score_mat, ind_match_dict):
        # multi-step observations for relationship probability estimation
        box_inds = ind_match_dict.keys()
        # update the relation pool
        rel_prob_mat = np.zeros(rel_score_mat.shape)
        for i in range(len(box_inds)):
            box_ind_i = box_inds[i]
            for j in range(i + 1, len(box_inds)):
                box_ind_j = box_inds[j]
                pool_ind_i = ind_match_dict[box_ind_i]
                pool_ind_j = ind_match_dict[box_ind_j]
                if pool_ind_i < pool_ind_j:
                    rel_score = rel_score_mat[:, box_ind_i, box_ind_j]
                    self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].update(rel_score, self.rel_kdes)
                    rel_prob_mat[:, box_ind_i, box_ind_j] = self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief
                    rel_prob_mat[:, box_ind_j, box_ind_i] = [self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[1],
                                                             self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[0],
                                                             self.rel_pool[(pool_ind_i, pool_ind_j)]["rel_belief"].belief[2],]
                else:
                    rel_score = rel_score_mat[:, box_ind_j, box_ind_i]
                    self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].update(rel_score, self.rel_kdes)
                    rel_prob_mat[:, box_ind_j, box_ind_i] = self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief
                    rel_prob_mat[:, box_ind_i, box_ind_j] = [self.rel_pool[(pool_ind_j, pool_ind_i)]["rel_belief"].belief[1],
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