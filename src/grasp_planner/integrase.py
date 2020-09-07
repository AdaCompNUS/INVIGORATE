#!/usr/bin/env python
import warnings

import rospy
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from scipy import optimize
import os
from torchvision.ops import nms
import time
# from stanfordcorenlp import StanfordCoreNLP
import pickle as pkl
import os.path as osp

from vmrn.model.utils.net_utils import leaf_and_descendant_stats, inner_loop_planning, relscores_to_visscores
from vmrn.model.rpn.bbox_transform import bbox_overlaps
from vmrn.model.utils.density_estimator import object_belief, gaussian_kde
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection
from ingress_srv.ingress_srv import Ingress

# -------- Constants ---------

# MODEL_NAME = "all_in_one_FixObj_NoScorePostProc_ShareW_NoRelClsGrad.pth"
# MODEL_PATH = "output/vmrdcompv1/res101"

BG_SCORE = 0.25

Q2={
    "type1": "I have not found the target. Where is it?", # COMMON FORMAT
    "type2": "I have not found the target. Where is it?",         # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean the {:s}?"
}

classes = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

# -------- Static ---------- 
br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')
classes_to_ind = dict(zip(classes, range(len(classes))))

# -------- Code ---------- 
# NEW VERSION with MAttNet
class INTEGRASE(object):
    def __init__(self):
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        rospy.wait_for_service('vmrn_server')
        rospy.wait_for_service('mattnet_server')
        self.obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        self.vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        self.grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)
        # self.ingress_client = Ingress()

        self.history_scores = []
        self.object_pool = []
        self.clue = None
        self.target_in_pool = None
        self._init_kde()
        print('INTEGRASE init finished!!')

    def _init_kde(self):
        cur_dir = osp.dirname(osp.abspath(__file__))
        with open(osp.join(cur_dir, 'density_esti_train_data.pkl')) as f:
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
        kde_pos = gaussian_kde(pos_data)
        kde_neg = gaussian_kde(neg_data)
        self.kdes = [kde_neg, kde_pos]

    def faster_rcnn_client(self, img):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls

    def vmrn_client(self, img, bbox):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat, res.grasps

    def mattnet_client(self, img, bbox, cls, expr):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.grounding(img_msg, bbox, cls, expr)
        return res.ground_prob

    def score_to_prob(self, score):
        print("Grounding Score: ")
        print(score.tolist())
        prob = torch.nn.functional.softmax(10 * torch.from_numpy(score), dim=0)
        return prob.numpy()

    def bbox_filter(self, bbox, cls):
        # apply NMS
        keep = nms(torch.from_numpy(bbox[:, :-1]), torch.from_numpy(bbox[:, -1]), 0.7)
        keep = keep.view(-1).numpy().tolist()
        for i in range(bbox.shape[0]):
            if i not in keep and bbox[i][-1] > 0.9:
                keep.append(i)
        bbox = bbox[keep]
        cls = cls[keep]
        return bbox, cls

    def bbox_match(self, bbox, prev_bbox, scores, prev_scores, mode="hungarian"):
        # TODO: apply Hungarian algorithm to match boxes
        if prev_bbox.size == 0:
            return {}
        ovs = bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
        ind_match_dict = {}
        if mode=="heuristic":
            # match bboxes between two steps.
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

        elif mode=="hungarian":
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

    def update_belief(self, belief, a, ans, data):
        ground_belief = belief["ground_prob"].cpu().numpy()
        leaf_desc_belief = belief["leaf_desc_prob"].cpu().numpy()
        num_box = ground_belief.shape[0] - 1
        ans = ans.lower()

        if a < 2 * num_box:
            return belief
        # Q1
        elif a < 3 * num_box:
            if ans in {"yes", "yeah", "yep", "sure"}:
                ground_belief[:] = 0
                ground_belief[a - 2 * num_box] = 1
            elif ans in {"no", "nope", "nah"}:
                ground_belief[a - 2 * num_box] = 0
                ground_belief /= np.sum(ground_belief)
        # Q2
        else:
            if ans in {"yes", "yeah", "yep", "sure"}:
                ground_belief[-1] = 0
                ground_belief /= np.sum(ground_belief)
            else:
                ground_belief[:] = 0
                ground_belief[-1] = 1
                img = data["img"]
                bbox = data["bbox"]
                cls = data["cls"]
                expr = ans

                self.clue = expr
                t_ground = self.mattnet_client(img, bbox, cls, expr)
                ind_match_dict = data["mapping"]
                for i, score in enumerate(t_ground):
                    obj_ind = ind_match_dict[i]
                    self.object_pool[obj_ind]["clue"] = self.clue
                    self.object_pool[obj_ind]["clue_belief"].reset()
                    self.object_pool[obj_ind]["clue_belief"].update(score, self.kdes)
                pcand = [self.object_pool[ind_match_dict[i]]["clue_belief"].belief[1] for i in range(num_box)]
                t_ground = self.p_cand_to_belief_mc(pcand)
                t_ground = np.expand_dims(t_ground, 0)
                leaf_desc_belief[:, -1] = (t_ground * leaf_desc_belief[:, :-1]).sum(-1)

        belief["ground_prob"] = torch.from_numpy(ground_belief)
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_belief)
        return belief

    def grasp_filter(self, boxes, grasps):
        keep_g = []
        for i, b in enumerate(boxes):
            g = grasps[i]
            bcx = (b[0] + b[2]) / 2
            bcy = (b[1] + b[3]) / 2
            gcx = (g[:, 0] + g[:, 2] + g[:, 4] + g[:, 6]) / 4
            gcy = (g[:, 1] + g[:, 3] + g[:, 5] + g[:, 7]) / 4
            dis = np.power(gcx - bcx, 2) + np.power(gcy - bcy, 2)
            selected = np.argmin(dis)
            keep_g.append(g[selected])
        return np.array(keep_g)

    def single_step_perception_new(self, img, expr, cls_filter=None):
        tb = time.time()
        obj_result = self.faster_rcnn_client(img)
        bboxes = np.array(obj_result[1]).reshape(-1, 4 + 32)
        cls = np.array(obj_result[2]).reshape(-1, 1)
        bboxes, cls = self.bbox_filter(bboxes, cls)

        scores = bboxes[:, 4:].reshape(-1, 32)
        bboxes = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, cls], axis=-1)
        num_box = bboxes.shape[0]

        prev_boxes = np.array([b["bbox"] for b in self.object_pool])
        prev_scores = np.array([b["cls_scores"] for b in self.object_pool])
        ind_match_dict = self.bbox_match(bboxes, prev_boxes, scores, prev_scores)
        not_matched = set(range(bboxes.shape[0])) - set(ind_match_dict.keys())
        ignored = set(range(prev_boxes.shape[0])) - set(ind_match_dict.values())

        # updating the information of matched bboxes
        for k, v in ind_match_dict.items():
            self.object_pool[v]["bbox"] = bboxes[k]
            self.object_pool[v]["cls_scores"] = scores[k]

        # add history ignored bounding boxes.
        for i, o in enumerate(self.object_pool):
            if i not in ind_match_dict.values() and o["removed"] == False and o["ground_belief"] > 0.5:
                bboxes = np.append(bboxes, o["bbox"])
                scores = np.append(scores, o["cls_scores"])
                ind_match_dict[num_box] = i
                num_box += 1
        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1, 32)

        # initialize newly detected bboxes
        for i in not_matched:
            new_box = {}
            new_box["bbox"] = bboxes[i]
            new_box["cls_scores"] = scores[i]
            new_box["cand_belief"] = object_belief()
            new_box["ground_belief"] = 0.
            new_box["ground_scores_history"] = []
            new_box["clue"] = None
            new_box["clue_belief"] = object_belief()
            # whether this box has been confirmed by user's answer
            # True: confirmed to be the target
            # False: confirmed not to be the target
            # None: not confirmed
            if self.target_in_pool:
                new_box["is_target"] = False
            else:
                new_box["is_target"] = None
            new_box["removed"] = False
            self.object_pool.append(new_box)
            ind_match_dict[i] = len(self.object_pool) - 1

        # detect vmr and grasp pose
        rel_result = self.vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        grasps = np.array(rel_result[2]).reshape((num_box, 5, -1))
        grasps = self.grasp_filter(bboxes, grasps)

        # TODO: updating the relationship probability according to the new observation
        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            leaf_desc_prob = leaf_and_descendant_stats(torch.from_numpy(rel_score_mat) * triu_mask).numpy()

        # visual grounding
        ground_score = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, 4].reshape(-1).tolist(), expr)
        for i, score in enumerate(ground_score):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["cand_belief"].update(score, self.kdes)
            self.object_pool[obj_ind]["ground_scores_history"].append(score)
        pcand = [self.object_pool[ind_match_dict[i]]["cand_belief"].belief[1] for i in range(num_box)]
        ground_result = self.p_cand_to_belief_mc(pcand)
        ground_result = np.append(ground_result, 1. - ground_result.sum())

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += scores[i][classes_to_ind[class_str]]
            if box_score < 0.02:
                ground_result[i] = 0.
        ground_result /= ground_result.sum()

        # 2. incorporate QA history
        ground_result_backup = ground_result.copy()
        for k, v in ind_match_dict.items():
            if self.object_pool[v]["is_target"] == True:
                ground_result[:] = 0.
                ground_result[k] = 1.
            elif self.object_pool[v]["is_target"] == False:
                ground_result[k] = 0.
        if self.target_in_pool:
            ground_result[-1] = 0
        assert ground_result.sum() > 0
        ground_result /= ground_result.sum()

        # update ground belief
        for k, v in ind_match_dict.items():
            self.object_pool[v]["ground_belief"] = ground_result[k]

        # 3. incorporate the provided clue by the user
        if self.clue is not None:
            t_ground = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, 4].reshape(-1).tolist(), self.clue)
            for i, score in enumerate(t_ground):
                obj_ind = ind_match_dict[i]
                self.object_pool[obj_ind]["clue_belief"].update(score, self.kdes)
            pcand = [self.object_pool[ind_match_dict[i]]["clue_belief"].belief[1] for i in range(num_box)]
            t_ground = self.p_cand_to_belief_mc(pcand)
            t_ground = np.expand_dims(t_ground, 0)
            leaf_desc_prob[:, -1] = (t_ground * leaf_desc_prob[:, :-1]).sum(-1)

        print("Perception Time Consuming: " + str(time.time() - tb) + "s")
        return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, ind_match_dict, grasps

    def p_cand_to_belief_mc(self, pcand, sample_num=100000):
        pcand = torch.Tensor(pcand).reshape(1, -1)
        pcand = pcand.repeat(sample_num, 1)
        sampled = torch.bernoulli(pcand)
        sampled_sum = sampled.sum(-1)
        sampled[sampled_sum > 0] /= sampled_sum[sampled_sum > 0].unsqueeze(-1)
        sampled = np.clip(sampled.mean(0).cpu().numpy(), 0.01, 0.99)
        if sampled.sum() > 1:
            sampled /= sampled.sum()
        return sampled

    def single_step_perception(self, img, expr, prevs=None, cls_filter=None):
        tb = time.time()
        obj_result = self.faster_rcnn_client(img)
        bboxes = np.array(obj_result[1]).reshape(-1, 4 + 32)
        cls = np.array(obj_result[2]).reshape(-1, 1)
        bboxes, cls = self.bbox_filter(bboxes, cls)

        scores = bboxes[:, 4:].reshape(-1, 32)
        bboxes = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, cls], axis=-1)

        ind_match_dict = {}
        if prevs is not None:
            ind_match_dict = self.bbox_match(bboxes, prevs["bbox"])
            self.history_scores[-1]["mapping"] = ind_match_dict
            not_matched = set(range(bboxes.shape[0])) - set(ind_match_dict.keys())
            ignored = set(range(prevs["bbox"].shape[0])) - set(ind_match_dict.values())

            # ignored = list(ignored - {prevs["actions"]})
            # bboxes = np.concatenate([bboxes, prevs["bbox"][ignored]], axis=0)
            # cls = np.concatenate([cls, prevs["cls"][ignored]], axis=0)
            prevs["qa_his"] = self.qa_his_mapping(prevs["qa_his"], ind_match_dict, not_matched, ignored)

        num_box = bboxes.shape[0]

        rel_result = self.vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        if prevs is not None:
            # modify the relationship probability according to the new observation
            rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] += \
                prevs["rel_score_mat"][:, ind_match_dict.values()][:, :, ind_match_dict.values()]
            rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] /= 2

        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            leaf_desc_prob = leaf_and_descendant_stats(torch.from_numpy(rel_score_mat) * triu_mask).numpy()

        ground_score = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), cls.reshape(-1).tolist(), expr)
        bg_score = BG_SCORE
        ground_score += (bg_score,)
        ground_score = np.array(ground_score)
        self.history_scores.append({"scores" : ground_score})
        if prevs is not None:
            # ground_score[ind_match_dict.keys()] += prevs["ground_score"][ind_match_dict.values()]
            # ground_score[ind_match_dict.keys()] /= 2
            ground_score[ind_match_dict.keys()] = np.maximum(
                prevs["ground_score"][ind_match_dict.values()],
                ground_score[ind_match_dict.keys()])
        ground_result = self.score_to_prob(ground_score)

        # utilize the answered questions to correct grounding results.
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += scores[i][classes_to_ind[class_str]]
            if box_score < 0.02:
                ground_result[i] = 0
        ground_result /= ground_result.sum()

        if prevs is not None:
            ground_result_backup = ground_result.copy()
            for k in prevs["qa_his"].keys():
                if k == "bg":
                    # target has already been detected in the last step
                    for i in not_matched:
                        ground_result[i] = 0
                    ground_result[-1] = 0
                elif k == "clue":
                    clue = prevs["qa_his"]["clue"]
                    t_ground = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), cls.reshape(-1).tolist(), clue)
                    t_ground += (BG_SCORE,)
                    t_ground = self.score_to_prob(np.array(t_ground))
                    t_ground = np.expand_dims(t_ground, 0)
                    leaf_desc_prob[:, -1] = (t_ground[:, :-1] * leaf_desc_prob[:, :-1]).sum(-1)
                else:
                    ground_result[k] = 0

            if ground_result.sum() > 0:
                ground_result /= ground_result.sum()
            else:
                # something wrong with the matching process. roll back
                for i in not_matched:
                    ground_result[i] = ground_result_backup[i]
                ground_result[-1] = ground_result_backup[-1]

        print("Perception Time Consuming: " + str(time.time() - tb) + "s")

        if prevs is not None:
            return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, prevs["qa_his"]
        else:
            return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, {}

    def qa_his_mapping(self, qa_his, ind_match_dict, not_matched, ignored):
        new_qa_his = {}
        for key, v in ind_match_dict.items():
            if v in qa_his.keys():
                new_qa_his[key] = qa_his[v]

        if "bg" in qa_his.keys():
            new_qa_his["bg"] = qa_his["bg"]
            for i in not_matched:
                new_qa_his[i] = 0

        if "clue" in qa_his.keys(): new_qa_his["clue"] = qa_his["clue"]
        return new_qa_his

