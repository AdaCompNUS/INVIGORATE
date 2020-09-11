'''
TODO
1. bug where only 1 object is detected
2. question asking for where is it?
3. grasp random objects when background has high prob?
4. The questions to be asked
5. clue is persistent??
6. what is this? if (target_prob[:-1] > 0.02).sum() == 1:
'''

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
import copy

from libraries.density_estimator.density_estimator import object_belief, gaussian_kde
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection

# -------- Constants ---------

# MODEL_NAME = "all_in_one_FixObj_NoScorePostProc_ShareW_NoRelClsGrad.pth"
# MODEL_PATH = "output/vmrdcompv1/res101"

BG_SCORE = 0.25

Q2={
    "type1": "I have not found the target. Where is it?",          # COMMON FORMAT
    "type2": "I have not found the target. Where is it?",          # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean the {:s}?"
}

CLASSES = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

CLASSES_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

class Invigorate():
    def __init__(self, robot):
        self._robot = robot
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        rospy.wait_for_service('vmrn_server')
        rospy.wait_for_service('mattnet_server')
        self.obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        self.vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        self.grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)
        
        self.br = CvBridge()

        self.history_scores = []
        self.object_pool = []
        self.target_in_pool = None
        self._init_kde()
        self.belief = {}

    def perceive_img(self, img, expr):
        '''
        @return bboxes,         [Nx5]
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
        num_bbox, bboxes, classes, cls_scores = self._faster_rcnn_client(img)
        bboxes = np.array(bboxes).reshape(num_bbox, -1)
        classes = np.array(classes).reshape(num_bbox, 1)
        scores = np.array(cls_scores).reshape(num_bbox, -1)
        bboxes, classes, scores = self._bbox_filter(bboxes, classes, scores)
        num_box = bboxes.shape[0]
        print('Step 1: faster-rcnn object detection completed!')

        bboxes = bboxes[:, :4]
        bboxes_and_classes = np.concatenate([bboxes, classes], axis=-1)

        prev_bboxes_and_classes = np.array([b["bbox_and_cls"] for b in self.object_pool])
        prev_scores = np.array([b["cls_scores"] for b in self.object_pool])
        ind_match_dict = self._bbox_match(bboxes_and_classes, prev_bboxes_and_classes, scores, prev_scores)
        not_matched = set(range(bboxes_and_classes.shape[0])) - set(ind_match_dict.keys())
        ignored = set(range(prev_bboxes_and_classes.shape[0])) - set(ind_match_dict.values())

        # updating the information of matched bboxes
        for k, v in ind_match_dict.items():
            self.object_pool[v]["bbox_and_cls"] = bboxes_and_classes[k]
            self.object_pool[v]["cls_scores"] = scores[k]

        # add history bounding boxes that are not detected in this timestamp 
        for i, o in enumerate(self.object_pool):
            if i not in ind_match_dict.values() and o["removed"] == False and o["ground_belief"] > 0.5:
                bboxes_and_classes = np.append(bboxes_and_classes, o["bbox"])
                scores = np.append(scores, o["cls_scores"])
                ind_match_dict[num_box] = i
                num_box += 1
        # bboxes = bboxes.reshape(-1, 5)
        # scores = scores.reshape(-1, 32)
        bboxes_and_classes = bboxes_and_classes.reshape(num_box, -1)
        scores = scores.reshape(num_box, -1)

        # initialize newly detected bboxes
        for i in not_matched:
            new_box = {}
            new_box["bbox"] = bboxes_and_classes[i]
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

        # decompose bbox and classes again
        bboxes = bboxes_and_classes[:, :4]
        classes = bboxes_and_classes[:, :-1]

        # detect vmr and grasp pose
        rel_result = self._vmrn_client(img, bboxes.reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box)) # [NxN]
        rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box)) #[3xNxN], 3 diff kinds of relationship
        grasps = np.array(rel_result[2]).reshape((num_box, 5, -1))
        grasps = self._grasp_filter(bboxes, grasps) 
        print('Step 2: mrt and grasp pose detection completed')

        # visual grounding
        grounding_scores = self._mattnet_client(img, bboxes.reshape(-1).tolist(), classes.reshape(-1).tolist(), expr)

        observations = {}
        observations['img'] = img
        observations['expr'] = expr
        observations['num_box'] = num_box
        observations['bboxes'] = bboxes
        observations['classes'] = classes
        observations['ind_match_dict'] = ind_match_dict
        observations['det_scores'] = scores
        observations['rel_mat'] = rel_mat
        observations['rel_score_mat'] = rel_score_mat
        observations['grounding_scores'] = grounding_scores
        observations['grasps'] = grasps

        self.observations = observations
        return observations

    def estimate_state_with_observation(self, observations):
        bboxes = observations['bboxes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        ind_match_dict = observations['ind_match_dict']

        num_box = observations['bboxes'].shape[0]

        # Estimate leaf_and_desc_prob
        # TODO: updating the relationship probability according to the new observation
        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            leaf_desc_prob = self._leaf_and_descendant_stats(torch.from_numpy(rel_score_mat) * triu_mask).numpy()

        for i, score in enumerate(grounding_scores):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["cand_belief"].update(score, self.kdes)
            self.object_pool[obj_ind]["ground_scores_history"].append(score)
        pcand = [self.object_pool[ind_match_dict[i]]["cand_belief"].belief[1] for i in range(num_box)]
        target_prob = self._cal_target_prob_from_p_cand(pcand)
        target_prob = np.append(target_prob, 1. - target_prob.sum()) # append bg
        print('Step 3: raw grounding completed')
        print('raw target_prob: {}'.format(target_prob))

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += det_scores[i][CLASSES_TO_IND[class_str]]
            if box_score < 0.02:
                target_prob[i] = 0.
        target_prob /= target_prob.sum()
        print('Step 3.1: class name filter completed')
        print('target_prob : {}'.format(target_prob))

        # 2. incorporate QA history
        # target_prob = target_prob.copy()
        for k, v in ind_match_dict.items():
            if self.object_pool[v]["is_target"] == True:
                target_prob[:] = 0.
                target_prob[k] = 1.
            elif self.object_pool[v]["is_target"] == False:
                target_prob[k] = 0.
        if self.target_in_pool:
            target_prob[-1] = 0
        assert target_prob.sum() > 0
        target_prob /= target_prob.sum()
        print('Step 3.2: incorporate QA history completed')
        print('target_prob: {}'.format(target_prob))

        # update target_prob
        for k, v in ind_match_dict.items():
            self.object_pool[v]["target_prob"] = target_prob[k]

        self.belief['leaf_desc_prob'] = leaf_desc_prob
        self.belief['target_prob'] = target_prob

    def estimate_state_with_user_answer(self, action, answer):
        target_prob = self.belief["target_prob"]
        leaf_desc_prob = self.belief["leaf_desc_prob"]
        num_box = self.observations['num_box']
        ind_match_dict = self.observations['ind_match_dict']
        ans = answer.lower()

        # Q1
        if self.get_action_type(action, num_box) == 'Q1':
            target_idx = action - 2 * num_box
            if ans in {"yes", "yeah", "yep", "sure"}:
                # set non-target
                target_prob[:] = 0
                for obj in self.object_pool:
                    obj["is_target"] = False
                # set target
                target_prob[target_idx] = 1
                self.object_pool[ind_match_dict[target_idx]]["is_target"] = True
            elif ans in {"no", "nope", "nah"}:
                target_prob[target_idx] = 0
                target_prob /= np.sum(target_prob)
                self.object_pool[ind_match_dict[target_idx]]["is_target"] = False
        # Q2
        elif self.get_action_type(action, num_box) == 'Q2':
            target_idx = np.argmax(target_prob[:-1])
            if ans in {"yes", "yeah", "yep", "sure"}:
                target_prob[:] = 0
                target_prob[target_idx] = 1
                target_prob /= np.sum(target_prob)
                self.object_pool[ind_match_dict[obj_ind]]["is_target"] = True
            else:
                self.object_pool[ind_match_dict[target_idx]]["is_target"] = False

                target_prob[:] = 0
                target_prob[-1] = 1

                leaf_desc_prob = self._estimate_state_with_user_clue(answer)

        self.belief["target_prob"] = torch.from_numpy()
        self.belief["leaf_desc_prob"] = torch.from_numpy()
        return belief

    def plan(self, planning_depth=3):
        # convert to torch # TODO numpy??
        belief = self.belief.copy()
        belief['target_prob'] = torch.from_numpy(self.belief['target_prob'])
        belief['leaf_desc_prob'] = torch.from_numpy(self.belief['leaf_desc_prob'])

        num_obj = belief["target_prob"].shape[0] - 1 # exclude the virtual node
        penalty_for_asking = -2
        # ACTIONS: Do you mean ... ? (num_obj) + Where is the target ? (1) + grasp object (num_obj)
        def grasp_reward_estimate(belief):
            # reward of grasping the corresponding object
            # return is a 1-D tensor including num_obj elements, indicating the reward of grasping the corresponding object.
            ground_prob = belief["target_prob"]
            leaf_desc_tgt_prob = (belief["leaf_desc_prob"] * ground_prob.unsqueeze(0)).sum(-1)
            leaf_prob = torch.diag(belief["leaf_desc_prob"])
            not_leaf_prob = 1. - leaf_prob
            target_prob = ground_prob
            leaf_tgt_prob = leaf_prob * target_prob
            leaf_desc_prob = leaf_desc_tgt_prob - leaf_tgt_prob
            leaf_but_not_desc_tgt_prob = leaf_prob - leaf_desc_tgt_prob

            # grasp and the end
            r_1 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-10) + leaf_desc_prob * (-10)\
                    + leaf_tgt_prob * (0)
            r_1 = r_1[:-1] # exclude the virtual node

            # grasp and not the end
            r_2 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-6) + leaf_desc_prob * (0)\
                    + leaf_tgt_prob * (-10)
            r_2 = r_2[:-1]  # exclude the virtual node
            return torch.cat([r_1, r_2], dim=0)

        def belief_update(belief):
            I = torch.eye(belief["target_prob"].shape[0]).type_as(belief["target_prob"])
            updated_beliefs = []
            # Do you mean ... ?
            # Answer No
            beliefs_no = belief["target_prob"].unsqueeze(0).repeat(num_obj + 1, 1)
            beliefs_no *= (1. - I)
            beliefs_no /= torch.clamp(torch.sum(beliefs_no, dim = -1, keepdim=True), min=1e-10)
            # Answer Yes
            beliefs_yes = I.clone()
            for i in range(beliefs_no.shape[0] - 1):
                updated_beliefs.append([beliefs_no[i], beliefs_yes[i]])

            # Is the target detected? Where is it?
            updated_beliefs.append([beliefs_no[-1], I[-1],])
            return updated_beliefs

        def is_onehot(vec, epsilon = 1e-2):
            return (torch.abs(vec - 1) < epsilon).sum().item() > 0

        def estimate_q_vec(belief, current_d):
            if current_d == planning_depth - 1:
                q_vec = grasp_reward_estimate(belief)
                return q_vec
            else:
                # branches of grasping
                q_vec = grasp_reward_estimate(belief).tolist()
                ground_prob = belief["target_prob"]
                new_beliefs = belief_update(belief)
                new_belief_dict = copy.deepcopy(belief)

                # Q1
                for i, new_belief in enumerate(new_beliefs[:-1]):
                    q = 0
                    for j, b in enumerate(new_belief):
                        new_belief_dict["target_prob"] = b
                        # branches of asking questions
                        if is_onehot(b):
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        else:
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                        if j == 0:
                            # Answer is No
                            q += t_q * (1 - ground_prob[i])
                        else:
                            # Answer is Yes
                            q += t_q * ground_prob[i]
                    q_vec.append(q.item())

                # Q2
                q = 0
                new_belief = new_beliefs[-1]
                for j, b in enumerate(new_belief):
                    new_belief_dict["ground_prob"] = b
                    if j == 0:
                        # target has been detected
                        if is_onehot(b):
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        else:
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                        q += t_q * (1 - ground_prob[-1])
                    else:
                        new_belief_dict["leaf_desc_prob"][:, -1] = new_belief_dict["leaf_desc_prob"][:, :-1].sum(-1) / num_obj
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        q += t_q * ground_prob[-1]
                q_vec.append(q.item())
                return torch.Tensor(q_vec).type_as(belief["ground_prob"])

        q_vec = estimate_q_vec(belief, 0)
        print("Q Value for Each Action: ")
        print(q_vec.tolist()[:num_obj])
        print(q_vec.tolist()[num_obj:2*num_obj])
        print(q_vec.tolist()[2*num_obj:3*num_obj])
        print(q_vec.tolist()[3*num_obj])
        best_action = torch.argmax(q_vec).item()
        return best_action

    def transit_state(self, action):
        action_type = self.get_action_type(action)
        if action_type == 'Q1' or action_type == 'Q2':
            # asking question does not change state
            return
        else:
            # mark object as being removed
            self.object_pool[ind_match[action % num_box]]["removed"] = True

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

    def _faster_rcnn_client(self, img):
        img_msg = self.br.cv2_to_imgmsg(img)
        res = self.obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.cls_scores

    def _vmrn_client(self, img, bbox):
        img_msg = self.br.cv2_to_imgmsg(img)
        res = self.vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat, res.grasps

    def _mattnet_client(self, img, bbox, cls, expr):
        img_msg = self.br.cv2_to_imgmsg(img)
        res = self.grounding(img_msg, bbox, cls, expr)
        return res.ground_prob

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

    def _grasp_filter(self, boxes, grasps):
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

    def _bbox_filter(self, bbox, cls, cls_scores):
        # apply NMS
        keep = nms(torch.from_numpy(bbox[:, :-1]), torch.from_numpy(bbox[:, -1]), 0.7)
        keep = keep.view(-1).numpy().tolist()
        for i in range(bbox.shape[0]):
            if i not in keep and bbox[i][-1] > 0.9:
                keep.append(i)
        bbox = bbox[keep]
        cls = cls[keep]
        cls_scores = cls_scores[keep]
        return bbox, cls, cls_scores

    def _bbox_match(self, bbox, prev_bbox, scores, prev_scores, mode="hungarian"):
        # TODO: apply Hungarian algorithm to match boxes
        if prev_bbox.size == 0:
            return {}
        ovs = _bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
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

        def _bbox_overlaps(anchors, gt_boxes):
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

    def _leaf_and_descendant_stats(self, rel_prob_mat, sample_num = 1000):
        # TODO: Numpy may support a faster implementation.
        def sample_trees(rel_prob, sample_num=1):
            return torch.multinomial(rel_prob, sample_num, replacement=True)

        cuda_data = False
        if rel_prob_mat.is_cuda:
            # this function runs much faster on CPU.
            cuda_data = True
            rel_prob_mat = rel_prob_mat.cpu()

        rel_prob_mat = rel_prob_mat.permute((1, 2, 0))
        mrt_shape = rel_prob_mat.shape[:2]
        rel_prob = rel_prob_mat.view(-1, 3)
        rel_valid_ind = rel_prob.sum(-1) > 0

        # sample mrts
        samples = sample_trees(rel_prob[rel_valid_ind], sample_num) + 1
        mrts = torch.zeros((sample_num,) + mrt_shape).type_as(samples)
        mrts = mrts.view(sample_num, -1)
        mrts[:, rel_valid_ind] = samples.permute((1,0))
        mrts = mrts.view((sample_num,) + mrt_shape)
        f_mats = (mrts == 1)
        c_mats = (mrts == 2)
        adj_mats = f_mats + c_mats.transpose(1,2)

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
            return desc_mat.transpose(0,1)

        leaf_desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
        count = 0
        for adj_mat in adj_mats:
            if no_cycle(adj_mat):
                desc_mat = descendants(adj_mat)
                leaf_desc_mat = desc_mat * (adj_mat.sum(1, keepdim=True) == 0)
                leaf_desc_prob += leaf_desc_mat
                count += 1
        leaf_desc_prob = leaf_desc_prob / count
        if cuda_data:
            leaf_desc_prob = leaf_desc_prob.cuda()
        return leaf_desc_prob
    
    def get_action_type(self, action, num_box=None):
        if num_box is None:
            num_box = self.observations['num_box']

        if action < num_box:
            return 'GRASP_AND_END'
        elif action < 2 * num_box:
            return 'GRASP_AND_CONTINUE'
        elif action < 3 * num_box:
            return 'Q1'
        else:
            return 'Q2'

    def _estimate_state_with_user_clue(self, clue):
        # 3. incorporate the provided clue by the user
        # clue contains the tentative target contained in anwer of user for 'where is '
        
        leaf_desc_prob = self.belief['leaf_desc_prob']
        img = self.observations['img']
        bboxes = self.observations['bboxes']
        classes = self.observations['classes']
        ind_match_dict = self.observations['ind_match_dict']
        num_box = self.observations['num_box']

        tentative_ground = self._mattnet_client(img, bboxes, classes, clue) # find clue object
        for i, score in enumerate(tentative_ground):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["clue_belief"].update(score, self.kdes)
        pcand = [self.object_pool[ind_match_dict[i]]["clue_belief"].belief[1] for i in range(num_box)]
        tentative_ground = self.p_cand_to_belief_mc(pcand)
        tentative_ground = np.expand_dims(tentative_ground, 0)
        leaf_desc_prob[:, -1] = (tentative_ground * leaf_desc_prob[:, :-1]).sum(-1) # assume l&d prob of target == l%d prob of bg
        print('Update leaf_desc_prob with clue completed')
        
        self.belief['leaf_desc_prob'] = leaf_desc_prob