

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
from sklearn.cluster import KMeans
from scipy import optimize

from libraries.density_estimator.density_estimator import object_belief, gaussian_kde
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection
from config.config import *

# -------- Settings ---------
DEBUG = True

# -------- Constants ---------


# -------- Statics ---------
def dbg_print(text):
    if DEBUG:
        print(text)

# -------- Code ---------
class Invigorate():
    def __init__(self):
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        rospy.wait_for_service('vmrn_server')
        rospy.wait_for_service('mattnet_server')
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        self._vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        self._grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)

        self._br = CvBridge()

        self.history_scores = []
        self.object_pool = []
        self.target_in_pool = None
        self._init_kde()
        self.belief = {}
        self.clue = None

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

        # object detection
        bboxes, classes, scores = self._object_detection(img)
        ind_match_dict, not_matched = self._bbox_post_process(bboxes, scores)
        num_box = bboxes.shape[0]
        print('Perceive_img: _object_detection finished')

        # relationship and grasp detection
        rel_mat, rel_score_mat, grasps = self._mrt_detection(img, bboxes)
        print('Perceive_img: mrt and grasp detection finished')

        # grounding
        grounding_scores = self._multi_step_grounding(img, bboxes, expr, ind_match_dict)
        print('Perceive_img: mattnet grounding finished')

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
        num_box = observations['num_box']

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
        # TODO. This means candidate belief estimation has problem
        # if after name filter, all prob sum to zero, the object is in background
        if target_prob.sum() == 0.0:
            target_prob[-1] = 1.0
        target_prob /= target_prob.sum()
        print('Step 3.1: class name filter completed')
        print('target_prob : {}'.format(target_prob))

        # 2. incorporate QA history
        target_prob_backup = target_prob.copy()
        for k, v in ind_match_dict.items():
            if self.object_pool[v]["confirmed"] == True:
                if self.object_pool[v]["ground_belief"] == 1.:
                    target_prob[:] = 0.
                    target_prob[k] = 1.
                elif self.object_pool[v]["ground_belief"] == 0.:
                    target_prob[k] = 0.
        if target_prob.sum() > 0:
            target_prob /= target_prob.sum()
        else:
            # something wrong with the matching process. roll back
            for i in observations['not_matched']:
                target_prob[i] = target_prob_backup[i]
            target_prob[-1] = target_prob_backup[-1]
            target_prob /= target_prob.sum()

        # update target_prob
        for k, v in ind_match_dict.items():
            self.object_pool[v]["target_prob"] = target_prob[k]

        print('Step 3.2: incorporate QA history completed')
        print('target_prob: {}'.format(target_prob))

        # 3. incorporate the provided clue by the user
        if self.clue is not None:
            leaf_desc_prob = self._estimate_state_with_user_clue(self.clue)

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
            dbg_print("Invigorate: handling answer for Q1")
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
            dbg_print("Invigorate: handling answer for Q2")
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

                self.belief["leaf_desc_prob"] = leaf_desc_prob
                leaf_desc_prob = self._estimate_state_with_user_clue(answer)

        self.belief["target_prob"] = target_prob
        self.belief["leaf_desc_prob"] = leaf_desc_prob

    def decision_making_heuristic(self):
        def choose_target(target_prob):
            if len(target_prob.shape) == 1:
                target_prob = target_prob.reshape(-1, 1)
            cluster_res = KMeans(n_clusters=2).fit_predict(target_prob)
            mean0 = target_prob[cluster_res==0].mean()
            mean1 = target_prob[cluster_res==1].mean()
            pos_label = 0 if mean0 > mean1 else 1
            pos_num = (cluster_res==pos_label).sum()
            if pos_num > 1:
                return ("Q1_{:d}".format(np.argmax(target_prob[:-1].reshape(-1))))
            else:
                if cluster_res[-1] == pos_label:
                    return "Q2"
                else:
                    return "G_{:d}".format(np.argmax(target_prob[:-1].reshape(-1)))

        num_box = self.observations['num_box']
        target_prob = self.belief['target_prob']
        leaf_desc_prob = self.belief['leaf_desc_prob']

        action = choose_target(target_prob.copy())
        if action.startswith("Q"):
            if action == "Q2":
                action = 3 * num_box
            else:
                action = int(action.split("_")[1]) + 2 * num_box
        else:
            selected_obj = int(action.split("_")[1])
            l_d_probs = leaf_desc_prob[:, selected_obj]
            current_tgt = np.argmax(l_d_probs)
            if current_tgt == selected_obj:
                # grasp and end program
                action = current_tgt
            else:
                # grasp and continue
                action = current_tgt + num_box

            # if the action is asking Q2, it is necessary to check whether the previous answer is useful.
            # if useful, the robot will use the previous answer instead of requiring a new answer from the user.
            # if a != 3 * num_box or self.clue is None:
            #     self.result_container.append(
            #         save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, self.data_viewer))

        return action

    def transit_state(self, action):
        num_box = self.observations['num_box']
        ind_match_dict = self.observations['ind_match_dict']

        action_type = self.get_action_type(action)
        if action_type == 'Q1' or action_type == 'Q2':
            # asking question does not change state
            return
        else:
            # mark object as being removed
            self.object_pool[ind_match_dict[action % num_box]]["removed"] = True

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
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.cls_scores

    def _vmrn_client(self, img, bbox):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat, res.grasps

    def _mattnet_client(self, img, bbox, cls, expr):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._grounding(img_msg, bbox, cls, expr)
        return res.ground_prob

    def _object_detection(self, img):
        obj_result = self._faster_rcnn_client(img)
        num_box = obj_result[0]
        bboxes = np.array(obj_result[1]).reshape(num_box, -1)
        print('_object_detection: bboxes {}'.format(bboxes))
        classes = np.array(obj_result[2]).reshape(num_box, 1)
        class_scores = np.array(obj_result[3]).reshape(num_box, -1)
        bboxes, classes, class_scores = self._bbox_filter(bboxes, classes, class_scores)

        return bboxes, classes, class_scores

    def _bbox_post_process(self, bboxes, scores):
        prev_boxes = np.array([b["bbox"] for b in self.object_pool])
        ind_match_dict = self._bbox_match(bboxes, prev_boxes)
        not_matched = set(range(bboxes.shape[0])) - set(ind_match_dict.keys())
        # updating the information of matched bboxes
        for k, v in ind_match_dict.items():
            self.object_pool[v]["bbox"] = bboxes[k]
            self.object_pool[v]["cls_scores"] = scores[k]
        # initialize newly detected bboxes
        for i in not_matched:
            new_box = self._init_object(bboxes[i], scores[i])
            self.object_pool.append(new_box)
            ind_match_dict[i] = len(self.object_pool) - 1
        return ind_match_dict, not_matched

    def _init_object(self, bbox, score):
        new_box = {}
        new_box["bbox"] = bbox
        new_box["cls_scores"] = score
        new_box["cand_belief"] = object_belief()
        new_box["ground_belief"] = 0.
        new_box["ground_scores_history"] = []
        new_box["clue"] = None
        new_box["clue_belief"] = object_belief()
        if self.target_in_pool:
            new_box["confirmed"] = True
        else:
            new_box["confirmed"] = False  # whether this box has been confirmed by user's answer
        new_box["removed"] = False
        return new_box

    def _mrt_detection(self, img, bboxes):
        num_box = bboxes.shape[0]
        # print(num_box)
        rel_result = self._vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        # print(rel_result[1])
        if num_box == 1: # TODO hack!!
            rel_score_mat = (0.0, 0.0, 0.0)
        else:
            rel_score_mat = rel_result[1]
        rel_score_mat = np.array(rel_score_mat).reshape((3, num_box, num_box))
        grasps = np.array(rel_result[2]).reshape((num_box, 5, -1))
        grasps = self._grasp_filter(bboxes, grasps)
        return rel_mat, rel_score_mat, grasps

    def _multi_step_grounding(self, img, bboxes, expr, ind_match_dict):
        grounding_scores = self._mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, -1].reshape(-1).tolist(), expr)
        return grounding_scores

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

        t_ground = self._mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, -1].reshape(-1).tolist(), clue)
        self.clue = clue
        for i, score in enumerate(t_ground):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["clue_belief"].update(score, self.kdes)
        pcand = [self.object_pool[ind_match_dict[i]]["clue_belief"].belief[1] for i in range(num_box)]
        t_ground = self._cal_target_prob_from_p_cand(pcand)
        t_ground = np.append(t_ground, 1. - t_ground.sum())
        cluster_res = KMeans(n_clusters=2).fit_predict(t_ground.reshape(-1, 1))
        mean0 = t_ground[cluster_res == 0].mean()
        mean1 = t_ground[cluster_res == 1].mean()
        pos_label = 0 if mean0 > mean1 else 1
        pos_num = (cluster_res == pos_label).sum()
        if pos_num == 1 and cluster_res[-1] == pos_label:
            # cannot successfully ground the clue object, reset clue
            self.clue = None
        else:
            t_ground = np.expand_dims(t_ground, 0)
            leaf_desc_prob[:, -1] = (t_ground * leaf_desc_prob[:, :-1]).sum(-1)
        print('Update leaf_desc_prob with clue completed')

        return leaf_desc_prob