#!/usr/bin/env python
import warnings
try:
    from rosapi.baxter_api import *
    from rosapi.calibrate import calibrate_kinect
    from rosapi.kinect_subscriber import kinect_reader
except:
    warnings.warn("Baxter interface not available")

import rospy
from model.utils.net_utils import leaf_and_descendant_stats, inner_loop_planning, relscores_to_visscores

from faster_rcnn_detector.srv import ObjectDetection
from vmrn_old.srv import VmrDetection
from ingress_msgs.srv import MAttNetGrounding
from model.utils.data_viewer import dataViewer

import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from sklearn.cluster import KMeans
import os
import datetime
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_overlaps

import time
from stanfordcorenlp import StanfordCoreNLP
import pickle as pkl
from model.utils.density_estimator import object_belief, gaussian_kde

br = CvBridge()
# nlp = StanfordCoreNLP('nlpserver/stanford-corenlp')

YCROP = (350, 700)
XCROP = (620, 1020)
MODEL_NAME = "all_in_one_FixObj_NoScorePostProc_ShareW_NoRelClsGrad.pth"
MODEL_PATH = "output/vmrdcompv1/res101"

BG_SCORE = 0.25

Q2={
    "type1": "Is the target inside what I see? If not, where is it?", # COMMON FORMAT
    "type2": "I have not found the target yet. Where is it?",         # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean the {:s}?"
}

classes = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

classes_to_ind = dict(zip(classes, range(len(classes))))


def split_long_string(in_str, len_thresh=30):
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


def vis_action(action_str, shape, draw_arrow=False):
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
        cv2.arrowedLine(im, (0, mid_line), (im.shape[1], mid_line), (0, 0, 0), thickness=2, tipLength=0.03)
    return im

def save_visualization(img, bboxes, rel_mat, rel_score_mat, expr, ground_prob, a, data_viewer, im_id = None, tgt_size=500):
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
    num_box = bboxes.shape[0]

    # object detection
    cls = bboxes[:, -1]
    object_det_img = data_viewer.draw_objdet(img_show.copy(), vis_bboxes, list(range(cls.shape[0])))

    # relationship detection
    rel_det_img = data_viewer.draw_mrt(img_show.copy(), rel_mat, class_names= ground_prob.tolist()[:-1],
                                       rel_score=rel_score_mat, with_img=False, rel_img_size=500)
    rel_det_img = cv2.resize(rel_det_img, (img_show.shape[1], img_show.shape[0]))

    # grounding
    print("Grounding Probability: ")
    print(ground_prob.tolist())
    ground_img = data_viewer.draw_grounding_probs(img_show.copy(), expr, vis_bboxes, ground_prob[:-1])

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

# NEW VERSION with MAttNet
class INTEGRASE(object):
    def __init__(self):
        self.obj_det = self._init_client('faster_rcnn_server', ObjectDetection)
        self.vmr_det = self._init_client('vmrn_server', VmrDetection)
        self.grounding = self._init_client('mattnet_server', MAttNetGrounding)

        self.data_viewer = dataViewer(classes)

        self.history_scores = []
        self.object_pool = []
        self.clue = None
        self.target_in_pool = None
        self._init_kde()
        self.result_container = []

    def _init_kde(self):
        with open("density_esti_train_data.pkl") as f:
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

    def _init_client(self, srv_name, srv_type):
        rospy.wait_for_service(srv_name)
        try:
            client = rospy.ServiceProxy(srv_name, srv_type)
            return client
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def faster_rcnn_client(self, img):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.obj_det(img_msg)
        return res.num_box, res.bbox, res.cls

    def vmrn_client(self, img, bbox):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat

    def mattnet_client(self, img, bbox, cls, expr):
        img_msg = br.cv2_to_imgmsg(img)
        res = self.grounding(img_msg, bbox, cls, expr)
        return res.ground_prob

    def object_detection(self, img):
        obj_result = self.faster_rcnn_client(img)
        bboxes = np.array(obj_result[1]).reshape(-1, 4 + 32)
        cls = np.array(obj_result[2]).reshape(-1, 1)
        bboxes, cls = self.bbox_filter(bboxes, cls)

        scores = bboxes[:, 4:].reshape(-1, 32)
        bboxes = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, cls], axis=-1)
        return bboxes, scores

    def bbox_filter(self, bbox, cls):
        # apply NMS
        keep = nms(torch.from_numpy(bbox[:, :-1]), torch.from_numpy(bbox[:, -1]), 0.7)
        keep = keep.view(-1).numpy().tolist()
        for i in range(bbox.shape[0]):
            if i not in keep and bbox[i][-1] > 0.8:
                keep.append(i)
        bbox = bbox[keep]
        cls = cls[keep]
        return bbox, cls

    def bbox_match(self, bbox, prev_bbox):
        # TODO: apply Hungarian algorithm to match boxes
        # match bboxes between two steps.
        if prev_bbox.size == 0:
            return {}

        ovs = bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
        cls_mask = np.zeros(ovs.shape, dtype=np.uint8)
        for i, cls in enumerate(bbox[:, -1]):
            cls_mask[i][prev_bbox[:, -1] == cls] = 1
        ovs_mask = (ovs > 0.8)
        ovs *= ((cls_mask + ovs_mask) > 0)

        mapping = np.argsort(ovs, axis=-1)[:, ::-1]
        ovs_sorted = np.sort(ovs, axis=-1)[:, ::-1]
        matched = (np.max(ovs, axis=-1) > 0.5)
        occupied = {i: False for i in range(mapping.shape[-1])}
        ind_match_dict = {}
        for i in range(mapping.shape[0]):
            if matched[i]:
                for j in range(mapping.shape[-1]):
                    if not occupied[mapping[i][j]] and ovs_sorted[i][j] > 0.5:
                        ind_match_dict[i] = mapping[i][j]
                        occupied[mapping[i][j]] = True
                        break
                    elif ovs_sorted[i][j] <= 0.5:
                        break
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
            img = data["img"]
            bbox = data["bbox"]
            cls = data["cls"]
            ind_match_dict = data["mapping"]
            expr = ans
            # re-grounding with the clue given by the user
            t_ground = self.mattnet_client(img, bbox, cls, expr)
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

    def bbox_post_process(self, bboxes, scores):
        prev_boxes = np.array([b["bbox"] for b in self.object_pool])
        ind_match_dict = self.bbox_match(bboxes, prev_boxes)
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

    def mrt_detection(self, img, bboxes):
        num_box = bboxes.shape[0]
        rel_result = self.vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        # TODO: updating the relationship probability according to the new observation
        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            leaf_desc_prob = leaf_and_descendant_stats(torch.from_numpy(rel_score_mat) * triu_mask).numpy()
        return rel_mat, rel_score_mat, leaf_desc_prob

    def multi_step_grounding(self, img, bboxes, expr, ind_match_dict):
        num_box = bboxes.shape[0]
        mattnet_score = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, -1].reshape(-1).tolist(), expr)
        for i, score in enumerate(mattnet_score):
            obj_ind = ind_match_dict[i]
            self.object_pool[obj_ind]["cand_belief"].update(score, self.kdes)
            self.object_pool[obj_ind]["ground_scores_history"].append(score)
        pcand = [self.object_pool[ind_match_dict[i]]["cand_belief"].belief[1] for i in range(num_box)]
        ground_result = self.p_cand_to_belief_mc(pcand)
        ground_result = np.append(ground_result, 1. - ground_result.sum())
        return ground_result

    def p_cand_to_belief_mc(self, pcand, sample_num=100000):
        pcand = torch.Tensor(pcand).reshape(1, -1)
        pcand = pcand.repeat(sample_num, 1)
        sampled = torch.bernoulli(pcand)
        sampled_sum = sampled.sum(-1)
        sampled[sampled_sum > 0] /= sampled_sum[sampled_sum > 0].unsqueeze(-1)
        sampled = sampled.mean(0).cpu().numpy()
        if sampled.sum() > 1:
            sampled /= sampled.sum()
        return sampled

    def single_step_perception(self, img, expr, cls_filter=None):
        tb = time.time()

        # object detection
        bboxes, scores = self.object_detection(img)
        ind_match_dict, not_matched = self.bbox_post_process(bboxes, scores)
        num_box = bboxes.shape[0]
        # relationship detection
        rel_mat, rel_score_mat, leaf_desc_prob = self.mrt_detection(img, bboxes)
        # grounding
        ground_result = self.multi_step_grounding(img, bboxes, expr, ind_match_dict)

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += scores[i][classes_to_ind[class_str]]
            if box_score < 0.02:
                ground_result[i] = 0
        ground_result /= ground_result.sum()

        # 2. incorporate QA history
        ground_result_backup = ground_result.copy()
        for k, v in ind_match_dict.items():
            if self.object_pool[v]["confirmed"] == True:
                if self.object_pool[v]["ground_belief"] == 1.:
                    ground_result[:] = 0.
                    ground_result[k] = 1.
                elif self.object_pool[v]["ground_belief"] == 0.:
                    ground_result[k] = 0.
        if ground_result.sum() > 0:
            ground_result /= ground_result.sum()
        else:
            # something wrong with the matching process. roll back
            for i in not_matched:
                ground_result[i] = ground_result_backup[i]
            ground_result[-1] = ground_result_backup[-1]
            ground_result /= ground_result.sum()
        # update ground belief
        for k, v in ind_match_dict.items():
            self.object_pool[v]["ground_belief"] = ground_result[k]

        # 3. incorporate the provided clue by the user
        if self.clue is not None:
            t_ground = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, -1].reshape(-1).tolist(), self.clue)
            for i, score in enumerate(t_ground):
                obj_ind = ind_match_dict[i]
                self.object_pool[obj_ind]["clue_belief"].update(score, self.kdes)
            pcand = [self.object_pool[ind_match_dict[i]]["clue_belief"].belief[1] for i in range(num_box)]
            t_ground = self.p_cand_to_belief_mc(pcand)
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

        print("Perception Time Consuming: " + str(time.time() - tb) + "s")
        return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_result, ind_match_dict

    def decision_making_with_planning(self, img, expr, cls_filter):
        # perception
        bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_result, ind_match = \
            self.single_step_perception(img, expr, cls_filter=cls_filter)
        num_box = bboxes.shape[0]

        # outer-loop planning: in each step, grasp the leaf-descendant node.
        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        belief = {}
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob)
        belief["ground_prob"] = torch.from_numpy(ground_result)
        # inner-loop planning, with a sequence of questions and a last grasping.
        while (True):
            a = inner_loop_planning(belief)
            self.result_container.append(
                save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, self.data_viewer))
            if a < 2 * num_box:
                return a, a < num_box
            else:
                data = {"img": img,
                        "bbox": bboxes[:, :4].reshape(-1).tolist(),
                        "cls": bboxes[:, 4].reshape(-1).tolist(),
                        "mapping": ind_match}
                ans = raw_input("Your answer: ")
                self.result_container[-1]["answer"] = split_long_string("User's Answer: " + ans.upper())

                if a < 3 * num_box:
                    # we use binary variables to encode the answer of q1 questions.
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        for i, v in enumerate(self.object_pool):
                            if i == ind_match[a - 2 * num_box]:
                                self.object_pool[i]["is_target"] = True
                            else:
                                self.object_pool[i]["is_target"] = False
                    else:
                        obj_ind = ind_match[a - 2 * num_box]
                        self.object_pool[obj_ind]["is_target"] = False
                else:
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        # s_ing_client.target_in_pool = True
                        # for i, v in enumerate(s_ing_client.object_pool):
                        #     if i not in ind_match.values():
                        #         s_ing_client.object_pool[i]["is_target"] = False
                        if (ground_result[:-1] > 0.02).sum() == 1:
                            obj_ind = ind_match[np.argmax(ground_result[:-1])]
                            self.object_pool[obj_ind]["is_target"] = True
                    else:
                        # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                        ans = ans[6:]
                        # for i in ind_match.values():
                        #     s_ing_client.object_pool[i]["is_target"] = False
                        self.clue = ans
                        if (ground_result[:-1] > 0.02).sum() == 1:
                            obj_ind = ind_match[np.argmax(ground_result[:-1])]
                            self.object_pool[obj_ind]["is_target"] = False
                belief = self.update_belief(belief, a, ans, data)

    def decision_making_heuristic(self, img, expr, cls_filter):
        def choose_target(ground_scores):
            if len(ground_scores.shape) == 1:
                ground_scores = ground_scores.reshape(-1, 1)
            cluster_res = KMeans(n_clusters=2).fit_predict(ground_scores)
            mean0 = ground_scores[cluster_res==0].mean()
            mean1 = ground_scores[cluster_res==1].mean()
            pos_label = 0 if mean0 > mean1 else 1
            pos_num = (cluster_res==pos_label).sum()
            if pos_num > 1:
                return ("Q1_{:d}".format(np.argmax(ground_scores[:-1].reshape(-1))))
            else:
                if cluster_res[-1] == pos_label:
                    return "Q2"
                else:
                    return "G_{:d}".format(np.argmax(ground_scores[:-1].reshape(-1)))
        # perception
        bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_result, ind_match = \
            self.single_step_perception(img, expr, cls_filter=cls_filter)
        num_box = bboxes.shape[0]

        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        belief = {}
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob)
        belief["ground_prob"] = torch.from_numpy(ground_result)
        while(True):
            a = choose_target(ground_result.copy())
            if a.startswith("Q2"):
                if a == "Q2":
                    a = 3 * num_box
                else:
                    a = int(a.split("_")[1]) + 2 * num_box
            else:
                selected_obj = int(a.split("_")[1])
                l_d_probs = leaf_desc_prob[:, selected_obj]
                current_tgt = np.argmax(l_d_probs)
                if current_tgt == selected_obj:
                    # grasp and end program
                    a = current_tgt
                else:
                    # grasp and continue
                    a = current_tgt + num_box

            # if the action is asking Q2, it is necessary to check whether the previous answer is useful.
            # if useful, the robot will use the previous answer instead of requiring a new answer from the user.
            if a != 3 * num_box or self.clue is None:
                self.result_container.append(
                    save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, self.data_viewer))

            if a < 2 * num_box:
                return a, (a < num_box)
            else:
                data = {"img": img,
                        "bbox": bboxes[:, :4].reshape(-1).tolist(),
                        "cls": bboxes[:, 4].reshape(-1).tolist(),
                        "mapping": ind_match}

                # if previous answer is useful, it is unnecessary to require a new answer.
                if a == 3 * num_box and self.clue is not None:
                    ans = self.clue
                else:
                    ans = raw_input("Your answer: ")
                    self.result_container[-1]["answer"] = split_long_string("User's Answer: " + ans.upper())

                if a < 3 * num_box:
                    # we use binary variables to encode the answer of q1 questions.
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        for i, v in enumerate(self.object_pool):
                            if i == ind_match[a - 2 * num_box]:
                                self.object_pool[i]["is_target"] = True
                            else:
                                self.object_pool[i]["is_target"] = False
                    else:
                        obj_ind = ind_match[a - 2 * num_box]
                        self.object_pool[obj_ind]["is_target"] = False
                else:
                    # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                    ans = ans[6:]
                    self.clue = ans
                belief = self.update_belief(belief, a, ans, data)