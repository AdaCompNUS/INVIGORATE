# this file includes two caption wrappers from invigorate
#   1. INGRESS Service
#   3. INVIGORATE Service
# Originally, this file is used to test the captioning performance of
# INVIGORATE. One can also use these services for any purpose.
import abc
from abc import ABCMeta, abstractmethod
from ingress_srv.ingress_srv import Ingress
from invigorate.invigorate_models.invigorate_ijrr_v6 import InvigorateIJRRV6
from invigorate.config.config import CLASSES
from invigorate.libraries.caption_generator.caption_generator import generate_caption
import rospy
import numpy as np
import torch

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    anchors = torch.tensor(anchors).float()
    gt_boxes = torch.tensor(gt_boxes).float()

    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps.numpy()

class INGRESSService():
    def __init__(self):
        # to make it a fair comparison, we here use
        # the object detector in INVIGORATE to get
        # object classes
        self.invigorate = InvigorateIJRRV6()

    def generate_captions(self, img, bboxes):
        original_bboxes = bboxes.copy()
        bboxes, _, scores = self.invigorate.object_detection(img, bboxes)
        classes = scores[:, 1:].argmax(axis=1, ) + 1

        overlaps = iou(original_bboxes, bboxes)
        ori_ind, det_ind = np.where(overlaps > 0.95)
        ori_to_det = dict(zip(ori_ind.tolist(), det_ind.tolist()))

        class_names = [CLASSES[int(i.item())] for i in classes]
        num_obj = len(class_names)
        generated_rel_questions = []
        generated_self_questions = []

        for b_i, (b, name) in enumerate(zip(bboxes, class_names)):
            rel_cap = generate_caption(
                img, bboxes, classes, b_i, name, cap_type='rel')
            generated_rel_questions.append(rel_cap)
            self_cap = generate_caption(
                img, bboxes, classes, b_i, name, cap_type='self')
            generated_self_questions.append(self_cap)

        generated_rel_questions_sorted = [
            generated_rel_questions[ori_to_det[ori_i]]
            for ori_i in range(num_obj)]

        generated_self_questions_sorted = [
            generated_self_questions[ori_to_det[ori_i]]
            for ori_i in range(num_obj)]

        return generated_self_questions_sorted, generated_rel_questions_sorted


class INVIGORATEService():
    def __init__(self):
        self.invigorate = InvigorateIJRRV6()

    def generate_captions(self, img, bboxes):
        original_bboxes = bboxes.copy()
        bboxes, _, scores = self.invigorate.object_detection(img, bboxes)
        classes = scores[:, 1:].argmax(axis=1, ) + 1

        overlaps = iou(original_bboxes, bboxes)
        ori_ind, det_ind = np.where(overlaps > 0.95)
        ori_to_det = dict(zip(ori_ind.tolist(), det_ind.tolist()))

        class_names = [CLASSES[int(i.item())] for i in classes]
        num_obj = len(class_names)
        self.invigorate.belief['cls_scores_list'] = [[i] for i in scores]
        generated_questions = []

        for i in range(num_obj):
            name = class_names[i]
            subject_tokens = self.invigorate.expr_processor.find_subject(name)

            # initialize captioner
            self.invigorate.subject = subject_tokens
            self.invigorate.pos_expr = ''

            # generate caption candidates
            self.invigorate.question_captions_generation(img, bboxes, classes)
            cand_captions = self.invigorate.belief['questions']

            # choose the best one
            self.invigorate.match_question_to_object(img, bboxes, classes, cand_captions)

            generated_questions.append(
                self.invigorate.belief['questions'][i]
            )

        generated_questions_sorted = [
            generated_questions[ori_to_det[ori_i]]
            for ori_i in range(num_obj)]

        return generated_questions_sorted




