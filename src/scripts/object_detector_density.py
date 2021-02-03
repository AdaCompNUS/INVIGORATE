import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(this_dir, '../'))

# import matplotlib
# matplotlib.use('Agg')
import rospy
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
import os
import time
import xml.etree.ElementTree as ET
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import pickle
from config.config import *
from invigorate_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection, Grounding
from libraries.tools.refer.refer import REFER

br = CvBridge()
DBG_PRINT = True

try:
    from ingress_srv.ingress_srv import Ingress
except:
    import warnings
    warnings.warn("INGRES model has not been imported successfully.")

def dbg_print(string):
    if DBG_PRINT:
        print(string)

def faster_rcnn_detection(img, bbox):
    rospy.wait_for_service('faster_rcnn_server')
    try:
        obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = obj_det(img_msg, False, bbox.reshape(-1).tolist())

        num_box, bboxes, classes, scores = res.num_box, res.bbox, res.cls, res.cls_scores
        bboxes = np.array(bboxes).reshape(num_box, 4)
        classes = np.array(classes).reshape(num_box, 1)
        scores = np.array(scores).reshape(num_box, -1)
        dbg_print('classes: {}'.format(classes))

        return bboxes, classes, scores

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def parse_pascalvoc_labels(anno_file_path):
    tree = ET.parse(anno_file_path)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int32)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        cls = obj.find('name').text.lower().strip()
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(cls)

    return boxes, gt_classes

def postag_analysis(sent):
    text = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(text)
    return pos_tags

def find_subject(expr):
    pos_tags = postag_analysis(expr)

    subj_tokens = []

    # 1. Try to find the first noun phrase before any preposition
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
        elif postag in {"IN", "TO", "RP"}:
            break

    # 2. Otherwise, return all words before the first preposition
    assert subj_tokens == []
    for i, (token, postag) in enumerate(pos_tags):
        if postag in {"IN", "TO", "RP"}:
            break
        if postag in {"DT"}:
            continue
        subj_tokens.append(token)

    return subj_tokens

def initialize_cls_filter(subject):
    subj_str = ''.join(subject)
    cls_filter = []
    for i, cls in enumerate(CLASSES):
        cls_str = ''.join(cls.split(" "))
        if cls_str in subj_str or subj_str in cls_str:
            cls_filter.append((i, cls))
    assert len(cls_filter) <= 1
    return cls_filter

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
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

    return overlaps

def collect_density_data(datasets="nus_single_2+nus_single_3"):
    datasets = datasets.split("+")
    all_scores_pos = []
    all_scores_neg = []
    for dataset in datasets:
        grounding_label_path = osp.join(ROOT_DIR, "images", dataset, "grounding_gt.npy")
        labeled = np.load(grounding_label_path, allow_pickle=True, encoding="latin1")
        all_labels = list(labeled)

        det_res = {}
        for i, label in enumerate(all_labels):
            print("{:d}/{:d}".format(i, len(all_labels)))

            # object classification
            im_path = label["file_path"]
            img_name = im_path.split("/")[-1]
            img_id = ".".join(img_name.split(".")[:-1])
            label_dir = osp.join("/".join(im_path.split("/")[:-2]), "labels")
            label_file = osp.join(label_dir, img_id + ".xml")
            bboxes, classes = parse_pascalvoc_labels(label_file)
            gt_ind = iou(torch.as_tensor(label["bbox"][None]).float(), torch.as_tensor(bboxes).float()).reshape(-1).argmax().item()

            if img_name in det_res:
                det_boxes, det_classes, det_scores = det_res[img_name]
            else:
                img = cv2.imread(im_path)
                det_boxes, det_classes, det_scores = faster_rcnn_detection(img, bboxes)
                det_res[img_name] = (det_boxes, det_classes, det_scores)

            # extract cls_filter
            subject = find_subject(label["expr"])
            cls_filter = initialize_cls_filter(subject)
            if len(cls_filter) == 0:
                continue

            gt_cls = cls_filter[0][0]
            gt_cls_scores = det_scores[:, gt_cls].tolist()
            pos_score = gt_cls_scores[gt_ind]
            neg_scores = gt_cls_scores[:gt_ind] + gt_cls_scores[gt_ind+1:]
            all_scores_pos.append(pos_score)
            all_scores_neg += neg_scores

    return all_scores_pos, all_scores_neg

all_scores_pos, all_scores_neg = collect_density_data()
with open("object_density_estimtion.pkl", "wb") as f:
    pickle.dump({
        "pos": all_scores_pos,
        "neg": all_scores_neg
    }, f)

