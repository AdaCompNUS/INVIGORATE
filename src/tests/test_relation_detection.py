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
import pickle

from config.config import *
from invigorate_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection, Grounding
from libraries.tools.refer.refer import REFER

try:
    from ingress_srv.ingress_srv import Ingress
except:
    import warnings
    warnings.warn("INGRES model has not been imported successfully.")

br = CvBridge()

# ------- Settings ------
DBG_PRINT = False

def dbg_print(string):
    if DBG_PRINT:
        print(string)

def faster_rcnn_detection(img):
    rospy.wait_for_service('faster_rcnn_server')
    try:
        obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = obj_det(img_msg, True)

        num_box, bboxes, classes, _ = res.num_box, res.bbox, res.cls, res.box_feats
        dbg_print(bboxes)
        dbg_print(num_box)
        bboxes = np.array(bboxes).reshape(num_box, 4)
        classes = np.array(classes).reshape(num_box, 1)
        dbg_print('classes: {}'.format(classes))

        return num_box, bboxes, classes

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def vmrn_client(img, bbox):
    rospy.wait_for_service('vmrn_server')
    try:
        vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        img_msg = br.cv2_to_imgmsg(img)
        num_box = bbox.shape[0]
        bbox = bbox.reshape(-1).tolist()
        res = vmr_det(img_msg, bbox)
        rel_mat = np.array(res.rel_mat).reshape(num_box, num_box)
        rel_score_mat = np.array(res.rel_score_mat).reshape(3, num_box, num_box)
        return rel_mat, rel_score_mat
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def vilbert_obr_client(img, bbox):
    rospy.wait_for_service('vilbert_obr_server')
    try:
        vmr_det = rospy.ServiceProxy('vilbert_obr_server', VmrDetection)
        img_msg = br.cv2_to_imgmsg(img)
        num_box = bbox.shape[0]
        bbox = bbox.reshape(-1).tolist()
        res = vmr_det(img_msg, bbox)
        rel_mat = np.array(res.rel_mat).reshape(num_box, num_box)
        rel_score_mat = np.array(res.rel_score_mat).reshape(3, num_box, num_box)
        return rel_mat, rel_score_mat
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def draw_single_bbox(img, bbox, bbox_color=(163, 68, 187), text_str="", test_bg_color = None):
    if test_bg_color is None:
        test_bg_color = bbox_color
    bbox = tuple(bbox)
    bbox = tuple([int(i) for i in bbox])
    text_rd = (bbox[2], bbox[1] + 25)
    cv2.rectangle(img, bbox[0:2], bbox[2:4], bbox_color, 2)
    cv2.rectangle(img, bbox[0:2], text_rd, test_bg_color, -1)
    cv2.putText(img, text_str, (bbox[0], bbox[1] + 20),
                cv2.FONT_HERSHEY_PLAIN,
                2, (255, 255, 255), thickness=2)
    return img

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

def visualize_for_label(img_path):

    img_name = img_path.split("/")[-1]
    img_id = ".".join(img_name.split(".")[:-1])
    img_dir = "/".join(img_path.split("/")[:-1])
    label_dir = osp.join("/".join(img_path.split("/")[:-2]), "labels")
    label_file = osp.join(label_dir, img_id + ".xml")

    img = PIL.Image.open(img_path).convert('RGB')
    img = np.array(img)
    bboxes, classes = parse_pascalvoc_labels(label_file)

    img_to_show = img.copy()
    img_h, img_w = img_to_show.shape[:2]
    target_w = 960
    scaler = float(target_w) / float(img_w)
    img_to_show = cv2.resize(img_to_show, (target_w, int(img_h * scaler)))
    bboxes_to_show = bboxes.copy() * scaler

    for i, bbox in enumerate(bboxes_to_show):
        bbox = bbox.astype(np.int32).tolist()
        cls = classes[i]
        print(i, cls, bbox)
        draw_single_bbox(img_to_show, bbox, text_str=cls + "-" + str(i))

    plt.axis('off')
    plt.imshow(img_to_show)
    plt.show()

    # EXAMPLE: "1 2 2 4 1 5" means there are three relationships:
    #        1 is the child of 2,
    #        2 is the child of 4,
    #        1 is the child of 5
    rels = raw_input("Please input all the correct relationships:")
    rels = rels.split(" ")
    rels = [int(i) for i in rels if i != ""]

    def rel_seq_to_mat(rel_str, num_box):
        rel_mat = 3 * np.ones((num_box, num_box), dtype=np.int32)
        rels = np.array(rel_str, dtype=np.int32).reshape(-1, 2)
        for r in rels:
            rel_mat[r[0], r[1]] = 2
            rel_mat[r[1], r[0]] = 1
        rel_mat *= (1 - np.eye(num_box, dtype=np.int32))
        return rel_mat

    rel_mat = rel_seq_to_mat(rels, bboxes.shape[0])

    return bboxes, rel_mat


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


def label_obrs_ycb_single():

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "ycb_single", "obr_gt.npy")
    img_dir = osp.join(ROOT_DIR, "images", "ycb_single", "images")
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.split(".")[-1] in {"png", "jpg", "jpeg"}]
    for i, im_name in enumerate(img_list):

        im_path = osp.join(ROOT_DIR, "images", "ycb_single", "images", im_name)
        bboxes, rel_mat = visualize_for_label(im_path)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "rel_mat": rel_mat,
            "bbox": bboxes,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(img_list)))

    return all_labels


def label_obrs_nus_single_2():

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "nus_single_2", "obr_gt.npy")
    img_dir = osp.join(ROOT_DIR, "images", "nus_single_2", "images")
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.split(".")[-1] in {"png", "jpg", "jpeg"}]
    for i, im_name in enumerate(img_list):

        im_path = osp.join(ROOT_DIR, "images", "nus_single_2", "images", im_name)
        bboxes, rel_mat = visualize_for_label(im_path)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "rel_mat": rel_mat,
            "bbox": bboxes
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(img_list)))

    return all_labels


def label_obrs_nus_single_3():

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "nus_single_3", "obr_gt.npy")
    img_dir = osp.join(ROOT_DIR, "images", "nus_single_3", "images")
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.split(".")[-1] in {"png", "jpg", "jpeg"}]
    for i, im_name in enumerate(img_list):

        im_path = osp.join(ROOT_DIR, "images", "nus_single_3", "images", im_name)
        bboxes, rel_mat = visualize_for_label(im_path)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "rel_mat": rel_mat,
            "bbox": bboxes,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(img_list)))

    return all_labels


def label_obrs_nus_single():

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "nus_single", "obr_gt.npy")
    img_dir = osp.join(ROOT_DIR, "images", "nus_single", "images")
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.split(".")[-1] in {"png", "jpg", "jpeg"}]
    for i, im_name in enumerate(img_list):

        im_path = osp.join(ROOT_DIR, "images", "nus_single", "images", im_name)
        bboxes, rel_mat = visualize_for_label(im_path)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "rel_mat": rel_mat,
            "bbox": bboxes,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(img_list)))

    return all_labels


def label_obrs_vmrd_like_single():

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "vmrd_like_single", "obr_gt.npy")
    img_dir = osp.join(ROOT_DIR, "images", "vmrd_like_single", "images")
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.split(".")[-1] in {"png", "jpg", "jpeg"}]
    for i, im_name in enumerate(img_list):

        im_path = osp.join(ROOT_DIR, "images", "vmrd_like_single", "images", im_name)
        bboxes, rel_mat = visualize_for_label(im_path)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "rel_mat": rel_mat,
            "bbox": bboxes,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(img_list)))

    return all_labels


def check_grounding_labels(split="nus_single"):

    label_path = osp.join(ROOT_DIR, "images", split, "obr_gt.npy")
    if osp.exists(label_path):
        return np.load(label_path, allow_pickle=True)
    else:
        if split == "nus_single":
            return label_obrs_nus_single()
        elif split == "vmrd_like_single":
            return label_obrs_vmrd_like_single()
        elif split == "nus_single_2":
            return label_obrs_nus_single_2()
        elif split == "nus_single_3":
            return label_obrs_nus_single_3()
        elif split == "ycb_single":
            return label_obrs_ycb_single()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    rospy.init_node('test')

    gt_labels = check_grounding_labels(split="nus_single_3")
    gt_labels = gt_labels[:100]

    acc_mattnet = 0.
    acc_vilbert = 0.

    vis=True

    vmrn_count = 0
    vilbert_count = 0
    count = 0

    p_count = 0
    vilbert_p_count = 0
    vmrn_p_count = 0

    c_count = 0
    vilbert_c_count = 0
    vmrn_c_count = 0

    n_count = 0
    vilbert_n_count = 0
    vmrn_n_count = 0

    collected_scores_vmrn = []
    collected_scores_vilbert = []

    for i, gt in enumerate(gt_labels):
        im_id = gt["file_name"]
        rel_mat = gt["rel_mat"]
        gt_box = gt["bbox"]
        im_path = gt["file_path"]

        img_cv = cv2.imread(im_path)

        vmrn_rel_mrt, vmrn_rel_scores = vmrn_client(img_cv, gt_box)
        for o1 in range(rel_mat.shape[0]):
            for o2 in range(rel_mat.shape[0]):
                if o1 == o2:
                    continue
                else:
                    collected_scores_vmrn.append({"gt": rel_mat[o1, o2], "det_score": vmrn_rel_scores[:, o1, o2]})

        vilbert_rel_mrt, vilbert_rel_scores = vilbert_obr_client(img_cv, gt_box)
        for o1 in range(rel_mat.shape[0]):
            for o2 in range(rel_mat.shape[0]):
                if o1 == o2:
                    continue
                else:
                    collected_scores_vilbert.append({"gt": rel_mat[o1, o2], "det_score": vilbert_rel_scores[:, o1, o2]})

        num_box = gt_box.shape[0]
        vmrn_count += (vmrn_rel_mrt == rel_mat).sum() - num_box
        vilbert_count += (vilbert_rel_mrt == rel_mat).sum() - num_box
        count += num_box * num_box - num_box

        num_p = (rel_mat == 1).sum()
        vmrn_p_count += (vmrn_rel_mrt[np.where(rel_mat == 1)] == rel_mat[np.where(rel_mat == 1)]).sum()
        vilbert_p_count += (vilbert_rel_mrt[np.where(rel_mat == 1)] == rel_mat[np.where(rel_mat == 1)]).sum()
        p_count += num_p

        num_c = (rel_mat == 2).sum()
        vmrn_c_count += (vmrn_rel_mrt[np.where(rel_mat == 2)] == rel_mat[np.where(rel_mat == 2)]).sum()
        vilbert_c_count += (vilbert_rel_mrt[np.where(rel_mat == 2)] == rel_mat[np.where(rel_mat == 2)]).sum()
        c_count += num_c

        num_n = (rel_mat == 3).sum()
        vmrn_n_count += (vmrn_rel_mrt[np.where(rel_mat == 3)] == rel_mat[np.where(rel_mat == 3)]).sum()
        vilbert_n_count += (vilbert_rel_mrt[np.where(rel_mat == 3)] == rel_mat[np.where(rel_mat == 3)]).sum()
        n_count += num_n

        print("ViLBERT correct: {:d}/{:d}, VMRN correct: {:d}/{:d}".format(vilbert_count, count, vmrn_count, count))
        print("ViLBERT p correct: {:d}/{:d}, VMRN p correct: {:d}/{:d}".format(vilbert_p_count, p_count, vmrn_p_count, p_count))
        print("ViLBERT c correct: {:d}/{:d}, VMRN c correct: {:d}/{:d}".format(vilbert_c_count, c_count, vmrn_c_count, c_count))

        print('!!!!! test {} of {} complete'.format(i + 1, len(gt_labels)))

    print("ViLBERT acc: {:.3f}, VMRN acc: {:.3f}".
          format(float(vilbert_count) / float(count), float(vmrn_count) / float(count)))

    print("ViLBERT parent acc: {:.3f}, VMRN parent acc: {:.3f}".
          format(float(vilbert_p_count) / float(p_count), float(vmrn_p_count) / float(p_count)))

    print("ViLBERT child acc: {:.3f}, VMRN child acc: {:.3f}".
          format(float(vilbert_c_count) / float(c_count), float(vmrn_c_count) / float(c_count)))

    print("ViLBERT norel acc: {:.3f}, VMRN norel acc: {:.3f}".
          format(float(vilbert_n_count) / float(n_count), float(vmrn_n_count) / float(n_count)))

    with open("rel_dens_vmrn.pkl", "wb") as f:
        pickle.dump(collected_scores_vmrn, f)

    with open("rel_dens_vilbert.pkl", "wb") as f:
        pickle.dump(collected_scores_vilbert, f)