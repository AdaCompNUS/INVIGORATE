import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

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

from config.config import *
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection, ViLBERTGrounding
try:
    from ingress_srv.ingress_srv import Ingress
except:
    import warnings
    warnings.warn("INGRES model has not been imported successfully.")

br = CvBridge()

# ------- Settings ------
DBG_PRINT = False

TEST_CLS_NAME_FILTER = False
TEST_CAPTION_GENERATION = False
TEST_INGRESS_REFEXP = False

# ------- Constants -------
AMBIGUOUS_THRESHOLD = 0.1
WITHBG = False
BG_SCORE_VILBERT = -2.
BG_SCORE_MATTNET = 0.25

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
        res = vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def mattnet_client(img, bbox, classes, expr):
    rospy.wait_for_service('mattnet_server')
    try:
        grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)
        img_msg = br.cv2_to_imgmsg(img)
        res = grounding(img_msg, bbox, classes, expr)
        return res.ground_prob
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def vilbert_grounding_client(img, bbox, expr):
    rospy.wait_for_service('vilbert_grounding_server')
    try:
        grounding = rospy.ServiceProxy('vilbert_grounding_server', ViLBERTGrounding)
        img_msg = br.cv2_to_imgmsg(img)
        res = grounding(img_msg, bbox, expr)
        return res.ground_prob
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def ingress_client(img, bbox, expr):
    print(bbox)
    ingress_client = Ingress()
    res = ingress_client.ground_img_with_bbox(img, bbox, expr)
    return res

def vis_action(action_str, shape):
    im = 255. * np.ones(shape)
    cv2.putText(im, action_str, (0, im.shape[0] / 2),
                cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 0, 0), thickness=1)
    return im

def draw_single_bbox(img, bbox, bbox_color=(163, 68, 187), text_str="", test_bg_color = None):
    if test_bg_color is None:
        test_bg_color = bbox_color
    bbox = tuple(bbox)
    text_rd = (bbox[2], bbox[1] + 25)
    cv2.rectangle(img, bbox[0:2], bbox[2:4], bbox_color, 2)
    cv2.rectangle(img, bbox[0:2], text_rd, test_bg_color, -1)
    cv2.putText(img, text_str, (bbox[0], bbox[1] + 20),
                cv2.FONT_HERSHEY_PLAIN,
                2, (255, 255, 255), thickness=2)
    return img


def draw_ingress_res(img, boxes, context_boxes_idxs, captions):
    draw_img = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    for (count, idx) in enumerate(context_boxes_idxs):
        x1 = int(boxes[idx, 0])
        y1 = int(boxes[idx, 1])
        x2 = int(boxes[idx, 2])
        y2 = int(boxes[idx, 3])

        if count == 0:
            # top result
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 12)
        else:
            # context boxes
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if y1 - 15 > 5:
            cv2.putText(draw_img, captions[count],
                        (x1 + 6, y1 - 15), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(draw_img, captions[count],
                        (x1 + 6, y1 + 5), font, 1, (255, 255, 255), 2)
    return draw_img

def xyxy_to_xywh(bboxes):
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0] = bboxes[:, 0]
    new_bboxes[:, 1] = bboxes[:, 1]
    new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return new_bboxes

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


# def visualize_results(img, expr, g_mattnet=None, g_vilbert=None, g_ingress=None):
#     assert not (g_mattnet is None and g_ingress is None and g_vilbert is None)
#     ############ visualize
#     # resize img for visualization
#     data_viewer = DataViewer(CLASSES)
#
#     scalar = 500. / min(img_cv.shape[:2])
#     img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
#     vis_bboxes = bboxes * scalar
#     vis_bboxes[:, -1] = bboxes[:, -1]
#
#     # object detection
#     object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(classes.shape[0])))
#     cv2.imwrite("object_det.png", object_det_img)
#
#     # grounding
#     if TEST_REFER_EXPRESSION:
#         ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_prob.numpy())
#         # cv2.imwrite("ground.png", ground_img)
#     else:
#         ground_img = np.zeros((img_cv.shape), np.uint8)
#
#     # ingress
#     ingress_img = draw_ingress_res(img_cv.copy(), vis_bboxes, context_idxs, captions[0])
#
#     blank_img = np.zeros((img_cv.shape), np.uint8)
#
#     # save result
#     final_img = np.concatenate([np.concatenate([object_det_img, blank_img], axis = 1),
#                                 np.concatenate([ground_img, ingress_img], axis=1)], axis = 0)
#     out_dir = osp.join(ROOT_DIR, "images/output")
#     save_name = im_id.split(".")[0] + "_result.png"
#     save_path = os.path.join(out_dir, save_name)
#     i = 1
#     while (os.path.exists(save_path)):
#         i += 1
#         save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
#         save_path = os.path.join(out_dir, save_name)
#     cv2.imwrite(save_path, final_img)
#     # cv2.imshow('img', final_img)
#     # cv2.waitkey(0)
#     # cv2.destroyAllWindows()

def test_grounding(img_cv, expr):
    # get object proposals
    num_box, bboxes, classes = faster_rcnn_detection(img_cv)

    # grounding with MAttNet
    ground_scores_mattnet = mattnet_client(img_cv, bboxes.reshape(-1).tolist(), classes.reshape(-1).tolist(), expr)
    if WITHBG:
        ground_scores_mattnet = list(ground_scores_mattnet)
        ground_scores_mattnet.append(BG_SCORE_MATTNET)
    with torch.no_grad():
        ground_prob_mattnet = torch.nn.functional.softmax(torch.tensor(ground_scores_mattnet), dim=0)
    dbg_print('MAttNet ground_scores: {}'.format(ground_scores_mattnet))
    dbg_print('MAttNet ground_prob: {}'.format(ground_prob_mattnet))

    # grounding with ViLBERT
    ground_scores_vilbert = vilbert_grounding_client(img_cv, bboxes.reshape(-1).tolist(), expr)
    if WITHBG:
        ground_scores_vilbert = list(ground_scores_vilbert)
        ground_scores_vilbert.append(BG_SCORE_VILBERT)
    with torch.no_grad():
        ground_prob_vilbert = torch.nn.functional.softmax(torch.tensor(ground_scores_vilbert), dim=0)
    dbg_print('ViLBERT ground_scores: {}'.format(ground_scores_vilbert))
    dbg_print('ViLBERT ground_prob: {}'.format(ground_prob_vilbert))

    if TEST_INGRESS_REFEXP:
        # grounding with INGRESS
        bbox_2d_xywh = xyxy_to_xywh(bboxes[:, :4])
        _, top_idx_ingress, context_idxs_ingress, captions_ingress = ingress_client(img_cv, bbox_2d_xywh.tolist(), expr)
        dbg_print('INGRESS grounding sorted: {}'.format(top_idx_ingress))
        dbg_print('INGRESS context idxs: {}'.format(context_idxs_ingress))
        dbg_print('INGRESS captions: {}'.format(captions_ingress))

    def cls_filter(ground_prob, ground_scores):
        for i in range(len(ground_scores) - 1): # exclude background
            cls_name = CLASSES[classes[i][0]]
            if cls_name not in expr or expr not in cls_name: # simple filter
                ground_prob[i] = 0.0
                ground_scores[i] = -float('inf')

        ground_prob /= torch.sum(ground_prob)
        dbg_print('filtered_ground_prob: {}'.format(ground_prob))
        return ground_prob, ground_scores

    if TEST_CLS_NAME_FILTER:
        ground_prob_mattnet, ground_scores_mattnet = cls_filter(ground_prob_mattnet, ground_scores_mattnet)
        ground_prob_vilbert, ground_scores_vilbert = cls_filter(ground_prob_vilbert, ground_scores_vilbert)

        if TEST_INGRESS_REFEXP:
            filtered_top_idxs_ingress = []
            filtered_context_idxs_ingress = []
            for i, idx in enumerate(top_idx_ingress):
                cls_name = CLASSES[classes[idx][0]]
                if cls_name in expr or expr in cls_name:  # simple filter
                    filtered_top_idxs_ingress.append(idx)
                    filtered_context_idxs_ingress.append(context_idxs_ingress[i])

            top_idx_ingress = filtered_top_idxs_ingress
            context_idxs_ingress = filtered_context_idxs_ingress
            dbg_print('filtered top_idxs: {}'.format(top_idx_ingress))

    if TEST_INGRESS_REFEXP:
        return bboxes[np.argmax(ground_scores_mattnet)], bboxes[np.argmax(ground_scores_vilbert)], bboxes[top_idx_ingress[0]]
    else:
        return bboxes[np.argmax(ground_scores_mattnet)], bboxes[np.argmax(ground_scores_vilbert)]


def visualize_for_label(img_path, expr):

    img_name = img_path.split("/")[-1]
    img_id = ".".join(img_name.split(".")[:-1])
    img_dir = "/".join(img_path.split("/")[:-1])
    label_dir = osp.join("/".join(img_path.split("/")[:-2]), "labels")
    label_file = osp.join(label_dir, img_id + ".xml")

    img = PIL.Image.open(img_path).convert('RGB')
    img = np.array(img)
    bboxes, classes = parse_pascalvoc_labels(label_file)

    img_to_show = img.copy()
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32).tolist()
        cls = classes[i]
        print(i, cls, bbox)
        draw_single_bbox(img_to_show, bbox, text_str=cls + "-" + str(i))

    plt.axis('off')
    plt.imshow(img_to_show)
    plt.show()

    print("Referring Expression: {:s}".format(expr))
    box_id = raw_input("Please input the correct bbox id:")
    box_id = int(box_id)

    return bboxes[box_id]


def label_groundings_vmrd_like_single():

    test_set = \
        [['0.png', 'cup under the apple'],
         ['0.png', 'yellow cup'],
         ['0.png', 'blue glasses'],
         ['0.png', 'black mouse under the toothpaste'],
         ['0.png', 'remote'],

         ['1.jpg', 'wallet'],
         ['1.jpg', 'blue box under the toothpaste'],

         ['3.png', 'remote'],
         ['3.png', 'brown coffee cup'],
         ['3.png', 'cup beside the green toothpaste'],
         ['3.png', 'white mouse'],
         ['3.png', 'mouse beside the brown cup'],

         ['10.png', 'white remote'],
         ['10.png', 'yellow and blue toothpaste'],
         ['10.png', 'yellow cup'],
         ['10.png', 'mouse'],
         ['10.png', 'black remote'],

         ['23.png', 'banana'],
         ['23.png', 'green can'],
         ['23.png', 'yellow cup'],
         ['23.png', 'cup beside the banana'],

         ['28.png', 'banana on top'],
         ['28.png', 'the bottom apple'],
         ['28.png', 'red can under the banana'],
         ['28.png', 'green can beside the banana'],
         ['28.png', 'left apple'],
         ['28.png', 'banana to the left'],

         ['39.png', 'apple on top of the cup'],
         ['39.png', 'green can'],
         ['39.png', 'yellow cup'],
         ['39.png', 'cup to the left of the banana'],

         ['56.jpg', 'bottom green cup'],
         ['56.jpg', 'red can'],
         ['56.jpg', 'can at the bottom of the bowl'],
         ['56.jpg', 'leftmost cup'],

         ['57.jpg', 'cup under the apple'],
         ['57.jpg', 'brown and green box'],
         ['57.jpg', 'box under the banana'],

         ['59.jpg', 'banana'],
         ['59.jpg', 'the red cup'],

         ['61.jpg', 'the bottom apple'],
         ['61.jpg', 'apple on top of the banana'],
         ['61.jpg', 'blue cup'],
         ['61.jpg', 'banana'],
         ]

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "vmrd_like_single", "grounding_gt.npy")

    for i, t in enumerate(test_set):
        im_name = t[0]
        expr = t[1]

        im_path = osp.join(ROOT_DIR, "images", "vmrd_like_single", "images", im_name)
        bbox = visualize_for_label(im_path, expr)

        all_labels.append({
            "img_name": im_name,
            "expr": expr,
            "bbox": bbox,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(test_set)))

    return all_labels


def check_grounding_labels(split="vmrd_like_single"):
    label_path = osp.join(ROOT_DIR, "images", split, "grounding_gt.npy")
    if osp.exists(label_path):
        return np.load(label_path, allow_pickle=True)
    else:
        if split == "vmrd_like_single":
            return label_groundings_vmrd_like_single()


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

if __name__ == "__main__":
    rospy.init_node('test')

    gt_labels = check_grounding_labels(split="vmrd_like_single")

    acc_mattnet = 0.
    acc_vilbert = 0.

    for i, gt in enumerate(gt_labels):
        im_id = gt["img_name"]
        expr = gt["expr"]
        gt_box = gt["bbox"]

        img_path = osp.join(ROOT_DIR, "images/" + "vmrd_like_single/" + "images/" + im_id)
        img_cv = cv2.imread(img_path)
        mattnet_bbox, vilbert_bbox = test_grounding(img_cv, expr)

        mattnet_score = iou(torch.tensor(mattnet_bbox[None, :]).float(), torch.tensor(gt_box[None, :]).float()).item()
        if mattnet_score > 0.5:
            acc_mattnet += 1

        vilbert_score = iou(torch.tensor(vilbert_bbox[None, :]).float(), torch.tensor(gt_box[None, :]).float()).item()
        if vilbert_score > 0.5:
            acc_vilbert += 1

        print('!!!!! test {} of {} complete'.format(i + 1, len(gt_labels)))
        print("MAttNet score: {:.3f}, ViLBERT score: {:.3f}".format(mattnet_score, vilbert_score))

    acc_mattnet /= len(gt_labels)
    acc_vilbert /= len(gt_labels)
    print("MAttNet Acc: {:.3f}, ViLBERT Acc: {:.3f}".format(acc_mattnet, acc_vilbert))