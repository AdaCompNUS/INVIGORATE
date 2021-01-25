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

from config.config import *
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection, ViLBERTGrounding
from libraries.tools.refer.refer import REFER

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
TEST_VLBERT_REFEXP = False

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

def vilbert_obr_client(img, bbox):
    rospy.wait_for_service('vilbert_obr_server')
    try:
        vmr_det = rospy.ServiceProxy('vilbert_obr_server', VmrDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat
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
    bbox = tuple([int(i) for i in bbox])
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

    if TEST_VLBERT_REFEXP:
        ground_scores_vlbert = vlbert_grounding_client(img_cv, bboxes.reshape(-1).tolist(), expr)
        if WITHBG:
            ground_scores_vlbert = list(ground_scores_vlbert)
            ground_scores_vlbert.append(BG_SCORE_VLBERT)
        with torch.no_grad():
            ground_prob_vlbert = torch.nn.functional.softmax(torch.tensor(ground_scores_vlbert), dim=0)
        dbg_print('ViLBERT ground_scores: {}'.format(ground_scores_vlbert))
        dbg_print('ViLBERT ground_prob: {}'.format(ground_prob_vlbert))

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

def label_grounidngs_nus_multiple():
    pass


def label_groundings_nus_single():

    test_set = \
        [
            ['1.jpeg', 'blue box'],
            ['1.jpeg', 'black remote'],
            ['1.jpeg', 'white box'],
            ['1.jpeg', 'the apple'],
            ['1.jpeg', 'the bottle'],

            ['2.jpeg', 'the knife'],
            ['2.jpeg', 'black mouse'],
            ['2.jpeg', 'banana under the remote'],
            ['2.jpeg', 'the cup'],

            ['3.jpeg', 'bottle with a pink cap'],
            ['3.jpeg', 'red box'],
            ['3.jpeg', 'blue notebook'],
            ['3.jpeg', 'the bottle'],

            ['4.jpeg', 'the red box'],
            ['4.jpeg', 'yellow cup'],
            ['4.jpeg', 'cup under the white mouse'],
            ['4.jpeg', 'black mouse'],

            ['5.jpeg', 'the red box'],
            ['5.jpeg', 'the knife'],
            ['5.jpeg', 'the bottle'],
            ['5.jpeg', 'the box under the knife'],

            ['6.jpeg', 'the white box'],
            ['6.jpeg', 'the bottle'],
            ['6.jpeg', 'the pink knife'],
            ['6.jpeg', 'banana'],

            ['7.jpeg', 'the left toothbrush'],
            ['7.jpeg', 'the yellow knife'],

            ['8.jpeg', 'the blue notebook'],
            ['8.jpeg', 'the banana'],
            ['8.jpeg', 'the pink knife'],
            ['8.jpeg', 'the knife on top of the banana'],

            ['9.jpeg', 'the blue toothbrush'],
            ['9.jpeg', 'the bottle'],
            ['9.jpeg', 'the right knife'],

            ['10.jpeg', 'the bottom apple'],
            ['10.jpeg', 'the top box'],

            ['11.jpeg', 'the wooden block'],
            ['11.jpeg', 'the yellow bottle'],
            ['11.jpeg', 'blue cup'],
            ['11.jpeg', 'mouse'],

            ['12.jpeg', 'the bottom box'],
            ['12.jpeg', 'the left knife'],
            ['12.jpeg', 'banana'],
            ['12.jpeg', 'bottle'],

            ['13.jpeg', 'the remote'],
            ['13.jpeg', 'the notebook'],

            ['14.jpeg', 'the yellow bottle'],
            ['14.jpeg', 'the white mouse'],
            ['14.jpeg', 'the white box'],
            ['14.jpeg', 'the top right mouse'],
            ['14.jpeg', 'the mouse to the left of the white box'],

            ['15.jpeg', 'the box under the pink knife'],
            ['15.jpeg', 'the blue notebook'],
            ['15.jpeg', 'the box under the bottle'],
            ['15.jpeg', 'the bottle'],
        ]

    all_labels = []
    save_path = osp.join(ROOT_DIR, "images", "nus_single", "grounding_gt.npy")

    for i, t in enumerate(test_set):
        im_name = t[0]
        expr = t[1]

        im_path = osp.join(ROOT_DIR, "images", "nus_single", "images", im_name)
        bbox = visualize_for_label(im_path, expr)

        all_labels.append({
            "file_name": im_name,
            "file_path": im_path,
            "expr": expr,
            "bbox": bbox,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(test_set)))

    return all_labels

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
            "file_name": im_name,
            "file_path": im_path,
            "expr": expr,
            "bbox": bbox,
        })

        np.save(save_path, all_labels)
        print("Finished: {:d}/{:d}".format(i, len(test_set)))

    return all_labels


def generate_coco_gt_labels(data_root, data_dir, task="RefCOCO", split = "val", save_dir = "coco_feats"):

    if task.startswith("RefCOCOg"):
        refer = REFER(data_root, dataset="refcocog", splitBy="umd")
    elif task.startswith("RefCOCO+"):
        refer = REFER(data_root, dataset="refcoco+", splitBy="unc")
    else:
        refer = REFER(data_root, dataset="refcoco", splitBy="unc")
    ref_ids = refer.getRefIds(split=split)
    # remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
    # remove_ids = [int(x) for x in remove_ids]

    all_img_list1 = os.listdir(os.path.join(data_dir, "train2017"))
    all_img_list2 = os.listdir(os.path.join(data_dir, "val2017"))
    all_img_list3 = os.listdir(os.path.join(data_dir, "test2017"))

    all_img_dict = {name: os.path.join(data_dir, "train2017") for name in all_img_list1 if name.endswith("jpg")}
    all_img_dict2 = {name: os.path.join(data_dir, "val2017") for name in all_img_list2 if name.endswith("jpg")}
    all_img_dict3 = {name: os.path.join(data_dir, "test2017") for name in all_img_list2 if name.endswith("jpg")}
    all_img_dict.update(all_img_dict2)
    all_img_dict.update(all_img_dict3)

    all_labels = []
    for ref_id in ref_ids:
        ref = refer.Refs[ref_id]
        image_id = ref["image_id"]
        ref_id = ref["ref_id"]
        refBox = refer.getRefBox(ref_id)
        filename = str(image_id).rjust(12, "0") + ".jpg"
        for sent in ref["sentences"][:1]:
            anno_dict = {}
            anno_dict["file_name"] = filename
            anno_dict["file_path"] = os.path.join(all_img_dict[anno_dict["file_name"]], anno_dict["file_name"])
            anno_dict["bbox"] = np.array([refBox[0], refBox[1], refBox[0] + refBox[2], refBox[1] + refBox[3]])
            anno_dict["expr"] = sent["raw"]
            all_labels.append(anno_dict)

    return all_labels


def check_grounding_labels(split="vmrd_like_single"):
    if split not in {"refcoco_val"}:
        label_path = osp.join(ROOT_DIR, "images", split, "grounding_gt.npy")
        if osp.exists(label_path):
            return np.load(label_path, allow_pickle=True)
        else:
            if split == "vmrd_like_single":
                return label_groundings_vmrd_like_single()
            elif split == "nus_single":
                return label_groundings_nus_single()
    else:
        dataroot = "/data1/zhb/datasets/refcoco"
        datadir = "/data0/svc4/code/INVIGORATE/vilbert/datasets/coco/coco/"
        all_labels = generate_coco_gt_labels(dataroot, datadir)
        return all_labels

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

def visualize_results(img, expr, gt, g_mattnet=None, g_vilbert=None, g_ingress=None):
    assert not (g_mattnet is None and g_ingress is None and g_vilbert is None)
    draw_single_bbox(img, (1, 1, img.shape[1]-1, img.shape[0]-1), text_str=expr)

    draw_single_bbox(img, gt, text_str="ground truth", bbox_color=(0, 255, 0))
    if g_vilbert is not None:
        draw_single_bbox(img, g_vilbert, text_str="vilbert", bbox_color=(255, 0, 0))
    if g_mattnet is not None:
        draw_single_bbox(img, g_mattnet, text_str="mattnet", bbox_color=(0, 0, 255))
    if g_ingress is not None:
        draw_single_bbox(img, g_ingress, text_str="ingress")

    plt.axis('off')
    plt.imshow(img[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    rospy.init_node('test')

    gt_labels = check_grounding_labels(split="nus_single")
    gt_labels = gt_labels[:100]

    acc_mattnet = 0.
    acc_vilbert = 0.

    vis=True

    for i, gt in enumerate(gt_labels):
        im_id = gt["file_name"]
        expr = gt["expr"]
        gt_box = gt["bbox"]
        im_path = gt["file_path"]

        img_cv = cv2.imread(im_path)
        mattnet_bbox, vilbert_bbox = test_grounding(img_cv, expr)

        mattnet_score = iou(torch.tensor(mattnet_bbox[None, :]).float(), torch.tensor(gt_box[None, :]).float()).item()
        if mattnet_score > 0.5:
            acc_mattnet += 1

        vilbert_score = iou(torch.tensor(vilbert_bbox[None, :]).float(), torch.tensor(gt_box[None, :]).float()).item()
        if vilbert_score > 0.5:
            acc_vilbert += 1

        if vis:
            visualize_results(img_cv.copy(), expr, gt_box, g_vilbert=vilbert_bbox, g_mattnet=mattnet_bbox)

        print('!!!!! test {} of {} complete'.format(i + 1, len(gt_labels)))
        print("MAttNet score: {:.3f}, ViLBERT score: {:.3f}".format(mattnet_score, vilbert_score))

    acc_mattnet /= len(gt_labels)
    acc_vilbert /= len(gt_labels)
    print("MAttNet Acc: {:.3f}, ViLBERT Acc: {:.3f}".format(acc_mattnet, acc_vilbert))