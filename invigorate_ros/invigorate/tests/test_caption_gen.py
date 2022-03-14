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
from invigorate_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection, Grounding

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


CLASSES = ['__background__',  # always index 0
           'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
           'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
           'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
           'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
           'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch', 'ball', 'toy',
           'spoon', 'carrot', 'scissors']

CLASSES_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

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

def ingress_client(img, bbox):
    pass

def invigorate_client(img, bbox):
    pass

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

def visualize_for_label_captions(img_path, bbox, cls):
    img = PIL.Image.open(img_path).convert('RGB')
    img = np.array(img)
    img_to_show = img.copy()

    img_h, img_w = img_to_show.shape[:2]
    target_w = 640
    scaler = float(target_w) / float(img_w)
    img_to_show = cv2.resize(img_to_show, (target_w, int(img_h * scaler)))
    bboxes_to_show = bbox.copy() * scaler

    bbox = bbox.astype(np.int32).tolist()
    print("Object infos: {}, {}".format(cls, bbox))
    draw_single_bbox(img_to_show, bboxes_to_show, text_str=cls)

    plt.axis('off')
    plt.imshow(img_to_show)
    plt.show()

    captions = raw_input("Please input the captions:")
    captions = captions.split(";")
    captions = [c.strip() for c in captions if len(c.strip()) > 0]

    return captions

def label_captions_rss():
    save_path = osp.join(ROOT_DIR, "dataset", "caption", "caption_labels.npy")
    all_labels = []

    img_dir = osp.join(ROOT_DIR, "dataset", "caption", "images")
    img_list = os.listdir(img_dir)
    for i, im_name in enumerate(img_list):

        if not im_name.endswith("png"):
            continue

        im_path = osp.join(ROOT_DIR, "dataset", "caption", "images", im_name)

        img_name = im_path.split("/")[-1]
        img_id = ".".join(img_name.split(".")[:-1])
        label_dir = osp.join("/".join(im_path.split("/")[:-2]), "objlabels")
        label_file = osp.join(label_dir, img_id + ".xml")
        bboxes, classes = parse_pascalvoc_labels(label_file)

        for i in range(bboxes.shape[0]):
            captions = visualize_for_label_captions(im_path, bboxes[i], classes[i])
            for expr in captions:
                all_labels.append({
                    "file_name": im_name,
                    "file_path": im_path,
                    "expr": expr,
                    "bbox": bboxes[i],
                })

            np.save(save_path, all_labels)
            print("Finished: {:d}/{:d}".format(i, bboxes.shape[0]))

    return all_labels

def visualize_results(img, expr, gt=None, g_mattnet=None, g_vilbert=None, g_ingress=None, score=None):
    pass

def main_test_captioning():
    pass
if __name__ == "__main__":
    label_captions_rss()