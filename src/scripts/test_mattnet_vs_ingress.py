import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
import os
import time

from config.config import *
from libraries.data_viewer.data_viewer import DataViewer
# from vmrn.model.utils.net_utils import relscores_to_visscores
from vmrn_msgs.srv import MAttNetGrounding, ObjectDetection, VmrDetection
from ingress_srv.ingress_srv import Ingress

br = CvBridge()

# ------- Settings ------
DBG_PRINT = True

TEST_OBJECT_DETECTION = True
TEST_REFER_EXPRESSION = True
TEST_CLS_NAME_FILTER = False
TEST_CAPTION_GENERATION = False
TEST_MRT_DETECTION = True
TEST_GRASP_POLICY = False

# ------- Constants -------
AMBIGUOUS_THRESHOLD = 0.1
BG_SCORE = 0

def dbg_print(string):
    if DBG_PRINT:
        print(string)

def faster_rcnn_client(img):
    rospy.wait_for_service('faster_rcnn_server')
    try:
        obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = obj_det(img_msg, True)
        return res.num_box, res.bbox, res.cls, res.box_feats
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

def test_obj_manipulation(img_cv, expr):
    # test object detection
    if TEST_OBJECT_DETECTION:
        tb = time.time()
        num_box, bboxes, classes, _ = faster_rcnn_client(img_cv)
        dbg_print(bboxes)
        dbg_print(num_box)
        bboxes = np.array(bboxes).reshape(-1, 5)
        classes = np.array(classes).reshape(-1, 1)
        dbg_print('classes: {}'.format(classes))
        bbox_2d = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, classes], axis=-1)
    else:
        dbg_print('TEST_OBJECT_DETECTION is false, quit')
        return

    if TEST_REFER_EXPRESSION:
        ground_scores = list(mattnet_client(img_cv, bboxes[:, :4].reshape(-1).tolist(), bboxes[:, -1].reshape(-1).tolist(), expr))
        ground_scores.append(BG_SCORE)
        ground_prob = torch.nn.functional.softmax(10 * torch.Tensor(ground_scores), dim=0)
        dbg_print('ground_scores: {}'.format(ground_scores))
        dbg_print('ground_prob: {}'.format(ground_prob))
    else:
        dbg_print('TEST_REFER_EXPRESSION is false, skip')

    bbox_2d_xywh = xyxy_to_xywh(bbox_2d)
    _, top_idx, context_idxs, captions = ingress_client(img_cv, bbox_2d_xywh.tolist(), expr)
    dbg_print('top_idx: {}'.format(top_idx))
    dbg_print('context_idxs: {}'.format(context_idxs))
    dbg_print('captions: {}'.format(captions))

    if TEST_CLS_NAME_FILTER:
        for i in range(len(ground_scores) - 1): # exclude background
            cls_name = CLASSES[classes[i][0]]
            if cls_name not in expr: # simple filter
                ground_prob[i] = 0.0
                ground_scores[i] = -float('inf')

        ground_prob /= torch.sum(ground_prob)
        dbg_print('filtered_ground_prob: {}'.format(ground_prob))

        filtered_context_idxs =[]
        for i in context_idxs:
            cls_name = CLASSES[classes[i][0]]
            if cls_name in expr: # simple filter
                filtered_context_idxs.append(i)
        context_idxs = filtered_context_idxs
        dbg_print('filtered context_idxs: {}'.format(context_idxs))

    ############ visualize
    # resize img for visualization
    data_viewer = DataViewer(CLASSES)

    scalar = 500. / min(img_cv.shape[:2])
    img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    vis_bboxes = bboxes * scalar
    vis_bboxes[:, -1] = bboxes[:, -1]

    # object detection
    object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(classes.shape[0])))
    cv2.imwrite("object_det.png", object_det_img)

    # grounding
    if TEST_REFER_EXPRESSION:
        ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_prob.numpy())
        # cv2.imwrite("ground.png", ground_img)
    else:
        ground_img = np.zeros((img_cv.shape), np.uint8)

    # ingress
    ingress_img = draw_ingress_res(img_cv.copy(), vis_bboxes, context_idxs, captions[0])

    blank_img = np.zeros((img_cv.shape), np.uint8)

    # save result
    final_img = np.concatenate([np.concatenate([object_det_img, blank_img], axis = 1),
                                np.concatenate([ground_img, ingress_img], axis=1)], axis = 0)
    out_dir = osp.join(ROOT_DIR, "images/output")
    save_name = im_id.split(".")[0] + "_result.png"
    save_path = os.path.join(out_dir, save_name)
    i = 1
    while (os.path.exists(save_path)):
        i += 1
        save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
        save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, final_img)
    # cv2.imshow('img', final_img)
    # cv2.waitkey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('test')

    # user input
    img_array = ['']

    # im_id = '1.png'
    # expr = 'cup under banana'

    # img_cv = cv2.imread("../images/" + im_id)

    # test(img_cv, expr)

    TESTS = [['1.png', 'cup under banana'], ['1.png', 'cup under apple'],
             ['13.png', 'remote'], ['13.png', 'cup'], ['13.png', 'white mouse'],
             ['15.png', 'white mouse'], ['15.png', 'black mouse'], ['15.png', 'mouse on the left'],
             ['21.png', 'banana on top'], ['21.png', 'banana below'], ['21.png', 'apple'],
             ['36.png', 'apple under banana'], ['36.png', 'cup under apple'], ['36.png', 'cup under banana'],
            #  ['37.png', 'apple under banana'],
            #  ['38.png', 'apple under banana'],
             ['60.jpg', 'apple on the left'], ['60.jpg', 'apple on the right'], ['60.jpg', 'blue cup'], ['60.jpg', 'green cup'],
             ['table.png', 'bottle next to banana'], ['table.png', 'top left bottle'],
             ['101.png', 'the box'], ['102.png', 'banana'], ['103.png', 'banana'],
             ['104.png', 'the white box'], ['105.png', 'the notebook'], ['106.png', 'banana'],
             ['107.png', 'the red box'], ['108.png', 'the white box'],  ['109.png', 'the red box'], ['110.png', 'the white mouse']
            ]

    for i, test in enumerate(TESTS):
        im_id = test[0]
        expr = test[1]
        img_path = osp.join(ROOT_DIR, "images/" + im_id)
        img_cv = cv2.imread(img_path)
        test_obj_manipulation(img_cv, expr)
        print('!!!!! test {} of {} complete'.format(i + 1, len(TESTS)))