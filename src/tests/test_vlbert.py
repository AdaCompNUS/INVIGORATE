import matplotlib
matplotlib.use('Agg')

import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import time

from config.config import *
from vmrn_msgs.srv import *
from libraries.data_viewer.data_viewer import DataViewer

br = CvBridge()

# ------- Settings ------
DBG_PRINT = False

TEST_OBJECT_DETECTION = True
TEST_REFER_EXPRESSION = True
TEST_CLS_NAME_FILTER = True
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
        res = obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.box_feats
    except rospy.ServiceException as e:
        # print("Service call failed: %s"%e)
        pass

def vlbert_client(img, bboxes, expr, num_box):
    rospy.wait_for_service('vlbert_service')
    try:
        vlbert_client = rospy.ServiceProxy('vlbert_service', VLBert)
        req = VLBertRequest()
        img_msg = br.cv2_to_imgmsg(img)
        req.img = img_msg
        req.bboxes = bboxes.flatten().tolist()
        req.expr = expr

        resp = vlbert_client(req)
        ground_prob = np.array(list(resp.ground_prob))
        # TODO 
        # rel_score_mat = np.array(list(resp.rel_score_mat)).reshape(num_box, num_box)
        # rel_mat = 
        rel_score_mat = None
        rel_mat = None
        return ground_prob, rel_score_mat, rel_mat
    except rospy.ServiceException as e:
        # print("Service call failed: %s"%e)
        pass


def test_obj_manipulation(img_cv, expr):
    tb = time.time()
    num_box, bboxes, classes, bbox_feats = faster_rcnn_client(img_cv)
    # dbg_print(bboxes)
    print(num_box)
    bboxes_with_score = np.array(bboxes).reshape(-1, 5)
    classes = np.array(classes).reshape(-1, 1)
    print('classes: {}'.format(classes))
    bboxes = bboxes_with_score[:, :4]

    data_viewer = DataViewer(CLASSES)

    grounding_scores, rel_score_mat, rel_mat = vlbert_client(img_cv, bboxes, expr, num_box)
    print(grounding_scores)
    print(rel_score_mat)
    print(rel_mat)

    ############ visualize

    # resize img for visualization
    scalar = 500. / min(img_cv.shape[:2])
    img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    bboxes_with_cls = np.concatenate([bboxes, classes], axis=-1)
    vis_bboxes = bboxes_with_cls * scalar
    vis_bboxes[:, -1] = bboxes_with_cls[:, -1]
    
    # object detection
    print('vis_bboxes: {}'.format(vis_bboxes))
    object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(classes.shape[0])))

    cv2.imwrite("object_det.png", object_det_img)

    vis_rel_score_mat = data_viewer.relscores_to_visscores(rel_score_mat)

    blank_img = np.zeros((img_cv.shape), np.uint8)

    # relation
    rel_det_img = blank_img
    # rel_det_img = data_viewer.draw_mrt(img_cv.copy(), rel_mat, rel_score = vis_rel_score_mat)
    # rel_det_img = cv2.resize(rel_det_img, (img_cv.shape[1], img_cv.shape[0]))

    # grounding
    grounding_scores = np.append(grounding_scores, 0) # dummy background score
    dbg_print('ground_scores: {}'.format(grounding_scores))
    ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, grounding_scores)

    # save result
    final_img = np.concatenate([np.concatenate([object_det_img, rel_det_img], axis = 1),
                                np.concatenate([ground_img, blank_img], axis=1)], axis = 0)
    # out_dir = "../images/output"
    # save_name = im_id.split(".")[0] + "_result.png"
    # save_path = os.path.join(out_dir, save_name)
    # i = 1
    # while (os.path.exists(save_path)):
    #     i += 1
    #     save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
    #     save_path = os.path.join(out_dir, save_name)
    # cv2.imwrite(save_path, final_img)
    cv2.imshow('img', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('test')

    img_cv = cv2.imread("../images/33.png")
    expr = 'banana'
    test_obj_manipulation(img_cv, expr)
