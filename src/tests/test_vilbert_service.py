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
from scipy.special import softmax

from config.config import *
from invigorate_msgs.srv import *
from libraries.data_viewer.data_viewer import DataViewer

from rls_perception_msgs.srv import *

br = CvBridge()

# ------- Settings ------
DBG_PRINT = True

# ------- Constants -------

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

# def detectron_client(img):
#     rospy.wait_for_service('rls_perception_services/object_detection_srv')
#     try:
#         obj_det = rospy.ServiceProxy('rls_perception_services/object_detection_srv', DetectObjects)
#         img_msg = br.cv2_to_imgmsg(img)
#         res = obj_det(img_msg)
#         num_box = len(res.objects)
#         bboxes = []
#         cls = []

#         for object in res.objects:
#             bbox = list(object.bbox)
#             bbox.append(object.prob)
#             bboxes.append(bbox)
#             cls.append(object.class_name)
#         return num_box, bboxes, cls, None
#     except rospy.ServiceException as e:
#         # print("Service call failed: %s"%e)
#         pass

def vilbert_client(img, bboxes, expr, num_box):
    rospy.wait_for_service('vilbert_grounding_service')
    try:
        vilbert_client = rospy.ServiceProxy('vilbert_grounding_service', Grounding)
        req = GroundingRequest()
        img_msg = br.cv2_to_imgmsg(img)
        req.img = img_msg
        req.bboxes = bboxes.flatten().tolist()
        req.expr = expr

        resp = vilbert_client(req)
        grounding_scores = np.array(list(resp.grounding_scores))
        # rel_score_mat = np.array(list(resp.rel_score_mat)).reshape(3, num_box, num_box)
        # rel_mat = np.argmax(rel_score_mat, axis=0) + 1
        return grounding_scores
    except rospy.ServiceException as e:
        # print("Service call failed: %s"%e)
        pass

def test_obj_manipulation(img_cv, expr, im_id):
    tb = time.time()
    num_box, bboxes, classes, bbox_feats = faster_rcnn_client(img_cv)
    # num_box, bboxes, classes, bbox_feats = detectron_client(img_cv)
    bboxes_with_score = np.array(bboxes).reshape(-1, 5)
    classes = np.array(classes).reshape(-1, 1)
    bboxes = bboxes_with_score[:, :4]
    dbg_print(bboxes)
    dbg_print(num_box)
    dbg_print('classes: {}'.format(classes))

    data_viewer = DataViewer(CLASSES)

    grounding_scores= vilbert_client(img_cv, bboxes, expr, num_box)
    dbg_print(grounding_scores)

    # name replacement trick:
    # ground_prob = softmax(grounding_scores)
    # for i in range(len(ground_prob)): # exclude background
    #     cls_name = CLASSES[classes[i][0]]
    #     if cls_name not in expr: # simple filter
    #         ground_prob[i] = 0.0

    # ground_prob /= np.sum(ground_prob)
    # dbg_print('filtered_ground_prob: {}'.format(ground_prob))
    ground_prob = grounding_scores

    ############ visualize

    # resize img for visualization
    scalar = 500. / min(img_cv.shape[:2])
    img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    bboxes_with_cls = np.concatenate([bboxes, classes], axis=-1)
    vis_bboxes = bboxes_with_cls * scalar
    vis_bboxes[:, -1] = bboxes_with_cls[:, -1]

    # blank img filler
    blank_img = np.zeros((img_cv.shape), np.uint8)

    # object detection
    dbg_print('vis_bboxes: {}'.format(vis_bboxes))
    object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(classes.shape[0])))

    # relation
    rel_det_img = blank_img
    # vis_rel_score_mat = data_viewer.relscores_to_visscores(rel_score_mat)
    # rel_det_img = data_viewer.draw_mrt(img_cv.copy(), rel_mat, rel_score = vis_rel_score_mat, with_img=False)
    # rel_det_img = cv2.resize(rel_det_img, (img_cv.shape[1], img_cv.shape[0]))

    # grounding
    ground_prob = np.append(ground_prob, -100.0) # dummy background score
    dbg_print('ground_prob: {}'.format(ground_prob))
    ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_prob)

    final_img = np.concatenate([np.concatenate([object_det_img, rel_det_img], axis = 1),
                                np.concatenate([ground_img, blank_img], axis=1)], axis = 0)

    ## save result
    out_dir = "../images/output"
    save_name = im_id.split(".")[0] + "_result.png"
    save_path = os.path.join(out_dir, save_name)
    i = 1
    while (os.path.exists(save_path)):
        i += 1
        save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
        save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, final_img)

    # Show images (Optional)
    # cv2.imshow('img', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('test_vlbert')

    # Unit Test
    # img_cv = cv2.imread(r"/media/peacock-rls/My Passport/datasets/coco/coco/images/train2014/COCO_train2014_000000575461.jpg")
    # img_cv = cv2.imread("../images/107.png")
    # expr = 'blue box'
    # test_obj_manipulation(img_cv, expr, 'test')

    # Batch Test with Vis
    TESTS = [['1.png', 'cup under banana'], ['1.png', 'cup under apple'],
             ['13.png', 'remote'], ['13.png', 'cup'], ['13.png', 'white mouse'],
             ['15.png', 'white mouse'], ['15.png', 'black mouse'], ['15.png', 'mouse on the left'],
             ['21.png', 'banana on top'], ['21.png', 'banana below'], ['21.png', 'apple'],
             ['36.png', 'apple under banana'], ['36.png', 'cup under apple'], ['36.png', 'cup under banana'],
             ['37.png', 'apple under banana'],
             ['38.png', 'apple under banana'],
             ['60.jpg', 'apple on the left'], ['60.jpg', 'apple on the right'], ['60.jpg', 'blue cup'], ['60.jpg', 'green cup'],
             ['table.png', 'bottle next to banana'], ['table.png', 'top left bottle']
            ]

    for i, test in enumerate(TESTS):
        im_id = test[0]
        expr = test[1]
        img_cv = cv2.imread("../images/" + im_id)
        test_obj_manipulation(img_cv, expr, im_id)
        print('!!!!! test {} of {} complete'.format(i + 1, len(TESTS)))
