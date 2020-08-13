import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('../faster_rcnn_detector/VMRN/Visual-Manipulation-Relationship-Network-Pytorch/')

import rospy
from model.utils.data_viewer import dataViewer
from model.utils.net_utils import leaf_and_descendant_stats, inner_loop_planning, relscores_to_visscores

from faster_rcnn_detector.srv import ObjectDetection
from vmrn_old.srv import VmrDetection
from vmrn_msgs.srv import MAttNetGrounding
from ingress_srv.ingress_srv import Ingress

import cv2
from cv_bridge import CvBridge
br = CvBridge()

import torch
import numpy as np
import os

import time

# ------- Settings ------
TEST_OBJECT_DETECTION = True
TEST_REFER_EXPRESSION = True
TEST_CAPTION_GENERATION = True
TEST_MRT_DETECTION = False
TEST_GRASP_POLICY = False

def faster_rcnn_client(img):
    rospy.wait_for_service('faster_rcnn_server')
    try:
        obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = obj_det(img_msg)
        return res.num_box, res.bbox, res.cls
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

def mattnet_client(img, bbox, cls, expr):
    rospy.wait_for_service('mattnet_server')
    try:
        grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)
        img_msg = br.cv2_to_imgmsg(img)
        res = grounding(img_msg, bbox, cls, expr)
        return res.ground_prob
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

# def densecap_client(img, bbox, classes, expr):
#     print(bbox)
#     ingress_client = Ingress()
#     filtered_box, _, _, selected_idxes = ingress_client.load_img_and_filter_bbox(img, bbox, expr)
#     filtered_cls = np.take(np.array(classes), selected_idxes, axis=0)
#     return filtered_box, filtered_cls

def caption_generation_client(img, bbox, target_box_id):
    print(bbox)
    ingress_client = Ingress()
    top_caption, top_context_box_idx = ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
    return top_caption, top_context_box_idx

def vis_action(action_str, shape):
    im = 255. * np.ones(shape)
    cv2.putText(im, action_str, (0, im.shape[0] / 2),
                cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 0), thickness=2)
    return im


def test(img_cv, expr):
    # test object detection
    if TEST_OBJECT_DETECTION:
        tb = time.time()
        obj_result = faster_rcnn_client(img_cv)
        print(obj_result[1])
        num_box = obj_result[0]
        print(num_box)
        bboxes = np.array(obj_result[1]).reshape(-1, 5)
        cls = np.array(obj_result[2]).reshape(-1, 1)
        bbox_2d = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, cls], axis=-1)
    else:
        print('TEST_OBJECT_DETECTION is false, quit')
        return
    
    if TEST_MRT_DETECTION:
        rel_result = vmrn_client(img_cv, obj_result[1])
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
    else:
        print('TEST_MRT_DETECTION is false, skip')
    
    if TEST_REFER_EXPRESSION:
        filtered_box = obj_result[1]
        filtered_cls = obj_result[2]
        ground_result = mattnet_client(img_cv, filtered_box, filtered_cls, expr)
        bg_score = 0.3
        ground_result += (bg_score,)
        ground_result = torch.nn.functional.softmax(10 * torch.Tensor(ground_result), dim=0)
    else:
        print('TEST_REFER_EXPRESSION is false, skip')

    if TEST_CAPTION_GENERATION:
        print(ground_result)
        ground_result = torch.sort(ground_result, descending=True)
        ground_result_idx = torch.argsort(ground_result, descending=True)
        if (ground_result[0] - ground_result[1] < 0.1):
            target_box_ind = ground_result_idx[0]
            # target_box_ind = 0 # hard code now.
            top_caption, top_context_box_idxs = caption_generation_client(img_cv, bbox_2d, target_box_ind)
            print('top_caption: {}'.format(top_caption))
            print('top_context_box_idxs: {}'.format(top_context_box_idxs))

    if TEST_GRASP_POLICY:
        belief = {}
        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            rel_score_mat = torch.from_numpy(rel_score_mat)
            rel_score_mat *= triu_mask
            belief["leaf_desc_prob"] = leaf_and_descendant_stats(rel_score_mat)
        belief["ground_prob"] = ground_result
        a = inner_loop_planning(belief)
        print('TEST_GRASP_POLICY is false, skip')

    ############ visualize
    # resize img for visualization
    scalar = 500. / min(img_cv.shape[:2])
    img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    vis_bboxes = bboxes * scalar
    vis_bboxes[:, -1] = bboxes[:, -1]

    # temporarily
    if top_context_box_idxs == vis_bboxes.shape[0]:
        # top context is the whole image:
        vis_bboxes = np.take(vis_bboxes, [target_box_ind, 4], axis=0)
    else:
        vis_bboxes = np.take(vis_bboxes, [target_box_ind, top_context_box_idxs], axis=0)    

    # object detection
    object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(cls.shape[0])))
    cv2.imwrite("object_det.png", object_det_img)

    # relationship detection
    if TEST_MRT_DETECTION:
        rel_det_img = data_viewer.draw_mrt(img_cv.copy(), rel_mat, rel_score = vis_rel_score_mat)
        rel_det_img = cv2.resize(rel_det_img, (img_cv.shape[1], img_cv.shape[0]))
        cv2.imwrite("relation_det.png", rel_det_img)
    else:
        rel_det_img = np.zeros((img_cv.shape), np.uint8)

    # grounding
    if TEST_REFER_EXPRESSION:
        print("Grounding Score: ")
        print(ground_result)
        print("Grounding Probability: ")
        print(ground_result.tolist())
        ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_result[:-1].numpy())
        cv2.imwrite("ground.png", ground_img)
    else:
        ground_img = np.zeros((img_cv.shape), np.uint8)

    if TEST_CAPTION_GENERATION:
        caption_img = vis_action(top_caption, img_cv.shape)
    else:
        caption_img = np.zeros((img_cv.shape), np.uint8)

    if TEST_GRASP_POLICY:
        print("Optimal Action:")
        if a < num_box:
            action_str = "Grasping object " + str(a)
        elif a < 2 * num_box:
            action_str ="Asking Q1 for " + str(a-num_box) + "th object"
        else:
            action_str ="Asking Q2"
        print("Time Consuming: "+ str(time.time() - tb) + "s")
        action_img = vis_action(action_str, img_cv.shape)
    else:
        action_img = np.zeros((img_cv.shape), np.uint8)

    # save result
    final_img = np.concatenate([object_det_img, caption_img], axis = 1)
    # final_img = np.concatenate([np.concatenate([object_det_img, rel_det_img], axis = 1),
    #                             np.concatenate([ground_img, action_img], axis=1),], axis = 0)
    out_dir = "../images/output"
    save_name = im_id.split(".")[0] + "_result.png"
    save_path = os.path.join(out_dir, save_name)
    i = 1
    while (os.path.exists(save_path)):
        i += 1
        save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
        save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, final_img)

if __name__ == "__main__":
    rospy.init_node('test')

    classes = ['__background__',  # always index 0
                 'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                 'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                 'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                 'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                 'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']
    data_viewer = dataViewer(classes)

    # user input
    img_array = ['']

    im_id = "37.png"
    expr = "black mouse"

    img_cv = cv2.imread("../images/" + im_id)

    test(img_cv, expr)

