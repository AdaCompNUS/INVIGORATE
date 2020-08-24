import matplotlib
matplotlib.use('Agg')

import sys
import os.path as osp
import rospy
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
import os
import time

import vmrn._init_path
from vmrn.model.utils.data_viewer import dataViewer
# from vmrn.model.utils.net_utils import relscores_to_visscores
from vmrn_msgs.srv import MAttNetGroundingV2, ObjectDetection, VmrDetection
from ingress_srv.ingress_srv import Ingress

br = CvBridge()

# ------- Settings ------
DBG_PRINT = False

TEST_OBJECT_DETECTION = True
TEST_REFER_EXPRESSION = True
TEST_CLS_NAME_FILTER = True
TEST_CAPTION_GENERATION = True
TEST_MRT_DETECTION = False
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

def mattnet_client(img, expr, img_data):
    rospy.wait_for_service('mattnet_server_v2')
    try:
        grounding = rospy.ServiceProxy('mattnet_server_v2', MAttNetGroundingV2)
        img_msg = br.cv2_to_imgmsg(img)
        res = grounding(img_msg, expr, img_data)
        return res.ground_scores
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

# def densecap_client(img, bbox, classes, expr):
#     print(bbox)
#     ingress_client = Ingress()
#     filtered_box, _, _, selected_idxes = ingress_client.load_img_and_filter_bbox(img, bbox, expr)
#     filtered_cls = np.take(np.array(classes), selected_idxes, axis=0)
#     return filtered_box, filtered_cls

def caption_generation_client(img, bbox, target_box_id):
    # dbg_print(bbox)
    ingress_client = Ingress()
    top_caption, top_context_box_idx = ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
    return top_caption, top_context_box_idx

def vis_action(action_str, shape):
    im = 255. * np.ones(shape)
    cv2.putText(im, action_str, (0, im.shape[0] / 2),
                cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 0, 0), thickness=1)
    return im

def form_rel_caption_sentence(obj_cls, cxt_obj_cls, rel_caption):
    obj_name = CLASSES[obj_cls]
    cxt_obj_name = CLASSES[cxt_obj_cls]
    
    if cxt_obj_cls == 0:
        rel_caption_sentence = '{} {} (of image)'.format(obj_name, rel_caption)
    else:
        rel_caption_sentence = '{} {} of {}'.format(obj_name, rel_caption, cxt_obj_name)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def test_obj_manipulation(img_cv, expr):
    # test object detection
    if TEST_OBJECT_DETECTION:
        tb = time.time()
        num_box, bboxes, classes, bbox_feats = faster_rcnn_client(img_cv)
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
    
    if TEST_MRT_DETECTION:
        # TODO!!!
        # rel_result = vmrn_client(img_cv, obj_result[1])
        # rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        # rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        # vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        pass
    else:
        dbg_print('TEST_MRT_DETECTION is false, skip')
    
    if TEST_REFER_EXPRESSION:
        ground_scores = list(mattnet_client(img_cv, expr, bbox_feats))
        ground_scores.append(BG_SCORE)
        ground_prob = torch.nn.functional.softmax(10 * torch.Tensor(ground_scores), dim=0)
        dbg_print('ground_scores: {}'.format(ground_scores))
        dbg_print('ground_prob: {}'.format(ground_prob))
    else:
        dbg_print('TEST_REFER_EXPRESSION is false, skip')

    if TEST_CLS_NAME_FILTER:
        for i in range(len(ground_scores) - 1): # exclude background
            cls_name = CLASSES[classes[i][0]]
            if cls_name not in expr: # simple filter
                ground_prob[i] = 0.0
                ground_scores[i] = -float('inf')
        
        ground_prob /= torch.sum(ground_prob)
        dbg_print('filtered_ground_prob: {}'.format(ground_prob))

    if TEST_CAPTION_GENERATION:
        ground_scores_sorted = sorted(ground_scores, reverse=True)
        # ground_prob_sorted, ground_prob_idx = torch.sort(ground_prob, descending=True)
        if (ground_scores_sorted[0] - ground_scores_sorted[1] < AMBIGUOUS_THRESHOLD): # this is temporary
            # target_box_ind = ground_prob_idx[0]
            target_box_ind = ground_scores.index(ground_scores_sorted[0])
            if target_box_ind == len(ground_scores) - 1:
                # target_box_ind = ground_scores.index(ground_scores_sorted[1])
                top_caption = 'background'
                top_context_box_idxs = -1
                target_box_ind = -1
                dbg_print('top_caption: {}'.format(top_caption))
            else:
                dbg_print('grounding is ambiguous, generate captions for object {}'.format(target_box_ind))
                # target_box_ind = 0 # hard code now.
                top_caption, top_context_box_idxs = caption_generation_client(img_cv, bbox_2d, target_box_ind)
                dbg_print('top_caption: {}'.format(top_caption))
                dbg_print('top_context_box_idxs: {}'.format(top_context_box_idxs))
                if top_context_box_idxs == len(ground_scores) - 1:
                    # top context is background
                    caption_sentence = form_rel_caption_sentence(classes[target_box_ind][0], 0, top_caption)
                else:
                    caption_sentence = form_rel_caption_sentence(classes[target_box_ind][0], classes[top_context_box_idxs][0], top_caption)
                dbg_print('caption_sentence: {}'.format(caption_sentence))
        else:
            top_caption = ''
            top_context_box_idxs = -1
            target_box_ind = -1
            dbg_print('not ambiguous, skip caption generation')

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
        dbg_print('TEST_GRASP_POLICY is false, skip')

    ############ visualize
    # resize img for visualization
    scalar = 500. / min(img_cv.shape[:2])
    img_cv = cv2.resize(img_cv, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    vis_bboxes = bboxes * scalar
    vis_bboxes[:, -1] = bboxes[:, -1]

    # object detection
    object_det_img = data_viewer.draw_objdet(img_cv.copy(), vis_bboxes, list(range(classes.shape[0])))
    cv2.imwrite("object_det.png", object_det_img)

    # relationship detection
    if TEST_MRT_DETECTION:
        rel_det_img = data_viewer.draw_mrt(img_cv.copy(), rel_mat, rel_score = vis_rel_score_mat)
        rel_det_img = cv2.resize(rel_det_img, (img_cv.shape[1], img_cv.shape[0]))
        # cv2.imwrite("relation_det.png", rel_det_img)
    else:
        rel_det_img = np.zeros((img_cv.shape), np.uint8)

    # grounding
    if TEST_REFER_EXPRESSION:
        ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_prob.numpy())
        # cv2.imwrite("ground.png", ground_img)
    else:
        ground_img = np.zeros((img_cv.shape), np.uint8)

    if TEST_CAPTION_GENERATION and top_caption != '':
        # temporarily
        # if top_context_box_idxs == vis_bboxes.shape[0]:
        #     # top context is the whole image:
        #     vis_bboxes = np.take(vis_bboxes, [target_box_ind, 4], axis=0)
        # else:
        #     vis_bboxes = np.take(vis_bboxes, [target_box_ind, top_context_box_idxs], axis=0)
        if top_caption == 'background':
            caption_vis_sentence = top_caption
        else:
            caption_vis_sentence = 'for object {}: {}'.format(target_box_ind, caption_sentence)
        caption_img = vis_action(caption_vis_sentence, img_cv.shape)
    else:
        caption_img = np.zeros((img_cv.shape), np.uint8)

    if TEST_GRASP_POLICY:
        dbg_print("Optimal Action:")
        if a < num_box:
            action_str = "Grasping object " + str(a)
        elif a < 2 * num_box:
            action_str ="Asking Q1 for " + str(a-num_box) + "th object"
        else:
            action_str ="Asking Q2"
        dbg_print("Time Consuming: "+ str(time.time() - tb) + "s")
        action_img = vis_action(action_str, img_cv.shape)
    else:
        action_img = np.zeros((img_cv.shape), np.uint8)

    blank_img = np.zeros((img_cv.shape), np.uint8)

    # save result
    final_img = np.concatenate([np.concatenate([object_det_img, rel_det_img], axis = 1),
                                np.concatenate([ground_img, caption_img], axis=1),
                                np.concatenate([action_img, blank_img], axis=1)], axis = 0)
    out_dir = "../images/output"
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

    CLASSES = ['__background__',  # always index 0
                 'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                 'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                 'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                 'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                 'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']
    data_viewer = dataViewer(CLASSES)

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
             ['table.png', 'bottle next to banana'], ['table.png', 'top left bottle']
            ]

    for i, test in enumerate(TESTS):
        im_id = test[0]
        expr = test[1]
        img_cv = cv2.imread("../images/" + im_id)
        test_obj_manipulation(img_cv, expr)
        print('!!!!! test {} of {} complete'.format(i + 1, len(TESTS)))



