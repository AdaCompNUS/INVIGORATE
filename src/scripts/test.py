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
TEST_OBJECT_DETECTION = True
TEST_REFER_EXPRESSION = True
TEST_CAPTION_GENERATION = True
TEST_MRT_DETECTION = False
TEST_GRASP_POLICY = False

# ------- Constants -------
AMBIGUOUS_THRESHOLD = 0.2

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

def form_rel_caption_sentence(obj_cls, cxt_obj_cls, rel_caption):
    print(obj_cls, cxt_obj_cls)
    obj_name = CLASSES[obj_cls]
    cxt_obj_name = CLASSES[cxt_obj_cls]

    rel_caption_sentence = '{} {} of {}'.format(obj_name, rel_caption, cxt_obj_name)
    rel_caption_sentence.replace('.', '')
    return rel_caption_sentence
    

# def single_step_perception(self, img, expr, prevs=None, cls_filter=None):
#         tb = time.time()
#         obj_result = self.faster_rcnn_client(img)
#         bboxes = np.array(obj_result[1]).reshape(-1, 4 + 32)
#         cls = np.array(obj_result[2]).reshape(-1, 1)
#         bboxes, cls = self.bbox_filter(bboxes, cls)

#         scores = bboxes[:, 4:].reshape(-1, 32)
#         bboxes = bboxes[:, :4]
#         bboxes = np.concatenate([bboxes, cls], axis=-1)

#         ind_match_dict = {}
#         if prevs is not None:
#             ind_match_dict = self.bbox_match(bboxes, prevs["bbox"])
#             self.history_scores[-1]["mapping"] = ind_match_dict
#             not_matched = set(range(bboxes.shape[0])) - set(ind_match_dict.keys())
#             ignored = set(range(prevs["bbox"].shape[0])) - set(ind_match_dict.values())

#             # ignored = list(ignored - {prevs["actions"]})
#             # bboxes = np.concatenate([bboxes, prevs["bbox"][ignored]], axis=0)
#             # cls = np.concatenate([cls, prevs["cls"][ignored]], axis=0)
#             prevs["qa_his"] = self.qa_his_mapping(prevs["qa_his"], ind_match_dict, not_matched, ignored)

#         num_box = bboxes.shape[0]

#         rel_result = self.vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
#         rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
#         rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
#         if prevs is not None:
#             # modify the relationship probability according to the new observation
#             rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] += \
#                 prevs["rel_score_mat"][:, ind_match_dict.values()][:, :, ind_match_dict.values()]
#             rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] /= 2

#         with torch.no_grad():
#             triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
#             triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
#             leaf_desc_prob = leaf_and_descendant_stats(torch.from_numpy(rel_score_mat) * triu_mask).numpy()

#         ground_score = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), cls.reshape(-1).tolist(), expr)
#         bg_score = BG_SCORE
#         ground_score += (bg_score,)
#         ground_score = np.array(ground_score)
#         self.history_scores.append({"scores" : ground_score})
#         if prevs is not None:
#             # ground_score[ind_match_dict.keys()] += prevs["ground_score"][ind_match_dict.values()]
#             # ground_score[ind_match_dict.keys()] /= 2
#             ground_score[ind_match_dict.keys()] = np.maximum(
#                 prevs["ground_score"][ind_match_dict.values()],
#                 ground_score[ind_match_dict.keys()])
#         ground_result = self.score_to_prob(ground_score)

#         # utilize the answered questions to correct grounding results.
#         for i in range(bboxes.shape[0]):
#             box_score = 0
#             for class_str in cls_filter:
#                 box_score += scores[i][classes_to_ind[class_str]]
#             if box_score < 0.02:
#                 ground_result[i] = 0
#         ground_result /= ground_result.sum()

#         if prevs is not None:
#             ground_result_backup = ground_result.copy()
#             for k in prevs["qa_his"].keys():
#                 if k == "bg":
#                     # target has already been detected in the last step
#                     for i in not_matched:
#                         ground_result[i] = 0
#                     ground_result[-1] = 0
#                 elif k == "clue":
#                     clue = prevs["qa_his"]["clue"]
#                     t_ground = self.mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), cls.reshape(-1).tolist(), clue)
#                     t_ground += (BG_SCORE,)
#                     t_ground = self.score_to_prob(np.array(t_ground))
#                     t_ground = np.expand_dims(t_ground, 0)
#                     leaf_desc_prob[:, -1] = (t_ground[:, :-1] * leaf_desc_prob[:, :-1]).sum(-1)
#                 else:
#                     ground_result[k] = 0

#             if ground_result.sum() > 0:
#                 ground_result /= ground_result.sum()
#             else:
#                 # something wrong with the matching process. roll back
#                 for i in not_matched:
#                     ground_result[i] = ground_result_backup[i]
#                 ground_result[-1] = ground_result_backup[-1]

#         print("Perception Time Consuming: " + str(time.time() - tb) + "s")

#         if prevs is not None:
#             return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, prevs["qa_his"]
#         else:
#             return bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, {}


def test(img_cv, expr):
    # test object detection
    if TEST_OBJECT_DETECTION:
        tb = time.time()
        num_box, bboxes, classes, bbox_feats = faster_rcnn_client(img_cv)
        print(bboxes)
        print(num_box)
        bboxes = np.array(bboxes).reshape(-1, 5)
        classes = np.array(classes).reshape(-1, 1)
        print('classes: {}'.format(classes))
        bbox_2d = bboxes[:, :4]
        bboxes = np.concatenate([bboxes, classes], axis=-1)
    else:
        print('TEST_OBJECT_DETECTION is false, quit')
        return
    
    if TEST_MRT_DETECTION:
        # TODO!!!
        # rel_result = vmrn_client(img_cv, obj_result[1])
        # rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        # rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
        # vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        pass
    else:
        print('TEST_MRT_DETECTION is false, skip')
    
    if TEST_REFER_EXPRESSION:
        ground_scores = list(mattnet_client(img_cv, expr, bbox_feats))
        bg_score = 0.3
        ground_scores.append(bg_score)
        ground_prob = torch.nn.functional.softmax(10 * torch.Tensor(ground_scores), dim=0)
        print('ground_scores: {}'.format(ground_scores))
        print('ground_prob: {}'.format(ground_prob))
    else:
        print('TEST_REFER_EXPRESSION is false, skip')

    if TEST_CAPTION_GENERATION:
        ground_scores_sorted = sorted(ground_scores, reverse=True)
        # ground_prob_sorted, ground_prob_idx = torch.sort(ground_prob, descending=True)
        if (ground_scores_sorted[0] - ground_scores_sorted[1] < AMBIGUOUS_THRESHOLD): # this is temporary
            # target_box_ind = ground_prob_idx[0]
            target_box_ind = ground_scores.index(ground_scores_sorted[0])
            if target_box_ind == len(ground_scores) - 1:
                target_box_ind = ground_scores.index(ground_scores_sorted[1])

            print('grounding is ambiguous, generate captions for object {}'.format(target_box_ind))
            # target_box_ind = 0 # hard code now.
            top_caption, top_context_box_idxs = caption_generation_client(img_cv, bbox_2d, target_box_ind)
            print('top_caption: {}'.format(top_caption))
            print('top_context_box_idxs: {}'.format(top_context_box_idxs))
            caption_sentence = form_rel_caption_sentence(classes[target_box_ind][0], classes[top_context_box_idxs][0], top_caption)
            print('caption_sentence: {}'.format(caption_sentence))
        else:
            top_caption = ''
            top_context_box_idxs = -1
            target_box_ind = -1
            print('not ambiguous, skip caption generation')

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
        ground_img = data_viewer.draw_grounding_probs(img_cv.copy(), expr, vis_bboxes, ground_prob[:-1].numpy())
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
        caption_img = vis_action(str(target_box_ind) + " " + caption_sentence, img_cv.shape)
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

    im_id = "60.jpg"
    expr = "cup"

    img_cv = cv2.imread("../images/" + im_id)

    test(img_cv, expr)

