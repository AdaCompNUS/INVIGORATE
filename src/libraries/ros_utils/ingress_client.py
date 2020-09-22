#!/usr/bin/env python

import rospy
from roslib import message
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import actionlib
import copy
import numpy as np
import math
from argparse import ArgumentParser
from datetime import datetime

import ingress_msgs.msg
import ingress_msgs.srv


# Settings
DEBUG = False
PUBLISH_DEBUG_RESULT = True

# Constants
VALID_MIN_CLUSTER_SIZE = 500  # points
VALID_MIN_RELEVANCY_SCORE = 0.05
VALID_MIN_METEOR_SCORE = 0.1
NAME_REPLACEMENT_METEOR_SCORE_THRESHOLD = 0.1


def dbg_print(text):
    if DEBUG:
        print(text)


class IngressClient():
    def __init__(self):
        # wait for grounding load server
        self._load_client = actionlib.SimpleActionClient(
            'dense_refexp_load', ingress_msgs.msg.DenseRefexpLoadAction)
        rospy.loginfo("DemoManager: 1. Waiting for dense_refexp_load action server ...")
        self._self_captions = []
        self._load_client.wait_for_server()

        # wait for ground bbox load server
        self._load_bbox_client = actionlib.SimpleActionClient(
            'dense_refexp_load_bboxes', ingress_msgs.msg.DenseRefexpLoadBBoxesAction)
        rospy.loginfo("1. Waiting for dense_refexp_load_bboxes action server ...")
        self._load_bbox_client.wait_for_server()

        # wait for relevancy clustering server
        self._relevancy_client = actionlib.SimpleActionClient(
            'relevancy_clustering', ingress_msgs.msg.RelevancyClusteringAction)
        rospy.loginfo("DemoManager: 2. Waiting for relevancy_clustering action server ...")
        self._relevancy_client.wait_for_server()

        # wait for grounding query server
        self._query_client = actionlib.SimpleActionClient(
            'boxes_refexp_query', ingress_msgs.msg.BoxesRefexpQueryAction)
        rospy.loginfo("DemoManager: 3. Waiting for boxes_refexp_query action server ...")
        self._rel_captions = []
        self._query_client.wait_for_server()

        # meteor score client:
        self._meteor_score_client = rospy.ServiceProxy(
            'meteor_score', ingress_msgs.srv.MeteorScore)

        # publisher for Ingress grounding results
        self._grounding_result_pub = rospy.Publisher(
            'ingress/dense_refexp_result', Image, queue_size=10, latch=True)

        # demo ready!
        rospy.loginfo("Ingress ready!")

    def _ground_load(self, image):
        '''
        load the image into the grounding model
        input: image
        output: bounding boxes
        '''

        goal = ingress_msgs.msg.DenseRefexpLoadGoal(image)
        rospy.loginfo("_ground_load: sending goal")
        self._load_client.send_goal(goal)
        rospy.loginfo("_ground_load: waiting for result")
        self._load_client.wait_for_result()
        rospy.loginfo("_ground_load: getting result")
        load_result = self._load_client.get_result()

        boxes = np.reshape(load_result.boxes, (-1, 4))
        losses = np.array(load_result.scores)
        self._ungrounded_captions = np.array(load_result.captions)

        return boxes, losses

    def _process_captions(self, relevancy_result, query_result, context_boxes_idxs, query):
        '''
        helper function to organize captions
        '''

        all_orig_idx = relevancy_result.all_orig_idx

        semantic_softmax = np.array(relevancy_result.softmax_probs)
        # this is disgusting.... puke! You should be ashamed of yourself Mohit.
        semantic_softmax_orig_idxs = [-1.] * len(all_orig_idx)
        for count, idx in enumerate(all_orig_idx):
            semantic_softmax_orig_idxs[idx] = semantic_softmax[count]

        top_idxs = context_boxes_idxs
        spatial_captions = query_result.predicted_captions
        spatial_softmax = np.array(query_result.probs) / np.sum(np.array(query_result.probs))

        dbg_print(all_orig_idx)
        dbg_print(semantic_softmax_orig_idxs)
        dbg_print(semantic_softmax)
        dbg_print(spatial_softmax)
        dbg_print(top_idxs)

        sem_captions = []
        sem_probs = []
        rel_captions = []
        rel_probs = []
        for (count, idx) in enumerate(top_idxs):
            if idx in context_boxes_idxs:
                sem_captions.append(self._ungrounded_captions[idx])
                sem_probs.append(semantic_softmax_orig_idxs[idx])
                if count < len(spatial_softmax) and count < len(spatial_captions):
                    rel_captions.append(spatial_captions[count].replace(
                        query, '').replace('.', '').strip())
                    rel_probs.append(spatial_softmax[count])
                else:
                    rospy.logwarn("Semantic and Spatial captions+prob lengths dont match!")

        return [sem_captions, sem_probs, rel_captions, rel_probs]

    def _ground_query(self, expr, boxes):
        '''
        query a specific expression with the grounding model
        input: expr
        output: best bounding box index, other box indexes sorted by grounding scores
        '''

        query = expr
        rospy.loginfo("Grounding Input String: " + query)

        # send expression for relevancy clustering
        incorrect_idxs = []

        # dialog storage
        self._relevancy_result = None
        self._context_boxes_idxs = None

        # relevancy
        relevancy_goal = ingress_msgs.msg.RelevancyClusteringGoal(query, incorrect_idxs)
        self._relevancy_client.send_goal(relevancy_goal)
        self._relevancy_client.wait_for_result()
        self._relevancy_result = self._relevancy_client.get_result()
        # selection_orig_idx = self._relevancy_client.get_result().selection_orig_idx
        selection_orig_idx = list(self._relevancy_result.selection_orig_idx)
        meteor_scores = list(self._relevancy_result.meteor_scores)
        rospy.loginfo("after relevancy clustering: bbox index {}".format(selection_orig_idx))
        rospy.loginfo("after relevancy clustering: meteor scores {}".format(meteor_scores))

        # clean
        # num_points_arr = [list(self._segment_pc(boxes[idx]))[1] for idx in selection_orig_idx]
        # print num_points_arr
        # pruned_selection_orig_idx = [i for i, idx in enumerate(selection_orig_idx) if num_points_arr[i] > VALID_MIN_CLUSTER_SIZE]
        # selection_orig_idx = list(np.take(selection_orig_idx, pruned_selection_orig_idx))
        # self._relevancy_result.selection_orig_idx = list(selection_orig_idx)

        # clean by threshold
        # semantic_softmax = np.array(self._relevancy_result.softmax_probs)

        # all_orig_idx = self._relevancy_result.all_orig_idx
        # # this is disgusting.... puke! You should be ashamed of yourself Mohit.
        # semantic_softmax_orig_idxs = [-1.] * len(all_orig_idx)
        # for count, idx in enumerate(all_orig_idx):
        #     semantic_softmax_orig_idxs[idx] = semantic_softmax[count]

        # pruned_selection_orig_idx = [i for i, idx in enumerate(
        #     selection_orig_idx) if semantic_softmax_orig_idxs[idx] > VALID_MIN_RELEVANCY_SCORE]
        # selection_orig_idx = list(np.take(selection_orig_idx, pruned_selection_orig_idx))
        # self._relevancy_result.selection_orig_idx = selection_orig_idx
        # rospy.loginfo("pruned index {}".format(selection_orig_idx))

        # clean by threshold on meteor score
        pruned_selection_orig_idx = [idx for i, idx in enumerate(selection_orig_idx)
                                     if meteor_scores[i] > VALID_MIN_METEOR_SCORE]
        selection_orig_idx = pruned_selection_orig_idx
        rospy.loginfo("pruned index {}".format(selection_orig_idx))

        if len(selection_orig_idx) == 0:
            rospy.logwarn("Ingress_srv: no object detected")
            return 0, [], None

        # ground the query
        selected_boxes = np.take(boxes, selection_orig_idx, axis=0)
        query_goal = ingress_msgs.msg.BoxesRefexpQueryGoal(
            query, np.array(selected_boxes).ravel(), selection_orig_idx, incorrect_idxs)
        self._query_client.send_goal(query_goal)
        self._query_client.wait_for_result()
        query_result = self._query_client.get_result()

        # grounding results: indexes of most likely bounding boxes
        top_idx = query_result.top_box_idx
        context_boxes_idxs = [top_idx] + list(query_result.context_boxes_idxs)
        self._context_boxes_idxs = list(context_boxes_idxs)

        rospy.loginfo("after relation query: bbox index {}".format(self._context_boxes_idxs))
        rospy.loginfo("after relation query: top idx {}".format(top_idx))

        # preprocess captions
        captions = self._process_captions(
            self._relevancy_result, query_result, context_boxes_idxs, query)

        return top_idx, context_boxes_idxs, captions

    def _reground_query(self, expr, boxes):
        '''
        ground new expr but with same boxes (for dialog interaction)
        NOTE: must be called after _ground_query
        '''

        query = expr
        if self._relevancy_result == None:
            rospy.logerr(
                "Grounding server wasn't initialized. Can't reground without initial grounding from _ground_query function")
            return None

        rospy.loginfo("Regrounding Input String: " + query)

        if query == '':
            return None, None, [None, [1.0]*len(self._relevancy_result.selection_orig_idx), None, [1.0]*len(self._relevancy_result.selection_orig_idx)]

        # send expression for relevancy clustering
        incorrect_idxs = []
        relevancy_goal = ingress_msgs.msg.RelevancyClusteringGoal(query, incorrect_idxs)
        self._relevancy_client.send_goal(relevancy_goal)
        self._relevancy_client.wait_for_result()
        new_relevancy_result = self._relevancy_client.get_result()
        # selection_orig_idx = self._relevancy_client.get_result().selection_orig_idx

        # ground the new query with old relevancy clustering
        selection_orig_idx = self._relevancy_result.selection_orig_idx
        selected_boxes = np.take(boxes, selection_orig_idx, axis=0)
        query_goal = ingress_msgs.msg.BoxesRefexpQueryGoal(
            query, np.array(selected_boxes).ravel(), selection_orig_idx, incorrect_idxs)
        self._query_client.send_goal(query_goal)
        self._query_client.wait_for_result()
        query_result = self._query_client.get_result()

        # grounding results: indexes of mostly bounding boxes
        top_idx = query_result.top_box_idx
        context_boxes_idxs = [top_idx] + list(query_result.context_boxes_idxs)

        # preprocess captions
        captions = self._process_captions(
            new_relevancy_result, query_result, self._context_boxes_idxs, query)

        return top_idx, context_boxes_idxs, captions

    def _publish_grounding_result(self, boxes, context_boxes_idxs, captions=None):
        '''
        debugging visualization of the bounding box outputs from the grounding model
        '''

        if not PUBLISH_DEBUG_RESULT:
            return

        # Input validity check
        if len(boxes) == 0 or len(context_boxes_idxs) == 0:
            return

        # Deepcopy is important here, otherwise self._img_msg will be contaminated.
        draw_img = copy.deepcopy(CvBridge().imgmsg_to_cv2(self._img_msg))
        for (count, idx) in enumerate(context_boxes_idxs):

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0] + boxes[idx][2])
            y2 = int(boxes[idx][1] + boxes[idx][3])

            if count == 0:  # top idx
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            elif count < len(context_boxes_idxs):
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # add captions
            if captions is not None:
                sem_captions, _, _, _ = captions
                font = cv2.FONT_HERSHEY_DUPLEX
                print("x1 {}, y1 {}".format(x1, y1))
                if y1 - 15 > 5:
                    cv2.putText(draw_img, sem_captions[count],
                                (x1 + 6, y1 - 15), font, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(draw_img, sem_captions[count],
                                (x1 + 6, y1 + 5), font, 1, (255, 255, 255), 2)

        # cv2.imshow('image', draw_img)
        # cv2.waitKey(50)
        # cv2.destroyAllWindows()
        result_img = CvBridge().cv2_to_imgmsg(draw_img, encoding='bgr8')
        self._grounding_result_pub.publish(result_img)
        rospy.loginfo("grounding result published")

    """
    def _replace_object_names(self, captions, true_names):
        '''
        Replace object name in captions if it is not close to the true object name
        '''

        # input validity check:
        if len(captions) != len(true_names) + 1:
            # captions include a caption for the whole image, therefore, the length is 1 more than len(true_names)
            rospy.logerr("_replace_object_names: captions len != true_names len")
            return False

        for i in range(len(true_names)):
            meteor_score = self._get_meteor_score(captions[i], true_names[i])
            rospy.loginfo("captions: {}, true_names: {}, score {}".format(
                captions[i], true_names[i], meteor_score))
            if meteor_score < NAME_REPLACEMENT_METEOR_SCORE_THRESHOLD:
                # replace the whole caption
                captions[i] = true_names[i]
                # change score to 1.0 to indicate that the name is replaced
                self._ungrounded_caption_scores[i] = 1.0

                # TODO replace noun only
                # l = captions[i].split() # split string into list
                # noun_idx = self._get_noun_idx(l)
                # l[noun_idx] = true_names[i]
                # captions[i] = " ".join(l) # join list into string

        return True

    def _get_meteor_score(self, text1, text2):
        resp = self._meteor_score_client(text1, text2)
        return resp.score


    def _get_noun_idx(self, str_list):
        # Hardcode to find noun
        # between 'a', 'the' / 'colour' and 'on', 'in'
        res_idx = -1
        for i in reversed(range(len(str_list))):
            if str_list[i] in COLOUR_LIST or str_list[i] in PREFIX_LIST:
                res_idx = i + 1
                break

            if str_list[i] in PREPOSITION_LIST:
                res_idx = i - 1


        return
    """

    def ground(self, image, expr):
        '''
        run the full grounding pipeline
        @param image, cv2 img
        @param expr, string, user input to describe the target object
        @return boxes, the detected bounding boxes
                top_idx, the index of the most likely boudning box in boxes
                context_idx, maps from captions index tp bbox indexs.
                             The first value in context_idx is the index of bounding box for the first value in sem_captions.
                             For example, sem_caption for most likely bbox is sem_caption[context_idx.index(top_idx)]
                captions, (sem_captions, sem_probs, rel_captions, rel_probs) tuple
        '''

        # Input validity check
        if image is None:
            rospy.logerr("DemoMovo/ground: image is none, skip")
            return None

        self._img_msg = CvBridge().cv2_to_imgmsg(image, "rgb8")
        bboxes, losses = self._ground_load(self._img_msg)

        # if user exprssion is empty, just generate captions for image
        if expr == '':
            rospy.loginfo("Ingress: empty query string received, returning ungrounded result")
            top_idx = 0
            context_idxs = [i for i in range(0, len(bboxes)) if losses[i] > 0]
            captions = ([self._ungrounded_captions[i]
                         for i in context_idxs], [losses[i] for i in context_idxs], [], None)
            self._publish_grounding_result(
                bboxes, context_idxs, captions)  # visualization of RViz
            return bboxes, top_idx, context_idxs, captions

        # else if user expression is not empty, ground user query
        else:
            top_idx, context_idxs, captions = self._ground_query(expr, bboxes)

        self._publish_grounding_result(bboxes, context_idxs)  # visualization of RViz
        return bboxes, top_idx, context_idxs, captions

    def ground_img_with_bbox(self, image, bboxes, expr='', true_names=None):
        '''
        Ground images with predefined bounding box
        @param image, cv2_img
        @param bbox, a list of bbox each in [top, left, width, height] format
        @param expr, user expression
        @param true_names, a list of true object names in bboxes. If the list is not None, the caption of bbox will be replaced by the true name
                           if it is far away from true name.
        @return boxes, the detected bounding boxes
                top_idx, the index of the most likely boudning box in boxes
                context_idx, maps from captions index tp bbox indexs.
                             The first value in context_idx is the index of bounding box for the first value in sem_captions.
                             For example, sem_caption for most likely bbox is sem_caption[context_idx.index(top_idx)]
                captions, (sem_captions, sem_probs, rel_captions, rel_probs) tuple
        '''

        # Input validity check
        if image is None:
            rospy.logerr("DemoMovo/ground: image is none, skip")
            return None

        # preprocess data
        # convert image to ros image
        self._img_msg = CvBridge().cv2_to_imgmsg(image, "rgb8")
        # convert 2d bbox list to 1d
        bboxes_1d = [i for bbox in bboxes for i in bbox]
        print(bboxes_1d)

        # load image, extract and store feature vectors for each bounding box
        goal = ingress_msgs.msg.DenseRefexpLoadBBoxesGoal()
        goal.input = self._img_msg
        goal.boxes = bboxes_1d
        if true_names is not None:
            # replace object names
            rospy.loginfo("Ingress: replacing object names!")
            goal.bbox_obj_names = true_names

        self._load_bbox_client.send_goal(goal)
        self._load_bbox_client.wait_for_result()
        load_result = self._load_bbox_client.get_result()

        rospy.loginfo("ground_img_with_bbox, result received")
        rospy.loginfo("bbox captions: {}".format(load_result.captions))
        rospy.loginfo("bbox caption scores: {}".format(load_result.scores))
        self._ungrounded_captions = np.array(load_result.captions)
        self._ungrounded_caption_scores = np.array(load_result.scores)

        # if user exprssion is empty, just generate captions for image
        if expr == '':
            rospy.loginfo("Ingress: empty query string received, returning ungrounded result")
            top_idx = 0
            context_idxs = [i for i in range(0, len(bboxes))]  # bbox index
            captions = ([self._ungrounded_captions[i]
                         for i in context_idxs], [self._ungrounded_caption_scores[i] for i in context_idxs], [], None)
            self._publish_grounding_result(bboxes, context_idxs, captions)  # visualization of RViz
            return bboxes, top_idx, context_idxs, captions

        # else if user expression is not empty, ground user query
        else:
            top_idx, context_idxs, captions = self._ground_query(expr, bboxes)

        self._publish_grounding_result(bboxes, context_idxs)  # visualization of RViz
        return bboxes, top_idx, context_idxs, captions


if __name__ == '__main__':
    rospy.init_node('ingress_ros')

    try:
        ingress = Ingress()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ingress exit!!")