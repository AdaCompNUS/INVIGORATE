#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import xml.etree.ElementTree as ET
import actionlib
import ingress_msgs.msg
import copy
import numpy as np
import os

import ingress_srv.ingress_srv as ingress_srv


def get_bboxes_from_xml(xml_path):
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()

    bboxes = []
    bbox_obj_names = []
    for obj in root.findall('object'):
        bbox_tmp = obj.find('bndbox')
        bbox = []
        bbox.append(float(bbox_tmp.find('xmin').text))
        bbox.append(float(bbox_tmp.find('ymin').text))
        bbox.append(float(bbox_tmp.find('xmax').text) - float(bbox_tmp.find('xmin').text))
        bbox.append(float(bbox_tmp.find('ymax').text) - float(bbox_tmp.find('ymin').text))
        bboxes.append(bbox)

        obj_name = obj.find('name').text
        bbox_obj_names.append(obj_name)

    rospy.loginfo("bboxes found: {}".format(bboxes))
    rospy.loginfo("bbox_obj_names : {}".format(bbox_obj_names))
    return bboxes, bbox_obj_names


if __name__ == '__main__':
    try:
        rospy.init_node('Grounding', anonymous=True)

        ingress_service = ingress_srv.Ingress()

        while not rospy.is_shutdown():
            try:
                # get image
                img_path = raw_input("Enter path to image (Enter q to exit): ")
                if img_path == 'q':
                    break
                if img_path == '':
                    img_path = '/home/peacock-rls/work/rls_robot_ws/src/services/planning_services/ingress/examples/images/01076.jpg'
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                xml_path = os.path.splitext(img_path)[0] + '.xml'
                bboxes, bbox_obj_names = get_bboxes_from_xml(xml_path)
                # bboxes = [[282, 330, 98, 102], [238, 328, 114, 53],
                #           [98, 260, 96, 99],   [193, 210, 135, 135],
                #           [417, 244, 75, 196], [200, 238, 231, 36]]
                # bbox_obj_names = []
                # bbox_obj_names = ['box', 'box', 'remote controller', 'cup']

            except Exception as e:
                rospy.logerr(e)

            expr = raw_input("Enter user expression: ")

            # with name replacement
            if len(bbox_obj_names) > 0:
                bboxes, top_idx, context_idxs, captions = ingress_service.ground_img_with_bbox(
                    img, bboxes, expr, true_names=bbox_obj_names)
            # without name replacement
            else:
                bboxes, top_idx, context_idxs, captions = ingress_service.ground_img_with_bbox(
                    img, bboxes, expr)

            if len(context_idxs) == 0 or captions is None:
                rospy.loginfo("no object found")
                continue

            sem_captions, sem_probs, rel_captions, rel_probs = captions
            # rospy.loginfo("top index = {}, context_idx = {}".format(top_idx, context_idxs))
            rospy.loginfo("sem_captions: {}".format(sem_captions))
            rospy.loginfo("sem_probs: {}".format(sem_probs))
            rospy.loginfo("rel_captions: {}".format(rel_captions))
            rospy.loginfo("rel_probs: {}".format(rel_probs))

            rospy.loginfo("Top bbox is {}".format(bboxes[top_idx]))
            rospy.loginfo(
                "Top bbox self-referential caption: {}".format(sem_captions[context_idxs.index(top_idx)]))
            rospy.loginfo("Top bbox relational caption: {}".format(
                sem_captions[context_idxs.index(top_idx)]))

    except rospy.ROSInterruptException:
        pass
