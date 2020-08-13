#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys

import ingress_srv.ingress_srv as ingress_srv

if __name__ == '__main__':
    rospy.init_node("ingress_srv_test")
    _cv_bridge = CvBridge()
    ingress_service = ingress_srv.Ingress()
    img = cv2.imread(sys.argv[1])
    img_msg = _cv_bridge.cv2_to_imgmsg(img)
    boxes, top_idx, context_idxs, captions = ingress_service.ground(img_msg, '')
    sem_captions, self_probs, rel_captions, rel_probs = captions
    rospy.loginfo("Self-Referrential Captions: " + str(sem_captions))
    rospy.loginfo("Self-Referrential Probabilities: " + str(self_probs))

    raw_input("press anything to exit")
