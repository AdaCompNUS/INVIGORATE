import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp

import fetch_api

XCROP = (100, 430)
YCROP = (100, 540)

class FetchRobot():
    def __init__(self):
        self._br = CvBridge()

    def read_imgs(self):
        img_msg = rospy.wait_for_message('/head_camera/rgb/image_rect_color', Image)
        depth_img_msg = rospy.wait_for_message('/head_camera/depth/image_rect', Image)
        img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        print('img_size : {}'.format(img.shape)) # 480x640
        img = img[XCROP[0]:XCROP[1], YCROP[0]:YCROP[1]]
        print('img_size : {}'.format(img.shape))
        depth = self._br.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        return img, depth

    def grasp(self, grasp):
        print('Dummy execution of grasp {}'.format(grasp))
    
    def say(self, text):
        print('Dummy execution of say: {}'.format(text))

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input('Enter: ')
        return text
