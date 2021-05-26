import rospy
import cv2
import os.path as osp
import logging
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from config.config import *
from libraries.utils.log import LOGGER_NAME

USE_REALSENSE = False

YCROP = (470, 1000) # 1080
XCROP = (700, 1460) # 1920

logger = logging.getLogger(LOGGER_NAME)

class DummyRobot():
    def __init__(self):
        self._br = CvBridge()

    def read_imgs(self):
        if USE_REALSENSE:
            img_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10)
            img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            logger.info('img_size : {}'.format(img.shape))
            img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
            logger.info('img_size : {}'.format(img.shape))
            depth = None

        else:
            img_name = raw_input('Enter img name: ')
            img = cv2.imread(osp.join(ROOT_DIR, 'images/' + img_name))
            depth = None

        return img, depth

    def grasp(self, grasp, is_target = False):
        print('Dummy execution of grasp {}'.format(grasp))
        return True

    def say(self, text):
        print('Dummy execution of say: {}'.format(text))
        return True

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input("Enter: ")
        return text

    def move_arm_to_home(self):
        print('Dummy execution of move_arm_to_home')
        return