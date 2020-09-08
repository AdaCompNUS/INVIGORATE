import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp

# import fetch_api

class FetchRobot():
    def __init__(self):
        self._br = CvBridge()

    def read_imgs(self):
        # img_msg = rospy.wait_for_message('', Image)
        # depth_img_msg = rospy.wait_for_message('', Image)
        # img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        # depth = self._br.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')

        cur_dir = osp.dirname(osp.abspath(__file__))
        img = cv2.imread(osp.join(cur_dir, '../images/' + '1.png'))
        depth = None
        return img, depth

    def grasp(self, grasp):
        print('dummy execution of grasp {}'.format(grasp))