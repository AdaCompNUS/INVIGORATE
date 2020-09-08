import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp

REAL_ROBOT = True

if REAL_ROBOT:
    import fetch_api

YCROP = (250, 650)
XCROP = (650, 1050)

class FetchRobot():
    def __init__(self):
        self._br = CvBridge()

    def read_imgs(self):
        if REAL_ROBOT:
            img_msg = rospy.wait_for_message('/head_camera/rgb/image_rect_color', Image)
            depth_img_msg = rospy.wait_for_message('/head_camera/depth/image_rect', Image)
            img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            print(img.shape)
            depth = self._br.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        else:
            cur_dir = osp.dirname(osp.abspath(__file__))
            img = cv2.imread(osp.join(cur_dir, '../images/' + '60.jpg'))
            depth = None
        return img, depth

    def grasp(self, grasp):
        print('Dummy execution of grasp {}'.format(grasp))
    
    def say(self, text):
        print('Dummy execution of say: {}'.format(text))

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input("Enter: ")
        return text
