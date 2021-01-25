import rospy
from cv_bridge import CvBridge
import numpy as np

from invigorate_msgs.srv import *

class VMRNClient():
    def __init__(self):
        self._br = CvBridge()
        self._vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)

    def detect_obr(self, img, bbox):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat