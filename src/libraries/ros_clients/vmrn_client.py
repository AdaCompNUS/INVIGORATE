import rospy
from cv_bridge import CvBridge
import numpy as np

from invigorate_msgs.srv import *

class VMRNClient():
    def __init__(self):
        self._br = CvBridge()
        self._vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)

    def detect_obr(self, img, bboxes):
        img_msg = self._br.cv2_to_imgmsg(img)
        bboxes = bboxes.flatten().tolist()
        res = self._vmr_det(img_msg, bboxes)
        self._res_cache = res
        return res.rel_mat, res.rel_score_mat

    def detect_grasps(self, img, bboxes, get_cache=True):
        if get_cache:
            res = self._res_cache
        else:
            img_msg = self._br.cv2_to_imgmsg(img)
            bboxes = bboxes.flatten().tolist()
            res = self._vmr_det(img_msg, bboxes)

        return res.grasps
