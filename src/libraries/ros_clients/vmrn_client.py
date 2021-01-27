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
        num_box = bboxes.shape[0]
        bboxes = bboxes.flatten().tolist()
        res = self._vmr_det(img_msg, bboxes)
        self._res_cache = res
        rel_mat = np.array(res.rel_mat).reshape((num_box, num_box))
        if num_box == 1: # TODO hack!!
            rel_score_mat = (0.0, 0.0, 0.0)
        else:
            rel_score_mat = res.rel_score_mat
        rel_score_mat = np.array(rel_score_mat).reshape((3, num_box, num_box))

        return rel_mat, rel_score_mat

    def detect_grasps(self, img, bboxes, get_cache=True):
        num_box = bboxes.shape[0]
        if get_cache:
            res = self._res_cache
        else:
            img_msg = self._br.cv2_to_imgmsg(img)
            bboxes = bboxes.flatten().tolist()
            res = self._vmr_det(img_msg, bboxes)

        grasps = np.array(res.grasps).reshape((num_box, 5, -1))
        return grasps
