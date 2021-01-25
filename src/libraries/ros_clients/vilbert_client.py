import rospy
from cv_bridge import CvBridge
import numpy as np

from invigorate_msgs.srv import *

class VilbertClient():
    def __init__(self):
        self._br = CvBridge()
        self._vilbert_client = rospy.ServiceProxy('vilbert_grounding_service', Grounding)

    def ground(self, img, bboxes, expr):
        req = GroundingRequest()
        img_msg = self._br.cv2_to_imgmsg(img)
        req.img = img_msg
        req.bboxes = bboxes.flatten().tolist()
        req.expr = expr

        resp = self._vilbert_client(req)
        grounding_scores = np.array(list(resp.grounding_scores))

        return grounding_scores