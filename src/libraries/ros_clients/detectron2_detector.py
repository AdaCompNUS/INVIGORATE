import rospy
from cv_bridge import CvBridge

from invigorate_msgs.srv import *

class Detectron2Client():
    def __init__(self):
        self._br = CvBridge()
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)

    def detect_objects(self, img):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.cls_scores