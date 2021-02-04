import rospy
from cv_bridge import CvBridge

from invigorate_msgs.srv import *

class FasterRCNNClient():
    def __init__(self):
        self._br = CvBridge()
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)

    def detect_objects(self, img, rois=None):
        if rois is not None:
            rois = rois.flatten().tolist()
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._obj_det(img_msg, False, rois)
        return res.num_box, res.bbox, res.cls, res.cls_scores