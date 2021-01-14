import rospy
import numpy as np
import logging

from config.config import *
from libraries.utils.log import LOGGER_NAME
from vmrn_msgs.srv import *

from tpn import TPN

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code --------

class Baseline(TPN):
    def __init__(self):
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        rospy.wait_for_service('vmrn_server')
        rospy.wait_for_service('vlbert_server')
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        self._vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        self._grounding = rospy.ServiceProxy('vlbert_server', VLBert)

    def perceive_img(self, img, expr):
        '''
        @return bboxes,         [Nx5], xyxy
                scores,         [Nx1]
                rel_mat,        [NxN]
                rel_score_mat,  [3xNxN]
                leaf_desc_prob, [N+1xN+1]
                ground_score,   [N+1]
                ground_result,  [N+1]
                ind_match_dict,
                grasps,         [Nx5x8], 5 grasps for every object, every grasp is x1y1, x2y2, x3y3, x4y4
        '''

        # object detection
        bboxes, classes, scores = self._object_detection(img)
        if bboxes is None:
            logger.warning("WARNING: nothing is detected")
            return None
        logger.info('Perceive_img: _object_detection finished')

        # relationship and grasp detection
        rel_mat, rel_score_mat, grasps = self._mrt_detection(img, bboxes)
        logger.info('Perceive_img: mrt and grasp detection finished')

        # object and relationship detection post process
        ind_match_dict, not_matched = self._bbox_post_process(bboxes, scores, rel_score_mat)
        num_box = bboxes.shape[0]
        logger.info('Perceive_img: post process of object and mrt detection finished')

        # grounding
        grounding_scores = self._visual_grounding(img, bboxes, expr)
        logger.info('Perceive_img: vlbert grounding finished')

        observations = {}
        observations['img'] = img
        observations['expr'] = expr
        observations['num_box'] = num_box
        observations['bboxes'] = bboxes
        observations['classes'] = classes
        observations['ind_match_dict'] = ind_match_dict
        observations['not_matched'] = not_matched
        observations['det_scores'] = scores
        observations['rel_mat'] = rel_mat
        observations['rel_score_mat'] = rel_score_mat
        observations['grounding_scores'] = grounding_scores
        observations['grasps'] = grasps

        self.observations = observations
        return observations

    def _mrt_detection(self, img, bboxes):
        num_box = bboxes.shape[0]
        # logger.info(num_box)
        rel_result = self._vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
        rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
        # logger.info(rel_result[1])
        if num_box == 1: # TODO hack!!
            rel_score_mat = (0.0, 0.0, 0.0)
        else:
            rel_score_mat = rel_result[1]
        rel_score_mat = np.array(rel_score_mat).reshape((3, num_box, num_box))
        grasps = np.array(rel_result[2]).reshape((num_box, 5, -1))
        grasps = self._grasp_filter(bboxes, grasps)
        return rel_mat, rel_score_mat, grasps

    def _visual_grounding(self, img, bboxes, expr):
        return self._vlbert_client(img, bboxes[:, :4].reshape(-1).tolist(), expr)

    def _faster_rcnn_client(self, img):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.cls_scores

    def _vmrn_client(self, img, bbox):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat, res.grasps

    def _vlbert_client(self, img, bbox, expr):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._grounding(img_msg, bbox, expr)
        return res.grounding_scores

