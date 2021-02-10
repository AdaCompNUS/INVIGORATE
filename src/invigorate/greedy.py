
import numpy as np
import logging

from config.config import *
from libraries.utils.log import LOGGER_NAME

from .invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code --------

class Greedy(Invigorate):
    '''
    Singelstep, greedy
    '''
    def estimate_state_with_img(self, img, expr):
        logger.info("Greedy: estimate_state_with_observation")

        # multistep object detection
        self.singlestep_object_detection(img)

        # multistep grounding
        self.singlestep_grounding(img, expr)

        # multistep obr detection
        self.singlestep_obr_detection(img)

        # grasp detection
        # Note, this is not multistep
        self.grasp_detection(img)

    def singlestep_object_detection(self, img):
        bboxes, classes, scores = self._object_detection(img)

        self.belief["num_obj"] = len(bboxes)
        self.belief["bboxes"] = bboxes
        self.belief["classes"] = classes
        self.belief["cls_scores"] = scores
        logger.info("multistep_object_detection finished:")
        logger.info("bboxes: {}".format(self.belief["bboxes"]))
        logger.info("classes: {}".format(self.belief["classes"]))

    def singlestep_grounding(self, img, expr):
        num_obj = self.belief["num_obj"]
        classes = self.belief["classes"]
        bboxes = self.belief["bboxes"]
        cls_scores = self.belief["cls_scores"]

        # visual grounding
        grounding_scores = self._vis_ground_client.ground(img, bboxes, expr, classes)
        logger.info('Step 1: raw grounding completed')
        logger.info("grounding_scores: {}".format(grounding_scores))

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        target_prob = np.array(grounding_scores)
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        for i in range(num_obj):
            box_score = 0
            for class_str in cls_filter:
                box_score += cls_scores[i][CLASSES_TO_IND[class_str]]
            if box_score < 0.02:
                target_prob[i] = float('-inf')
        logger.info('Step 2: class name filter completed')
        logger.info('target_prob: {}'.format(target_prob))

        # Estimate grounding score greedily
        max_ind = np.argmax(target_prob)
        target_prob = np.zeros(len(target_prob) + 1)
        target_prob[max_ind] = 1
        logger.info('After greedy: target_prob: {}'.format(target_prob))

        self.belief['target_prob'] = target_prob

    def singlestep_obr_detection(self, img):
        bboxes = self.belief['bboxes']

        # detect obr
        rel_mat, rel_score_mat = self._rel_det_client.detect_obr(img, bboxes)

        # Estimate leaf_and_desc_prob greedily
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        rel_prob_mat = np.zeros(rel_score_mat.shape)
        rel_prob_mat[rel_score_mat - rel_score_mat.max(axis=0) == 0] = 1
        logger.debug("after greedy: rel_score_mat: {}".format(rel_prob_mat))
        # assert (rel_prob_mat.sum(axis=0) == 1).sum() == rel_prob_mat[0].size

        self.belief['rel_prob'] = rel_prob_mat

    def grasp_detection(self, img):
        # get current belief
        bboxes = self.belief["bboxes"]
        num_obj = self.belief["num_obj"]

        # grasp
        grasps = self._grasp_det_client.detect_grasps(img, bboxes)
        grasps = self._grasp_filter(bboxes, grasps)
        logger.info('Perceive_img: grasp detection finished')

        self.belief["grasps"] = grasps

    def plan_action(self):
        return self.decision_making_greedy()

    def decision_making_greedy(self):
        logger.info("Greedy: decision_making_greedy")

        target_prob = self.belief['target_prob']
        rel_prob = self.belief['rel_prob']
        num_box = len(self.step_infos["bboxes"])
        leaf_desc_prob,_, _, _, _ = self._get_leaf_desc_prob_from_rel_mat(rel_prob)

        # choose grasp action greedily, ignoring background
        action = "G_{:d}".format(np.argmax(target_prob[:-1].reshape(-1)))

        # post process action (grasp only)
        selected_obj = int(action.split("_")[1])
        l_d_probs = leaf_desc_prob[:, selected_obj]
        current_tgt = np.argmax(l_d_probs)
        if current_tgt == selected_obj:
            # grasp and end program
            action = current_tgt
            logger.info("Greedy, grasp {} and end".format(current_tgt))
        else:
            # grasp and continue
            action = current_tgt + num_box
            logger.info("Greedy, grasp {} and continue".format(current_tgt))

        return action

    def transit_state(self, action):
        # clear history
        self.object_pool = []
