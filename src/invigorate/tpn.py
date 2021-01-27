
import numpy as np
import logging

from config.config import *
from libraries.utils.log import LOGGER_NAME

from invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code --------

class TPN(Invigorate):
    '''
    Heuristic, No POMDP
    '''

    def estimate_state_with_observation(self, observations):
        logger.info("TPN: estimate_state_with_observation")

        img = observations['img']
        bboxes = observations['bboxes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        ind_match_dict = observations['ind_match_dict']
        num_box = observations['num_box']

        # Estimate leaf_and_desc_prob greedily
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        rel_prob_mat = np.zeros(rel_score_mat.shape)
        rel_prob_mat[rel_score_mat - rel_score_mat.max(axis=0) == 0] = 1
        # assert (rel_prob_mat.sum(axis=0) == 1).sum() == rel_prob_mat[0].size

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        target_prob = np.array(grounding_scores)
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += det_scores[i][CLASSES_TO_IND[class_str]]
            if box_score < 0.02:
                target_prob[i] = float('-inf')
        logger.info('Step 2: class name filter completed')
        logger.info('target_prob: {}'.format(target_prob))

        # Estimate grounding score greedily
        max_ind = np.argmax(target_prob)
        target_prob = np.zeros(len(target_prob) + 1)
        target_prob[max_ind] = 1
        logger.info('target_prob: {}'.format(target_prob))

        self.belief['target_prob'] = target_prob
        self.belief['rel_prob'] = rel_prob_mat

    def plan_action(self):
        return self.decision_making_greedy()

    def decision_making_greedy(self):
        logger.info("TPN: decision_making_greedy")

        num_box = self.observations['num_box']
        target_prob = self.belief['target_prob']
        rel_prob = self.belief['rel_prob']
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
        else:
            # grasp and continue
            action = current_tgt + num_box

        return action