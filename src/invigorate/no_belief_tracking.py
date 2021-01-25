import numpy as np
import logging
import torch
import torch.nn.functional as f

from config.config import *
from libraries.utils.log import LOGGER_NAME

from invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

class NoBeliefTracking(Invigorate):

    def _cal_target_prob_from_ground_score(self, ground_scores):
        bg_score = 0.25
        ground_scores = np.append(ground_scores, bg_score)
        return f.softmax(torch.FloatTensor(ground_scores), dim=0).numpy()

    def estimate_state_with_observation(self, observations):
        logger.info("NoBeliefTracking: estimate_state_with_observation")

        img = observations['img']
        bboxes = observations['bboxes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        ind_match_dict = observations['ind_match_dict']
        num_box = observations['num_box']

        # Estimate leaf_and_desc_prob and target_prob according to multi-step observations
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        # NOTE: here no multi-step for both rel_prob_mat and target_prob
        rel_prob_mat = rel_score_mat
        target_prob = self._cal_target_prob_from_ground_score(np.array(grounding_scores))
        logger.info('Step 1: raw grounding completed')
        logger.info('raw target_prob: {}'.format(target_prob))
        logger.info('raw rel_prob_mat: {}'.format(rel_prob_mat))

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += det_scores[i][CLASSES_TO_IND[class_str]]
            if box_score < 0.02:
                target_prob[i] = 0.
        # TODO. This means candidate belief estimation has problem
        # if after name filter, all prob sum to zero, the object is in background
        if target_prob.sum() == 0.0:
            target_prob[-1] = 1.0
        target_prob /= target_prob.sum()
        logger.info('Step 2: class name filter completed')
        logger.info('target_prob : {}'.format(target_prob))

        # 2. incorporate QA history
        target_prob_backup = target_prob.copy()
        for k, v in ind_match_dict.items():
            # if self.object_pool[v]["confirmed"] == True:
            #     if self.object_pool[v]["ground_belief"] == 1.:
            #         target_prob[:] = 0.
            #         target_prob[k] = 1.
            #     elif self.object_pool[v]["ground_belief"] == 0.:
            #         target_prob[k] = 0.
            if self.object_pool[v]["is_target"] == 1:
                target_prob[:] = 0.
                target_prob[k] = 1.
            elif self.object_pool[v]["is_target"] == 0:
                target_prob[k] = 0.
        # sanity check
        if target_prob.sum() > 0:
            target_prob /= target_prob.sum()
        else:
            # something wrong with the matching process. roll back
            for i in observations['not_matched']:
                target_prob[i] = target_prob_backup[i]
            target_prob[-1] = target_prob_backup[-1]
            target_prob /= target_prob.sum()

        # update target_prob
        for k, v in ind_match_dict.items():
            self.object_pool[v]["target_prob"] = target_prob[k]

        logger.info('Step 3: incorporate QA history completed')
        logger.info('target_prob: {}'.format(target_prob))

        self.belief['target_prob'] = target_prob
        self.belief['rel_prob'] = rel_prob_mat

class NoBeliefTracking2(NoBeliefTracking):
    def _cal_target_prob_from_ground_score(self, ground_scores):
        bg_score = 0.25
        ground_scores = np.append(ground_scores, bg_score)
        ground_scores *= 10 # NOTE!!! the difference is here
        return f.softmax(torch.FloatTensor(ground_scores), dim=0).numpy()
