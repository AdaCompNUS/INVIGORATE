
import numpy as np
import logging
from sklearn.cluster import KMeans

from config.config import *
from libraries.utils.log import LOGGER_NAME

from .invigorate_rss import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code --------

class Heuristic(Invigorate):
    '''
    Heuristic, No POMDP
    '''

    def plan_action(self):
        '''
        @return action, int, if 0 < action < num_obj, grasp obj with index action and end
                             if num_obj < action < 2*num_obj, grasp obj with index action and continue
                             if 2*num_obj < action < 3*num_obj, ask questions about obj with index action
        '''
        return self._decision_making_heuristic()

    def _decision_making_heuristic(self):
        pool_to_det, det_to_pool, num_obj = self._get_valid_obj_candidates()
        target_prob = self.belief['target_prob']
        rel_prob = self.belief['rel_prob']
        leaf_desc_prob,_, _, _, _ = self._get_leaf_desc_prob_from_rel_mat(rel_prob)
        logger.info("decision_making_heuristic: ")

        # select action using K-means
        if len(target_prob.shape) == 1:
            target_prob = target_prob.reshape(-1, 1)
        cluster_res = KMeans(n_clusters=2).fit_predict(target_prob)
        mean0 = target_prob[cluster_res==0].mean()
        mean1 = target_prob[cluster_res==1].mean()
        pos_label = 0 if mean0 > mean1 else 1
        pos_num = (cluster_res==pos_label).sum()
        if pos_num > 1:
            # multiple candidates, ask question
            action = np.argmax(target_prob[:-1].reshape(-1)) + 2 * num_obj
        else:
            # only one target, grasp its most probable leaf and descendant
            target = np.argmax(target_prob.reshape(-1))
            l_d_probs = leaf_desc_prob[:, target]
            current_tgt = np.argmax(l_d_probs)
            if current_tgt == target:
                # grasp and end program
                action = current_tgt
            else:
                # grasp and continue
                action = current_tgt + num_obj
            logger.info("target = {}, its leaf_and_desc = {}".format(target, current_tgt))

        return action