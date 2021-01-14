
import numpy as np
import logging

from config.config import *
from libraries.utils.log import LOGGER_NAME

from invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

# -------- Code --------

class TPN(Invigorate):

    def estimate_state_with_observation(self, observations):
        logger.info("InvigorateHeuristic: estimate_state_with_observation")
        img = observations['img']
        bboxes = observations['bboxes']
        det_scores = observations['det_scores']
        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        expr = observations['expr']
        ind_match_dict = observations['ind_match_dict']
        num_box = observations['num_box']

        # Estimate leaf_and_desc_prob and target_prob greedily
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        rel_prob_mat = np.zeros(rel_score_mat.shape)
        rel_prob_mat[rel_score_mat - rel_score_mat.max(axis=0) == 0] = 1
        # assert (rel_prob_mat.sum(axis=0) == 1).sum() == rel_prob_mat[0].size
        leaf_desc_prob, _, _, _, _ = self._get_leaf_desc_prob_from_rel_mat(rel_prob_mat, 1)
        target_prob = np.array(grounding_scores)

        # grounding result postprocess.
        # 1. filter scores belonging to unrelated objects
        cls_filter = [cls for cls in CLASSES if cls in expr or expr in cls]
        for i in range(bboxes.shape[0]):
            box_score = 0
            for class_str in cls_filter:
                box_score += det_scores[i][CLASSES_TO_IND[class_str]]
            if box_score < 0.02:
                target_prob[i] = -10.
        logger.info('Step 2: class name filter completed')
        logger.info('target_prob : {}'.format(target_prob))

        # estimate grounding score greedily
        max_ind = np.argmax(target_prob)
        target_prob = np.zeros(len(target_prob) + 1)
        target_prob[max_ind] = 1
        logger.info('target_prob : {}'.format(target_prob))
        logger.info('leaf_desc_prob: \n{}'.format(leaf_desc_prob))
    
    def plan_action(self):
        return self.decision_making_heuristic()

    def decision_making_heuristic(self):
        # def choose_action(target_prob):
        #     if len(target_prob.shape) == 1:
        #         target_prob = target_prob.reshape(-1, 1)
        #     cluster_res = KMeans(n_clusters=2).fit_predict(target_prob)
        #     mean0 = target_prob[cluster_res==0].mean()
        #     mean1 = target_prob[cluster_res==1].mean()
        #     pos_label = 0 if mean0 > mean1 else 1
        #     pos_num = (cluster_res==pos_label).sum()
        #     if pos_num > 1:
        #         return ("Q1_{:d}".format(np.argmax(target_prob[:-1].reshape(-1))))
        #     else:
        #         if cluster_res[-1] == pos_label:
        #             return "Q2"
        #         else:
        #             return "G_{:d}".format(np.argmax(target_prob[:-1].reshape(-1)))
        #     # return "G_{:d}".format(np.argmax(target_prob[:-1].reshape(-1)))

        num_box = self.observations['num_box']
        target_prob = self.belief['target_prob']
        rel_prob = self.belief['rel_prob']
        clue_prob = self.belief['clue_prob']
        leaf_desc_prob,_, _, _, _ = self._get_leaf_desc_prob_from_rel_mat(rel_prob)
        logger.info("decision_making_heuristic: ")
        if clue_prob is not None:
            clue_leaf_desc_prob = self._get_clue_leaf_desc_prob(leaf_desc_prob, clue_prob)
            logger.debug('clue_leaf_desc_prob: {}'.format(clue_leaf_desc_prob))

        # choose grasp action greedily, ignoring background
        action = "G_{:d}".format(np.argmax(target_prob[:-1].reshape(-1))) 

        # post process action
        if action.startswith("Q"):
            if action == "Q2":
                if self.clue is None:
                    if self.q2_num_asked < MAX_Q2_NUM:
                        action = 3 * num_box
                        self.q2_num_asked += 1
                    else:
                        # here we:
                        # 1. cannot ground the clue object successfully.
                        # 2. cannot ground the target successfully.
                        # 3. have asked too many Q2s.
                        # therefore, we should grasp a random leaf object and continue.
                        l_probs = np.diagonal(leaf_desc_prob)
                        current_tgt = np.argmax(l_probs)
                        action = current_tgt + num_box
                else:
                    # clue is not None, the robot will grasp the clue object first
                    l_d_probs = clue_leaf_desc_prob
                    current_tgt = np.argmax(l_d_probs)
                    # grasp and continue
                    action = current_tgt + num_box
            else:
                action = int(action.split("_")[1]) + 2 * num_box
        else:
            selected_obj = int(action.split("_")[1])
            l_d_probs = leaf_desc_prob[:, selected_obj]
            current_tgt = np.argmax(l_d_probs)
            if current_tgt == selected_obj:
                # grasp and end program
                action = current_tgt
            else:
                # grasp and continue
                action = current_tgt + num_box

            # if the action is asking Q2, it is necessary to check whether the previous answer is useful.
            # if useful, the robot will use the previous answer instead of requiring a new answer from the user.
            # if a != 3 * num_box or self.clue is None:
            #     self.result_container.append(
            #         save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, self.data_viewer))

        return action