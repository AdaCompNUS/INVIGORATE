class InvigorateMultiSingleStepComparison(Invigorate):
    def _cal_target_prob_from_ground_score(self, ground_scores):
        bg_score = 0.25
        ground_scores = np.append(ground_scores, bg_score)
        return f.softmax(torch.FloatTensor(ground_scores), dim=0).numpy()

    def estimate_state_with_observation_singlestep(self, observations):
        logger.info("Singlestep: estimate_state_with_observation")

        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']

        # Estimate leaf_and_desc_prob and target_prob according to multi-step observations
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        # NOTE: here no multi-step for both rel_prob_mat and target_prob
        rel_prob_mat = rel_score_mat
        leaf_desc_prob = self._get_leaf_desc_prob_from_rel_mat(rel_prob_mat)
        target_prob = self._cal_target_prob_from_ground_score(np.array(grounding_scores))
        logger.info('Step 1: raw grounding completed')
        logger.info('raw target_prob: {}'.format(target_prob))
        logger.info('raw leaf_desc_prob: \n{}'.format(leaf_desc_prob))

        return target_prob, rel_prob_mat

    def estimate_state_with_observation_multistep(self, observations):
        logger.info("Multistep: estimate_state_with_observation")

        grounding_scores = observations['grounding_scores']
        rel_score_mat = observations['rel_score_mat']
        ind_match_dict = observations['ind_match_dict']

        # Estimate leaf_and_desc_prob and target_prob according to multi-step observations
        logger.debug("grounding_scores: {}".format(grounding_scores))
        logger.debug("rel_score_mat: {}".format(rel_score_mat))
        rel_prob_mat = self._multi_step_mrt_estimation(rel_score_mat, ind_match_dict)
        leaf_desc_prob = self._get_leaf_desc_prob_from_rel_mat(rel_prob_mat)
        target_prob = self._cal_target_prob_from_grounding_score(grounding_scores, ind_match_dict)
        target_prob /= target_prob.sum()
        logger.info('Step 1: raw grounding completed')
        logger.debug('raw target_prob: {}'.format(target_prob))
        logger.debug('raw leaf_desc_prob: \n{}'.format(leaf_desc_prob))

        self.belief['leaf_desc_prob'] = leaf_desc_prob
        self.belief['target_prob'] = target_prob
        self.belief['clue_leaf_desc_prob'] = None

        return target_prob, rel_prob_mat