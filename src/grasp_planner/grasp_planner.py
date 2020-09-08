import torch

from density_estimator import object_belief, gaussian_kde

class GraspPlanner():
    def __init__(self):
        self.belief = {}
        self.actions = []

    def init_belief(self, observation):
        '''
        initialize belief according to observation
        '''
        rel_prob_mat = observation['rel_prob']
        self.belief['leaf_desc_prob'] = self._cal_leaf_desc_prob(rel_prob_mat)

    def update_belief(self, observation):
        '''
        Update current belief according to new observation
        '''
        pass

    def _cand_prob_to_belief_mc(self, pcand, sample_num=100000):
        pcand = torch.Tensor(pcand).reshape(1, -1)
        pcand = pcand.repeat(sample_num, 1)
        sampled = torch.bernoulli(pcand)
        sampled_sum = sampled.sum(-1)
        sampled[sampled_sum > 0] /= sampled_sum[sampled_sum > 0].unsqueeze(-1)
        sampled = np.clip(sampled.mean(0).cpu().numpy(), 0.01, 0.99)
        if sampled.sum() > 1:
            sampled /= sampled.sum()
        return sampled

    def plan_action(self):
        action = self._inner_loop_planning(self.belief)
        return action

    def _inner_loop_planning(belief, planning_depth=3):
        num_obj = belief["ground_prob"].shape[0] - 1 # exclude the virtual node
        penalty_for_asking = -2
        # ACTIONS: Do you mean ... ? (num_obj) + Where is the target ? (1) + grasp object (num_obj)
        def grasp_reward_estimate(belief):
            # reward of grasping the corresponding object
            # return is a 1-D tensor including num_obj elements, indicating the reward of grasping the corresponding object.
            ground_prob = belief["ground_prob"]
            leaf_desc_tgt_prob = (belief["leaf_desc_prob"] * ground_prob.unsqueeze(0)).sum(-1)
            leaf_prob = torch.diag(belief["leaf_desc_prob"])
            not_leaf_prob = 1. - leaf_prob
            target_prob = ground_prob
            leaf_tgt_prob = leaf_prob * target_prob
            leaf_desc_prob = leaf_desc_tgt_prob - leaf_tgt_prob
            leaf_but_not_desc_tgt_prob = leaf_prob - leaf_desc_tgt_prob

            # grasp and the end
            r_1 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-10) + leaf_desc_prob * (-10)\
                    + leaf_tgt_prob * (0)
            r_1 = r_1[:-1] # exclude the virtual node

            # grasp and not the end
            r_2 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-6) + leaf_desc_prob * (0)\
                    + leaf_tgt_prob * (-10)
            r_2 = r_2[:-1]  # exclude the virtual node
            return torch.cat([r_1, r_2], dim=0)

        def belief_update(belief):
            I = torch.eye(belief["ground_prob"].shape[0]).type_as(belief["ground_prob"])
            updated_beliefs = []
            # Do you mean ... ?
            # Answer No
            beliefs_no = belief["ground_prob"].unsqueeze(0).repeat(num_obj + 1, 1)
            beliefs_no *= (1. - I)
            beliefs_no /= torch.clamp(torch.sum(beliefs_no, dim = -1, keepdim=True), min=1e-10)
            # Answer Yes
            beliefs_yes = I.clone()
            for i in range(beliefs_no.shape[0] - 1):
                updated_beliefs.append([beliefs_no[i], beliefs_yes[i]])

            # Is the target detected? Where is it?
            updated_beliefs.append([beliefs_no[-1], I[-1],])
            return updated_beliefs

        def is_onehot(vec, epsilon = 1e-2):
            return (torch.abs(vec - 1) < epsilon).sum().item() > 0

        def estimate_q_vec(belief, current_d):
            if current_d == planning_depth - 1:
                q_vec = grasp_reward_estimate(belief)
                return q_vec
            else:
                # branches of grasping
                q_vec = grasp_reward_estimate(belief).tolist()
                ground_prob = belief["ground_prob"]
                new_beliefs = belief_update(belief)
                new_belief_dict = copy.deepcopy(belief)

                # Q1
                for i, new_belief in enumerate(new_beliefs[:-1]):
                    q = 0
                    for j, b in enumerate(new_belief):
                        new_belief_dict["ground_prob"] = b
                        # branches of asking questions
                        if is_onehot(b):
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        else:
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                        if j == 0:
                            # Answer is No
                            q += t_q * (1 - ground_prob[i])
                        else:
                            # Answer is Yes
                            q += t_q * ground_prob[i]
                    q_vec.append(q.item())

                # Q2
                q = 0
                new_belief = new_beliefs[-1]
                for j, b in enumerate(new_belief):
                    new_belief_dict["ground_prob"] = b
                    if j == 0:
                        # target has been detected
                        if is_onehot(b):
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        else:
                            t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                        q += t_q * (1 - ground_prob[-1])
                    else:
                        new_belief_dict["leaf_desc_prob"][:, -1] = new_belief_dict["leaf_desc_prob"][:, :-1].sum(-1) / num_obj
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                        q += t_q * ground_prob[-1]
                q_vec.append(q.item())
                return torch.Tensor(q_vec).type_as(belief["ground_prob"])
        q_vec = estimate_q_vec(belief, 0)
        print("Q Value for Each Action: ")
        print(q_vec.tolist()[:num_obj])
        print(q_vec.tolist()[num_obj:2*num_obj])
        print(q_vec.tolist()[2*num_obj:3*num_obj])
        print(q_vec.tolist()[3*num_obj])
        return torch.argmax(q_vec).item()

    def _cal_leaf_desc_prob(self, rel_prob_mat, sample_num = 1000):
        '''
        Sample MRT to find leaf and descent probability of
        '''
        # TODO: Numpy may support a faster implementation.
        def sample_trees(rel_prob, sample_num=1):
            return torch.multinomial(rel_prob, sample_num, replacement=True)

        cuda_data = False
        if rel_prob_mat.is_cuda:
            # this function runs much faster on CPU.
            cuda_data = True
            rel_prob_mat = rel_prob_mat.cpu()

        rel_prob_mat = rel_prob_mat.permute((1, 2, 0))
        mrt_shape = rel_prob_mat.shape[:2]
        rel_prob = rel_prob_mat.view(-1, 3)
        rel_valid_ind = rel_prob.sum(-1) > 0

        # sample mrts
        samples = sample_trees(rel_prob[rel_valid_ind], sample_num) + 1
        mrts = torch.zeros((sample_num,) + mrt_shape).type_as(samples)
        mrts = mrts.view(sample_num, -1)
        mrts[:, rel_valid_ind] = samples.permute((1,0))
        mrts = mrts.view((sample_num,) + mrt_shape)
        f_mats = (mrts == 1)
        c_mats = (mrts == 2)
        adj_mats = f_mats + c_mats.transpose(1,2)

        def no_cycle(adj_mat):
            keep_ind = (adj_mat.sum(0) > 0)
            if keep_ind.sum() == 0:
                return True
            elif keep_ind.sum() == adj_mat.shape[0]:
                return False
            adj_mat = adj_mat[keep_ind][:, keep_ind]
            return no_cycle(adj_mat)

        def descendants(adj_mat):
            def find_children(node, adj_mat):
                return torch.nonzero(adj_mat[node]).view(-1).tolist()

            def find_descendant(node, adj_mat, visited, desc_mat):
                if node in visited:
                    return visited, desc_mat
                else:
                    desc_mat[node][node] = 1
                    for child in find_children(node, adj_mat):
                        visited, desc_mat = find_descendant(child, adj_mat, visited, desc_mat)
                        desc_mat[node] = (desc_mat[node] | desc_mat[child])
                    visited.append(node)
                return visited, desc_mat

            roots = torch.nonzero(adj_mat.sum(0) == 0).view(-1).tolist()
            visited = []
            desc_mat = torch.zeros(mrt_shape).type_as(adj_mat).long()
            for root in roots:
                visited, desc_list = find_descendant(root, adj_mat, visited, desc_mat)
            return desc_mat.transpose(0,1)

        leaf_desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
        count = 0
        for adj_mat in adj_mats:
            if no_cycle(adj_mat):
                desc_mat = descendants(adj_mat)
                leaf_desc_mat = desc_mat * (adj_mat.sum(1, keepdim=True) == 0)
                leaf_desc_prob += leaf_desc_mat
                count += 1
        leaf_desc_prob = leaf_desc_prob / count
        if cuda_data:
            leaf_desc_prob = leaf_desc_prob.cuda()
        return leaf_desc_prob

    def bbox_match(self, bbox, prev_bbox, scores, prev_scores, mode="hungarian"):
        # TODO: apply Hungarian algorithm to match boxes
        if prev_bbox.size == 0:
            return {}
        ovs = bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
        ind_match_dict = {}
        if mode=="heuristic":
            # match bboxes between two steps.
            cls_mask = np.zeros(ovs.shape, dtype=np.uint8)
            for i, cls in enumerate(bbox[:, -1]):
                cls_mask[i][prev_bbox[:, -1] == cls] = 1
            ovs_mask = (ovs > 0.8)
            ovs *= ((cls_mask + ovs_mask) > 0)
            mapping = np.argsort(ovs, axis=-1)[:, ::-1]
            ovs_sorted = np.sort(ovs, axis=-1)[:, ::-1]
            matched = (np.max(ovs, axis=-1) > 0.5)
            occupied = {i: False for i in range(mapping.shape[-1])}
            for i in range(mapping.shape[0]):
                if matched[i]:
                    for j in range(mapping.shape[-1]):
                        if not occupied[mapping[i][j]] and ovs_sorted[i][j] > 0.5:
                            ind_match_dict[i] = mapping[i][j]
                            occupied[mapping[i][j]] = True
                            break
                        elif ovs_sorted[i][j] <= 0.5:
                            break

        elif mode=="hungarian":
            ov_cost = 1. - ovs
            # normalize scores
            scores /= np.expand_dims(np.linalg.norm(scores, axis=-1), axis=-1)
            prev_scores /= np.expand_dims(np.linalg.norm(prev_scores, axis=-1), axis=-1)
            scores_cost = np.expand_dims(scores, 1) * np.expand_dims(prev_scores, 0)
            scores_cost = 1 - scores_cost.sum(-1)
            cost = 0.6 * ov_cost + 0.4 * scores_cost
            mapping = optimize.linear_sum_assignment(cost)

            thresh = 0.5
            for i in range(mapping[0].size):
                ind1 = mapping[0][i]
                ind2 = mapping[1][i]
                if cost[ind1][ind2] < thresh:
                    ind_match_dict[ind1] = ind2

        return ind_match_dict

