import torch

def inner_loop_planning(belief, planning_depth=3):
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