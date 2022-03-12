import collections

from sklearn.neighbors import KernelDensity as kde
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os.path as osp

class gaussian_kde(object):
    def __init__(self, data, bandwidth=0.05):
        self.training_data = data
        self.data_dim = self.training_data.shape[-1]
        self.bandwidth = bandwidth
        self.kde = kde(kernel='gaussian', bandwidth=self.bandwidth).fit(self.training_data)

    def update(self, new_data):
        self.training_data = np.concatenate([self.training_data, new_data], axis = 0)
        self.kde.fit(self.training_data)
        return self

    def comp_prob(self, x):
        if isinstance(x, (float, np.float, np.float32, np.float64)):
            assert self.data_dim == 1
            x = np.array([[x]])
        elif isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            if self.data_dim == 1:
                if x.ndim == 1:
                    x = np.expand_dims(x, axis = -1)
            else:
                if x.ndim == 1:
                    x = np.expand_dims(x, axis = 0)
            assert x.shape[-1] == self.data_dim and x.ndim == 2
        else:
            raise RuntimeError("Unsupported data type. The input should be a float, a list or a numpy array.")
        x = self.kde.score_samples(x)
        return x.squeeze()

class object_belief(object):
    def __init__(self,
                 confirmed=False,
                 independent_neg=False):

        if not confirmed:
            self._belief = np.array([0.5, 0.5])
        else:
            # if an object is initialized with confirmed = True, it means
            # that another object has already been confirmed to be the target.
            # hence, this object will be already completely excluded.
            self._belief = np.array([1., 0.])

        self._independent_neg = independent_neg
        # only when an negative expression has been observed, will the
        # self._belief_neg be enabled
        self._enable_neg = False
        if self._independent_neg:
            # the belief that this object matches the negative expression
            self._belief_neg = np.array([0.5, 0.5])

        # This placeholder 'cls_llh' is prepared for object detector,
        # hence in INVIGORATE, the object detection score can also
        # help to filter out incorrect objects.
        # However, since results of object detection changes
        # at every time step, we update it at every step and use it
        # to update the posterior.
        # In the future, we can consider to introduce object results
        # from multiple detection steps as the object class likelihood.
        self.cls_llh = np.array([0.5, 0.5])

        self.low_thr = 0.01

        # confirmed by an instance-specific question
        self.confirmed = confirmed

    @property
    def belief(self):
        # handle special cases
        # here the cls_llh has a higher priority
        if self.cls_llh[0] == 0 and self.cls_llh[1] > 0:
            self._belief[:] = [0, 1]
            return self._belief
        elif self.cls_llh[0] > 0 and self.cls_llh[1] == 0:
            self._belief[:] = [1, 0]
            return self._belief
        else:
            assert self.cls_llh[0] > 0 and self.cls_llh[1] > 0

        if not self._independent_neg or not self._enable_neg:
            belief = self._belief.copy()
        else:
            pos_belief = self._belief[1] * self._belief_neg[0]
            belief = np.array([1-pos_belief, pos_belief])

        belief *= self.cls_llh
        belief /= belief.sum()
        return belief

    def update(self, score, kde, is_pos=True):

        # update negative expression enabling state
        if self._independent_neg and not is_pos and not self.confirmed:
            self._enable_neg = True
        else:
            self._enable_neg = False

        # update with observation
        if not self. _independent_neg:
            if is_pos:
                neg_prob = np.exp(kde[0].comp_prob(score))
                pos_prob = np.exp(kde[1].comp_prob(score))
            else:
                neg_prob = np.exp(kde[1].comp_prob(score))
                pos_prob = np.exp(kde[0].comp_prob(score))
            self.update_with_likelihood([neg_prob, pos_prob])
        else:
            neg_prob = np.exp(kde[0].comp_prob(score))
            pos_prob = np.exp(kde[1].comp_prob(score))
            if is_pos:
                self.update_with_likelihood(
                    [neg_prob, pos_prob], update_pos=True)
            else:
                self.update_with_likelihood(
                    [neg_prob, pos_prob], update_pos=False)

        return self.belief

    def update_cls_llh(self, llh):
        self.cls_llh[:] = llh

    def update_with_likelihood(self,
                               likelihood,
                               enable_low_thresh=True,
                               update_pos=True):

        if not update_pos:
            assert self._independent_neg, \
                "To update negative belief, you need to initialize " \
                "it first"

        # for the completely confirmed objects, no update is needed
        if self.confirmed:
            belief = self._belief
            assert belief[0] in {0., 1.}
            return belief

        def update_belief(belief, likelihood, confirmed, enable_low_thresh, low_thr):
            if likelihood[0] == 0 and likelihood[1] > 0:
                belief[:] = [0., 1.]
            elif likelihood[0] > 0 and likelihood[1] == 0:
                belief[:] = [1., 0.]
            else:
                assert likelihood[0] > 0 and likelihood[1] > 0
                # self.confirmed is initialized as False. As long as it has been
                # set to True during a task, it as well as the belief will never
                # change any more.
                if not confirmed:
                    belief *= likelihood
                    belief /= belief.sum()
                    if enable_low_thresh:
                        belief = np.clip(belief, low_thr, 1 - low_thr)
                    else:
                        belief = np.clip(belief, 0., 1.)
            return belief

        def check_belief(belief):
            assert not np.isnan(belief).any(), \
                "Please check: \n cls_llh: {} \n belief: {} \n belief_neg: {}".format(
                    self.cls_llh.tolist(), self._belief.tolist(),
                    self._belief_neg.tolist() if self._independent_neg else []
                )

        # likelihood should only contain 2 values
        assert len(likelihood) == 2

        # handle special cases
        if not self._independent_neg or update_pos:
            self._belief = update_belief(
                self._belief, likelihood, self.confirmed,
                enable_low_thresh, self.low_thr)
        else:
            self._belief_neg = update_belief(
                self._belief_neg, likelihood, self.confirmed,
                enable_low_thresh, self.low_thr)

        belief = self.belief
        check_belief(belief)

        return belief

    def reset(self, confirmed=False):
        self.cls_llh = np.array([0.5, 0.5])
        self._belief = np.array([0.5, 0.5])
        if self._independent_neg:
            self._belief_neg = np.array([0.5, 0.5])
        self.confirmed = confirmed
        self._enable_neg = False

    def update_linguistic(self, answer, match_prob, epsilon=0.01, confirmed=False):
        # an extension in invigorate_IJRR_v1, not used for other versions
        # for original INVIGORATE:
        #        match_prob = 1 for the object being asked,
        #        match_prob = 0 else.

        # firstly scale match_prob, which is equivalent to scale the observation model
        match_prob = self._remap_match_prob(match_prob, epsilon)

        if answer:
            # answer is yes
            likelihood = [epsilon * (1 - match_prob), match_prob]
        else:
            # answer is no
            likelihood = [1 - epsilon * (1 - match_prob), 1 - match_prob]

        # if confirmed is enabled, meaning that the object belief will not change anymore
        # the likelihood should be deterministic
        if confirmed:
            assert (likelihood[0] == 0. and likelihood[1] > 0) or \
                   (likelihood[0] > 0 and likelihood[1] == 0.)

        belief = self.update_with_likelihood(likelihood, enable_low_thresh=False)

        # update confirm state.
        # once an object has been confirmed, it will never change any more
        if not self.confirmed:
            self.confirmed = confirmed
            if confirmed: self._enable_neg = False

        return belief

    def _remap_match_prob(self, match_prob, epsilon):
        # scale match_prob so that the stationary point is 0.5
        # instead of the unstable epsilon / (1 + epsilon)
        mid_point = epsilon / (1 + epsilon)
        if match_prob < 0.5:
            match_prob = match_prob / 0.5 * mid_point
        else:
            match_prob = (match_prob - 0.5) / 0.5 * (1 - mid_point) + mid_point
        return match_prob

class relation_belief(object):
    def __init__(self,
                 bbox1=None,
                 bbox2=None):
        self._belief = np.array([1./3., 1./3., 1./3.])
        # if two objects have some relationship, the bboxes
        # should be near each other.
        # if two objects do not have relationship, the bboxes
        # are possibly segregated.
        self.bbox_llh = np.array([[1.,      0.],
                                  [1.,      0.],
                                  [0.99,    0.01]])

        self.bbox1 = bbox1
        self.bbox2 = bbox2

    def _box_iou(self, bbox1, bbox2):

        left_int, top_int, right_uni, bottom_uni  = np.maximum(bbox1, bbox2)
        left_uni, top_uni, right_int, bottom_int = np.minimum(bbox1, bbox2)

        if right_int <= left_int or bottom_int <= top_int:
            return 0
        else:
            iou = (bottom_int - top_int) * (right_int - left_int) / \
                  ((bottom_uni - top_uni) * (right_uni - left_uni))
            return iou

    def _is_near(self, bbox1, bbox2):
        bound_thresh = 0
        iou = self._box_iou(
            bbox1 + [-bound_thresh, -bound_thresh, bound_thresh, bound_thresh],
            bbox2 + [-bound_thresh, -bound_thresh, bound_thresh, bound_thresh])
        return iou > 0

    @property
    def belief(self):
        if self.bbox1 is None or self.bbox2 is None:
            return self._belief
        else:
            if self._is_near(self.bbox1, self.bbox2):
                belief = self._belief * self.bbox_llh[:, 0]
            else:
                belief = self._belief * self.bbox_llh[:, 1]
            belief /= np.linalg.norm(belief)
            return belief

    def update(self, score, kde, bbox1=None, bbox2=None):
        if bbox1 is not None and bbox2 is not None:
            self.bbox1[:] = bbox1
            self.bbox2[:] = bbox2

        MIN_PROB = 0.05

        parent_llh = np.exp(kde[0].comp_prob(score))
        child_llh = np.exp(kde[1].comp_prob(score))
        norel_llh = np.exp(kde[2].comp_prob(score))
        # posterior
        self._belief *= [parent_llh, child_llh, norel_llh]
        self._belief /= self._belief.sum()

        # clip the prob to make sure that the probability is reasonable
        if self._belief.min() < MIN_PROB:
            _indicator = (self._belief < MIN_PROB)
            res_sum = (self._belief * (1 - _indicator)).sum()

            for i, _ in enumerate(_indicator):
                if self._belief[i] < MIN_PROB:
                    self._belief[i] = MIN_PROB
                else:
                    self._belief[i] = self._belief[i] / res_sum * (1 - MIN_PROB * _indicator.sum())

        return self.belief

    def reset(self):
        self._belief = np.array([0.333, 0.333, 0.334])

if __name__=="__main__":
    # this_dir = osp.dirname(osp.abspath(__file__))
    # with open(osp.join(this_dir, '../../invigorate/density_esti_train_data.pkl')) as f:
    #     data = pkl.load(f)
    # data = data["ground"]
    #
    # pos_data = []
    # neg_data = []
    # for d in data:
    #     for i, score in enumerate(d["scores"]):
    #         if str(i) in d["gt"]:
    #             pos_data.append(score)
    #         else:
    #             neg_data.append(score)
    # pos_data = np.expand_dims(np.array(pos_data), axis=-1)
    # pos_data = np.sort(pos_data, axis=0)[5:-5]
    # neg_data = np.expand_dims(np.array(neg_data), axis=-1)
    # neg_data = np.sort(neg_data, axis=0)[5:-5]
    #
    # kde_pos = gaussian_kde(pos_data)
    # kde_neg = gaussian_kde(neg_data)
    #
    # x = (np.arange(100).astype(np.float32) / 100 - 0.5) * 2
    # y = np.array([kde_pos.comp_prob(x[i]) for i in range(len(x))])
    # plt.plot(x, y, ls="-", lw=2, label="positive")
    # y = np.array([kde_neg.comp_prob(x[i]) for i in range(len(x))])
    # plt.plot(x, y, ls="-", lw=2, label="negative")
    # plt.xlabel("Grounding score")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.show()

    this_dir = osp.dirname(osp.abspath(__file__))
    # with open(osp.join(this_dir, '../../scripts/density_esti_train_data.pkl')) as f:
    with open(osp.join(this_dir, '../../scripts/rel_nus.pkl')) as f:
        data = pkl.load(f)

    parents = []
    childs = []
    norels = []
    for i, d in enumerate(data["gt_mrts"]):
        num_rels = d.shape[1] * d.shape[2]
        d = d.reshape((3, -1))
        det = data["det_mrts"][i].reshape((3, -1))
        for n_rel in range(num_rels):
            r = d[:, n_rel]
            if r[0] == 1:
                parents.append(det[:, n_rel])
            elif r[1] == 1:
                childs.append(det[:, n_rel])
            elif r[2] == 1:
                norels.append(det[:, n_rel])

    parents = np.array(parents)[:, :1]
    childs = np.array(childs)[:, 1:2]
    rels = childs + parents
    rels = np.array(rels)[:, :2].sum(axis = -1)
    rels = np.expand_dims(rels, axis = -1)
    norels = 1 - np.array(norels)[:, 2:]

    kde_p = gaussian_kde(parents, bandwidth=0.05)
    kde_c = gaussian_kde(childs, bandwidth=0.05)
    kde_r = gaussian_kde(rels, bandwidth=0.05)
    kde_n = gaussian_kde(norels, bandwidth=0.05)

    x = (np.arange(100).astype(np.float32) / 100)
    y = np.array([kde_r.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="rel")
    y = np.array([kde_n.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="no_rel")
    plt.xlabel("Rel confidence score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()