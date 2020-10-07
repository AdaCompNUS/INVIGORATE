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
        x = np.exp(self.kde.score_samples(x))
        return x.squeeze()

class object_belief(object):
    def __init__(self):
        self.belief = np.array([0.5, 0.5])

    def update(self, score, kde):
        neg_prob = kde[0].comp_prob(score)
        pos_prob = kde[1].comp_prob(score)
        self.belief *= [neg_prob, pos_prob]
        self.belief /= self.belief.sum()
        self.belief = np.clip(self.belief, 0.01, 0.99)
        return self.belief

    def reset(self):
        self.belief = np.array([0.5, 0.5])

class relation_belief(object):
    def __init__(self):
        self.belief = np.array([0.333, 0.333, 0.334])

    def update(self, score, kde):
        parent_llh = kde[0].comp_prob(score)
        child_llh = kde[1].comp_prob(score)
        norel_llh = kde[2].comp_prob(score)
        # posterior
        self.belief *= [parent_llh, child_llh, norel_llh]
        self.belief /= self.belief.sum()

        # clip the prob to make sure that the probability is reasonable
        if self.belief.min() < 0.1:
            _indicator = (self.belief < 0.1)
            res_sum = (self.belief * (1 - _indicator)).sum()

            for i, _ in enumerate(_indicator):
                if self.belief[i] < 0.1:
                    self.belief[i] = 0.1
                else:
                    self.belief[i] = self.belief[i] / res_sum * (1 - 0.1 * _indicator.sum())
        return self.belief

    def reset(self):
        self.belief = np.array([0.333, 0.333, 0.334])

if __name__=="__main__":
    this_dir = osp.dirname(osp.abspath(__file__))
    with open(osp.join(this_dir, 'density_esti_train_data.pkl')) as f:
        data = pkl.load(f)
    data = data["ground"]

    pos_data = []
    neg_data = []
    for d in data:
        for i, score in enumerate(d["scores"]):
            if str(i) in d["gt"]:
                pos_data.append(score)
            else:
                neg_data.append(score)
    pos_data = np.expand_dims(np.array(pos_data), axis=-1)
    pos_data = np.sort(pos_data, axis=0)[5:-5]
    neg_data = np.expand_dims(np.array(neg_data), axis=-1)
    neg_data = np.sort(neg_data, axis=0)[5:-5]

    kde_pos = gaussian_kde(pos_data)
    kde_neg = gaussian_kde(neg_data)

    x = (np.arange(100).astype(np.float32) / 100 - 0.5) * 2
    y = np.array([kde_pos.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="positive")
    y = np.array([kde_neg.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="negative")
    plt.legend()
    plt.show()
