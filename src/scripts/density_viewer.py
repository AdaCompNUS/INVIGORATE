from src.libraries.density_estimator.density_estimator import gaussian_kde
import numpy as np
import pickle
import matplotlib.pyplot as plt

grounding_data_path = "../tests/ground_density_estimation.pkl"
rel_data_path = "../tests/rel_dens_vmrn.pkl"
rel_data_path_vilbert = "../tests/rel_dens_vilbert.pkl"

def vis_grounding_density(grounding_data_path, upper=10.0, lower=-20.0):
    # vis grounding scores
    grounding_data = pickle.load(open(grounding_data_path, "rb"))
    grounding_pos = np.array([d["scores"][int(d["gt"][0])] for d in grounding_data])
    grounding_neg = []
    for d in grounding_data:
        grounding_neg += d["scores"][:int(d["gt"][0])].tolist() + d["scores"][int(d["gt"][0])+1:].tolist()
    grounding_neg = np.array(grounding_neg)
    kde_grounding_pos = gaussian_kde(grounding_pos[:, None], bandwidth=1.0)
    kde_grounding_neg = gaussian_kde(grounding_neg[:, None], bandwidth=1.0)
    x = np.arange(1000).astype(np.float32) / 1000 * (upper - lower) + lower
    x_dens_neg = kde_grounding_neg.comp_prob(x)
    x_dens_pos = kde_grounding_pos.comp_prob(x)
    plt.figure(figsize=[20,9])
    plt.plot(x, x_dens_pos, label="pos",color="#F08080")
    plt.plot(x, x_dens_neg, label="neg",color="#0B7093")
    plt.legend()
    plt.show()

# vis grounding scores
def vis_relation_density(rel_data_path, upper=1.0, lower=0.0, mode=None):
    if mode is None:
        # vis the density for "no rel" and "rel"
        relation_data = pickle.load(open(rel_data_path, "rb"))
        relation_pos = np.array([1 - d["det_score"][2] for d in relation_data if d["gt"] in {1, 2}])
        relation_neg = np.array([1 - d["det_score"][2] for d in relation_data if d["gt"] == 3])
        kde_rel_pos = gaussian_kde(relation_pos[:, None], bandwidth=0.1)
        kde_rel_neg = gaussian_kde(relation_neg[:, None], bandwidth=0.1)
        x = np.arange(1000).astype(np.float32) / 1000 * (upper - lower) + lower
        x_dens_neg = kde_rel_neg.comp_prob(x)
        x_dens_pos = kde_rel_pos.comp_prob(x)
        plt.figure(figsize=[20,9])
        plt.plot(x, x_dens_pos, label="pos",color="#F08080")
        plt.plot(x, x_dens_neg, label="neg",color="#0B7093")
        plt.legend()
        plt.show()

# vis_relation_density(rel_data_path, upper=2.0, lower=-1.0)
vis_grounding_density(grounding_data_path)