from src.libraries.density_estimator.density_estimator import gaussian_kde
import numpy as np
import pickle
import matplotlib.pyplot as plt

grounding_data_path = "../model/ground_density_estimation.pkl"
mattnett_data_path = "../model/ground_density_estimation_mattnet.pkl"
rel_data_path = "../model/rel_dens_vmrn.pkl"
rel_data_path_vilbert = "../model/rel_dens_vilbert.pkl"
detection_data_path = "../model/object_density_estimation.pkl"

def vis_grounding_density(grounding_data_path, upper=1.0, lower=-1.0):
    # vis grounding scores
    grounding_data = pickle.load(open(grounding_data_path, "rb"))["ground"]
    grounding_pos = []
    for d in grounding_data :
        if 'NONE' not in d["gt"]:
            for gt in d["gt"]:
                ind = int(gt)
                grounding_pos.append(d["scores"][ind])
    grounding_pos = np.array(grounding_pos)
    grounding_neg = []
    for d in grounding_data:
        if 'NONE' not in d["gt"]:
            for ind in range(d["scores"].shape[0]):
                if str(ind) not in d["gt"]:
                    grounding_neg.append(d["scores"][ind])
    grounding_neg = np.array(grounding_neg)
    kde_grounding_pos = gaussian_kde(grounding_pos[:, None], bandwidth=0.01)
    kde_grounding_neg = gaussian_kde(grounding_neg[:, None], bandwidth=0.01)
    x = np.arange(1000).astype(np.float32) / 1000 * (upper - lower) + lower
    x_dens_neg = kde_grounding_neg.comp_prob(x) * grounding_neg.shape[0]
    x_dens_pos = kde_grounding_pos.comp_prob(x) * grounding_pos.shape[0]
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

def vis_detection_density(detection_data_path, upper=1.0, lower=0.0):
    # vis grounding scores
    detection_data = pickle.load(open(detection_data_path, "rb"))
    detection_pos = np.array(detection_data["pos"])
    detection_neg = np.array(detection_data["neg"])
    kde_detection_pos = gaussian_kde(detection_pos[:, None], bandwidth=0.2)
    kde_detection_neg = gaussian_kde(detection_neg[:, None], bandwidth=0.2)
    x = np.arange(1000).astype(np.float32) / 1000 * (upper - lower) + lower
    x_dens_neg = np.exp(kde_detection_neg.comp_prob(x))#  * detection_neg.shape[0] / detection_pos.shape[0]

    x_dens_pos = np.exp(kde_detection_pos.comp_prob(x))
    print(np.sum(0.001 * x_dens_pos))

    x_tgt_prob = x_dens_pos / (x_dens_pos + x_dens_neg)
    plt.figure(figsize=[20,9])
    plt.plot(x, x_dens_pos, label="pos",color="#F08080")
    plt.plot(x, x_dens_neg, label="neg",color="#0B7093")
    plt.plot(x, x_tgt_prob, label="prob_mapping", color="#000000")
    plt.legend()
    plt.show()



# vis_relation_density(rel_data_path, upper=2.0, lower=-1.0)
# vis_grounding_density(mattnett_data_path)
vis_detection_density(detection_data_path)