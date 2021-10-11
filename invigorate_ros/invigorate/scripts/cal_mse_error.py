import math
import numpy as np
import pickle
import re
from ast import literal_eval

def cal_tgt_loss(gt_tgt_prob, pred_tgt_prob):
    loss = 0.0
    for i in range(len(gt_tgt_prob)):
        loss += math.fabs(gt_tgt_prob[i] - pred_tgt_prob[i])
    return loss

def cal_rel_loss(gt_rel_prob, pred_rel_prob):
    N = gt_rel_prob.shape[1]
    loss = 0.0
    gt_rel_prob = gt_rel_prob.reshape(-1)
    pred_rel_prob = pred_rel_prob.reshape(-1)
    for i in range(len(gt_rel_prob)):
        loss += math.fabs(gt_rel_prob[i] - pred_rel_prob[i])
    loss /= (N * (N - 1))
    return loss

gt_tgt_probs = [[] for _ in range(10)]
st_tgt_probs = [[] for _ in range(10)]
mt_tgt_probs = [[] for _ in range(10)]
gt_tgt_probs[0].append(np.array([0, 0, 1, 0, 0]))
st_tgt_probs[0].append(np.array([0.18278658, 0.20364547, 0.23768567, 0.18604629, 0.18983606]))
mt_tgt_probs[0].append(np.array([0.13955913, 0.27348723, 0.41313897, 0.16681291, 0.00700176]))
gt_tgt_probs[0].append(np.array([0., 1., 0., 0., 0.]))
st_tgt_probs[0].append(np.array([0.21311957, 0.20922683, 0.18934439, 0.19588761, 0.19242156]))
mt_tgt_probs[0].append(np.array([0.30818002, 0.36387653, 0.13793399, 0.18921607, 0.0007934]))
gt_tgt_probs[0].append(np.array([0., 1., 0., 0., 0., 0.]))
st_tgt_probs[0].append(np.array([0.16700086, 0.18461856, 0.16464531, 0.22692688, 0.09480662, 0.16200179]))
mt_tgt_probs[0].append(np.array([0.2533422, 0.28087239, 0.17511924, 0.28074516, 0.00992101, 0.        ]))
gt_tgt_probs[0].append(np.array([0., 1., 0., 0.]))
st_tgt_probs[0].append(np.array([0.29890734, 0.2573317,  0.22151627, 0.22224472]))
mt_tgt_probs[0].append(np.array([0.38063884, 0.38088357, 0.23847759, 0.        ]))
gt_tgt_probs[0].append(np.array([1., 0., 0.]))
st_tgt_probs[0].append(np.array([0.35560083, 0.32545438, 0.31894475]))
mt_tgt_probs[0].append(np.array([0.599295, 0.398775, 0.00193 ]))
gt_tgt_probs[0].append(np.array([1., 0.]))
st_tgt_probs[0].append(np.array([0.5659443, 0.43405563]))
mt_tgt_probs[0].append(np.array([0.98948997, 0.01051003]))

for ind in range(2, 11):
    file = open("src/experiment/multistep_vs_singlestep/{}.txt".format(ind), "r")
    lines = file.readlines()
    for i, line in enumerate(lines):
        line = line.strip(' []\n')
        line_list = line.split(" ")
        val_list = [float(x) for x in line_list if x != '']
        val = np.array(val_list)
        if i % 3 == 0:
            gt_tgt_probs[ind - 1].append(val)
        elif i % 3 == 1:
            st_tgt_probs[ind - 1].append(val)
        else:
            mt_tgt_probs[ind - 1].append(val)
    file.close()

for i in range(len(gt_tgt_probs)):
    st_tgt_losses = []
    mt_tgt_losses = []
    for j in range(len(gt_tgt_probs[i])):
        st_tgt_loss = cal_tgt_loss(gt_tgt_probs[i][j], st_tgt_probs[i][j])
        mt_tgt_loss = cal_tgt_loss(gt_tgt_probs[i][j], mt_tgt_probs[i][j])
        st_tgt_losses.append(st_tgt_loss)
        mt_tgt_losses.append(mt_tgt_loss)
    print(st_tgt_losses)
    print(mt_tgt_losses)

# pickle_file_name = "src/experiment/multistep_vs_singlestep/result.pkl"
# pickle_file = open(pickle_file_name,'wb')
# pickle_data = {"gt_tgt_probs": gt_tgt_probs, "st_tgt_probs": st_tgt_probs, "mt_tgt_probs": mt_tgt_probs}
# pickle.dump(pickle_data, pickle_file)

print("---")

gt_rel_probs = [[] for _ in range(10)]
st_rel_probs = [[] for _ in range(10)]
mt_rel_probs = [[] for _ in range(10)]
for ind in range(1, 11):
    file = open("src/experiment/multistep_vs_singlestep/{}_rel.txt".format(ind), "r")
    data = file.read()
    data_list = data.split('!')
    for i, data in enumerate(data_list):
        data = re.sub(r"([^[])\s+([^]])", r"\1, \2", data)
        data = data.strip(', \n')
        data = np.array(literal_eval(data))
        data[data>0.9] = 0.9
        data[data<0.05] = 0.05
        for j in range(3):
            np.fill_diagonal(data[j], 0.0)
        if i % 3 == 0:
            gt_rel_probs[ind - 1].append(data)
        elif i % 3 == 1:
            st_rel_probs[ind - 1].append(data)
        else:
            mt_rel_probs[ind - 1].append(data)

    file.close()

for i in range(len(gt_rel_probs)):
    st_rel_losses = []
    mt_rel_losses = []
    for j in range(len(gt_rel_probs[i])):
        st_rel_loss = cal_rel_loss(gt_rel_probs[i][j], st_rel_probs[i][j])
        mt_rel_loss = cal_rel_loss(gt_rel_probs[i][j], mt_rel_probs[i][j])
        st_rel_losses.append(st_rel_loss)
        mt_rel_losses.append(mt_rel_loss)
    print(st_rel_losses)
    print(mt_rel_losses)