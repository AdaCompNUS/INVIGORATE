import pickle
import os
import numpy as np

LOG_DIR = "../experiment/invigorate_vs_vilbert"
EXP_IDX = 10

def generate_gt_tgt_prob(tgt_idx, num_box):
    gt_tgt_prob = np.zeros((num_box))
    for idx in tgt_idx:
        gt_tgt_prob[idx] = 1.0
    return gt_tgt_prob

for i in range(1, 11):
    exp_idx = EXP_IDX + i
    print(exp_idx)
    dir = os.path.join(LOG_DIR, str(exp_idx))
    entries = os.listdir(dir)
    for f in entries:
        if f.endswith("pkl"):
            print(f)
            infile = open(os.path.join(dir, f),'rb')
            data = pickle.load(infile)
            if "gt_tgt_prob" not in data:
                gt_target_str = raw_input("Enter idx for gt tgt:")
                if gt_target_str != "":
                    gt_target_list = gt_target_str.split(" ")
                    gt_targets = [int(i) for i in gt_target_list]
                else:
                    gt_targets = []
                num_obj = len(data["invigorate_tgt_prob"])
                gt_tgt_prob = generate_gt_tgt_prob(gt_targets, num_obj)
                data["gt_tgt_prob"] = gt_tgt_prob

                print("gt_tgt_prob", data["gt_tgt_prob"])
                print("invigorate_tgt_prob", data["invigorate_tgt_prob"])
                print("vilbert_tgt_prob", data["vilbert_tgt_prob"])

                infile.close()

                outfile = open(os.path.join(dir, f),'wb')
                pickle.dump(data, outfile)
                outfile.close()
            else:
                infile.close()


