#!/usr/bin/python
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")

import vlbert.invigorate._init_paths
from vlbert.common.utils.load import smart_load_model_state_dict
from vlbert.common.trainer import to_cuda
from vlbert.invigorate.data.transforms import build_transforms
from vlbert.invigorate.modules import *
from vlbert.external.pytorch_pretrained_bert import BertTokenizer
from vlbert.invigorate.function.config import config, update_config

from invigorate_msgs.srv import *
import rospy
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class vlbert_server(object):
    def __init__(self, args, config):

        self._init_network()
        self.transform = self._init_transform(config)
        self.task_ids = config.DATASET.TASK_ID

        s = rospy.Service('tpn_service', VLBert, self._vlbert_callback)
        print("READY TO RUN VL-BERT!")

    def _init_network(self):

        print('INITIALIZING MODEL')
        self.tokenizer = BertTokenizer.from_pretrained(
            './vlbert/model/pretrained_model/bert-base-uncased')
        ckpt_path = './vlbert/vl-bert_base_res101_refcoco-0019.model'

        # Set up model and load pretraiend weights
        self.model = eval(config.MODULE)(config)
        torch.cuda.set_device(0)
        self.model.cuda()
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        smart_load_model_state_dict(self.model, checkpoint['state_dict'])
        self.model.eval()
        print('DONE LOADING MODEL')

    def _init_transform(self, cfg, mode='test'):

        return build_transforms(cfg, mode)

    def _vlbert_callback(self, req):

        print("vlbert_service: request received")
        expr = req.expr
        img_msg = req.img
        img_cv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                            dtype=np.uint8, buffer=img_msg.data)  # bypass python3 cv_bridge
        # Swap channel to match training process
        img_cv = img_cv[:, :, [2, 1, 0]]
        img_cv = Image.fromarray(np.uint8(img_cv))

        bboxes = req.bboxes  # 1d array
        bboxes = np.array(bboxes).reshape(-1, 4)  # reshape into 2d array
        num_box = bboxes.shape[0]

        # run inference
        output = {}
        for task_id in self.task_ids:
            result = self._inference_ref(img_cv, bboxes, expr, task_id)
            output.update(result)

        resp = VLBertResponse()
        resp.grounding_scores = output['grounding_logits'].cpu().numpy().flatten().tolist()
        obr_prob = output['obr_probs'].cpu().numpy()

        # parse obr_prob into rel_score_mat
        # where rel_score_mat[0, i, j] is the probability of i being the parent of j,
        # where rel_score_mat[1, i, j] is the probability of i being the child of j,
        # where rel_score_mat[2, i, j] is the probability of i having no relation to j,
        rel_score_mat = np.zeros((3, num_box, num_box))
        obr_prob_idx = 0
        for i in range(num_box):
            for j in range(i + 1, num_box):
                rel_score_mat[0, i, j] = obr_prob[obr_prob_idx, 0]
                rel_score_mat[1, i, j] = obr_prob[obr_prob_idx, 1]
                rel_score_mat[2, i, j] = obr_prob[obr_prob_idx, 2]
                obr_prob_idx += 1
                rel_score_mat[0, j, i] = obr_prob[obr_prob_idx, 0]
                rel_score_mat[1, j, i] = obr_prob[obr_prob_idx, 1]
                rel_score_mat[2, j, i] = obr_prob[obr_prob_idx, 2]
                obr_prob_idx += 1
        assert obr_prob_idx == obr_prob.shape[0]

        resp.rel_score_mat = rel_score_mat.flatten().tolist()
        resp.rel_mat = []

        return resp

    def _tokenize_exp(self, exp):

        exp_retokens = self.tokenizer.tokenize(' '.join(exp))
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp_retokens)
        exp_ids = torch.as_tensor(exp_ids)

        return exp_ids

    def _process_data(self, image, boxes, exp, task_id=0):

        img_info = torch.from_numpy(
            np.array([image.size[0],image.size[1], 1.0, 1.0])).unsqueeze(0)
            #np.array([ 450, 600, 1.0, 1.0])).unsqueeze(0)

        if task_id == 0:
            exp_ids = self._tokenize_exp(exp)
            return self.transform(image, None, None, None, None)[0].unsqueeze(0), torch.from_numpy(boxes).unsqueeze(0).float(), img_info.float(),  exp_ids.unsqueeze(0)

        elif task_id == 1:
            return self.transform(image, None, None, None, None)[0].unsqueeze(0), torch.from_numpy(boxes).unsqueeze(0).float(), img_info.float()
        else:
            raise NotImplementedError

    def _key_filter(self, dict, key):

        return {k: v for k, v in dict.items() if k in key}

    @torch.no_grad()
    def _inference_ref(self, img, bboxes, expr, task_id):

        # test
        ref_ids = []
        pred_boxes = []

        expr = expr.split(' ')
        batch = self._process_data(img, bboxes, expr, task_id)
        batch = to_cuda(batch)
        output = self.model(batch, task_id)

        result = {}
        if task_id == 0:
            import pdb; pdb.set_trace()
            result.update(self._key_filter(output, ['grounding_logits']))
        elif task_id == 1:
            result.update(self._key_filter(output, ['obr_probs', 'pred_rels']))
        else:
            raise NotImplementedError

        return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Inference config parser')
    parser.add_argument('--cfg', type=str, help='path to config file')
    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    rospy.init_node('vlbert_server')
    vlbert_server(args, config)
    rospy.spin()
