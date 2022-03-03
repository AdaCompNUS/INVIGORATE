import numpy as np
import logging
import torch
import torch.nn.functional as f
from collections import OrderedDict

from config.config import *
from libraries.utils.log import LOGGER_NAME

from .invigorate_rss import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

class NoMultistep(Invigorate):

    def _bbox_post_process(self, bboxes, scores):
        # history objects information
        not_removed = [i for i, o in enumerate(self.object_pool) if not o["removed"]]
        no_rmv_to_pool = OrderedDict(zip(range(len(not_removed)), not_removed))
        prev_boxes = np.array([b["bbox"] for b in self.object_pool if not b["removed"]])
        prev_scores = np.array([np.array(b["cls_scores"]).mean(axis=0).tolist() for b in self.object_pool if not b["removed"]])

        # match newly detected bbox to the history objects
        det_to_no_rmv = self._bbox_match(bboxes, prev_boxes, scores, prev_scores)
        det_to_pool = {i: no_rmv_to_pool[v] for i, v in det_to_no_rmv.items()}
        pool_to_det = {v: i for i, v in det_to_pool.items()}
        not_matched = set(range(bboxes.shape[0])) - set(det_to_pool.keys())
        logger.info('_bbox_post_process: object from object pool selected for latter process: {}'.format(pool_to_det.keys()))

        target_confirmed = False
        for o in self.object_pool:
            if o["is_target"] == 1 and not o["removed"]:
                target_confirmed = True

        # updating the information of matched bboxes
        for k, v in det_to_pool.items():
            self.object_pool[v]["bbox"] = bboxes[k]
            self.object_pool[v]["cls_scores"] = [scores[k].tolist()] # Here instead of append, just change to singlestep
        for i in range(len(self.object_pool)):
            if not self.object_pool[i]["removed"] and i not in det_to_pool.values():
                # This only happens when the detected class lable is different with previous box
                logger.info("history bbox {} not matched!, class label different, deleting original object".format(i))
                self.object_pool[i]["removed"] = True

        # initialize newly detected bboxes, add to object pool
        for i in not_matched:
            new_box = self._init_object(bboxes[i], scores[i])
            self.object_pool.append(new_box)
            det_to_pool[i] = len(self.object_pool) - 1
            pool_to_det[len(self.object_pool) - 1] = i
            if target_confirmed:
                new_box["cand_belief"].belief[0] = 1.
                new_box["cand_belief"].belief[1] = 0.
            for j in range(len(self.object_pool[:-1])):
                # initialize relationship belief
                new_rel = self._init_relation(np.array([0.33, 0.33, 0.34]))
                self.rel_pool[(j, det_to_pool[i])] = new_rel

    def transit_state(self, action):
        action_type, target_idx = self.parse_action(action)
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates()

        if action_type == 'GRASP_AND_END' or action_type == 'GRASP_AND_CONTINUE':
            # clear object pool. Only objects that are confirmed by user answer remain.
            self.object_pool[det_to_pool[target_idx]]["removed"] = True
            object_pool = [obj for obj in self.object_pool if obj["is_target"] != -1]
            self.object_pool = object_pool

            # reset relation pool
            for k in self.rel_pool.keys():
                new_rel = self._init_relation(np.array([0.33, 0.33, 0.34]))
                self.rel_pool[k] = new_rel

        elif action_type == 'Q1':
            # asking question does not change state
            pass
    
        return True

class NoMultistepAll(Invigorate):

    def transit_state(self, action):
        action_type, target_idx = self.parse_action(action)
        pool_to_det, det_to_pool, obj_num = self._get_valid_obj_candidates()

        if action_type == 'GRASP_AND_END' or action_type == 'GRASP_AND_CONTINUE':
            # clear everything
            self.object_pool = []
            self.rel_pool = {}

        elif action_type == 'Q1':
            # asking question does not change state
            pass
    
        return True