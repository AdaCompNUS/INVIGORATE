# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import absolute_import
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg

from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg

from model.op2l.rois_pair_expanding_layer import _RoisPairExpandingLayer
from model.op2l.object_pairing_layer import _ObjPairLayer

class _OP2L(nn.Module):
    def __init__(self, pool_height, pool_width, pool_scaler, isex):
        super(_OP2L, self).__init__()
        self._isex = isex

        self.OP2L_rois_pairing = _RoisPairExpandingLayer()
        self.OP2L_object_pair = _ObjPairLayer(self._isex)

        self.OP2L_roi_pool = _RoIPooling(pool_height, pool_width, pool_scaler)
        self.OP2L_roi_align = RoIAlignAvg(pool_height, pool_width, pool_scaler)

    def forward(self, feats, rois, batch_size, obj_num):
        """
        :param feats: input features from basenet Channels: x W x H
        :param rois: object detection results: N x 4
        :return: object pair features: N(N-1) x 3 x 512 x 7 x 7

        algorithm:
        1 regions of intrests pairing
        2 roi pooling
        3 concate object pair features
        """
        paired_rois = self.OP2L_rois_pairing(rois, batch_size, obj_num)
        if cfg.VMRN.OP2L_POOLING_MODE == 'align':
            pooled_feat = self.OP2L_roi_align(feats, paired_rois.view(-1,5))
        elif cfg.VMRN.OP2L_POOLING_MODE == 'pool':
            pooled_feat = self.OP2L_roi_pool(feats, paired_rois.view(-1,5))
        obj_pair_feats = self.OP2L_object_pair(pooled_feat, batch_size, obj_num)
        return obj_pair_feats