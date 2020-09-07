"""
# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class _RCNN_header(nn.Module):
    def __init__(self, input_dim, n_classes, class_agnostic):
        super(_RCNN_header, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_agnostic = class_agnostic

    def forward(self, x):
        raise NotImplementedError
