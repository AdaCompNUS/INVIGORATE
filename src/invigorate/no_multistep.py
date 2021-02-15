import numpy as np
import logging
import torch
import torch.nn.functional as f

from config.config import *
from libraries.utils.log import LOGGER_NAME

from .invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

class NoMultistep(Invigorate):

    def transit_state(self, action):
        # clear object pool. Only objects that are confirmed by user answer remain.
        object_pool = [obj for obj in self.object_pool if obj["is_target"] == 1]
        self.object_pool = object_pool

