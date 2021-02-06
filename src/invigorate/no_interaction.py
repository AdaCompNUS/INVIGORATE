import numpy as np
import logging
import torch
import torch.nn.functional as f

from config.config import *
from libraries.utils.log import LOGGER_NAME

from .invigorate import Invigorate

# -------- Statics ---------
logger = logging.getLogger(LOGGER_NAME)

class NoInteraction(Invigorate):
    # TODO