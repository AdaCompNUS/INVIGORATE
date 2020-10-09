import logging
from datetime import datetime
import os.path as osp

from config.config import ROOT_DIR

LOGGER_NAME = 'invigorate_logger'

now = datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
LOG_FILE = osp.join(ROOT_DIR, "logs/demo_{}.log".format(date_time))
LOG_LEVEL = logging.DEBUG

logger = logging.getLogger(LOGGER_NAME)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(level=LOG_LEVEL)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(level=LOG_LEVEL)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


