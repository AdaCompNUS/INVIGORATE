import logging
from datetime import datetime
import os.path as osp
import sys

# this_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(this_dir, '../../'))

from config.config import ROOT_DIR

LOGGER_NAME = 'invigorate_logger'

now = datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
LOG_FILE = osp.join(ROOT_DIR, "logs/demo_{}.log".format(date_time))
LOG_LEVEL = logging.DEBUG

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(formatter)
file_handler.setLevel(LOG_LEVEL)
logger.addHandler(file_handler)

if __name__ == "__main__":
    logger.error("test")
    logger.warning("test")
    logger.info("test")
    logger.debug("test")

