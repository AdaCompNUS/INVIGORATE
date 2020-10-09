import logging

LOGGER_NAME = 'invigorate_logger'

LOG_FILE = "demo.log"
LOG_LEVEL = logging.DEBUG

logger = logging.getLogger(LOGGER_NAME)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


