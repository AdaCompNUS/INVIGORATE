import logging

LOG_FILE = "demo.log"
LOG_LEVEL = logging.DEBUG

logging.basicConfig(filename=LOG_FILE, encoding='utf-8', level=LOG_LEVEL)
logger = logging.getLogger('')

