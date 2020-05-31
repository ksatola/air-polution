# Module handler for logging
"""
Usage:
Import this module and use required levels like this:
from logger import logger
logger.info("Started")
logger.error("Operation failed.")
logger.debug("Encountered debug case")
"""

import logging.handlers

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(filename)s | %(lineno)d | %(funcName)s | %(asctime)s | %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

logFilePath = "air-pollution.log"
file_handler = logging.handlers.RotatingFileHandler(
    filename=logFilePath, maxBytes=1000000, backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
