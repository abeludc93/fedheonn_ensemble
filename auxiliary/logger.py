#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
auxiliary.logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logger module.
It runs as a singleton as Python automatically loads and runs the module once imported elsewhere.
Logs both to sys_out and to a log file on the current directory.
"""

# Standard libraries
import logging
import sys
import os
import time

# Define global variables
APP_NAME = "FedHEONN"
LOG_LEVEL = logging.DEBUG
LOG_LEVEL_CMD = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG
LOG_PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)
TIME_CREATION = time.localtime()
LOG_FILE = f"{APP_NAME}_{TIME_CREATION.tm_yday}.log"


# Create a console handler and set log level
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(LOG_LEVEL_CMD)

# Create a file handler and set log level
_fh = logging.FileHandler(f"{LOG_PATH}/{LOG_FILE}")
_fh.setLevel(LOG_LEVEL_FILE)

# Create a formatter and attach to handlers
_formatter_ch = logging.Formatter('%(levelname)-8s - %(message)s')
_formatter_fh = logging.Formatter('%(asctime)s - %(module)-10s - %(threadName)-10s - %(levelname)-8s - %(message)s')
_ch.setFormatter(_formatter_ch)
_fh.setFormatter(_formatter_fh)

# Create a logger and add handlers to it
logger = logging.getLogger(APP_NAME)
logger.setLevel(LOG_LEVEL)
logger.addHandler(_ch)
logger.addHandler(_fh)
