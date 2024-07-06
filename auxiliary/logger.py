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

# Define global variables
APP_NAME = "FedHEONN"
LOG_LEVEL_CMD = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG
LOG_FILE = "out.log"
LOG_PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)


# Create a console handler and set log level
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(LOG_LEVEL_CMD)

# Create a file handler and set log level
_fh = logging.FileHandler("{0}/{1}".format(LOG_PATH, LOG_FILE))
_fh.setLevel(LOG_LEVEL_FILE)

# Create a formatter and attach to handlers
_formatter_ch = logging.Formatter('[%(levelname)5s]: %(message)s')
_formatter_fh = logging.Formatter('%(asctime)s [%(levelname)5s]: %(message)s')
_ch.setFormatter(_formatter_ch)
_fh.setFormatter(_formatter_fh)

# Create a logger and add handlers to it
logger = logging.getLogger(APP_NAME)
logger.setLevel(LOG_LEVEL_CMD)
logger.addHandler(_ch)
logger.addHandler(_fh)
