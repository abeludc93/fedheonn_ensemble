#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
auxiliary.activation_functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains several activation functions
for the neural network.

:author: Oscar.
:version: 0.0.2
"""
# Third-party libraries
from numpy import log, exp, ones, divide, ndarray
# Application modules
from auxiliary.decorators import clamp_positive_range, clamp_unitary_range


""" Global variables as functions """
# LOG-SIGMOID transfer functions
LOGSIG = lambda x: 1 / (1 + (exp(-x)))
INV_LOGSIG = clamp_unitary_range(lambda x: -log(divide(1, x) - 1))
DER_LOGSIG = lambda x: 1 / ((1 + exp(-x)) ** 2) * exp(-x)
# RELU activation functions
RELU = lambda x: log(1 + exp(x))
INV_RELU = clamp_positive_range(lambda x: log(exp(x) - 1))
DER_RELU = lambda x: 1 / (1 + exp(-x))
# LINEAR activation functions
LINEAR = lambda x: x
INV_LINEAR = LINEAR
DER_LINEAR = lambda x: ones(len(x)) if type(x) is list or type(x) is ndarray else 1
