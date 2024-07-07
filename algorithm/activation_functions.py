#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
algorithms.activation_functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains several activation functions used within the neural network.
"""

# Standard libraries
from typing import Callable
from sys import float_info as sys_flt
# Third-party libraries
from numpy import log, exp, ones, divide, ndarray
# Application modules
from auxiliary.decorators import clamp_positive_range, clamp_unitary_range

# We avoid using the @decorator's notation for multiprocessing reasons (can't be pickled)

# LOG-SIGMOID transfer functions
def logsig(x):
    return 1 / (1 + (exp(-x)))
def _inv_logsig(x):
    return -log(divide(1, x) - 1)
def inv_logsig(x):
    return clamp_unitary_range(_inv_logsig)(x)
def der_logsig(x):
    return 1 / ((1 + exp(-x)) ** 2) * exp(-x)
# RELU activation functions
def relu(x):
    return log(1 + exp(x))
def _inv_relu(x):
    return log(exp(x) - 1)
def inv_relu(x):
    return clamp_positive_range(_inv_relu)(x)
def der_relu(x):
    return 1 / (1 + exp(-x))
# LINEAR activation functions
def linear(x):
    return x
def inv_linear(x):
    return x
def der_linear(x):
    return ones(len(x)) if type(x) is list or type(x) is ndarray else 1

# AUXILIARY FUNCTION:
def _load_act_fn(fn: str = "linear") -> (Callable, Callable, Callable):
    """
    _load_act_fn() Returns a tuple of three callable functions depending on the 'fn' string, representing
    the activation function desired for the neural network.

    :param fn: string for the activation function 'linear'|'relu'|'logs' (string)
    :return: tuple of three callable functions.
    """
    if fn == "linear":
        return linear, inv_linear, der_linear
    elif fn == "relu":
        return relu, inv_relu, der_relu
    elif fn == "logs":
        return logsig, inv_logsig, logsig
    else:
        raise ValueError(f"Activation function unknown: {fn}")
