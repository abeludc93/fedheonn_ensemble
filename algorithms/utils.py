#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
algorithms.utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
File containing auxiliary functions for the client/coordinator
neural network algorirthms

:author: Abel
:version: 0.0.1
"""
# Standard library modules
from typing import Callable
# Application modules
import auxiliary.activation_functions as act_fn


def _load_act_fn(fn: str = "linear") -> (Callable, Callable, Callable):
    """
    _load_act_fn() Returns a tuple of three callable functions depending on the 'fn' string, representing
    the activation function desired for the neural network.

    :param fn: string for the activation function 'linear'|'relu'|'logs' (string)
    :return: tuple of three callable functions.
    """
    if fn == "linear":
        return act_fn.LINEAR, act_fn.INV_LINEAR, act_fn.DER_LINEAR
    elif fn == "relu":
        return act_fn.RELU, act_fn.INV_RELU, act_fn.DER_RELU
    else:
        return act_fn.LOGSIG, act_fn.INV_LOGSIG, act_fn.LOGSIG
