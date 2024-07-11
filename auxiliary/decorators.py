#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
auxiliary.decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module exposes different decorators, pe., to clamp ranges or modify existing functions.
"""

# Standard libraries
from time import perf_counter
from typing import Callable
from sys import float_info as sys_flt
# Third-party libraries
import numpy as np
# Application modules
from auxiliary.logger import logger as log


def clamp_positive_range(func: Callable) -> Callable:
    """
    clamp_positive_range() Decorator to clamp the value of the arguments to positive values.

    :param func: function to be decorated
    :return: decorated function
    """
    lowest = sys_flt.epsilon
    def _wrapper(x):
        if type(x) is list or type(x) is np.ndarray:
            lst = []
            for item in x:
                lst.append(lowest) if item <= 0 else lst.append(item)
            return func(lst)
        else:
            if x <= 0:
                x = lowest
            return func(x)
    return _wrapper


def clamp_unitary_range(func: Callable) -> Callable:
    """
    clamp_unitary_range() Decorator to clamp the value of the arguments to open range (0, 1).

    :param func: function to be decorated
    :return: decorated function
    """
    lowest = sys_flt.epsilon
    highest = 1 - sys_flt.epsilon
    def _wrapper(x):
        if type(x) is list or type(x) is np.ndarray:
            lst = []
            for item in x:
                if not 0 < item < 1:
                    replace = lowest if item <= 0 else highest
                    lst.append(replace)
                else:
                    lst.append(item)
            return func(lst)
        else:
            if not 0 < x < 1:
                x = lowest if x <= 0 else highest
            return func(x)
    return _wrapper


def log_func(func: Callable) -> Callable:
    """
    Decorator to log all function arguments and return value/s.

    :param func: function to decorate
    :return: logged function
    """
    def _wrapper(*args, **kwargs):
        log.debug("Input arguments to %s: %s, %s", func.__name__, args, kwargs)
        output = func(*args, **kwargs)
        log.debug("Output to %s: %s", func.__name__, output)
        return output
    return _wrapper


def time_func(func: Callable) -> Callable:
    """
    Decorator to time the elapsed duration of a function call.

    :param func: function to decorate
    :return: fractional seconds
    """
    def _wrapper(*args, **kwargs):
        log.info("Starting timer to [%s]", func.__name__)
        start = perf_counter()
        output = func(*args, **kwargs)
        end = perf_counter()
        log.info("Elapsed time running [%s]: [%.3f] seconds", func.__name__, end - start)
        return output
    return _wrapper
