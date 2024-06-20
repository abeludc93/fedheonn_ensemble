#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
metrics.general
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains helper functions to prepare and calculate data in order
to call classification and regression metric functions.

:author: Oscar, Abel
:version: 0.0.2
"""
# Standard libraries
from abc import abstractmethod
# Third-party libraries

# Application modules
from auxiliary.logger import logger as log


class Metrics:

    def __init__(self, params: {} = None, dataset: str = None):
        self.params = params
        self.dataset = dataset

    def __str__(self):
        return f"Metrics general instance with params: {self.params}, {self.dataset}"

    def __repr__(self):
        return f"Metrics(params={self.params}, dataset={self.dataset})"

    # Helper functions
    def _extract_params(self, regression: bool = False):
        log.debug(f"Extracting parameters from dictionary: [length: {len(self.params)}]")
        self.x_train = self.params["x_train"]
        self.x_test = self.params["x_test"]
        self.d_train = self.params["d_train"]
        self.d_test = self.params["d_test"]
        self.y_train = self.params["y_train"]
        self.y_test = self.params["y_test"]
        if not regression:  # mandatory classification parameters
            self.threshold = self.params["threshold"]
            self.d_threshold = self.params["d_threshold"]
            self.desired_outputs = self.params["desired_outputs"]
            self.classification = self.params["classification"]
            self.n_classes = self.params["n_classes"]

    @abstractmethod
    def generate_metrics(self):
        if self.y_train is None or self.y_test is None:
            raise AttributeError("y_train|y_test")

    @abstractmethod
    def print_metrics(self):
        pass

    @abstractmethod
    def plot_metrics(self) -> None:
        pass

    @staticmethod
    def _is_classification(params: {}):
        return "classification" in params and params["classification"]

    @staticmethod
    def fill_params(x_train=None, x_test=None, d_train=None, d_test=None, y_train=None, y_test=None,
                    threshold=None, d_threshold=None, desired_outputs=None, classification=None, n_classes=None) -> dict:
        params = {"x_train": x_train, "x_test": x_test, "d_train": d_train, "d_test": d_test, "y_train": y_train,
                  "y_test": y_test, "threshold": threshold, "d_threshold": d_threshold,
                  "desired_outputs": desired_outputs, "classification": classification, "n_classes": n_classes}
        return params

    @staticmethod
    def run(params: {} = None, dataset: str = None):
        try:
            if Metrics._is_classification(params):
                log.debug("[Initializing ClassificationMetric]")
                from metrics.classification import ClassificationMetrics
                metric = ClassificationMetrics(params, dataset)
            else:
                log.debug("[Initializing RegressionMetric]")
                from metrics.regression import RegressionMetrics
                metric = RegressionMetrics(params, dataset)
            log.info("Generating metric data ...")
            metric.generate_metrics()
            log.info("Printing metrics ...")
            metric.print_metrics()
            log.info("Plotting metrics ...")
            metric.plot_metrics()
        except AttributeError as ae:
            log.error(f"No y_train and/or y_test predictions: {ae}")
        except KeyError as ke:
            log.error(f"Mandatory parameter not found: {ke}")
