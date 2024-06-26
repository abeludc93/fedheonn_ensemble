#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
metrics.regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains functions and procedures to obtain different
kinds of metrics in order to evaluate supervised regression problems

:author: Oscar, Abel
:version: 0.0.2
"""
# Standard libraries

# Third-party libraries
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, PredictionErrorDisplay
# Application modules
from metrics.general import Metrics
from auxiliary.logger import logger as log


class RegressionMetrics(Metrics):

    def __init__(self, params: {} = None, dataset: str = None):
        super().__init__(params, dataset)
        super()._extract_params()
        self.mse_test = None
        self.mse_percent_test = None
        self.r2_test = None
        self.mse_train = None
        self.mse_percent_train = None
        self.r2_train = None

    def __repr__(self):
        return f"RegressionMetrics(params={self.params}, dataset={self.dataset})"

    def __str__(self):
        return f"RegressionMetrics instance with params: {self.params} {self.dataset}"

    def generate_metrics(self):
        # Simulate predictions
        super().generate_metrics()
        # Vectorize predictions
        self.y_train = RegressionMetrics._vectorize(self.y_train)
        self.y_test = RegressionMetrics._vectorize(self.y_test)
        # Calculate MSE, R2
        self.mse_train = RegressionMetrics._calculate_mse(self.d_train, self.y_train)
        self.mse_test = RegressionMetrics._calculate_mse(self.d_test, self.y_test)
        self.mse_percent_train = RegressionMetrics._calculate_mae_percent(self.d_train, self.y_train)
        self.mse_percent_test = RegressionMetrics._calculate_mae_percent(self.d_test, self.y_test)
        self.r2_train = RegressionMetrics._calculate_r2_score(self.d_train, self.y_train)
        self.r2_test = RegressionMetrics._calculate_r2_score(self.d_test, self.y_test)

    def print_metrics(self):
        log.info(
            f"[REGRESSION REPORT DATASET [{self.dataset}]\n\t"
            f"[TRAIN SET]\n"
            f"MSE: {self.mse_train:.4e}\n"
            f"MAE%: {['{:.2%}'.format(i) for i in self.mse_percent_train]}\n"
            f"R2: {self.r2_train:.4f}\n\t"
            f"[TEST  SET]\n"
            f"MSE: {self.mse_test:.4e}\n"
            f"MAE%: {['{:.2%}'.format(i) for i in self.mse_percent_test]}\n"
            f"R2: {self.r2_test:.4f}\n")

    def plot_metrics(self) -> None:
        fig_pe, (ax1, ax2) = plt.subplots(1, 2)
        fig_pe.suptitle(f"Prediction Error Display for dataset: {self.dataset}", fontsize=14)
        ax1.set_title("Train set")
        RegressionMetrics._plot_prediction_err_display(self.d_train, self.y_train, ax1)
        ax2.set_title("Test set")
        RegressionMetrics._plot_prediction_err_display(self.d_test, self.y_test, ax2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _vectorize_predictions(y):
        return y.reshape(len(y)) if y.shape == (len(y), 1) else y

    @staticmethod
    def _calculate_mse(y_true, y_pred) -> float:
        # y_true data (from dataset) and y_pred (predictions) should match in length/shape: [m,1] - [m,]
        assert len(y_true) == len(y_pred)
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def _calculate_mae_percent(y_true, y_pred) -> float:
        # y_true data (from dataset) and y_pred (predictions) should match in length/shape: [m,1] - [m,]
        assert len(y_true) == len(y_pred)
        return mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')

    @staticmethod
    def _calculate_r2_score(y_true, y_pred) -> float:
        # y_true data (from dataset) and y_pred (predictions) should match in length/shape: [m,1] - [m,]
        assert len(y_true) == len(y_pred)
        return r2_score(y_true, y_pred)

    @staticmethod
    def _plot_prediction_err_display(y_true, y_pred, axes):
        # Graphic prediction error
        PredictionErrorDisplay.from_predictions(y_true, y_pred, ax=axes)

    @staticmethod
    def _vectorize(y):
        return y.reshape(len(y)) if y.shape == (len(y), 1) else y
