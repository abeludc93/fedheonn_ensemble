#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
metrics.general
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains exposes classes to determine metrics for supervised machine learning models.
"""

# Standard libraries
from abc import abstractmethod
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_percentage_error, PredictionErrorDisplay,\
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             RocCurveDisplay)
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
    def _extract_params(self):
        log.debug(f"Extracting parameters from dictionary: [length: {len(self.params)}]")
        self.x_train = self.params["x_train"]
        self.x_test = self.params["x_test"]
        self.d_train = self.params["d_train"]
        self.d_test = self.params["d_test"]
        self.y_train = self.params["y_train"]
        self.y_test = self.params["y_test"]
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
    def plot_metrics(self):
        pass

    @staticmethod
    def _is_classification(params: {}) -> bool:
        return "classification" in params and params["classification"]

    @staticmethod
    def fill_params(x_train=None, x_test=None, d_train=None, d_test=None, y_train=None, y_test=None, threshold=None,
                    d_threshold=None, desired_outputs=None, classification=False, n_classes=None) -> dict:
        params = {"x_train": x_train, "x_test": x_test, "d_train": d_train, "d_test": d_test, "y_train": y_train,
                  "y_test": y_test, "threshold": threshold, "d_threshold": d_threshold,
                  "desired_outputs": desired_outputs, "classification": classification, "n_classes": n_classes}
        return params

    @staticmethod
    def run(params: {} = None, dataset: str = None):
        try:
            if Metrics._is_classification(params):
                log.debug("[Initializing ClassificationMetric]")
                metric = ClassificationMetrics(params, dataset)
            else:
                log.debug("[Initializing RegressionMetric]")
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

# Supervised regression metrics
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

    def plot_metrics(self):
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

# Supervised classification metrics:
class ClassificationMetrics(Metrics):

    def __init__(self, params: {} = None, dataset: str = None):
        super().__init__(params, dataset)
        super()._extract_params()
        self.hits_train = None
        self.hits_test = None
        self.clazz_report_train = None
        self.clazz_report_test = None
        self.clazz_report_print = [None, None]
        self.roc_train = None
        self.roc_test = None

    def __repr__(self):
        return f"ClassificationMetrics(params={self.params}, dataset={self.dataset})"

    def __str__(self):
        return f"ClassificationMetrics instance with params: {self.params} {self.dataset}"

    def generate_metrics(self):
        # Simulate predictions
        super().generate_metrics()
        # Calculate labeled predictions and assign to instance variables
        self.hits_train, self.hits_test = self._calculate_hit_predictions()
        self.clazz_report_print[0], self.clazz_report_train = self._calculate_classification_report(self.d_train,
                                                                                                    self.y_train,
                                                                                                    self.hits_train)
        self.clazz_report_print[1], self.clazz_report_test = self._calculate_classification_report(self.d_test,
                                                                                                   self.y_test,
                                                                                                   self.hits_test)
        self.roc_train = self._calculate_roc_score(self.d_train, self.y_train)
        self.roc_test = self._calculate_roc_score(self.d_test, self.y_test)

    def print_metrics(self):
        roc_train = "{0:.2%}".format(self.roc_train) if self.roc_train is not None else 'CLASSES UNAVAILABLE'
        roc_test = "{0:.2%}".format(self.roc_test) if self.roc_test is not None else 'CLASSES UNAVAILABLE'
        log.info(f"[CLASSIFICATION REPORT DATASET [{self.dataset}]\n\t"
                 f"[TRAIN SET]\n{self.clazz_report_print[0]}\n"
                 f"ROC: {roc_train}\n\t"
                 f"[TEST  SET]\n{self.clazz_report_print[1]}\n"
                 f"ROC: {roc_test}\n")

    def plot_metrics(self) -> None:
        if self.n_classes == 2:
            fig_cm, (ax1, ax2) = plt.subplots(1, 2)
            fig_cm.suptitle(f"Confusion Matrix Display for dataset: {self.dataset}", fontsize=14)
            t_names = [f"lbl_{i}" for i in range(self.n_classes)]
            ax1.set_title("Train set")
            ClassificationMetrics._plot_confusion_matrix(self.d_train, self.y_train, t_names, ax1)
            ax2.set_title("Test set")
            ClassificationMetrics._plot_confusion_matrix(self.d_test, self.y_test, t_names, ax2)
            fig_roc, (ax3, ax4) = plt.subplots(1, 2)
            fig_roc.suptitle(f"ROC Curve Display for dataset: {self.dataset}", fontsize=14)
            ClassificationMetrics._plot_roc_curve(self.d_train, self.y_train, "Train set", ax3)
            ClassificationMetrics._plot_roc_curve(self.d_test, self.y_test, "Test set", ax4)
            plt.tight_layout()
            plt.show()
        else:
            log.info(f"Not plotting classification metrics for multiclass case: {self.n_classes}")

    def _calculate_hit_predictions(self) -> ():
        # Calculate number of hits for train & test data
        # That is when both labeled & predicted outputs overcome its respective thresholds
        hits_train = np.sum(self.y_train == self.d_train)
        hits_test = np.sum(self.y_test == self.d_test)
        return hits_train, hits_test

    def _calculate_classification_report(self, y_true, y_pred, hits) -> ():
        target_names = [f"lbl_{i}" for i in range(self.n_classes)]
        labels = [i for i in range(self.n_classes)]
        # For binary case, extract true positives directly
        if len(target_names) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            log.info(f"|TrueNegatives: {tn} |FalsePositives: {fp} |FalseNegatives: {fn} |TruePositives: {tp} "
                     f"|Hits: {hits}")
        report_print = classification_report(y_true, y_pred, target_names=target_names, labels=labels,
                                             zero_division=0.0,
                                             output_dict=False)
        report_dict = classification_report(y_true, y_pred, target_names=target_names, labels=labels, zero_division=0.0,
                                            output_dict=True)
        return report_print, report_dict

    def _calculate_roc_score(self, y_true, y_pred) -> float | None:
        # List with all available classes
        target_names = [i for i in range(self.n_classes)]
        # Check if true target data has all the available classes. If so, calculate ROC, else return None.
        n_classes_true = len(np.unique(y_true))
        if n_classes_true == self.n_classes:
            # One-hot multiclass
            y_t = np.zeros((len(y_true), n_classes_true))
            y = np.zeros((len(y_pred), n_classes_true))
            for i in range(len(y_pred)):
                y_t[i, y_true[i]] = 1
                y[i, y_pred[i]] = 1
            return roc_auc_score(y_t, y, labels=target_names, multi_class="ovr")
        else:
            log.warn("ROC_SCORE: 'y_true' target data is unbalanced and does not contain all available classes.")
            return None

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, target_names, axes):
        if len(target_names) > 1:
            # Graphic confusion matrix
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes,
                                                    colorbar=False, display_labels=target_names, cmap="Greys")

    @staticmethod
    def _plot_roc_curve(y_true, y_pred, name, axes):
        RocCurveDisplay.from_predictions(y_true, y_pred, ax=axes, name=name)
