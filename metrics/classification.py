#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
metrics.classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains functions and procedures to obtain different
kinds of metrics in order to evaluate supervised classification problems

:author: Oscar, Abel
:version: 0.0.2
"""
# Third-party library modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, \
    RocCurveDisplay
# Application modules
from auxiliary.logger import logger as log
from metrics.general import Metrics


class ClassificationMetrics(Metrics):

    def __init__(self, params: {} = None, dataset: str = None):
        super().__init__(params, dataset)
        super()._extract_params(regression=False)
        self.y_train_true = None
        self.y_train_pred = None
        self.hits_train = None
        self.y_test_true = None
        self.y_test_pred = None
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
        self.y_train_true, self.y_train_pred = self._calculate_labeled_predictions(self.y_train, self.d_train)
        self.y_test_true, self.y_test_pred = self._calculate_labeled_predictions(self.y_test, self.d_test)
        self.hits_train, self.hits_test = self._calculate_hit_predictions()
        self.clazz_report_print[0], self.clazz_report_train = self._calculate_classification_report(self.y_train_true,
                                                                                                    self.y_train_pred,
                                                                                                    self.hits_train)
        self.clazz_report_print[1], self.clazz_report_test = self._calculate_classification_report(self.y_test_true,
                                                                                                   self.y_test_pred,
                                                                                                   self.hits_test)
        self.roc_train = self._calculate_roc_score(self.y_train_true, self.y_train_pred, self.y_train)
        self.roc_test = self._calculate_roc_score(self.y_test_true, self.y_test_pred, self.y_test)

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
            ClassificationMetrics._plot_confusion_matrix(self.y_train_true, self.y_train_pred, t_names, ax1)
            ax2.set_title("Test set")
            ClassificationMetrics._plot_confusion_matrix(self.y_test_true, self.y_test_pred, t_names, ax2)
            fig_roc, (ax3, ax4) = plt.subplots(1, 2)
            fig_roc.suptitle(f"ROC Curve Display for dataset: {self.dataset}", fontsize=14)
            ClassificationMetrics._plot_roc_curve(self.y_train_true, self.y_train_pred, t_names, "Train set", ax3)
            ClassificationMetrics._plot_roc_curve(self.y_test_true, self.y_test_pred, t_names, "Test set", ax4)
            plt.tight_layout()
            plt.show()

    def _calculate_labeled_predictions(self, y_data, d_data) -> ():
        # y_data (from dataset) and d_data (from predictions) should match in shape
        assert y_data.shape == d_data.shape
        # desired outputs should be a list of two elements, floats, low & high, as desired null and hit values
        assert len(self.desired_outputs) == 2
        # Low | High values: first represent values that do not match and the second for those that do
        low, high = self.desired_outputs[0], self.desired_outputs[1]
        # Generate numpy arrays with predictions and true values for future use on metric functions
        arr_true = np.apply_along_axis(lambda arr, value: np.abs(arr - value).argmin(), 1, d_data, high)
        arr_pred = np.apply_along_axis(lambda arr, value: np.abs(arr - value).argmin(), 1, y_data, high)
        return arr_true, arr_pred

    def _calculate_hit_predictions(self) -> ():
        # Calculate number of hits for train & test data
        # That is when both labeled & predicted outputs overcome its respective thresholds
        hits_train = np.sum(np.logical_and(self.y_train >= self.threshold, self.d_train >= self.d_threshold) |
                            np.logical_and(self.y_train < self.threshold, self.d_train < self.d_threshold))
        hits_test = np.sum(np.logical_and(self.y_test >= self.threshold, self.d_test >= self.d_threshold) |
                           np.logical_and(self.y_test < self.threshold, self.d_test < self.d_threshold))
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

    def _calculate_roc_score(self, y_true, y_pred, y_prob) -> float | None:
        # List with all available classes
        target_names = [i for i in range(self.n_classes)]
        if self.n_classes > 2:
            # Turn threshold labels into probability data
            y_pred = y_prob / y_prob.sum(axis=1, keepdims=1)
        # Check if true target data has all the available classes. If so, calculate ROC, else return None.
        n_classes_true = len(np.unique(y_true))
        if n_classes_true == self.n_classes:
            return roc_auc_score(y_true, y_pred, labels=target_names, multi_class="ovr")
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
    def _plot_roc_curve(y_true, y_pred, target_names, name, axes):
        RocCurveDisplay.from_predictions(y_true, y_pred, ax=axes, name=name)
