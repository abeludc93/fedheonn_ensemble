#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
# Application modules
from algorithms.client.base import FedHEONN_client


class FedHEONN_classifier(FedHEONN_client):
    """FedHEONN client for classification tasks"""

    def fit(self, X, t_onehot):
        """Fit the model to data matrix X and target(s) t_onehot.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.

        t_onehot : ndarray of shape (n_samples, n_classes)
            The target values (class labels using one-hot encoding).
        """
        n, n_classes = t_onehot.shape

        # Transforms the values (0, 1) to (0.05, 0.95) to avoid the problem
        # of the inverse of the activation function at the extremes
        t_onehot = t_onehot * 0.90 + 0.05

        # A model is generated for each class
        for c in range(0, n_classes):
            M, US = self._fit(X, t_onehot[:, c])
            self.M.append(M)
            self.US.append(US)

    def predict(self, X):
        """Predict using FedHEONN model for classification.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes: values between 0 and n_classes-1
        """
        y_onehot = self._predict(X)
        y = np.apply_along_axis(lambda arr, value: np.abs(arr - value).argmin(), axis=0, arr=y_onehot, value=0.95)
        return y
