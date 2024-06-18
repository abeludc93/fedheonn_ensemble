#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Application modules
from algorithms.client.base import FedHEONN_client


class FedHEONN_regressor(FedHEONN_client):
    """FedHEONN client for regression tasks"""

    def fit(self, X, t):
        """Fit the model to data matrix X and target(s) t.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.

        t : ndarray of shape (n_samples, n_outputs)
            The target values.
        """
        n, n_outputs = t.shape

        # A model is generated for each output
        for o in range(0, n_outputs):
            M, US = self._fit(X, t[:, o])
            self.M.append(M)
            self.US.append(US)

    def predict(self, X):
        """Predict using FedHEONN model for regression.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        y = self._predict(X)
        return y
