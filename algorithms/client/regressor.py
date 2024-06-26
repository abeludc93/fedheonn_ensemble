#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
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

        X, t = self._preprocess(X, t)

        n, n_outputs = t.shape
        # Bagging
        if self.bagging and self.n_estimators > 1:
            for i in range(self.n_estimators):
                M_e, US_e = [], []
                X_bag, t_bag = self.bootstrap_sample(X, t)
                # A model is generated for each output
                for o in range(0, n_outputs):
                    M, US = self._fit(X_bag, t_bag[:, o])
                    M_e.append(M)
                    US_e.append(US)
                self.M.append(M_e)
                self.US.append(US_e)
        else:
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
        X = self._reshape(X).T

        if self.bagging and self.n_estimators > 1:
            y = []
            W_orig = self.W
            for i in range(self.n_estimators):
                self.W  = W_orig[i]
                y.append(self._predict(X))
            y = np.array(y).mean(axis=0)
            self.W = W_orig
        else:
            y = self._predict(X)

        return y.T
