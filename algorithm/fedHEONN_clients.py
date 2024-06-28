#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
import scipy as sp
import tenseal as ts
# Application modules
from algorithm.activation_functions import _load_act_fn


# Abstract client
class FedHEONN_client:
    def __init__(self, f='logs', encrypted: bool=True, sparse: bool=True, context=None, bagging: bool=False, n_estimators: int=None):
        """Constructor method"""
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)
        self.encrypted = encrypted  # Encryption hyperparameter
        self.sparse    = sparse     # Sparse hyperparameter
        self.M         = []
        self.US        = []
        self.W         = None
        self.context   = context
        self.bagging = bagging
        self.n_estimators = n_estimators

    def _fit(self, X, d):
        # Number of data points (n)
        n = np.size(X, 1)

        # The bias is included as the first input (first row)
        Xp = np.insert(X, 0, np.ones(n), axis=0)

        # Inverse of the neural function
        inv_d = self.f_inv(d)

        # Derivative of the neural function
        der_d = self.fderiv(inv_d)

        if self.sparse:
            # Diagonal sparse matrix
            F_sparse = sp.sparse.spdiags(der_d, 0, der_d.size, der_d.size, format="csr")

            # Matrix on which the Singular Value Decomposition will be calculated later
            H = Xp @ F_sparse

            # Singular Value Decomposition of H
            U, S, _ = sp.linalg.svd(H, full_matrices=False)

            # Calculate M
            M = Xp @ (F_sparse @ (F_sparse @ inv_d.T))
            M = M.flatten()
        else:
            # Diagonal matrix
            F = np.diag(der_d)

            # Matrix on which the Singular Value Decomposition will be calculated later
            H = Xp @ F

            # Singular Value Decomposition of H
            U, S, _ = sp.linalg.svd(H, full_matrices=False)

            # Calculate M
            M = Xp @ (F @ (F @ inv_d))

        # If the encrypted option is selected then the M vector is encrypted
        if self.encrypted:
            M = ts.ckks_vector(self.context, M)

        return M, U @ np.diag(S)

    def _predict(self, X):
        # Number of variables (m) and data points (n)
        m, n = X.shape

        # Number of output neurons
        n_outputs = len(self.W)

        y = np.empty((0, n), float)

        # For each output neuron
        for o in range(0, n_outputs):
            # If the weights are encrypted then they are decrypted to get the performance results
            if self.encrypted:
                W = np.array((self.W[o]).decrypt())
            else:
                W = self.W[o]
            # Neural Network Simulation
            y = np.vstack((y, self.f(W.transpose() @ np.insert(X, 0, np.ones(n), axis=0))))

        return y

    def get_param(self):
        return self.M, self.US

    def set_weights(self, W):
        self.W = W

    def set_context(self, context):
        """Method that sets the tenseal context used in this client instance.

        Parameters
        ----------
        context : tenseal context
        """
        self.context = context

    def clean_client(self):
        """Method that resets models auxiliary matrix's and weights"""
        self.M         = []
        self.US        = []
        self.W         = None

    @staticmethod
    def bootstrap_sample(X, d):
        """Function to create bootstrap samples"""
        n_samples = X.shape[1]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[:, indices], d[indices, :]

    @staticmethod
    def _reshape(arr):
        return arr.reshape(len(arr), 1) if arr.ndim == 1 else arr

    @staticmethod
    def _preprocess(X, t):
        X = FedHEONN_client._reshape(X).T
        t = FedHEONN_client._reshape(t)
        return X, t

class FedHEONN_regressor(FedHEONN_client):
    """FedHEONN client for regression tasks"""
    def fit(self, X, t):
        # Transpose and reshape train data
        X, t = self._preprocess(X, t)

        # Determine sample and output length
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
        # Transpose test data
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

class FedHEONN_classifier(FedHEONN_client):
    """FedHEONN client for classification tasks"""
    def fit(self, X, t_onehot):
        # Transpose and reshape train data
        X, t_onehot = self._preprocess(X, t_onehot)

        # Determine sample and number of classes
        n, n_classes = t_onehot.shape

        # Transforms the one-hot encoding, mapping values (0, 1) to (0.05, 0.95) to avoid
        # the problem of the inverse of the activation function at the extremes
        t_onehot = t_onehot * 0.90 + 0.05

        # Bagging
        if self.bagging and self.n_estimators > 1:
            for i in range(self.n_estimators):
                M_e, US_e = [], []
                X_bag, t_bag = self.bootstrap_sample(X, t_onehot)
                # A model is generated for each class
                for o in range(0, n_classes):
                    M, US = self._fit(X_bag, t_bag[:, o])
                    M_e.append(M)
                    US_e.append(US)
                self.M.append(M_e)
                self.US.append(US_e)
        else:
            # A model is generated for each class
            for c in range(0, n_classes):
                M, US = self._fit(X, t_onehot[:, c])
                self.M.append(M)
                self.US.append(US)

    def predict(self, X):
        # Transpose test data
        X = self._reshape(X).T

        if self.bagging and self.n_estimators > 1:
            y = []
            W_orig = self.W
            for i in range(self.n_estimators):
                self.W = W_orig[i]
                y.append(np.apply_along_axis(lambda arr, value: np.abs(arr - value).argmin(), axis=0, arr=self._predict(X), value=0.95))
            y = np.array(y)
            y = (np.count_nonzero(y, axis=0) >= self.n_estimators / 2).astype(int)
            self.W = W_orig
        else:
            y_onehot = self._predict(X)
            y = np.apply_along_axis(lambda arr, value: np.abs(arr - value).argmin(), axis=0, arr=y_onehot, value=0.95)

        return y.T
