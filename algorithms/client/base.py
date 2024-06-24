#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
import scipy as sp
import tenseal as ts
# Application modules
from algorithms.utils import _load_act_fn


class FedHEONN_client:
    """FedHEONN client class.

    Parameters
    ----------
    f : {'logs','relu','linear'}, default='logs'
        Activation function for the neurons.

    encrypted: bool, default=True
        Indicates if homomorphic encryption is used in the client or not.

    sparse: bool, default=True
        Indicates whether internal sparse matrices will be used during the training process.
        Recommended for large data sets.

    Attributes
    ----------
        encrypted : bool
            Specifies whether the client is using homomorphic encryption or not.
        sparse : bool
            Specifies whether the client is using internal sparse matrices or not.
        M : list of m vectors of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the m vector associated with the ith output neuron.
        US : list of U*S matrices of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the U*S matrices associated with the ith output neuron.
        W : list of weights of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights include the bias as first element.
    """

    def __init__(self,  f='logs', encrypted: bool = True, sparse: bool = True, context=None, bagging: bool = False,
                 n_estimators: int = None):
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
        """Private method to fit the model to data matrix X and target(s) d.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.

        d : ndarray of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        m  : auxiliar matrix for federated/incremental learning
        US : auxiliar matrix for federated/incremental learning
        """
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

            M = Xp @ (F_sparse @ (F_sparse @ inv_d.T))
            M = M.flatten()

        else:
            # Diagonal matrix
            F = np.diag(der_d)

            # Matrix on which the Singular Value Decomposition will be calculated later
            H = Xp @ F

            # Singular Value Decomposition of H
            U, S, _ = sp.linalg.svd(H, full_matrices=False)

            M = Xp @ (F @ (F @ inv_d))

        # If the encrypted option is selected then the M vector is encrypted
        if self.encrypted:
            M = ts.ckks_vector(self.context, M)

        return M, U @ np.diag(S)

    def _predict(self, X):
        """Predict using FedHEONN model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
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
        """Method that provides the values of m and U*S"""
        return self.M, self.US

    def set_weights(self, W):
        """Method that set the values of the weights.

        Parameters
        ----------
        W : list of ndarray of shape (n_outputs,)
            Each element of the list is an array with the weights associated to the corresponding output neuron.
        """
        self.W = W

    def set_context(self, context):
        """Method that set the tenseal context used in this client instance.

        Parameters
        ----------
        context : tenseal context
        """
        self.context = context

    def clean_client(self):
        self.M         = []
        self.US        = []
        self.W         = None

    @staticmethod
    def bootstrap_sample(X, d):
        """Function to create bootstrap samples"""
        n_samples = X.shape[1]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[:, indices], d[indices, :]
