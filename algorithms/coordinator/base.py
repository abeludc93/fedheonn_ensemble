#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
import scipy as sp
# Application modules
from algorithms.utils import _load_act_fn


class FedHEONN_coordinator:
    """FedHEONN coordinator class.

    Parameters
    ----------
    f : {'logs','relu','lin'}, default='logs'
        Activation function for the neurons of the clients.

    lam: regularization term, default=0
        Strength of the L2 regularization term.

    encrypted: bool, default=True
        Indicates if homomorphic encryption is used in the clients or not.

    sparse: bool, default=True
        Indicates whether sparse matrices will be used during the aggregation process.
        Recommended for large data sets.

    Attributes
    ----------
        lam : float
            Strength of the L2 regularization term.
        encrypted : bool
            Specifies whether the clients are using homomorphic encryption or not.
        sparse : bool
            Specifies whether the coordinator is using internal sparse matrices or not.
        W : list of weights of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights includes the bias as first element.
    """

    def __init__(self, f='logs', lam=0, encrypted=True, sparse=True):
        """Constructor method"""
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)
        self.lam = lam  # Regularization hyperparameter
        self.encrypted = encrypted  # Encryption hyperparameter
        self.sparse = sparse  # Sparse hyperparameter
        self.W = []

    def aggregate(self, M_list, US_list):
        """Method to aggregate the models of the clients in the federated learning.

        Parameters
        ----------
        M_list : list of shape (n_clients,)
            The list of m terms computed previously by a a set o clients.

        US_list : list of shape (n_clients,)
            The list of U*S terms computed previously by a a set o clients.
        """
        # Number of classes
        n_classes = len(M_list[0])

        # For each class the results of each client are aggregated
        for c in range(0, n_classes):

            # Initialization using the first element of the list
            M = M_list[0][c]
            US = US_list[0][c]

            M_rest = [item[c] for item in M_list[1:]]
            US_rest = [item[c] for item in US_list[1:]]

            # Aggregation of M and US from the second client to the last
            for M_k, US_k in zip(M_rest, US_rest):
                M = M + M_k
                U, S, _ = sp.linalg.svd(np.concatenate((US_k, US), axis=1), full_matrices=False)
                US = U @ np.diag(S)

            if self.sparse:
                I_ones = np.ones(np.size(S))
                I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format="csr")
                S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format="csr")
                aux2 = S_sparse * S_sparse + self.lam * I_sparse

                # Optimal weights: the order of the multiplications has been rearranged to optimize the speed
                if self.encrypted:
                    aux2 = aux2.toarray()
                    w = (M.matmul(U)).matmul((U @ np.linalg.pinv(aux2)).T)
                else:
                    aux2 = aux2.toarray()
                    w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M))
            else:
                # Optimal weights: the order of the multiplications has been rearranged to optimize the speed
                if self.encrypted:
                    w = (M.matmul(U)).matmul((U @ (np.diag(1 / (S * S + self.lam * (np.ones(np.size(S))))))).T)
                else:
                    w = U @ (np.diag(1 / (S * S + self.lam * (np.ones(np.size(S))))) @ (U.transpose() @ M))

            self.W.append(w)

    def send_weights(self):
        """ Method to get the weights of the aggregated model

        Returns
        -------
        W : list of ndarray of shape (n_outputs,)
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights include the bias as first element.
        """
        return self.W
