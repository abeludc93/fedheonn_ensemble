#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-party libraries
import numpy as np
import scipy as sp
# Application modules
from algorithm.activation_functions import _load_act_fn


class FedHEONN_coordinator:
    def __init__(self, f: str='logs', lam: float=0, encrypted: bool=True, sparse: bool=True, ensemble: {}=None ):
        """Constructor method"""
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)
        self.lam        = lam       # Regularization hyperparameter
        self.encrypted  = encrypted # Encryption hyperparameter
        self.sparse     = sparse    # Sparse hyperparameter
        self.ensemble   = ensemble  # Ensemble methods set
        self.W          = []
        # Incremental learning matrix's
        self.M_glb      = []
        self.U_glb      = []
        self.S_glb      = []

    def _aggregate(self, M_list, US_list):
        # Number of classes
        n_classes = len(M_list[0])

        # Optimal weights
        W_out = []

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

            W_out.append(w)

        return W_out

    def send_weights(self):
        return self.W

    def aggregate(self, M_list, US_list):
        # Aggregates entire M&US lists at once

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:
            # Aggregate each estimators output
            n_estimators = len(US_list[0])
            for i in range(n_estimators):
                M_base_lst = [M[i] for M in M_list]
                US_base_lst = [US[i] for US in US_list]
                self.W.append(self._aggregate(M_base_lst, US_base_lst))
        else:
            self.W = self._aggregate(M_list, US_list)

    def aggregate_partial(self, M_list, US_list):
        # Aggregates partial M&US lists to the model

        # Number of classes
        nclasses = len(M_list[0])

        # Flag to represent an initial or incremental aggregation (no global M|U|S beforehand)
        init = False

        # For each class the results of each client are aggregated
        for c in range(0, nclasses):

            if (not self.M_glb) or init:
                init = True
                # Initialization using the first element of the list
                M  = M_list[0][c]
                US = US_list[0][c]
                M_rest  = [item[c] for item in M_list[1:]]
                US_rest = [item[c] for item in US_list[1:]]
            else:
                assert nclasses == len(self.M_glb)
                M = self.M_glb[c]
                US = self.U_glb[c] @ np.diag(self.S_glb[c])
                M_rest  = [item[c] for item in M_list[:]]
                US_rest = [item[c] for item in US_list[:]]

            # Aggregation of M and US
            for M_k, US_k in zip(M_rest, US_rest):
                M = M + M_k
                U, S, _ = sp.linalg.svd(np.concatenate((US_k, US),axis=1), full_matrices=False)
                US = U @ np.diag(S)

            # Save contents
            if init:
                self.M_glb.append(M)
                self.U_glb.append(U)
                self.S_glb.append(S)
            else:
                self.M_glb[c] = M
                self.U_glb[c] = U
                self.S_glb[c] = S

    def calculate_weights(self):
        """
        Method to calculate the optimal weights of the ONN model for the current M & US matrix's.
        """

        # Reset optimal weights
        self.W = []

        # If there is model fitted data
        if self.M_glb and self.U_glb and self.S_glb:

            # Number of classes/outputs
            nclasses = len(self.M_glb)
            for c in range(0, nclasses):

                # For each output calculate optimal weights
                M = self.M_glb[c]
                U = self.U_glb[c]
                S = self.S_glb[c]

                if self.sparse:
                    I_ones = np.ones(np.size(S))
                    I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format = "csr")
                    S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format = "csr")
                    aux2 = S_sparse * S_sparse + self.lam * I_sparse
                    # Optimal weights: the order of the matrix and vector multiplications have been rearranged to optimize the speed
                    if self.encrypted:
                        aux2 = aux2.toarray()
                        w = (M.matmul(U)).matmul((U @ np.linalg.pinv(aux2)).T)
                    else:
                        aux2 = aux2.toarray()
                        w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M))
                else:
                    # Optimal weights: the order of the matrix and vector multiplications have been rearranged to optimize the speed
                    if self.encrypted:
                        w = (M.matmul(U)).matmul((U @ (np.diag(1/(S*S+self.lam*(np.ones(np.size(S))))))).T)
                    else:
                        w = U @ (np.diag(1/(S*S+self.lam*(np.ones(np.size(S))))) @ (U.transpose() @ M))

                # Append optimal weights
                self.W.append(w)