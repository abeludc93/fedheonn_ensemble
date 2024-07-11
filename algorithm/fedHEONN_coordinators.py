#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import time
import multiprocessing
import tempfile
import os
from itertools import repeat
# Third-party libraries
import numpy as np
import scipy as sp
from psutil import cpu_count
import tenseal as ts
# Application modules
from algorithm.activation_functions import _load_act_fn
from auxiliary.decorators import time_func
from auxiliary.logger import logger as log

class FedHEONN_coordinator:
    def __init__(self, f: str='logs', lam: float=0, encrypted: bool=True, sparse: bool=True, ensemble: {}=None,
                 parallel: bool=False):
        """Constructor method"""
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)
        self.lam        = lam       # Regularization hyperparameter
        self.encrypted  = encrypted # Encryption hyperparameter
        self.sparse     = sparse    # Sparse hyperparameter
        self.ensemble   = ensemble  # Ensemble methods set
        self.W          = []
        self.parallel   = parallel
        # Incremental learning matrix's
        self.M_glb      = []
        self.U_glb      = []
        self.S_glb      = []
        # Auxiliary list for attribute indexes
        self.idx_feats  = []

    @staticmethod
    def _aggregate(M_list, US_list, lam, sparse, encrypted, parallel=False, ctx_str=None):
        # Number of classes
        n_classes = len(M_list[0])
        n_clients = len(M_list)

        # Deserialize in case of encrypted and parallel configuration
        ctx = None
        if encrypted and parallel:
            ctx = FedHEONN_coordinator.load_context(ctx_str)
            for i in range(n_clients):
                M_list[i] = [ts.ckks_vector_from(ctx, m) for m in M_list[i]]


        # Optimal weights
        W_out = []

        # For each class the results of each client are aggregated
        for c in range(n_classes):

            # Initialization using the first element of the list
            M = M_list[0][c]
            US = US_list[0][c]

            M_rest = [item[c] for item in M_list[1:]]
            US_rest = [item[c] for item in US_list[1:]]

            if not (M_rest and US_rest):
                # Only one client
                U, S, _ = sp.linalg.svd(US, full_matrices=False)
            else:
                # Aggregation of M and US from the second client to the last
                for M_k, US_k in zip(M_rest, US_rest):
                    M = M + M_k
                    U, S, _ = sp.linalg.svd(np.concatenate((US_k, US), axis=1), full_matrices=False)
                    US = U @ np.diag(S)

            if sparse:
                I_ones = np.ones(np.size(S))
                I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format="csr")
                S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format="csr")
                aux2 = S_sparse * S_sparse + lam * I_sparse

                # Optimal weights: the order of the multiplications has been rearranged to optimize the speed
                if encrypted:
                    aux2 = aux2.toarray()
                    w = (M.matmul(U)).matmul((U @ np.linalg.pinv(aux2)).T)
                else:
                    aux2 = aux2.toarray()
                    w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M))
            else:
                # Optimal weights: the order of the multiplications has been rearranged to optimize the speed
                if encrypted:
                    w = (M.matmul(U)).matmul((U @ (np.diag(1 / (S * S + lam * (np.ones(np.size(S))))))).T)
                else:
                    w = U @ (np.diag(1 / (S * S + lam * (np.ones(np.size(S))))) @ (U.transpose() @ M))

            W_out.append(w.serialize() if encrypted and parallel else w)

        return W_out

    def send_weights(self):
        return self.W

    def calculate_idx_feats(self, n_estimators, n_features, p_features, b_features):
        assert n_estimators > 1
        assert p_features   > 0.0
        # No need to calculate index features if p_feat is 1.0 and no replacement is allowed (bootstrap)
        if self.ensemble and not (p_features == 1.0 and b_features is False):
            np.random.seed(42)
            self.idx_feats = []
            for i in range(n_estimators):
                self.idx_feats.append(np.sort(np.random.choice(n_features, size=int(n_features * p_features), replace=b_features)))

    def send_idx_feats(self):
        return self.idx_feats

    @time_func
    def aggregate(self, M_list, US_list):
        # Aggregates entire M&US lists at once

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:
            # Aggregate each estimators output
            n_estimators = len(US_list[0])
            n_clients = len(US_list)

            # Parallelized aggregation
            if self.parallel:
                ctx_str, ctx = None, None
                if self.encrypted:
                    ctx_str, ctx = FedHEONN_coordinator.save_context_from(M_list[0][0][0])
                M_base = []
                US_base = []
                for i in range(n_estimators):
                    M_base.append([M[i] for M in M_list])
                    US_base.append([US[i] for US in US_list])
                    if self.encrypted:
                        for j in range(n_clients):
                            M_base[i][j] = [m.serialize() for m in M_base[i][j]]
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                log.debug(f"\t\tDoing parallelized aggregation, number of estimators: {({n_estimators})}")
                pool = multiprocessing.Pool(processes=cpu)
                iterable = [[M_base[k], US_base[k], self.lam, self.sparse, self.encrypted, self.parallel, ctx_str]
                            for k in range(n_estimators)]
                # Blocks until ready, ordered results
                results = pool.starmap(FedHEONN_coordinator._aggregate, iterable)
                for w in results:
                    if self.encrypted:
                        self.W.append([ts.ckks_vector_from(ctx, w_item) for w_item in w])
                    else:
                        self.W.append(w)
                log.info(f"\t\tParallelized ({cpu}) aggregation done in: {time.perf_counter() - t_ini:.3f} s")
                if self.encrypted:
                    os.remove(ctx_str)
            else:
                for i in range(n_estimators):
                    M_base_lst = [M[i] for M in M_list]
                    US_base_lst = [US[i] for US in US_list]
                    self.W.append(FedHEONN_coordinator._aggregate(M_list=M_base_lst, US_list=US_base_lst, lam=self.lam,
                                                                  sparse=self.sparse, encrypted=self.encrypted))
        else:
            self.W = FedHEONN_coordinator._aggregate(M_list=M_list, US_list=US_list, lam=self.lam,
                                                     sparse=self.sparse, encrypted=self.encrypted)

    @time_func
    def aggregate_partial(self, M_list, US_list):
        # Aggregates partial M&US lists

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:
            # Aggregate each estimators output
            n_estimators = len(US_list[0])
            self.M_glb = [[] for i in range(n_estimators)] if not self.M_glb else self.M_glb
            self.U_glb = [[] for i in range(n_estimators)] if not self.U_glb else self.U_glb
            self.S_glb = [[] for i in range(n_estimators)] if not self.S_glb else self.S_glb
            for i in range(n_estimators):
                M_base_lst = [M[i] for M in M_list]
                US_base_lst = [US[i] for US in US_list]
                FedHEONN_coordinator._aggregate_partial(M_list=M_base_lst, US_list=US_base_lst,
                                                        M_glb=self.M_glb[i], U_glb=self.U_glb[i], S_glb=self.S_glb[i])
        else:
            FedHEONN_coordinator._aggregate_partial(M_list=M_list, US_list=US_list,
                                                    M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb)

    @staticmethod
    def _aggregate_partial(M_list, US_list, M_glb, U_glb, S_glb):
        # Aggregates partial M&US lists to the model

        # Number of classes
        nclasses = len(M_list[0])

        # Flag to represent an initial or incremental aggregation (no global M|U|S beforehand)
        init = False

        # For each class the results of each client are aggregated
        for c in range(nclasses):

            if not M_glb or init:
                init = True
                # Initialization using the first element of the list
                M  = M_list[0][c]
                US = US_list[0][c]
                M_rest  = [item[c] for item in M_list[1:]]
                US_rest = [item[c] for item in US_list[1:]]
            else:
                assert nclasses == len(M_glb)
                M = M_glb[c]
                US = U_glb[c] @ np.diag(S_glb[c])
                M_rest  = [item[c] for item in M_list[:]]
                US_rest = [item[c] for item in US_list[:]]


            if not (M_rest and US_rest):
                # Only one client
                U, S, _ = sp.linalg.svd(US, full_matrices=False)
            else:
                # Aggregation of M and US from the second client to the last
                for M_k, US_k in zip(M_rest, US_rest):
                    M = M + M_k
                    U, S, _ = sp.linalg.svd(np.concatenate((US_k, US),axis=1), full_matrices=False)
                    US = U @ np.diag(S)

            # Save contents
            if init:
                M_glb.append(M)
                U_glb.append(U)
                S_glb.append(S)
            else:
                M_glb[c] = M
                U_glb[c] = U
                S_glb[c] = S

    @time_func
    def calculate_weights(self):
        # Calculate weights
        self.W = []
        # Check for ensemble methods
        if self.ensemble and "bagging" in self.ensemble:
            # Calculate optimal weights for each estimator
            n_estimators = len(self.M_glb)
            for i in range(n_estimators):
                self.W.append(self._calculate_weights(M_glb=self.M_glb[i], U_glb=self.U_glb[i], S_glb=self.S_glb[i]))
        else:
            self.W = self._calculate_weights(M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb)

    def _calculate_weights(self, M_glb, U_glb, S_glb) -> []:
        """
        Method to calculate the optimal weights of the ONN model for the current M & US matrix's.
        """
        W_out = []

        # If there is model fitted data
        if M_glb and U_glb and S_glb:

            # Number of classes/outputs
            n_classes = len(M_glb)
            for c in range(n_classes):

                # For each output calculate optimal weights
                M = M_glb[c]
                U = U_glb[c]
                S = S_glb[c]

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
                W_out.append(w)

        return W_out

    def clean_coordinator(self):
        """Method that resets coordinators model auxiliary matrix's and weights"""
        self.W          = []
        # Incremental learning matrix's
        self.M_glb      = []
        self.U_glb      = []
        self.S_glb      = []

    @staticmethod
    def generate_ensemble_params():
        return {'bagging'}


    @staticmethod
    def load_context(ctx_str):
        print(f"Loading context from: ({ctx_str})")
        with open(ctx_str, "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

    @staticmethod
    def save_context_from(tenseal_vector):
        ctx = tenseal_vector.context()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_filename = tmp.name
        with open(tmp_filename, "wb") as ctx_32k:
            ctx_32k.write(ctx.serialize(save_public_key=True,
                                        save_secret_key=False,
                                        save_galois_keys=True,
                                        save_relin_keys=True))
        return tmp_filename, ctx