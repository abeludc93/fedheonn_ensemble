#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import time
import multiprocessing
import tempfile
import os
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
    def _aggregate(M_list, US_list, lam, sparse, encrypted):
        # Number of classes
        n_classes = len(M_list[0])

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

            W_out.append(w)

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
                t = time.perf_counter()
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
                log.debug(f"\t\tDoing parallelized aggregation, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")
                n_processes = min(cpu, n_estimators)
                M_base_groups  = self.split_list(M_base, n_processes)
                US_base_groups = self.split_list(US_base, n_processes)
                iterable = [[M_base_groups[k], US_base_groups[k], self.lam, self.sparse, self.encrypted, self.parallel, ctx_str]
                            for k in range(n_processes)]
                print(f"Preparing data for parallel process: {time.perf_counter()-t:.3f} s")
                t = time.perf_counter()
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Blocks until ready, ordered results
                    results = pool.starmap(FedHEONN_coordinator._aggregate_wrapper, iterable)
                    for w in results:
                        if self.encrypted:
                            for k in range(len(w)):
                                w[k] = [ts.ckks_vector_from(ctx, w_item) for w_item in w[k]]
                            self.W.extend(w)
                        else:
                            self.W.extend(w)
                log.info(f"\t\tParallelized ({n_processes}) aggregation done in: {time.perf_counter() - t_ini:.3f} s")
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

    @staticmethod
    def _aggregate_wrapper(M_list, US_list, lam, sparse, encrypted, parallel=False, ctx_str=None):
        n_estimators = len(M_list[0])
        # Load context in case of encrypted and parallel configuration
        ctx = None
        if encrypted and parallel:
            ctx = FedHEONN_coordinator.load_context(ctx_str)
        # Process each group
        W = []
        for i in range(len(M_list)):
            # De-serialize encrypted vectors in case of enc-multiprocessing
            if encrypted and parallel:
                for j in range(n_estimators):
                    M_list[i][j] = [ts.ckks_vector_from(ctx, m) for m in M_list[i][j]]
            w = FedHEONN_coordinator._aggregate(M_list[i], US_list[i], lam, sparse, encrypted)
            # Serialize again
            if encrypted and parallel:
                W.append([vector.serialize() for vector in w])
            else:
                W.append(w)

        return W

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

            if not self.parallel:

                for i in range(n_estimators):
                    M_base_lst = [M[i] for M in M_list]
                    US_base_lst = [US[i] for US in US_list]
                    FedHEONN_coordinator._aggregate_partial(M_list=M_base_lst, US_list=US_base_lst,
                                                            M_glb=self.M_glb[i], U_glb=self.U_glb[i], S_glb=self.S_glb[i])
            else:
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                log.debug(f"\t\tDoing parallelized partial aggregation, number of estimators: {({n_estimators})}, "
                          f"cpu-cores: {cpu}")
                n_processes = min(cpu, n_estimators)

                if self.encrypted:
                    iterable_svd = []
                    for i in range(n_estimators):
                        M_base_lst = [M[i] for M in M_list]
                        FedHEONN_coordinator._aggregate_partial(M_list=M_base_lst, US_list=None,
                                                                M_glb=self.M_glb[i], U_glb=None, S_glb=None,
                                                                parallel_option='m')

                        iterable_svd.append([None, [US[i] for US in US_list], None, self.U_glb[i], self.S_glb[i], 'svd'])

                    with multiprocessing.Pool(processes=n_processes) as pool:
                        # Blocks until ready, ordered results
                        tuple_results = pool.starmap(FedHEONN_coordinator._aggregate_partial, iterable_svd)
                    for i in range(n_estimators):
                        _, U, S = tuple_results[i]
                        self.U_glb[i], self.S_glb[i] = U, S
                    log.info(f"\t\tParallelized ({n_processes}) partial aggregation done in: {time.perf_counter()-t_ini:.3f} s")
                else:
                    iterable = []
                    for i in range(n_estimators):
                        M_base_lst = [M[i] for M in M_list]
                        US_base_lst = [US[i] for US in US_list]
                        iterable.append([M_base_lst, US_base_lst, self.M_glb[i], self.U_glb[i], self.S_glb[i]])
                    with multiprocessing.Pool(processes=n_processes) as pool:
                        # Blocks until ready, ordered results
                        tuple_results = pool.starmap(FedHEONN_coordinator._aggregate_partial, iterable)
                    for i in range(n_estimators):
                        M, U, S = tuple_results[i]
                        self.M_glb[i], self.U_glb[i], self.S_glb[i] = M, U, S

                    log.info(f"\t\tParallelized ({n_processes}) partial aggregation done in: {time.perf_counter()-t_ini:.3f} s")
        else:
            FedHEONN_coordinator._aggregate_partial(M_list=M_list, US_list=US_list,
                                                    M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb)

    @staticmethod
    def _aggregate_partial(M_list, US_list, M_glb, U_glb, S_glb, parallel_option='m_svd'):
        # Parallel_option == 'm_svd' | 'svd ' | 'm'

        # Aggregates partial M&US lists to the model

        # Number of classes
        n_classes = len(M_list[0]) if 'm' in parallel_option else len(US_list[0])

        # Flag to represent an initial or incremental aggregation (no global M|U|S beforehand)
        init = False

        # For each class the results of each client are aggregated
        for c in range(n_classes):

            if not (M_glb if 'm' in parallel_option else U_glb) or init:
                init = True
                # Initialization using the first element of the list
                if 'svd' in parallel_option:
                    US = US_list[0][c]
                    US_rest = [item[c] for item in US_list[1:]]
                if 'm' in parallel_option:
                    M = M_list[0][c]
                    M_rest = [item[c] for item in M_list[1:]]
            else:
                assert n_classes == len(M_glb) if 'm' in parallel_option else len(U_glb)
                if 'svd' in parallel_option:
                    US = U_glb[c] @ np.diag(S_glb[c])
                    US_rest = [item[c] for item in US_list[:]]
                if 'm' in parallel_option:
                    M = M_glb[c]
                    M_rest = [item[c] for item in M_list[:]]

            if 'svd' in parallel_option and not US_rest:
                # Only one client
                U, S, _ = sp.linalg.svd(US, full_matrices=False)
            else:
                # Aggregation of M and US from the second client to the last
                if 'svd' in parallel_option:
                    for US_k in US_rest:
                        U, S, _ = sp.linalg.svd(np.concatenate((US_k, US),axis=1), full_matrices=False)
                        US = U @ np.diag(S)
                if 'm' in parallel_option:
                    for M_k in M_rest:
                        M = M + M_k

            # Save contents
            if init:
                if 'm' in parallel_option:
                    M_glb.append(M)
                if 'svd' in parallel_option:
                    U_glb.append(U)
                    S_glb.append(S)
            else:
                if 'm' in parallel_option:
                    M_glb[c] = M
                if 'svd' in parallel_option:
                    U_glb[c] = U
                    S_glb[c] = S

        return M_glb, U_glb, S_glb


    @time_func
    def calculate_weights(self):
        # Calculate weights
        self.W = []
        # Check for ensemble methods
        if self.ensemble and "bagging" in self.ensemble:
            assert self.M_glb; assert self.U_glb; assert self.S_glb

            # Calculate optimal weights for each estimator
            n_estimators = len(self.M_glb)
            if self.parallel:
                t = time.perf_counter()
                ctx_str, ctx = None, None
                if self.encrypted: # TODO: check for previously saved context
                    ctx_str, ctx = FedHEONN_coordinator.save_context_from(self.M_glb[0][0])
                    for i in range(n_estimators):
                        self.M_glb[i] = [m.serialize() for m in self.M_glb[i]]
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                log.debug(f"\t\tDoing parallelized calculate_weights, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")
                n_processes = min(cpu, n_estimators)
                M_groups = self.split_list(self.M_glb, n_processes)
                U_groups = self.split_list(self.U_glb, n_processes)
                S_groups = self.split_list(self.S_glb, n_processes)
                iterable = [[M_groups[k], U_groups[k], S_groups[k], self.lam, self.sparse, self.encrypted, self.parallel, ctx_str]
                            for k in range(n_processes)]
                print(f"Preparing data for parallel process: {time.perf_counter() - t:.3f} s")
                t = time.perf_counter()
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Blocks until ready, ordered results
                    results = pool.starmap(FedHEONN_coordinator._calculate_weights_wrapper, iterable)
                    for w in results:
                        if self.encrypted:
                            for k in range(len(w)):
                                w[k] = [ts.ckks_vector_from(ctx, w_item) for w_item in w[k]]
                            self.W.extend(w)
                        else:
                            self.W.extend(w)
                log.info(f"\t\tParallelized ({n_processes}) calculate_weights done in: {time.perf_counter() - t_ini:.3f} s")
                if self.encrypted:
                    os.remove(ctx_str)
            else:
                for i in range(n_estimators):
                    self.W.append(self._calculate_weights(M_glb=self.M_glb[i], U_glb=self.U_glb[i], S_glb=self.S_glb[i],
                                                          sparse=self.sparse, lam=self.lam, encrypted=self.encrypted))
        else:
            assert self.M_glb; assert self.U_glb; assert self.S_glb

            self.W = self._calculate_weights(M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb,
                                             sparse=self.sparse, lam=self.lam, encrypted=self.encrypted)

    @staticmethod
    def _calculate_weights_wrapper(M_list, U_list, S_list, lam, sparse, encrypted, parallel=False, ctx_str=None):
        # Number of estimators
        n_estimators, n_groups = len(M_list[0]), len(M_list)
        # Load context in case of encrypted and parallel configuration
        ctx = None
        if encrypted and parallel:
            ctx = FedHEONN_coordinator.load_context(ctx_str)
        # Process each group
        W = []
        for i in range(n_groups):
            # De-serialize encrypted vectors in case of encrypted-multiprocessing
            if encrypted and parallel:
                M_list[i] = [ts.ckks_vector_from(ctx, m) for m in M_list[i]]
            w = FedHEONN_coordinator._calculate_weights(M_list[i], U_list[i], S_list[i], lam, sparse, encrypted)
            # Serialize again
            if encrypted and parallel:
                W.append([vector.serialize() for vector in w])
            else:
                W.append(w)

        return W


    @staticmethod
    def _calculate_weights(M_glb, U_glb, S_glb, lam, sparse, encrypted) -> []:
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

                if sparse:
                    I_ones = np.ones(np.size(S))
                    I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format = "csr")
                    S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format = "csr")
                    aux2 = S_sparse * S_sparse + lam * I_sparse
                    # Optimal weights: the order of the matrix and vector multiplications have been rearranged to optimize the speed
                    if encrypted:
                        aux2 = aux2.toarray()
                        w = (M.matmul(U)).matmul((U @ np.linalg.pinv(aux2)).T)
                    else:
                        aux2 = aux2.toarray()
                        w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M))
                else:
                    # Optimal weights: the order of the matrix and vector multiplications have been rearranged to optimize the speed
                    if encrypted:
                        w = (M.matmul(U)).matmul((U @ (np.diag(1/(S*S+lam*(np.ones(np.size(S))))))).T)
                    else:
                        w = U @ (np.diag(1/(S*S+lam*(np.ones(np.size(S))))) @ (U.transpose() @ M))

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
        log.info(f"Loading context from: ({ctx_str})")
        with open(ctx_str, "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

    @staticmethod
    def save_context_from(tenseal_vector):
        ctx = tenseal_vector.context()
        tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
        tmp_filename = tmp.name
        with open(tmp_filename, "wb") as ctx_32k:
            ctx_32k.write(ctx.serialize(save_public_key=True,
                                        save_secret_key=False,
                                        save_galois_keys=True,
                                        save_relin_keys=True))
        return tmp_filename, ctx

    @staticmethod
    def split_list(input_list, num_groups):

        list_len = len(input_list)
        base_size = list_len // num_groups
        extra_elements = list_len % num_groups

        sublists = []
        start = 0
        for i in range(num_groups):
            sublist_size = base_size + (1 if i < extra_elements else 0)
            sublists.append(input_list[start:start + sublist_size])
            start += sublist_size

        return sublists