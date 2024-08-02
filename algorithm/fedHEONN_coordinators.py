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
        # Auxiliary variable
        self.ctx_str    = None
        self.ctx_persist= False

    def __del__(self):
        if getattr(self, 'ctx_str', None) and not getattr(self, 'ctx_persist', False):
            FedHEONN_coordinator.delete_context(self.ctx_str)

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

    def aggregate(self, M_list, US_list):
        # Aggregates entire M&US lists at once

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:

            # Aggregate each estimators output
            n_estimators = len(US_list[0])
            n_clients = len(US_list)

            # Aggregation in parallel
            if self.parallel:
                # Get number of physical cores and start timer
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                # Number of pool processes
                n_processes = min(cpu, n_estimators)
                log.debug(f"\t\tDoing parallelized aggregation, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")

                # Save tenSEAL context to temp file for later use in multiprocessing
                ctx = None
                if self.encrypted:
                    if self.ctx_str is None:
                        # Save and retrieve context
                        self.ctx_str, ctx = FedHEONN_coordinator.save_context_from(M_list[0][0][0])
                    else:
                        # Already saved, no need to load from file, just retrieve it
                        ctx = M_list[0][0][0].context()

                # Prepare data for multiprocessing:
                # Arrange list of estimators M&US[i] matrix's
                M_base = []
                US_base = []
                for i in range(n_estimators):
                    M_base.append([M[i] for M in M_list])
                    US_base.append([US[i] for US in US_list])
                    # Serialize tenSEAL CKKS vectors for later use in multiprocessing
                    if self.encrypted:
                        for j in range(n_clients):
                            M_base[i][j] = [m.serialize() for m in M_base[i][j]]
                # Split data in iterable groups - one for each process
                M_base_groups  = self.split_list(M_base, n_processes)
                US_base_groups = self.split_list(US_base, n_processes)
                iterable = [[M_base_groups[k], US_base_groups[k], self.lam, self.sparse, self.encrypted, self.parallel, self.ctx_str]
                            for k in range(n_processes)]
                log.debug(f"\t\tPreparing data for parallel process: {time.perf_counter()-t_ini:.3f} s")

                # Multiprocessing POOL
                t_ini = time.perf_counter()
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Blocks until ready, ordered results
                    results = pool.starmap(FedHEONN_coordinator._aggregate_wrapper, iterable)
                    for w in results:
                        # If encrypted, we need to deserialize returned data
                        if self.encrypted:
                            for k in range(len(w)):
                                w[k] = [ts.ckks_vector_from(ctx, w_item) for w_item in w[k]]
                            self.W.extend(w)
                        else:
                            self.W.extend(w)
                log.debug(f"\t\tParallelized ({n_processes}) aggregation done in: {time.perf_counter() - t_ini:.3f} s")

                # Delete temporary file containing the public tenSEAL context
                if self.encrypted and not self.ctx_persist:
                    FedHEONN_coordinator.delete_context(self.ctx_str)
                    self.ctx_str = None

            else:
                # Aggregating in series (with bagging)
                for i in range(n_estimators):
                    M_base_lst = [M[i] for M in M_list]
                    US_base_lst = [US[i] for US in US_list]
                    self.W.append(FedHEONN_coordinator._aggregate(M_list=M_base_lst, US_list=US_base_lst, lam=self.lam,
                                                                  sparse=self.sparse, encrypted=self.encrypted))
        else:
            # Aggregating in series (without bagging)
            self.W = FedHEONN_coordinator._aggregate(M_list=M_list, US_list=US_list, lam=self.lam,
                                                     sparse=self.sparse, encrypted=self.encrypted)

    @staticmethod
    def _aggregate_wrapper(M_list, US_list, lam, sparse, encrypted, parallel=False, ctx_str=None):
        # Number of estimators and matrix's per group
        n_estimators, n_groups = len(M_list[0]), len(M_list)

        # Load context in case of encrypted and parallel configuration
        ctx = None
        if encrypted and parallel:
            ctx = FedHEONN_coordinator.load_context(ctx_str)

        # Process each group of M_list&US_list data
        W = []
        for i in range(n_groups):
            # De-serialize tenSEAL CKKS encrypted vectors for multiprocessing purposes
            if encrypted and parallel:
                for j in range(n_estimators):
                    M_list[i][j] = [ts.ckks_vector_from(ctx, m) for m in M_list[i][j]]

            # Aggregate and append returned optimal weights
            w = FedHEONN_coordinator._aggregate(M_list[i], US_list[i], lam, sparse, encrypted)
            if encrypted and parallel:
                # Serialize returned data (CKKS vectors) from pool processes
                W.append([vector.serialize() for vector in w])
            else:
                # Append plain results
                W.append(w)

        return W

    def aggregate_partial(self, M_list, US_list):
        # Aggregates partial M&US lists of matrix's

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:

            # Aggregate each estimators output
            n_estimators = len(US_list[0])
            self.M_glb = [[] for _ in range(n_estimators)] if not self.M_glb else self.M_glb
            self.U_glb = [[] for _ in range(n_estimators)] if not self.U_glb else self.U_glb
            self.S_glb = [[] for _ in range(n_estimators)] if not self.S_glb else self.S_glb

            # Partial aggregation in series (with bagging)
            if not self.parallel:

                for i in range(n_estimators):
                    M_base_lst = [M[i] for M in M_list]
                    US_base_lst = [US[i] for US in US_list]
                    FedHEONN_coordinator._aggregate_partial(M_list=M_base_lst, US_list=US_base_lst,
                                                            M_glb=self.M_glb[i], U_glb=self.U_glb[i],
                                                            S_glb=self.S_glb[i])
            # Partial aggregation in parallel (with bagging)
            else:
                # Get number of physical cores and start timer
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                # Number of pool processes
                n_processes = min(cpu, n_estimators)
                log.debug(f"\t\tDoing parallelized partial aggregation, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")

                # If M is encrypted data, parallelize only the partial SVD aggregation
                if self.encrypted:

                    # Prepare data for multiprocessing (iterable_svd) and aggregate M data IN SERIES
                    iterable_svd = []
                    for i in range(n_estimators):
                        # Collect M data and aggregate (parallel_option is 'm')
                        M_base_lst = [M[i] for M in M_list]
                        FedHEONN_coordinator._aggregate_partial(M_list=M_base_lst, US_list=None,
                                                                M_glb=self.M_glb[i], U_glb=None, S_glb=None,
                                                                parallel_option='m')
                        # Prepare multiprocess iterable with U&S data (parallel_option is 'svd')
                        iterable_svd.append([None, [US[i] for US in US_list], None, self.U_glb[i], self.S_glb[i], 'svd'])

                    # Multiprocessing POOL
                    with multiprocessing.Pool(processes=n_processes) as pool:
                        # Blocks until ready, ordered results
                        tuple_results = pool.starmap(FedHEONN_coordinator._aggregate_partial, iterable_svd)
                        # Place returned results in instance's attributes accordingly
                        for i in range(n_estimators):
                            _, U, S = tuple_results[i]
                            self.U_glb[i], self.S_glb[i] = U, S

                # If M is plain data, parallelize the whole process
                else:
                    # Prepare data for multiprocessing (iterable)
                    iterable = []
                    for i in range(n_estimators):
                        M_base_lst = [M[i] for M in M_list]
                        US_base_lst = [US[i] for US in US_list]
                        iterable.append([M_base_lst, US_base_lst, self.M_glb[i], self.U_glb[i], self.S_glb[i]])

                    # Multiprocessing POOL
                    with multiprocessing.Pool(processes=n_processes) as pool:
                        # Blocks until ready, ordered results
                        tuple_results = pool.starmap(FedHEONN_coordinator._aggregate_partial, iterable)
                        # Place returned results in instance's attributes accordingly
                        for i in range(n_estimators):
                            M, U, S = tuple_results[i]
                            self.M_glb[i], self.U_glb[i], self.S_glb[i] = M, U, S

                log.debug(f"\t\tParallelized ({n_processes}) partial aggregation done in: {time.perf_counter()-t_ini:.3f} s")
        else:
            # Partial aggregation in series (without bagging)
            FedHEONN_coordinator._aggregate_partial(M_list=M_list, US_list=US_list,
                                                    M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb)

    @staticmethod
    def _aggregate_partial(M_list, US_list, M_glb, U_glb, S_glb, parallel_option='m_svd'):
        # Parallel_option == 'm_svd' | 'svd ' | 'm'
        # 'm_svd':  aggregates all information, that is, sums M data and process US
        # 'm':      aggregates only M data
        # 'svd':    aggregates only US data

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

        # Return data (only for multiprocessing purposes, because pool processes can't modify the original matrix's)
        return M_glb, U_glb, S_glb

    def calculate_weights(self):
        # Calculate weights for the current coordinators M_glb, U_glb and S_glb data

        # Clean previous weights and assert there is data
        assert self.M_glb; assert self.U_glb; assert self.S_glb
        self.W = []

        # Check for ensemble methods of aggregation
        if self.ensemble and "bagging" in self.ensemble:

            # Calculate optimal weights for each estimator
            n_estimators = len(self.M_glb)

            # Calculate weights in parallel (with bagging)
            if self.parallel:
                # Get number of physical cores and start timer
                t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
                # Number of pool processes
                n_processes = min(cpu, n_estimators)
                log.debug(f"\t\tDoing parallelized calculate_weights, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")

                # Save tenSEAL context to temp file and serialize CKKS vectors for later use in multiprocessing
                ctx, M_glb_serialized = None, []
                if self.encrypted:
                    if self.ctx_str is None:
                        # Save and retrieve context
                        self.ctx_str, ctx = FedHEONN_coordinator.save_context_from(self.M_glb[0][0])
                    else:
                        # Already saved, no need to load context from file, just retrieve it from sample CKKS vector
                        ctx = self.M_glb[0][0].context()
                    for i in range(n_estimators):
                        M_glb_serialized.append([m.serialize() for m in self.M_glb[i]])

                # Prepare data for multiprocessing:
                # Split data in iterable groups - one for each process
                M_groups = self.split_list(M_glb_serialized if self.encrypted else self.M_glb, n_processes)
                U_groups = self.split_list(self.U_glb, n_processes)
                S_groups = self.split_list(self.S_glb, n_processes)
                iterable = [[M_groups[k], U_groups[k], S_groups[k], self.lam, self.sparse, self.encrypted, self.parallel, self.ctx_str]
                            for k in range(n_processes)]
                log.debug(f"\t\tPreparing data for parallel process: {time.perf_counter() - t_ini:.3f} s")

                # Multiprocessing POOL
                t_ini = time.perf_counter()
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Blocks until ready, ordered results
                    results = pool.starmap(FedHEONN_coordinator._calculate_weights_wrapper, iterable)
                    for w in results:
                        # If encrypted, we need to deserialize returned data
                        if self.encrypted:
                            for k in range(len(w)):
                                w[k] = [ts.ckks_vector_from(ctx, w_item) for w_item in w[k]]
                            self.W.extend(w)
                        else:
                            self.W.extend(w)
                log.debug(f"\t\tParallelized ({n_processes}) calculate_weights done in: {time.perf_counter() - t_ini:.3f} s")

            # Calculate weights in series (with bagging)
            else:
                for i in range(n_estimators):
                    self.W.append(self._calculate_weights(M_glb=self.M_glb[i], U_glb=self.U_glb[i], S_glb=self.S_glb[i],
                                                          sparse=self.sparse, lam=self.lam, encrypted=self.encrypted))
        else:
            # Calculate weights in series (without bagging)
            self.W = self._calculate_weights(M_glb=self.M_glb, U_glb=self.U_glb, S_glb=self.S_glb,
                                             sparse=self.sparse, lam=self.lam, encrypted=self.encrypted)

    @staticmethod
    def _calculate_weights_wrapper(M_list, U_list, S_list, lam, sparse, encrypted, parallel=False, ctx_str=None):
        # Number of estimators and matrix's per group
        n_estimators, n_groups = len(M_list[0]), len(M_list)

        # Load context in case of encrypted and parallel configuration
        ctx = None
        if encrypted and parallel:
            ctx = FedHEONN_coordinator.load_context(ctx_str)

        # Process each group of M&U&S list data
        W = []
        for i in range(n_groups):
            # De-serialize encrypted CKKS vectors in case of encrypted-multiprocessing
            if encrypted and parallel:
                M_list[i] = [ts.ckks_vector_from(ctx, m) for m in M_list[i]]

            # Aggregate and append returned optimal weights
            w = FedHEONN_coordinator._calculate_weights(M_list[i], U_list[i], S_list[i], lam, sparse, encrypted)
            if encrypted and parallel:
                # Serialize returned data (CKKS vectors) from pool processes
                W.append([vector.serialize() for vector in w])
            else:
                # Append plain results
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
        # Possible saved context
        if self.ctx_str is not None:
            FedHEONN_coordinator.delete_context(self.ctx_str)
            self.ctx_str = None

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
                idx_lst = np.sort(np.random.choice(n_features, size=int(n_features * p_features), replace=b_features)).tolist()
                self.idx_feats.append(idx_lst)

    def send_idx_feats(self):
        return self.idx_feats

    def get_ctx_str(self):
        return self.ctx_str

    def set_ctx_str(self, ctx_str):
        self.ctx_str = ctx_str

    def set_activation_functions(self, f:str):
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)

    def get_parameters(self):
        return {"f": self.f.__name__,
                "lam": self.lam,
                "encrypted": self.encrypted,
                "sparse": self.sparse,
                "bagging": self.ensemble is not None and "bagging" in self.ensemble,
                "parallel": self.parallel, "ctx_str": self.ctx_str}

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
        log.info(f"Saved tenSEAL context in: ({tmp_filename}) - {os.path.getsize(tmp_filename) / (1024 * 1024):.2f} MB")
        return tmp_filename, ctx

    @staticmethod
    def delete_context(ctx_str):
        if ctx_str and os.path.isfile(ctx_str):
            log.info(f"\t\tDeleting temporary file: {ctx_str}")
            os.remove(ctx_str)
        else:
            log.warn(f"\t\tCouldn't delete temporary file (empty or not found): {ctx_str}")

    @staticmethod
    def split_list(input_list, num_groups):
        # Split list into num_groups of equal or similar size
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