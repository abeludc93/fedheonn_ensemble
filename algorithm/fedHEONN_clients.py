#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import multiprocessing
from psutil import cpu_count
from itertools import repeat
import time
import os
import tempfile
# Third-party libraries
import numpy as np
import scipy as sp
import tenseal as ts
# Application modules
from algorithm.activation_functions import _load_act_fn
from auxiliary.decorators import time_func
from auxiliary.logger import logger as log

# Abstract client
class FedHEONN_client:

    def __init__(self, f='logs', encrypted: bool=True, sparse: bool=True, context: ts.Context=None, ensemble: {}=None,
                 parallel: bool=False):
        """Constructor method"""
        self.f, self.f_inv, self.fderiv = _load_act_fn(f)
        self.encrypted  = encrypted # Encryption hyperparameter
        self.sparse     = sparse    # Sparse hyperparameter
        self.ensemble   = ensemble  # Ensemble methods dictionary
        self.M          = []
        self.US         = []
        self.W          = None
        self.context    = context
        self.idx_feats  = []
        self.parallel   = parallel

    @staticmethod
    def _fit(X, d, f_inv, fderiv, sparse, encrypted, ts_context=None):

        # Number of data points (n)
        n = np.size(X, 1)
        # The bias is included as the first input (first row)
        Xp = np.insert(X, 0, np.ones(n), axis=0)
        # Inverse of the neural function
        inv_d = f_inv(d)
        # Derivative of the neural function
        der_d = fderiv(inv_d)

        if sparse:
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
        if encrypted and ts_context is not None:
            M = ts.ckks_vector(ts_context, M)

        return M, U @ np.diag(S)

    def _predict(self, X):
        # Number of variables (m) and data points (n)
        m, n = X.shape

        # Number of output neurons
        n_outputs = len(self.W)

        y = np.empty((0, n), float)

        # For each output neuron
        for o in range(n_outputs):
            # If the weights are encrypted then they are decrypted to get the performance results
            if self.encrypted:
                W = np.array((self.W[o]).decrypt())
            else:
                W = self.W[o]
            # Neural Network Simulation
            y = np.vstack((y, self.f(W.transpose() @ np.insert(X, 0, np.ones(n), axis=0))))

        return y

    def normal_fit(self, X: np.ndarray, t: np.ndarray):
        # Determine number of outputs/classes
        _, n_outputs = t.shape

        # A model is generated for each output/class
        for o in range(n_outputs):
            M, US = self._fit(X, t[:, o], self.f_inv, self.fderiv, self.sparse, self.encrypted, self.context)
            self.M.append(M)
            self.US.append(US)

    def bagging_fit(self, X: np.ndarray, t: np.ndarray):
        # Determine number of outputs/classes
        _, n_outputs = t.shape

        # Extract ensemble hyper-parameters (bagging)
        n_estimators, p_samples, b_samples, p_features, b_features = self._extract_ensemble_params()
        assert n_estimators > 1
        assert p_features   > 0.0
        assert p_samples    > 0.0
        # No need for feature-indexing if p_feats is 1.0 and no replacement is being done
        if p_features == 1.0 and b_features is False:
            self.set_idx_feats([slice(None) for i in range(n_estimators)])

        # Seed random generator:
        np.random.seed(n_estimators * n_outputs) # TODO eliminate feature when tests are done

        # Fitting in series or in parallel
        if self.parallel:

            # Parallelized fitting
            t_ini, cpu = time.perf_counter(), cpu_count(logical=False)
            n_processes = min(cpu, n_estimators)
            log.debug(f"\t\tDoing parallelized bagging, number of estimators: {({n_estimators})}, cpu-cores: {cpu}")
            zip_iterable = zip(repeat(X), repeat(t), repeat(p_samples), repeat(b_samples), repeat(n_outputs),
                               self.idx_feats, repeat(self.f_inv), repeat(self.fderiv), repeat(self.sparse),
                               repeat(self.encrypted), repeat(None))
            with multiprocessing.Pool(processes=n_processes) as pool:
                # Blocks until ready, ordered results
                results = pool.starmap(FedHEONN_client._bagging_fit, zip_iterable)
                log.debug(f"Bagging ({n_estimators} estimators) SVD-part done in : {time.perf_counter()-t_ini:.3f} s")
                t_enc = time.perf_counter()
                for idx, (M_e, US_e) in enumerate(results):
                    if self.encrypted:
                        # Encrypt M_e's
                        M_e = [ts.ckks_vector(self.context, M) for M in M_e]
                    if idx == len(results) - 1:
                        log.debug(f"Bagging ({n_estimators} estimators) ENC-part done in : {time.perf_counter()-t_enc:.3f} s")
                    # Append to master M&US matrix's
                    self.M.append(M_e)
                    self.US.append(US_e)
            log.debug(f"\t\tParallelized ({n_processes}) bagging fitting done in: {time.perf_counter() - t_ini:.3f} s")

        else:

            #Serialized fitting
            t_ini = time.perf_counter()
            log.debug(f"\t\tDoing serialized bagging fitting, number of estimators: {({n_estimators})}")
            # Arrange each estimator
            for idx in range(n_estimators):
                M_e, US_e = self._bagging_fit(X, t, p_samples, b_samples, n_outputs, self.idx_feats[idx],
                                              self.f_inv, self.fderiv, self.sparse, self.encrypted, self.context)
                # Append to master M&US matrix's
                self.M.append(M_e)
                self.US.append(US_e)
            log.debug(f"\t\tSerialized bagging fitting done in: {time.perf_counter() - t_ini:.3f} s")

    @staticmethod
    def _bagging_fit(X, t, p_samples, b_samples, n_outputs, idx_feats, f_inv, fderiv, sparse, encrypted, ctx=None):
        M_e, US_e = [], []
        X_bag, t_bag = FedHEONN_client._random_patches(X, t, idx_feats, p_samples, b_samples)
        # A model is generated for each output/class
        for o in range(n_outputs):
            M, US = FedHEONN_client._fit(X_bag, t_bag[:, o], f_inv, fderiv, sparse, encrypted, ctx)
            M_e.append(M)
            US_e.append(US)
        return M_e, US_e

    def bagging_predict(self, X:np.ndarray, n_estimators: int=None):
        # List of estimator's predictions
        predictions = []

        # Save original weights
        W_orig = self.W

        # For each estimator predict values
        for i in range(n_estimators):
            # Prepare weights and test data
            self.W = W_orig[i]
            X_predict = X[self.idx_feats[i], :]
            predictions.append(self._predict(X_predict))

        # Restore original weights and return predictions
        self.W = W_orig
        return predictions


    def get_param(self):
        return self.M, self.US

    def set_weights(self, W: list[np.ndarray | bytes] | list[list[np.ndarray | bytes]]):
        if self.encrypted:
            # Deserialize data and build back CKKS vectors
            if self.ensemble:
                # List of estimators within lists of vector's byte data (for each output)
                self.W = []
                for i in range(len(W)):
                    self.W.append([ts.ckks_vector_from(self.context, arr) for arr in W[i]])
            else:
                # List of vector's byte data (for each output)
                self.W = [ts.ckks_vector_from(self.context, arr) for arr in W]
        else:
            # Assign list of optimal weights
            self.W = W

    def set_weights_serialized(self, W_serialized: list[bytes] | list[list[bytes]]):
        if self.ensemble:
            self.W = []
            for i in range(len(W_serialized)):
                self.W.append([ts.ckks_vector_from(self.context, arr) for arr in W_serialized[i]])
        else:
            self.W = [ts.ckks_vector_from(self.context, arr) for arr in W_serialized]

    def set_context(self, context):
        self.context = context

    def set_idx_feats(self, idx_feats):
        self.idx_feats = idx_feats

    def clean_client(self):
        """Method that resets models auxiliary matrix's and weights"""
        self.M         = []
        self.US        = []
        self.W         = None
        self.idx_feats = []

    def _set_ensemble_params(self, ensemble=None):
        self.ensemble = {} if ensemble is None else ensemble

    def _extract_ensemble_params(self):
        if not self.ensemble:
            log.warn(f"No ensemble hyper-parameters dictionary found in this client!")
            return None
        else:
            n_estimators    = self.ensemble['bagging'] if 'bagging' in self.ensemble else 0
            p_samples       = self.ensemble['p_samples'] if 'p_samples' in self.ensemble else 1.0
            b_samples       = self.ensemble['bootstrap_samples'] if 'bootstrap_samples' in self.ensemble else True
            p_features      = self.ensemble['p_features'] if 'p_features' in self.ensemble else 1.0
            b_features      = self.ensemble['bootstrap_features'] if 'bootstrap_features' in self.ensemble else False
        log.debug(f"\t\t"
                  f"n_estimators: {n_estimators} "
                  f"(p_samples: {p_samples} b_samples: {b_samples}) "
                  f"(p_features:{p_features} b_features: {b_features})")
        return n_estimators, p_samples, b_samples, p_features, b_features

    @staticmethod
    def generate_ensemble_params(n_estimators=10, p_samples=1.0, b_samples=True, p_features=1.0, b_features=False):
        ensemble = {'bagging': n_estimators, 'p_samples': p_samples, 'bootstrap_samples': b_samples,
                    'p_features': p_features, 'bootstrap_features': b_features}
        return ensemble

    @staticmethod
    def _random_patches(X, d, idx_features, p_samples=1.0, bootstrap_samples=True):
        """Function to create random patches"""
        n_features, n_samples = X.shape
        idx_samples  = np.sort(np.random.choice(n_samples, size=int(n_samples * p_samples), replace=bootstrap_samples))
        log.debug(f"\t\t"
                  f"Unique sample indexes: {len(np.unique(idx_samples))} "
                  f"Unique feature indexes: {len(np.unique(idx_features))}"
                  f"Feature indexes:\n{idx_features}\n")
        return X[:,idx_samples][idx_features,:], d[idx_samples, :]

    @staticmethod
    def _reshape(arr):
        return arr.reshape(len(arr), 1) if arr.ndim == 1 else arr

    @staticmethod
    def _preprocess(X, t):
        X = FedHEONN_client._reshape(X).T
        t = FedHEONN_client._reshape(t)
        return X, t

    def save_context(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
        tmp_filename = tmp.name
        with open(tmp_filename, "wb") as ctx_32k:
            # If context is already public save_secret_key will act as False, but no error will be thrown
            ctx_32k.write(self.context.serialize(save_public_key=True, save_secret_key=True,
                                                 save_galois_keys=True, save_relin_keys=True))
        log.info(f"Saved tenSEAL context in: ({tmp_filename}) - {os.path.getsize(tmp_filename) / (1024 * 1024):.2f} MB")
        return tmp_filename

    @staticmethod
    def load_context_from(ctx_str):
        log.info(f"Loading context from: ({ctx_str})")
        with open(ctx_str, "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

    @staticmethod
    def delete_context(ctx_str):
        if ctx_str and os.path.isfile(ctx_str):
            log.info(f"\t\tDeleting temporary file: {ctx_str}")
            os.remove(ctx_str)
        else:
            log.warn(f"\t\tCouldn't delete temporary file (empty or not found): {ctx_str}")


class FedHEONN_regressor(FedHEONN_client):
    """FedHEONN client for regression tasks"""
    def fit(self, X, t):
        # Transpose and reshape train data
        X, t = self._preprocess(X, t)

        # Fit data
        if self.ensemble and "bagging" in self.ensemble:
            self.bagging_fit(X=X, t=t)
        else:
            self.normal_fit(X=X, t=t)

    def predict(self, X):
        # Transpose test data
        X = self._reshape(X).T

        # Fit data
        if self.ensemble and "bagging" in self.ensemble:
            n_estimators = self.ensemble["bagging"]
            assert  n_estimators > 1
            predictions = self.bagging_predict(X, n_estimators=n_estimators)
            y = self.simple_mean(predictions)
        else:
            y = self._predict(X)

        return y.T

    @staticmethod
    def simple_mean(list_predictions: list=None):
        # Convert list to numpy array and return its mean along the 0-axis
        return np.array(list_predictions).mean(axis=0)

class FedHEONN_classifier(FedHEONN_client):
    """FedHEONN client for classification tasks"""
    def fit(self, X, t_onehot):
        # Transpose and reshape train data
        X, t_onehot = self._preprocess(X, t_onehot)

        # Transforms the one-hot encoding, mapping values (0, 1) to (0.05, 0.95) to avoid
        # the problem of the inverse of the activation function at the extremes
        t_onehot = t_onehot * 0.90 + 0.05

        # Fit data
        if self.ensemble and "bagging" in self.ensemble:
            self.bagging_fit(X=X, t=t_onehot)
        else:
            self.normal_fit(X=X, t=t_onehot)

    @time_func
    def predict(self, X):
        # Transpose test data
        X = self._reshape(X).T

        # Fit data
        if self.ensemble and "bagging" in self.ensemble:
            n_estimators = self.ensemble["bagging"]
            assert n_estimators > 1
            predictions_onehot = self.bagging_predict(X, n_estimators=n_estimators)
            desired_predictions = [self.determine_desired_outputs(prediction) for prediction in predictions_onehot]
            y = self.majority_vote(list_predictions=desired_predictions)
        else:
            predictions_onehot = self._predict(X)
            y = self.determine_desired_outputs(predictions_onehot)

        return y.T

    @staticmethod
    def determine_desired_outputs(predictions: np.ndarray, desired_value: float=0.95):
        # Extract the closest prediction to 'desired_value' from each output of the model
        return np.apply_along_axis(lambda arr, val: np.abs(arr - val).argmin(), axis=0, arr=predictions, val=desired_value)

    @staticmethod
    def majority_vote(list_predictions: list=None):
        arr_predictions = np.array(list_predictions)
        vote, _ = sp.stats.mode(arr_predictions, axis=0)
        return vote.flatten()
