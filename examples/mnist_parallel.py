#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.decorators import time_func
from examples.utils import global_fit


# EXAMPLE AND MODEL HYPERPARAMETERS
def load_params():

    # Number of clients
    n_clients = 2
    # Encryption
    enc = False
    # Sparse matrices
    spr = True
    # Regularization
    lam = 0.01
    # Activation function
    f_act = 'linear'
    # IID or non-IID scenario (True or False)
    iid = True
    # Parallelized
    par = True
    # Ensemble
    bag = True
    # Bagging
    n_estimators = 50
    p_samples = 1.0
    b_samples = False
    p_feat = 1.0
    b_feat = False
    ens_client = {'bagging': n_estimators,
                  'bootstrap_samples': b_samples, 'p_samples': p_samples,
                  'bootstrap_features': b_feat, 'p_features': p_feat
                  } if bag else {}
    ens_coord = {'bagging'} if bag else {}

    ctx = None
    if enc:
        import tenseal as ts
        # Configuring the TenSEAL context for the CKKS encryption scheme
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40

    return n_clients, enc, spr, lam, f_act, par, n_estimators, p_feat, b_feat, ens_client, ens_coord, ctx, iid


# LOAD MNIST DATASET
def load_dataset(b_iid: bool=True):

    # Seed
    np.random.seed(1)

    # Create and split classification dataset
    digits = load_digits()

    # Flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    train_X, test_X, train_t, test_t = train_test_split(data, digits.target, test_size=0.3, random_state=42)

    # Data normalization (z-score): mean 0 and std 1
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Number of training and test data
    n = len(train_t)

    # Non-IID option: Sort training data by class
    if not b_iid:
        ind = np.argsort(train_t)
        train_t = train_t[ind]
        train_X = train_X[ind]
        print('non-IID scenario')
    else:
        ind_list = list(range(n))
        # Data are shuffled in case they come ordered by class
        np.random.shuffle(ind_list)
        train_X = train_X[ind_list]
        train_t = train_t[ind_list]
        print('IID scenario')

    # Number of classes
    n_classes = len(np.unique(train_t))

    # One hot encoding for the targets
    t_onehot = np.zeros((n, n_classes))
    for i, value in enumerate(train_t):
        t_onehot[i, value] = 1

    return train_X, t_onehot, test_X, test_t, n

@time_func
def main(n_clients, f_act, lam, enc, spr, ctx,
         n_estimators, par, ens_client, ens_coord, p_feat, b_feat,
         n, train_X, t_onehot, test_X, test_t):
    # Create the coordinator
    coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord)

    # Generate random indexes
    if ens_coord:
        n_attributes = train_X.shape[1]
        coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feat, b_feat)

    # Create a list of clients and fit clients with their local data
    lst_clients = []
    for i in range(0, n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client, parallel=par)
        print(f"Training client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        if ens_client:
            client.set_idx_feats(coordinator.send_idx_feats())
        # Fit client local data
        client.fit(train_X[rang], t_onehot[rang])
        lst_clients.append(client)

    # PERFORM GLOBAL FIT
    acc_glb, _ = global_fit(list_clients=lst_clients, coord=coordinator, testX=test_X, testT=test_t, regression=False)

    # Print model's metrics
    print(f"Test accuracy global: {acc_glb:0.2f}")

if __name__ == "__main__":
    _n_clients, _enc, _spr, _lam, _f_act, _par, _n_estimators, _p_feat, _b_feat, _ens_client, _ens_coord, _ctx, _iid = load_params()
    _train_X, _t_onehot, _test_X, _test_t, _n = load_dataset(_iid)
    main(_n_clients, _f_act, _lam, _enc, _spr, _ctx,
         _n_estimators, _par, _ens_client, _ens_coord, _p_feat, _b_feat,
         _n, _train_X, _t_onehot, _test_X, _test_t)