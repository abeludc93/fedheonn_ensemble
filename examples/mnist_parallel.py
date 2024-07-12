#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.decorators import time_func
from auxiliary.logger import logger as log
from examples.utils import global_fit, incremental_fit, load_mnist_digits


@time_func
def main():
    # Create the coordinator
    coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord, parallel=par_coord)

    # Train data length
    n, n_attributes = trainX.shape
    # Generate random indexes
    if ens_coord:
        coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feat, b_feat)

    # Create a list of clients and fit clients with their local data
    lst_clients = []
    for i in range(0, n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client, parallel=par)
        log.info(f"\t\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        if ens_client:
            client.set_idx_feats(coordinator.send_idx_feats())
        # Fit client local data
        client.fit(trainX[rang], trainY_onehot[rang])
        lst_clients.append(client)

    # PERFORM GLOBAL FIT
    acc_glb, _ = global_fit(list_clients=lst_clients, coord=coordinator, testX=testX, testT=testY, regression=False)
    #acc_inc, _ = incremental_fit(list_clients=lst_clients, coord=coordinator, ngroups=1, testX=testX, testT=testY,
    #                             regression=False, random_groups=False)

    # Print model's metrics
    log.info(f"Test accuracy global: {acc_glb:0.2f}")
    #log.info(f"Test accuracy incremental: {acc_inc:0.2f}")

if __name__ == "__main__":
    # ---- MODEL HYPERPARAMETERS----
    # Number of clients
    n_clients = 3
    # Encryption
    enc = True
    # Sparse matrices
    spr = True
    # Regularization
    lam = 0.01
    # Activation function
    f_act = 'logs'
    # IID or non-IID scenario (True or False)
    iid = True
    # Preprocess data
    pre = True
    # Parallelized
    par = False
    par_coord = True
    # Ensemble
    bag = True
    # Random Patches bagging parameters
    n_estimators = 20
    p_samples = 1.0
    b_samples = False
    p_feat = 1.0
    b_feat = False

    # Configuring the TenSEAL context for the CKKS encryption scheme
    ctx = None
    if enc:
        import tenseal as ts
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40
    # Prepare ensemble dictionaries
    ens_client = {'bagging': n_estimators,
                  'bootstrap_samples': b_samples, 'p_samples': p_samples,
                  'bootstrap_features': b_feat, 'p_features': p_feat
                  } if bag else {}
    ens_coord = {'bagging'} if bag else {}

    # Load dataset
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_mnist_digits(f_test_size=0.3, b_preprocess=pre, b_iid=iid)

    # Parallel MAIN FUNCTION
    main()
