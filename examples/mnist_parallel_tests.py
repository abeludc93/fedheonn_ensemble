#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import itertools
import os
import numpy as np
from time import perf_counter
import pandas as pd
from sklearn.metrics import accuracy_score
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.logger import logger as log
from examples.utils import load_mnist_digits
import tenseal as ts


def experiments():

    n_estimators_lst    = [2, 5]
    p_sf_lst = [0.5, 1]
    parallel_lst = [True, False]

    # Pandas dataframe dictionary
    df_dict = {"PARALLEL": [], "N_ESTIMATORS": [], "P_SAMPLES": [], "P_FEATS": [],
               "T_FIT": [], "T_AGG": [], "T_CALC": [], "METRIC": []}
    parallel_it = itertools.product(n_estimators_lst, p_sf_lst, parallel_lst)
    parallel_space = np.prod([len(i) for i in [n_estimators_lst, p_sf_lst, parallel_lst]])

    for idx, tuple_it in enumerate(parallel_it):
        log.info(f"PARALLEL ITER {idx+1} of {parallel_space}")
        n_estimators, p_sf, par_both = tuple_it
        p_samples = p_sf
        p_features = p_sf

        t_fit_arr, t_agg, t_calc, accuracy = run_experiment(n_estimators, p_samples, p_features, par_both)

        df_dict["PARALLEL"].append(par_both)
        df_dict["N_ESTIMATORS"].append(n_estimators)
        df_dict["P_SAMPLES"].append(p_samples)
        df_dict["P_FEATS"].append(p_features)
        df_dict["T_FIT"].append(np.array(t_fit_arr).mean())
        df_dict["T_AGG"].append(t_agg)
        df_dict["T_CALC"].append(t_calc)
        df_dict["METRIC"].append(accuracy)

    return df_dict


def run_experiment(T, p_s, p_f, par_both):
    # Parallel
    par_coord = par_both
    par = par_both

    # Prepare ensemble dictionaries
    ens_client = {'bagging': T,
                  'bootstrap_samples': False, 'p_samples': p_s,
                  'bootstrap_features': False, 'p_features': p_f
                  }
    ens_coord = {'bagging'}

    # Create the coordinator
    coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord, parallel=par_coord)

    # Train data length
    n, n_attributes = trainX.shape
    # Generate random indexes
    if ens_coord:
        coordinator.calculate_idx_feats(T, n_attributes, p_f, False)

    # Create a list of clients and fit clients with their local data
    lst_clients, fit_time = [], []
    for i in range(0, n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client, parallel=par)
        log.info(f"\t\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        if ens_client:
            client.set_idx_feats(coordinator.send_idx_feats())

        # Fit client local data
        t_fit_ini = perf_counter()
        client.fit(trainX[rang], trainY_onehot[rang])
        t_fit_end = perf_counter()
        fit_time.append(t_fit_end-t_fit_ini)
        lst_clients.append(client)

    # PERFORM INCREMENTAL FIT
    M_grp, US_grp = [], []
    for client in lst_clients:
        M_c, US_c = client.get_param()
        M_grp.append(M_c)
        US_grp.append(US_c)

    # Aggregate partial model info
    t_agg_ini = perf_counter()
    coordinator.aggregate_partial(M_list=M_grp, US_list=US_grp)
    t_agg_end = perf_counter()
    t_agg = t_agg_end-t_agg_ini

    # Calculate opt. weights
    t_calc_ini = perf_counter()
    coordinator.calculate_weights()
    t_calc_end = perf_counter()
    t_calc = t_calc_end-t_calc_ini

    # Metrics
    lst_clients[0].set_weights(coordinator.send_weights())
    test_y = lst_clients[0].predict(testX)
    acc_inc = 100 * accuracy_score(testY, test_y)

    # Print model's metrics
    log.info(f"Test accuracy incremental: {acc_inc:0.2f}")
    return np.array(fit_time), t_agg, t_calc, acc_inc


if __name__ == "__main__":
    # ---- MODEL HYPERPARAMETERS----
    # Number of clients
    n_clients = 4
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
    # Context
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    # Load dataset
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_mnist_digits(f_test_size=0.3, b_preprocess=pre, b_iid=iid)

    # Parallel MAIN FUNCTION
    dataframe = experiments()

    # Construct dataframe, sort and export data
    pd.set_option("display.precision", 8)
    df1 = pd.DataFrame(dataframe)
    log.info(f"RESULTS:\n{df1.head()}")
    filename = f"parallel_experiments.xlsx"
    file_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + filename)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="PARALLEL", index=False)
