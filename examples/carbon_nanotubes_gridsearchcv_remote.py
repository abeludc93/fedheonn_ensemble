#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from algorithm.fedHEONN_clients import FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.decorators import time_func
from auxiliary.logger import logger as log
from examples.utils import global_fit, load_carbon_nanotube
from itertools import product
import os
import time


def generate_grid_search_iterator(lambda_grid, n_estimators_grid,
                                  p_samples_grid=None, p_features_grid=None, b_sample_grid=None, b_features_grid=None):
    if p_features_grid is None:
        p_features_grid = [1.0]
    if p_samples_grid is None:
        p_samples_grid = [1.0]
    if b_features_grid is None:
        b_features_grid = [True, False]
    if b_sample_grid is None:
        b_sample_grid = [True, False]
    return product(lambda_grid, n_estimators_grid, b_sample_grid, b_features_grid, p_samples_grid, p_features_grid)

@time_func
def main():
    # Hyperparameter search grid
    lambda_lst          = [0.01, 0.1, 1, 10]
    n_estimators_lst    = [2, 5, 10, 25, 50, 75, 100, 200]
    p_samples_lst       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p_features_lst      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Ensemble method
    if bag:
        gs_space = np.prod([len(i) for i in [lambda_lst, n_estimators_lst, p_samples_lst, p_features_lst, [True,False], [True,False]]])
        gs_it = generate_grid_search_iterator(lambda_lst, n_estimators_lst, p_samples_lst, p_features_lst)
    else:
        gs_space = len(lambda_lst)
        gs_it = lambda_lst
    log.info(f"Grid search hyper-parameter space: {gs_space}")

    # Pandas dataframe dictionary
    df_dict = {"LAMBDA": [], "N_ESTIMATORS": [], "B_SAMPLES": [], "B_FEATS": [], "P_SAMPLES": [], "P_FEATS": [],
               "METRIC_MEAN": [], "METRIC_STD": []} if bag else {"LAMBDA": [], "METRIC_MEAN": [], "METRIC_STD": []}
    # MAIN LOOP
    for idx, tuple_it in enumerate(gs_it):
        # Construct parameters
        log.info(f"GS ITER {idx+1} of {gs_space}")
        if bag:
            lam, n_estimators, b_samples, b_feats, p_samples, p_feats = tuple_it
            ens_client = {'bagging': n_estimators,
                      'bootstrap_samples': b_samples, 'p_samples': p_samples,
                      'bootstrap_features': b_feats, 'p_features': p_feats
                      }
            ens_coord = {'bagging'}
        else:
            lam = tuple_it
            ens_client = {}
            ens_coord = {}

        # Create the coordinator
        coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord)
        # Generate random indexes
        if ens_coord:
            n_attributes = trainX.shape[1]
            coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feats, b_feats)

        # Cross-validation
        if kfold:
            cv = KFold(n_splits=split)
        else:
            cv = ShuffleSplit(n_splits=split, test_size=0.2, random_state=42)
        acc_glb_splits = []

        # CV Loop
        for it, (train_index, test_index) in  enumerate(cv.split(trainX, trainY)):

            # Get split indexes
            log.info(f"\tCross validation split: {it+1}")
            trainX_data, trainT_data = trainX[train_index], trainY[train_index]
            testX_data, testT_data = trainX[test_index], trainY[test_index]
            n_split = trainT_data.shape[0]

            # Create a list of clients and fit clients with their local data
            lst_clients = []
            for i in range(n_clients):

                # Split train equally data among clients
                rang = range(int(i * n_split / n_clients), int(i * n_split / n_clients) + int(n_split / n_clients))
                client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
                log.info(f"\t\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
                if ens_client:
                    client.set_idx_feats(coordinator.send_idx_feats())

                # Fit client local data
                client.fit(trainX_data[rang], trainT_data[rang])
                lst_clients.append(client)

            # Perform fit and predict validation split
            acc_glb, _ = global_fit(list_clients=lst_clients, coord=coordinator,
                                        testX=testX_data, testT=testT_data, regression=True)
            acc_glb_splits.append(acc_glb)

        # Add results to dataframe dictionary
        df_dict["LAMBDA"].append(lam)
        df_dict["METRIC_MEAN"].append(np.array(acc_glb_splits).mean())
        df_dict["METRIC_STD"].append(np.array(acc_glb_splits).std())
        if bag:
            df_dict["N_ESTIMATORS"].append(n_estimators)
            df_dict["B_SAMPLES"].append(b_samples)
            df_dict["B_FEATS"].append(b_feats)
            df_dict["P_SAMPLES"].append(p_samples)
            df_dict["P_FEATS"].append(p_feats)

    # Construct dataframe, sort and export data
    df = pd.DataFrame(df_dict)
    df.sort_values(by="METRIC_MEAN", inplace=True, ascending=False)
    lt = time.localtime()
    timestamp = f"{lt.tm_year}_{lt.tm_mon}_{lt.tm_mday}-{lt.tm_hour}_{lt.tm_min}"
    filename = f"carbon_nanotubes_gs_cv_{'bagging' if bag else 'normal'}_{timestamp}.xlsx"
    file_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + filename)
    df.to_excel(file_path, "RESULTS")
    print(df.head())

if __name__ == "__main__":
    # ----MODEL HYPERPARAMETERS----
    # Number of clients
    n_clients = 1
    # Encryption
    enc = False
    # Sparse matrices
    spr = True
    # Activation function
    f_act = 'linear'
    # IID or non-IID scenario (True or False)
    iid = True
    # Preprocess data
    pre = True
    # Cross-validation
    kfold = True
    split = 10
    # --------

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
    # Load dataset
    np.random.seed(1)
    trainX, trainY, testX, testY = load_carbon_nanotube(f_test_size=0.3, b_preprocess=pre)

    # HYPER-PARAMETER SEARCH GRID
    # Normal execution
    #bag = False
    #main()
    # Ensemble
    bag = True
    main()
