#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, KFold
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.decorators import time_func
from examples.utils import global_fit, incremental_fit
from itertools import product
import os
import time


# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 4
# Number of clients per group
n_groups = 2
# Randomize number of clients per group in range (n_groups/2, groups*2)
rnd = False
# Encryption
enc = False
# Sparse matrices
spr = True
# Activation function
f_act = 'logs'
# IID or non-IID scenario (True or False)
iid = True
# Ensemble
bag = True
# Cross-validation
kfold = True
split = 10


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

np.random.seed(1)
# Create and split classification dataset
digits = load_digits()
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_X, test_X, train_t, test_t = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)

# Non-IID option: Sort training data by class
if not iid:
    ind = np.argsort(train_t)
    train_t = train_t[ind]
    train_X = train_X[ind]
    print('non-IID scenario')
else:
    ind_list = list(range(n))
    # Data are shuffled in case they come ordered by class
    np.random.seed(1)
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

def generate_grid_search_iterator(lambda_grid, n_estimators_grid, p_samples_grid=None, p_features_grid=None,
                                  b_sample_grid=None, b_features_grid=None):
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
    lambda_grid = [0.01, 1]
    n_estimators_grid = [2, 10]
    p_samples_grid = [0.75, 1.0]
    p_features_grid = [0.75, 1.0]
    gs_it = generate_grid_search_iterator(lambda_grid, n_estimators_grid, p_samples_grid, p_features_grid)

    # Ensemble method
    if bag:
        gs_space = np.prod([len(i) for i in [lambda_grid, n_estimators_grid, p_samples_grid, p_features_grid,
                                             [True,False], [True,False]]])
    else:
        gs_space = len(lambda_grid)
    print(f"Grid search hyper-parameter space: {gs_space}")

    # Pandas dataframe dictionary
    df_dict = {"LAMBDA": [], "N_ESTIMATORS": [], "B_SAMPLES": [], "B_FEATS": [], "P_SAMPLES": [], "P_FEATS": [],
               "METRIC_MEAN": [], "METRIC_STD": []} if bag else {"LAMBDA": [], "METRIC_MEAN": [], "METRIC_STD": []}
    # MAIN LOOP
    for idx, tuple_it in enumerate(gs_it):
        # Construct parameters
        print(f"GS ITER {idx+1} of {gs_space}")
        lam, n_estimators, b_samples, b_feats, p_samples, p_feats = tuple_it
        ens_client = {'bagging': n_estimators,
                      'bootstrap_samples': b_samples, 'p_samples': p_samples,
                      'bootstrap_features': b_feats, 'p_features': p_feats
                      } if bag else {}
        ens_coord = {'bagging'} if bag else {}

        # Create the coordinator
        coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord)
        # Generate random indexes
        if ens_coord:
            n_attributes = train_X.shape[1]
            coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feats, b_feats)

        # Cross-validation
        if kfold:
            cv = KFold(n_splits=split)
        else:
            cv = ShuffleSplit(n_splits=split, test_size=0.2, random_state=42)
        acc_glb_splits, w_glb_splits, acc_inc_splits, w_inc_splits = [], [], [], []

        # CV Loop
        for it, (train_index, test_index) in  enumerate(cv.split(train_X, t_onehot)):

            # Get split indexes
            print(f"\tCross validation split: {it+1}")
            trainX_data, trainT_data = train_X[train_index], t_onehot[train_index]
            testX_data, testT_data = train_X[test_index], train_t[test_index]
            n_split = trainT_data.shape[0]

            # Create a list of clients and fit clients with their local data
            lst_clients = []
            for i in range(n_clients):

                # Split train equally data among clients
                rang = range(int(i * n_split / n_clients), int(i * n_split / n_clients) + int(n_split / n_clients))
                client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
                print(f"Training client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
                if ens_client:
                    client.set_idx_feats(coordinator.send_idx_feats())

                # Fit client local data
                client.fit(trainX_data[rang], trainT_data[rang])
                lst_clients.append(client)

            # Perform fit and predict validation split
            acc_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                                        testX=testX_data, testT=testT_data, regression=False)
            acc_inc, w_inc = incremental_fit(list_clients=lst_clients, ngroups=n_groups, coord=coordinator,
                                             testX=testX_data, testT=testT_data, regression=False, random_groups=rnd)
            acc_glb_splits.append(acc_glb)
            w_glb_splits.append(w_glb)
            acc_inc_splits.append(acc_inc)
            w_inc_splits.append(w_inc)

            # Clean coordinator incremental data for the next fold
            coordinator.clean_coordinator()

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
    file_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + f"mnist_gs_cv_{timestamp}.xlsx")
    df.to_excel(file_path, "RESULTS")
    print(df.head())

if __name__ == "__main__":
    main()