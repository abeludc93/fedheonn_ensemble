#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fedHEONN_clients import FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import global_fit, incremental_fit

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 10
# Number of clients per group
n_groups = 5
# Randomize number of clients per group in range (n_groups/2, groups*2)
rnd = False
# Encryption
enc = False
# Sparse matrices
spr = True
# Regularization
lam = 0.01
# Activation function
f_act = 'relu'
# Ensemble
bag = False  # bagging
n_estimators = 10
ens_client = {'bagging': n_estimators} if bag else {}
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

# Create and split classification dataset
X, y = fetch_california_housing(return_X_y=True)
train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)

# Create the coordinator
coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord)

# Create a list of clients and fit clients with their local data
lst_clients = []
for i in range(0, n_clients):
    # Split train equally data among clients
    rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
    client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
    print(f"Training client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
    # Fit client local data
    client.fit(train_X[rang], train_t[rang])
    lst_clients.append(client)

# PERFORM GLOBAL FIT
metric_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                            testX=test_X, testT=test_t, regression=True)
metric_inc, w_inc = incremental_fit(list_clients=lst_clients, ngroups=n_groups, coord=coordinator,
                                 testX=test_X, testT=test_t, regression=True, random_groups=rnd)
# Print model's metrics
print(f"Test MSE global: {metric_glb:0.2f}")
print(f"Test MSE incremental: {metric_inc:0.2f}")