#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Example of using FedHEONN method for a regression task, with and without incremental
grouping and multiple set of clients.
"""
# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from algorithm.fedHEONN_clients import FedHEONN_regressor
from examples.utils import global_fit, incremental_fit

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n1_clients = 1000
# Second set of clients
n2_clients = 50
# Number of clients per group
n_groups = 10
# Randomize number of clients per group
rnd = True
# Encryption
enc = True
# Sparse matrices
spr = True
# Regularization
lam = 0.01
# Activation function
f_act = 'linear'

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

# Create and split regression dataset
Data = pd.read_csv('../datasets/carbon_nanotubes.csv', delimiter=';')
X = Data.iloc[:, :-3].to_numpy()
y = Data.iloc[:, -3:].to_numpy()
#X, y = make_regression(n_samples=1000, n_features=2, n_targets=1, noise=5, random_state=42)
train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)
ntest = len(test_t)
    
# Number of outputs
_ , noutputs = train_t.shape

# Create the coordinator
coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, sparse=spr)

# Create a list of clients and fit clients with their local data
lst1_clients = []
for i in range(0, n1_clients):
    # Split train equally data among clients
    rang = range(int(i*n/n1_clients), int(i*n/n1_clients) + int(n/n1_clients))
    client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx)
    print('Training client:', i+1, 'of', n1_clients, '(', min(rang), '-', max(rang), ')')
    # Fit client local data
    client.fit(train_X[rang], train_t[rang])
    lst1_clients.append(client)

# Create different set of clients and fit them with their local data
lst2_clients = []
for i in range(0, n2_clients):
    rang = range(int(i*n/n2_clients), int(i*n/n2_clients) + int(n/n2_clients))
    client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx)
    print('Training client:', i+1, 'of', n2_clients, '(', min(rang), '-', max(rang), ')')
    client.fit(train_X[rang], train_t[rang])
    lst2_clients.append(client)

# PERFORM GLOBAL AND INCREMENTAL FIT
mse_glb, w_glb = global_fit(list_clients=lst1_clients, coord=coordinator,
                            testX=test_X, testT=test_t, regression=True)
mse_inc, w_inc = incremental_fit(list_clients=lst2_clients, ngroups=n_groups, coord=coordinator,
                                 testX=test_X, testT=test_t, regression=True, random_groups=rnd)
# Print model's metrics
print(f"Test MSE global: {mse_glb:0.8f}")
print(f"Test MSE incremental: {mse_inc:0.8f}")
# Check if weights from both models are equal
for i in range(len(w_glb)):
    # If encrypted, decrypt data
    if enc:
        w_glb[i] = np.array(w_glb[i].decrypt())
        w_inc[i] = np.array(w_inc[i].decrypt())
    # Dif. tolerance
    tol = abs(min(w_glb[i].min(), w_inc[i].min())) / 100
    check = np.allclose(w_glb[i], w_inc[i], atol=tol)
    print(f"Comparing W_glb[{i}] with W_inc[{i}]: {'OK' if check else 'KO'}")
    if not check:
        # Print relative difference amongst weight elements
        diff = abs((w_glb[i] - w_inc[i]) / w_glb[i] * 100)
        print(f"DIFF %: {['{:.2f}%'.format(val) for val in diff]}")
