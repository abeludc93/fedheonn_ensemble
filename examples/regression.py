#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Example of using FedHEONN method for a regression task.
"""
# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from examples.utils import global_fit
from algorithm.fedHEONN_clients import FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 100
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

# The data set is loaded (Carbon Nanotubes)
# Source: https://archive.ics.uci.edu/dataset/448/carbon+nanotubes
Data = pd.read_csv('../datasets/carbon_nanotubes.csv', delimiter=';')
Inputs = Data.iloc[:, :-3].to_numpy()
Targets = Data.iloc[:, -3:].to_numpy() # 3 outputs to predict
train_X, test_X, train_t, test_t = train_test_split(Inputs, Targets, test_size=0.3, random_state=42)

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
coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc)

# Create a list of clients and fit clients with their local data
lst_clients = []
for i in range(0, n_clients):
    # Split train equally data among clients
    rang = range(int(i*n/n_clients), int(i*n/n_clients) + int(n/n_clients))
    client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx)
    print('Training client:', i+1, 'of', n_clients, '(', min(rang), '-', max(rang), ')')
    # Fit client local data
    client.fit(train_X[rang], train_t[rang])
    lst_clients.append(client)

# PERFORM GLOBAL FIT
mse_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                            testX=test_X, testT=test_t, regression=True)

# Print model's metrics
print(f"Test MSE global: {mse_glb:0.8f}")
