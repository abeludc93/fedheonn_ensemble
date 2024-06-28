#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Example of using FedHEONN method for a multiclass classification task.
"""
# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import global_fit

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 10
# Encryption
enc = False
# Sparse matrices
spr = True
# Regularization
lam = 0.01
# Activation function
f_act = 'logs'
# IID or non-IID scenario (True or False)
iid = True
# Bagging
bag = True
# Number of estimators
n_estimators = 200

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
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)
ntest = len(test_t)

# Non-IID option: Sort training data by class
if not iid:
    ind = np.argsort(train_t)
    train_t = train_t[ind]
    train_X = train_X[ind]
    print('non-IID scenario')
else:        
    ind_list = list(range(n))
    # Data are shuffled in case they come ordered by class
    np.random.shuffle(ind_list)
    train_X  = train_X[ind_list]
    train_t = train_t[ind_list]
    print('IID scenario')
    
# Number of classes
n_classes = len(np.unique(train_t))

# One hot encoding for the targets
t_onehot = np.zeros((n, n_classes))
for i, value in enumerate(train_t):
    t_onehot[i, value] = 1

# Create the coordinator
coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, bagging=bag)

# Create a list of clients and fit clients with their local data
lst_clients = []
for i in range(0, n_clients):
    # Split train equally data among clients
    rang = range(int(i*n/n_clients), int(i*n/n_clients) + int(n/n_clients))
    client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, bagging=bag, n_estimators=n_estimators)
    print('Training client:', i+1, 'of', n_clients, '(', min(rang), '-', max(rang), ')')
    # Fit client local data
    client.fit(train_X[rang], train_t[rang])
    lst_clients.append(client)

# PERFORM GLOBAL FIT
acc_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                            testX=test_X, testT=test_t, regression=False)

# Print model's metrics
print(f"Test accuracy global: {acc_glb:0.8f}")
