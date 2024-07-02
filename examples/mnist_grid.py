#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from examples.utils import grid_search

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 10
# Encryption
enc = False
# Sparse matrices
spr = True
# Activation function
f_act = 'logs'
# IID or non-IID scenario (True or False)
iid = True

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
digits = load_digits()
# flatten the images
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

# Grid search hyperparameters
rng_lambda = np.logspace(np.log(0.01)/np.log(10), 1, 4)
rng_nestim = np.arange(2,42,10)
grid_search(range_lambda=rng_lambda, range_estimators=rng_nestim, n_clients=n_clients,
            trainX=train_X, testX=test_X, trainY=t_onehot, testY=test_t, regression=False,
            f_act=f_act, enc=enc, spr=spr, ctx=ctx)

