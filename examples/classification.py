#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Example of using FedHEONN method for a multiclass classification task.
"""
# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from examples.utils import global_fit
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator

# EXAMPLE AND MODEL HYPERPARAMETERS
# Number of clients
n_clients = 200
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


# The data set is loaded (Dry Bean Dataset)
# Source: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
# Article: https://www.sciencedirect.com/science/article/pii/S0168169919311573?via%3Dihub
Data = pd.read_excel('../datasets/Dry_Bean_Dataset.xlsx', sheet_name='Dry_Beans_Dataset')
Data['Class'] = Data['Class'].map({'BARBUNYA': 0, 'BOMBAY': 1, 'CALI': 2, 'DERMASON': 3, 'HOROZ': 4, 'SEKER': 5, 'SIRA': 6})        
Inputs = Data.iloc[:, :-1].to_numpy()
Labels = Data.iloc[:, -1].to_numpy()        
train_X, test_X, train_t, test_t = train_test_split(Inputs, Labels, test_size=0.3, random_state=42)

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
nclasses = len(np.unique(train_t))     

# One hot encoding for the targets
t_onehot = np.zeros((n, nclasses))
for i, value in enumerate(train_t):
    t_onehot[i, value] = 1

# Create the coordinator
coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, sparse=spr)

# Create a list of clients and fit clients with their local data
lst_clients = []
for i in range(0, n_clients):
    # Split train equally data among clients
    rang = range(int(i*n/n_clients), int(i*n/n_clients) + int(n/n_clients))
    client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx)
    print('Training client:', i+1, 'of', n_clients, '(', min(rang), '-', max(rang), ')')
    # Fit client local data
    client.fit(train_X[rang], t_onehot[rang])
    lst_clients.append(client)

# PERFORM GLOBAL FIT
acc_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                            testX=test_X, testT=test_t, regression=False)

# Print model's metrics
print(f"Test accuracy global: {acc_glb:0.8f}")
