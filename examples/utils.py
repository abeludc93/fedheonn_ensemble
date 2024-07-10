#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
incremental_utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module containing auxiliary functions used in incremental examples
"""
# Standard libraries
from random import seed, shuffle, randint
# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Application modules
from algorithm.fedHEONN_clients import FedHEONN_classifier, FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.logger import logger as log


# Seed random numbers
seed(1)


# Function to obtain prediction metrics for given (testX,testT) data using one client and a coordinator
def get_prediction(ex_client, coord, testX, testT, regression=True):
    # Send the weights of the aggregate model to example client
    ex_client.set_weights(coord.send_weights())
    # Predictions for the test set using one client
    test_y = ex_client.predict(testX)
    if regression:
        # Global MSE for the 3 outputs
        metric = 100 * mean_squared_error(testT, test_y) # TODO: no es un porcentaje
    else:
        # Global precision for all outputs
        metric = 100 * accuracy_score(testT, test_y)

    return metric


# Function that groups the given list of clients in batches of 'ngroups' clients
def group_clients(list_clients, ngroups, randomize=False):
    # List of groups
    groups = []
    # Flag to randomize the number of clients per group
    if not randomize:
        # Group said clients
        for i in range(0, len(list_clients), ngroups):
            log.info(f"\t\tGrouping clients: ({i}:{i + ngroups})")
            group = list_clients[i:i + ngroups]
            groups.append(group)
    else:
        # Group randomly said clients
        idx = 0
        while idx < len(list_clients):
            n_groups_rnd = randint(ngroups // 2, 2 * ngroups)
            if idx + n_groups_rnd > len(list_clients):
                n_groups_rnd = len(list_clients) - idx
            log.info(f"\t\tGrouping clients randomly: ({idx}:{idx + n_groups_rnd})")
            group = list_clients[idx:idx + n_groups_rnd]
            groups.append(group)
            idx += n_groups_rnd

    return groups


# Function that returns the auxiliary matrix's M & US of a group of clients
def get_params_group(group):
    # M&US group matrix's
    M_grp, US_grp = [], []
    for client in group:
        M_c, US_c = client.get_param()
        # Append to list of matrix's
        M_grp.append(M_c)
        US_grp.append(US_c)

    return M_grp, US_grp


# Function that performs an 'incremental' fit on the given list of clients, aggregating then as sequential batches and
# returning the mean squared error and optimal weights on the test data
def incremental_fit(list_clients, coord, ngroups, testX, testT, regression=True, random_groups=False):
    # Flag to make predictions after incrementally processing each group
    debug = True
    # Shuffle client list
    shuffle(list_clients)
    # Group clients
    groups = group_clients(list_clients, ngroups, randomize=random_groups)
    for ig, group in enumerate(groups):
        M_grp, US_grp = get_params_group(group=group)
        # Aggregate partial model info
        coord.aggregate_partial(M_list=M_grp, US_list=US_grp)
        # Calculate opt. weights and realize current predictions
        if debug:
            coord.calculate_weights()
            metric_debug = get_prediction(list_clients[0], coord, testX, testT, regression=regression)
            log.debug(f"\t\tTest MSE incremental (group {ig+1}): {metric_debug:0.4f}")
    # Calculate opt. weights
    coord.calculate_weights()
    # Metrics
    metric = get_prediction(list_clients[0], coord, testX, testT, regression=regression)

    return metric, coord.send_weights()


# Function that performs a 'global' fit on the given list of clients, aggregating them separately and returning the
# metric and optimal weights on the test data
def global_fit(list_clients, coord, testX, testT, regression=True):
    # Shuffle client list
    shuffle(list_clients)
    # Fit the clients with their local data
    M , US = [], []
    for client in list_clients:
        M_c, US_c = client.get_param()
        M.append(M_c)
        US.append(US_c)
    # The coordinator aggregates the information provided by the clients
    # and obtains the weights of the collaborative model
    coord.aggregate(M, US)
    # Metrics
    metric = get_prediction(list_clients[0], coord, testX, testT, regression=regression)

    return  metric, coord.send_weights()

# Function that compares model weights w1 and w2, checking if they are equal to a certain tolerance
def check_weights(w1, w2, encrypted):
    for i in range(len(w1)):
        # If encrypted, decrypt data
        if encrypted:
            w1[i] = np.array(w1[i].decrypt())
            w2[i] = np.array(w2[i].decrypt())
        # Dif. tolerance
        tol = abs(min(w1[i].min(), w2[i].min())) / 100
        check = np.allclose(w1[i], w2[i], atol=tol)
        log.info(f"\tComparing W_glb[{i}] with W_inc[{i}]: {'OK' if check else 'KO'}")
        if not check:
            # Print relative difference amongst weight elements
            diff = abs((w1[i] - w2[i]) / w1[i] * 100)
            log.debug(f"\t\tDIFF %: {['{:.2f}%'.format(val) for val in diff]}")

#Function used to create and fit a list of n_clients on train data trainX
def create_list_clients(n_clients, trainX, trainY, regression, f_act, enc, spr, ctx, ens_client, coord=None):
    n = len(trainY)
    # Create a list of clients and fit clients with their local data
    lst_clients = []
    for i in range(0, n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        if regression:
            client = FedHEONN_regressor(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
        else:
            client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
        log.info(f"\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        # Fit client local data
        if ens_client:
            client.set_idx_feats(coord.send_idx_feats())
        client.fit(trainX[rang], trainY[rang])

        lst_clients.append(client)

    return lst_clients

# Function to load and prepare MNIST dataset
def load_mnist_digits(f_test_size=0.3, b_preprocess=True, b_iid=True):
    log.info("[*] MNIST DIGITS DATASET [*]")
    # Create and split classification dataset
    digits = load_digits()
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    train_X, test_X, train_t, test_t = train_test_split(data, digits.target, test_size=f_test_size, random_state=42)

    if b_preprocess:
        # Data normalization (z-score): mean 0 and std 1
        log.info("\t\tData normalization (z-score)")
        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    # Number of training and test data
    n = len(train_t)

    # Non-IID option: Sort training data by class
    if not b_iid:
        log.info('\t\tnon-IID scenario')
        ind = np.argsort(train_t)
        train_t = train_t[ind]
        train_X = train_X[ind]
    else:
        # Data are shuffled in case they come ordered by class
        log.info('\t\tIID scenario')
        ind_list = list(range(n))
        np.random.seed(1)
        np.random.shuffle(ind_list)
        train_X = train_X[ind_list]
        train_t = train_t[ind_list]

    # Number of classes
    n_classes = len(np.unique(train_t))

    # One hot encoding for the targets
    t_onehot = np.zeros((n, n_classes))
    for i, value in enumerate(train_t):
        t_onehot[i, value] = 1

    return train_X, t_onehot, test_X, test_t, train_t

# Function to load and prepare Dry-Bean dataset
def load_carbon_nanotube(f_test_size=0.3, b_preprocess=True):
    log.info("[*] CARBON NANOTUBE DATASET [*]")
    # Read dataset
    Data = pd.read_csv('../datasets/carbon_nanotubes.csv', delimiter=';')
    Inputs = Data.iloc[:, :-3].to_numpy()
    Targets = Data.iloc[:, -3:].to_numpy()  # 3 outputs to predict

    # Split
    train_X, test_X, train_t, test_t = train_test_split(Inputs, Targets, test_size=f_test_size, random_state=42)

    if b_preprocess:
        # Data normalization (z-score): mean 0 and std 1
        log.info("\t\tData normalization (z-score)")
        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    return train_X, train_t, test_X, test_t

# Function to load and prepare Dry_Bean dataset
def load_dry_bean(f_test_size=0.3, b_preprocess=True, b_iid=True):
    log.info("[*] DRY BEAN DATASET [*]")
    # Read dataset
    Data = pd.read_excel('../datasets/Dry_Bean_Dataset.xlsx', sheet_name='Dry_Beans_Dataset')
    Data['Class'] = Data['Class'].map(
        {'BARBUNYA': 0, 'BOMBAY': 1, 'CALI': 2, 'DERMASON': 3, 'HOROZ': 4, 'SEKER': 5, 'SIRA': 6})
    Inputs = Data.iloc[:, :-1].to_numpy()
    Labels = Data.iloc[:, -1].to_numpy()
    train_X, test_X, train_t, test_t = train_test_split(Inputs, Labels, test_size=f_test_size, random_state=42)

    if b_preprocess:
        # Data normalization (z-score): mean 0 and std 1
        log.info("\t\tData normalization (z-score)")
        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    # Number of training and test data
    n = len(train_t)

    # Non-IID option: Sort training data by class
    if not b_iid:
        log.info('\t\tnon-IID scenario')
        ind = np.argsort(train_t)
        train_t = train_t[ind]
        train_X = train_X[ind]
    else:
        # Data are shuffled in case they come ordered by class
        log.info('\t\tIID scenario')
        ind_list = list(range(n))
        np.random.seed(1)
        np.random.shuffle(ind_list)
        train_X = train_X[ind_list]
        train_t = train_t[ind_list]

    # Number of classes
    n_classes = len(np.unique(train_t))

    # One hot encoding for the targets
    t_onehot = np.zeros((n, n_classes))
    for i, value in enumerate(train_t):
        t_onehot[i, value] = 1

    return train_X, t_onehot, test_X, test_t, train_t