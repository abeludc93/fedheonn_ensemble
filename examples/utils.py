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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Seed random numbers
seed(0)


# Function to obtain prediction metrics for given (testX,testT) data using one client and a coordinator
def get_prediction(ex_client, coord, testX, testT, regression=True):
    # Send the weights of the aggregate model to example client
    ex_client.set_weights(coord.send_weights())
    # Predictions for the test set using one client
    test_y = ex_client.predict(testX)
    if regression:
        # Global MSE for the 3 outputs
        metric = 100 * mean_squared_error(testT, test_y)
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
            print(f"Grouping clients: ({i}:{i + ngroups})")
            group = list_clients[i:i + ngroups]
            groups.append(group)
    else:
        # Group randomly said clients
        idx = 0
        while idx < len(list_clients):
            n_groups_rnd = randint(ngroups // 2, 2 * ngroups)
            if idx + n_groups_rnd > len(list_clients):
                n_groups_rnd = len(list_clients) - idx
            print(f"Grouping clients: ({idx}:{idx + n_groups_rnd})")
            group = list_clients[idx:idx + n_groups_rnd]
            groups.append(group)
            idx += n_groups_rnd
    return groups


# Function that returns the auxiliary matrix's M & US of a group of clients
def get_params_group(group):
    M_grp, US_grp = [], []
    for _, client in enumerate(group):
        M_c, US_c = client.get_param()
        M_grp.append(M_c)
        US_grp.append(US_c)
    return M_grp, US_grp


# Function that performs an 'incremental' fit on the given list of clients, aggregating then as sequential batches and
# returning the mean squared error and optimal weights on the test data
def incremental_fit(list_clients, coord, ngroups, testX, testT, regression=True, random_groups=False):
    # Flag to realize predictions after processing each group
    debug = True
    # Shuffle client list
    shuffle(list_clients)
    # Group clients
    groups = group_clients(list_clients, ngroups, randomize=random_groups)
    for ig, group in enumerate(groups):
        M_grp, US_grp = get_params_group(group=group)
        # Aggregate partial model info
        coord.aggregate_partial(M_list=M_grp, US_list=US_grp)
        # Calc. optim weights and realize current predictions
        if debug:
            coord.calculate_weights()
            print(f"\t***Test MSE incremental (group {ig+1}): "
                  f"{get_prediction(list_clients[0], coord, testX, testT):0.8f}")
    # Calculate optimal weights
    coord.calculate_weights()
    # Metrics
    metric = get_prediction(list_clients[0], coord, testX, testT, regression=regression)
    return metric, coord.send_weights()


# Function that performs a 'global' fit on the given list of clients, aggregating them separately and returning the
# mean squared error and optimal weights on the test data
def global_fit(list_clients, coord, testX, testT, regression=True):
    # Shuffle client list
    shuffle(list_clients)
    # Fit the clients with their local data
    M  = []
    US = []
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
