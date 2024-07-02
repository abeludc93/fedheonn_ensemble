#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
incremental_utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module containing auxiliary functions used in incremental examples
"""
import copy
# Standard libraries
from random import seed, shuffle, randint
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from algorithm.fedHEONN_clients import FedHEONN_classifier, FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator

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
                  f"{get_prediction(list_clients[0], coord, testX, testT, regression=regression):0.8f}")
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

# Function that compares optimal weights w1 and w2, checking if they are equal to a certain tolerance
def check_weights(w1, w2, encrypted):
    for i in range(len(w1)):
        # If encrypted, decrypt data
        if encrypted:
            w1[i] = np.array(w1[i].decrypt())
            w2[i] = np.array(w2[i].decrypt())
        # Dif. tolerance
        tol = abs(min(w1[i].min(), w2[i].min())) / 100
        check = np.allclose(w1[i], w2[i], atol=tol)
        print(f"Comparing W_glb[{i}] with W_inc[{i}]: {'OK' if check else 'KO'}")
        if not check:
            # Print relative difference amongst weight elements
            diff = abs((w1[i] - w2[i]) / w1[i] * 100)
            print(f"DIFF %: {['{:.2f}%'.format(val) for val in diff]}")

#Function used to create and fit a list of n_clients on train data trainX
def create_list_clients(n_clients, trainX, trainY, regression, f_act, enc, spr, ctx, ens_client):
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
        print(f"Training client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        # Fit client local data
        client.fit(trainX[rang], trainY[rang])
        lst_clients.append(client)

    return lst_clients

# Function used to perform
def grid_search(range_lambda, range_estimators, n_clients, trainX, testX, trainY, testY, regression, f_act, enc, spr, ctx):
    ens_coord = {'bagging'}
    ens_client = {'bagging':1}
    graph_x, graph_y, graph_z = [], [], []
    # Create hyperparam grid
    print(f"GRID-SEARCH over {len(range_lambda)} x {len(range_estimators)} = {len(range_lambda)*len(range_estimators)}")
    for i in range_lambda:
        graph_x_row, graph_y_row, graph_z_row = [], [], []
        for j in range_estimators:
            print(f"\tPerforming grid-search: lambda ({i:.2f}) - n_estimators ({j})")
            lam = i
            ens_client["bagging"] = j
            list_clients = create_list_clients(n_clients, trainX, trainY, regression, f_act, enc, spr, ctx, ens_client)
            coord = FedHEONN_coordinator(lam=lam, ensemble=ens_coord, f=f_act, encrypted=enc, sparse=spr)
            metric, _ = global_fit(list_clients, coord, testX, testY, regression)
            graph_x_row.append(i);graph_y_row.append(j);graph_z_row.append(metric)
            print(f"\tMetric achieved: {metric:.4f}\n")
        graph_x.append(graph_x_row);graph_y.append(graph_y_row);graph_z.append(graph_z_row)
    graph_x, graph_y, graph_z = np.array(graph_x), np.array(graph_y), np.array(graph_z)
    if regression:
        #MSE
        best = np.min(graph_z)
        pos_best = np.argwhere(graph_z == np.min(graph_z))[0]
    else:
        #ACCURACY
        best = np.max(graph_z)
        pos_best = np.argwhere(graph_z == np.max(graph_z))[0]

    print(f"Best metric: {best:.4f}")
    print(f"Optimum lambda: {graph_x[pos_best[0], pos_best[1]]:.2f}")
    print(f"Optimum n_estimators: {graph_y[pos_best[0], pos_best[1]]}")
    plot_grid_search(graph_x, graph_y, graph_z)

# Function that plots a heat-map of the grid-search results
def plot_grid_search(x,y,z):
    z_min, z_max = z.min(), z.max()
    z_rescaled = 100 * (z - z_min) / (z_max - z_min)
    point_sizes = z_rescaled + 10
    #plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=z, s=point_sizes, cmap='viridis', alpha=0.9)
    plt.colorbar(scatter, label='Valor de Z')
    plt.xscale('log')
    plt.xlabel(r'Regularization (\lambda)')
    plt.ylabel('No. of estimators')
    plt.title('Hyper-parameter grid-search:')
    plt.show()
