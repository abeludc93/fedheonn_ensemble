#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module containing auxiliary functions used in project's examples
"""
# Standard libraries
from random import seed, shuffle, randint
from itertools import product
import time
import os
# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
# Application modules
from algorithm.fedHEONN_clients import FedHEONN_classifier, FedHEONN_regressor
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.logger import logger as log


# Seed random numbers
seed(1)

# Function that yields an iterator with the cartesian product of the given iterables
def generate_grid_search_iterator(lambda_grid, n_estimators_grid,
                                  p_samples_grid=None, p_features_grid=None,
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

# Function to obtain prediction metrics for given (testX,testT) data using one client and a coordinator
def get_prediction(ex_client, coord, testX, testT, regression=True):
    # Send the weights of the aggregate model to example client
    ex_client.set_weights(coord.send_weights())
    # Predictions for the test set using one client
    test_y = ex_client.predict(testX)
    if regression:
        # Global MSE for the 3 outputs
        metric = mean_squared_error(testT, test_y)
    else:
        # Global precision for all outputs (percentage)
        metric = 100 * accuracy_score(testT, test_y)

    return metric


# Function that groups the given list of clients in batches of 'ngroups' clients
def group_clients(list_clients, ngroups, randomize=False):
    # List of groups
    groups = []
    # Flag to randomize the number of clients per group
    if not randomize:
        # Group clients
        for i in range(0, len(list_clients), ngroups):
            log.debug(f"\t\tGrouping clients: ({i}:{i + ngroups})")
            group = list_clients[i:i + ngroups]
            groups.append(group)
    else:
        # Group clients randomly
        idx = 0
        while idx < len(list_clients):
            n_groups_rnd = randint(ngroups // 2, 2 * ngroups)
            if idx + n_groups_rnd > len(list_clients):
                n_groups_rnd = len(list_clients) - idx
            log.debug(f"\t\tGrouping clients randomly: ({idx}:{idx + n_groups_rnd})")
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
    debug = False
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
        log.debug(f"\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
        # Fit client local data
        if ens_client:
            client.set_idx_feats(coord.send_idx_feats())
        client.fit(trainX[rang], trainY[rang])

        lst_clients.append(client)

    return lst_clients

def normalize_dataset(train_data, test_data):
    # Data normalization (z-score): mean 0 and std 1
    log.info("\t\tData normalization (z-score)")
    scaler = StandardScaler().fit(train_data)
    trainX = scaler.transform(train_data)
    testX = scaler.transform(test_data)
    return trainX, testX

def shuffle_iid(trainX, trainY, iid=True):
    # Number of training and test data
    n = len(trainY)
    # Non-IID option: Sort training data by class
    if not iid:
        log.info('\t\tnon-IID scenario')
        ind = np.argsort(trainY)
        train_X = trainX[ind]
        train_t = trainY[ind]
    else:
        # Data are shuffled in case they come ordered by class
        log.info('\t\tIID scenario')
        ind_list = list(range(n))
        np.random.seed(1)
        np.random.shuffle(ind_list)
        train_X = trainX[ind_list]
        train_t = trainY[ind_list]

    return train_X, train_t

def one_hot_encoding(trainY):
    # Number of classes
    n, n_classes = len(trainY), len(np.unique(trainY))
    # One hot encoding for the targets
    t_onehot = np.zeros((n, n_classes))
    for i, value in enumerate(trainY):
        t_onehot[i, value] = 1

    return t_onehot

def split_prepare_dataset(X, y, test_size, preprocess, iid, regression):
    # Traint-Test split
    train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalize if necessary
    if preprocess:
        train_X, test_X = normalize_dataset(train_data=train_X, test_data=test_X)

    # IID scenario
    train_X, train_t = shuffle_iid(trainX=train_X, trainY=train_t, iid=iid)

    # One-hot encoding
    if not regression:
        t_onehot = one_hot_encoding(trainY=train_t)
        return train_X, t_onehot, test_X, test_t, train_t
    else:
        return train_X, train_t, test_X, test_t

# Function to load and prepare MNIST dataset
def load_mnist_digits(f_test_size=0.3, b_preprocess=True, b_iid=True):

    log.info("[*] MNIST DIGITS DATASET [*]")
    # Load dataset
    digits = load_digits()
    X_data, y_data = digits.data, digits.target

    # Split, preprocess and encode
    return split_prepare_dataset(X=X_data, y=y_data,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)

# Function to load and prepare the Skin dataset
def load_skin_dataset(f_test_size=0.3, b_preprocess=True, b_iid=True):

    log.info("[*] SKIN DATASET [*]")
    # Create and split classification dataset
    skin_segmentation = fetch_ucirepo(id=229)
    X = skin_segmentation.data.features
    y = skin_segmentation.data.targets
    data = X.to_numpy()
    target = y.to_numpy()
    target = target.reshape(target.shape[0]) - 1 # Target classes original values: 1-2 => Target classes: 0-1

    # Split, preprocess and encode
    return split_prepare_dataset(X=data, y=target,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)

# Function to load and prepare carbon-nanotube dataset
def load_carbon_nanotube(f_test_size=0.3, b_preprocess=True):

    log.info("[*] CARBON NANOTUBE DATASET [*]")
    # Read dataset
    Data = pd.read_csv('../datasets/carbon_nanotubes.csv', delimiter=';')
    Inputs = Data.iloc[:, :-3].to_numpy()
    Targets = Data.iloc[:, -3:].to_numpy()  # 3 outputs to predict

    # Split, preprocess and encode
    return split_prepare_dataset(X=Inputs, y=Targets,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=False, regression=True)

# Function to load and prepare Dry_Bean dataset
def load_dry_bean(f_test_size=0.3, b_preprocess=True, b_iid=True):

    log.info("[*] DRY BEAN DATASET [*]")
    # Read dataset
    Data = pd.read_excel('../datasets/Dry_Bean_Dataset.xlsx', sheet_name='Dry_Beans_Dataset')
    Data['Class'] = Data['Class'].map(
        {'BARBUNYA': 0, 'BOMBAY': 1, 'CALI': 2, 'DERMASON': 3, 'HOROZ': 4, 'SEKER': 5, 'SIRA': 6})
    Inputs = Data.iloc[:, :-1].to_numpy()
    Labels = Data.iloc[:, -1].to_numpy()

    # Split, preprocess and encode
    return split_prepare_dataset(X=Inputs, y=Labels,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)

def load_mini_boone(f_test_size=0.3, b_preprocess=True, b_iid=True):

    log.info("[*] MINI BOONE DATASET [*]")
    # Read dataset
    with open('../datasets/MiniBooNE_PID.txt') as f:
        first_line = f.readline()
    first_line = first_line.rstrip('\n').lstrip(' ').split(' ')
    n_signal, n_background = int(first_line[0]), int(first_line[1])
    data = pd.read_table('../datasets/MiniBooNE_PID.txt', header=None, skiprows=1, sep='\s+')
    labels_signal, labels_background = np.zeros(n_signal, dtype=int), np.ones(n_background, dtype=int)

    Inputs = data.to_numpy()
    Labels = np.append(labels_signal, labels_background)

    # Split, preprocess and encode
    return split_prepare_dataset(X=Inputs, y=Labels,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)

# Function that performs a cross-validation grid-search on a certain classification dataset
def gridsearch_cv_classification(f_activ, sparse, encryption, context, cv_type, n_splits, bagging,
                                 train_X, train_Y_onehot, train_Y, clients):
    # Hyperparameter search grid
    lambda_lst          = [0.01, 0.1, 1, 10]
    n_estimators_lst    = [2, 5, 10, 25, 50, 75, 100, 200]
    p_samples_lst       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p_features_lst      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Ensemble method
    if bagging:
        gs_space = np.prod([len(i) for i in [lambda_lst, n_estimators_lst, p_samples_lst, p_features_lst, [True,False], [True,False]]])
        gs_it = generate_grid_search_iterator(lambda_lst, n_estimators_lst, p_samples_lst, p_features_lst)
    else:
        gs_space = len(lambda_lst)
        gs_it = lambda_lst
    log.info(f"Grid search hyper-parameter space: {gs_space}")

    # Pandas dataframe dictionary
    df_dict = {"LAMBDA": [], "N_ESTIMATORS": [], "B_SAMPLES": [], "B_FEATS": [], "P_SAMPLES": [], "P_FEATS": [],
               "METRIC_MEAN": [], "METRIC_STD": []}
    # MAIN LOOP
    for idx, tuple_it in enumerate(gs_it):
        # Construct parameters
        log.info(f"GS ITER {idx+1} of {gs_space}")
        if bagging:
            lam, n_estimators, b_samples, b_feats, p_samples, p_feats = tuple_it
            ens_client = {'bagging': n_estimators,
                      'bootstrap_samples': b_samples, 'p_samples': p_samples,
                      'bootstrap_features': b_feats, 'p_features': p_feats
                      }
            ens_coord = {'bagging'}
        else:
            lam = tuple_it
            ens_client = {}
            ens_coord = {}

        # Create the coordinator
        coordinator = FedHEONN_coordinator(f=f_activ, lam=lam, encrypted=encryption, ensemble=ens_coord)
        # Generate random indexes
        if ens_coord:
            n_attributes = train_X.shape[1]
            coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feats, b_feats)

        # Cross-validation
        if cv_type:
            cv = KFold(n_splits=n_splits)
        else:
            cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        acc_glb_splits = []

        # CV Loop
        for it, (train_index, test_index) in  enumerate(cv.split(train_X, train_Y_onehot)):

            # Get split indexes
            log.info(f"\tCross validation split: {it+1}")
            trainX_data, trainT_data = train_X[train_index], train_Y_onehot[train_index]
            testX_data, testT_data = train_X[test_index], train_Y[test_index]
            n_split = trainT_data.shape[0]

            # Create a list of clients and fit clients with their local data
            lst_clients = []
            for i in range(clients):

                # Split train equally data among clients
                rang = range(int(i * n_split / clients), int(i * n_split / clients) + int(n_split / clients))
                client = FedHEONN_classifier(f=f_activ, encrypted=encryption, sparse=sparse, context=context, ensemble=ens_client)
                log.debug(f"\t\tTraining client: {i+1} of {clients} ({min(rang)}-{max(rang)})")
                if ens_client:
                    client.set_idx_feats(coordinator.send_idx_feats())

                # Fit client local data
                client.fit(trainX_data[rang], trainT_data[rang])
                lst_clients.append(client)

            # Perform fit and predict validation split
            acc_glb, _ = global_fit(list_clients=lst_clients, coord=coordinator, testX=testX_data, testT=testT_data, regression=False)
            acc_glb_splits.append(acc_glb)

        # Add results to dataframe dictionary
        df_dict["LAMBDA"].append(lam)
        df_dict["METRIC_MEAN"].append(np.array(acc_glb_splits).mean())
        df_dict["METRIC_STD"].append(np.array(acc_glb_splits).std())
        df_dict["N_ESTIMATORS"].append(n_estimators if bagging else None)
        df_dict["B_SAMPLES"].append(b_samples if bagging else None)
        df_dict["B_FEATS"].append(b_feats if bagging else None)
        df_dict["P_SAMPLES"].append(p_samples if bagging else None)
        df_dict["P_FEATS"].append(p_feats if bagging else None)

    return df_dict

# Function that exports grid-search-cv results to an Excel file
def export_dataframe_results(dict_no_bag, dict_bag, dataset_name, regression=False):

    # Construct dataframe, sort and export data
    pd.set_option("display.precision", 8)
    df1 = pd.DataFrame(dict_no_bag)
    df1.sort_values(by="METRIC_MEAN", inplace=True, ascending=regression)
    log.info(f"WITHOUT BAGGING:\n{df1.head()}")
    df2 = pd.DataFrame(dict_bag)
    df2.sort_values(by="METRIC_MEAN", inplace=True, ascending=regression)
    log.info(f"WITH BAGGING:\n{df2.head()}")

    # ExcelWriter to export
    lt = time.localtime()
    timestamp = f"{lt.tm_year}{lt.tm_mon}{lt.tm_mday}_{lt.tm_hour}{lt.tm_min}"
    filename = f"GridSearchCV_{dataset_name}_{timestamp}.xlsx"
    file_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + filename)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="CLASSIC", index=False)
        df2.to_excel(writer, sheet_name="ENSEMBLE", index=False)
