#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
examples.codecarbon_mnist_incremental
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Script that performs experiments on the entire MNIST ,
comparing the carbon footprint of different classic and federated models:
    - MultiLayer Perceptron (MLPC) derived from sklearn and tensorflow
    - FedHEONN classic/ensemble, with both plain and encrypted scenarios considered
    -

Some examples were borrowed from:
https://github.com/mlco2/codecarbon/blob/master/examples/
All credit due to the CodeCarbon team.
"""
import numpy as np
import pandas as pd
from algorithm.fedHEONN_clients import FedHEONN_classifier, FedHEONN_client
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from auxiliary.logger import logger as log
from examples.utils import load_mnist_digits_full, group_clients, get_params_group, get_prediction
from codecarbon import EmissionsTracker
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tenseal as ts


class BaseClient:
    def __init__(self, client: FedHEONN_classifier = None, train_x: np.ndarray = None, train_y: np.ndarray = None):
        self.client = client
        self.train_x = train_x
        self.train_y = train_y

    def fit(self):
        self.client.fit(self.train_x, self.train_y)

def fedheonn(cc_project_name: str,  # CodeCarbon project name
             ctx: ts.Context,       # TenSEAL context
             n_clients: int = 1,    # Number of clients
             n_groups: int = 1,     # Number of clients per group (used for partial/incremental fitting)
             enc: bool = False,     # Encryption
             lam: float = 0.01,     # Regularization
             f_act: str = 'logs',   # Activation function
             bag: bool = False,     # Ensemble
             n_estimators: int = 0,     # Random Patches parameters
             p_samples: float = .0,
             b_samples: bool = False,
             p_feat: float = .0,
             b_feat: bool = False,
             par_client: bool = False,  # Parallelism
             par_coord: bool = False
             ):

    # Context
    ctx = ctx if enc else None

    # Prepare ensemble dictionaries
    ens_client = FedHEONN_client.generate_ensemble_params(n_estimators, p_samples, b_samples, p_feat, b_feat) if bag else {}
    ens_coord = {'bagging'} if bag else {}

    # Create the coordinator
    coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord, parallel=par_coord)

    # Train data length
    n, n_attributes = trainX.shape

    # Generate random indexes
    if ens_coord:
        coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feat, b_feat)

    # Create a list of clients and fit clients with their local data
    lst_clients = []
    for i in range(n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        client = FedHEONN_classifier(f=f_act, encrypted=enc, context=ctx, ensemble=ens_client, parallel=par_client)
        if ens_client:
            client.set_idx_feats(coordinator.send_idx_feats())
        lst_clients.append(BaseClient(client, train_x=trainX[rang], train_y=trainY_onehot[rang]))

    # CodeCarbon
    try:
        tracker = EmissionsTracker(project_name=cc_project_name, measure_power_secs=1)

        # Group clients
        acc_lst = []
        groups = group_clients(lst_clients, n_groups, randomize=False)
        for ig, group in enumerate(groups):
            # Fit
            tracker.start_task(f"fit_group_{ig+1}")
            for clt in group:
                clt.fit()
            tracker.stop_task()

            # Aggregate partial model info
            tracker.start_task(f"aggregate_partial_group_{ig + 1}")
            M_grp, US_grp = get_params_group(group=[clt.client for clt in group])
            coordinator.aggregate_partial(M_list=M_grp, US_list=US_grp)
            tracker.stop_task()

            # Calculate opt. weights and realize current predictions
            tracker.start_task(f"calc_optim_weights_group_{ig + 1}")
            coordinator.calculate_weights()
            tracker.stop_task()
            tracker.start_task(f"predict_{ig + 1}")
            acc = get_prediction(lst_clients[0].client, coordinator, testX, testY, regression=False)
            log.info(f"\tAccuracy TEST incremental (group {ig+1}): {acc:0.2f}")
            acc_lst.append(acc)
            tracker.stop_task()
    finally:
        _ = tracker.stop()

    return acc_lst


def sklearn_mlpc(cc_project_name: str, f_act: str = 'relu', n_groups: int = 1):
    # Create MLPC Classifier
    model = MLPClassifier(hidden_layer_sizes=(128, 64),
                          activation=f_act,
                          max_iter=500,
                          tol=1e-4,
                          alpha=1e-4,
                          solver="sgd",
                          verbose=True,
                          random_state=42)

    # CodeCarbon
    acc_lst = []
    try:
        tracker = EmissionsTracker(project_name=cc_project_name, measure_power_secs=1)
        trainX_parts = np.array_split(trainX, n_groups, axis=0)
        trainY_parts = np.array_split(trainY, n_groups, axis=0)
        for i in range(n_groups):
            # Fit model
            trainX_accumulated = np.vstack(tuple(trainX_parts[j] for j in range(i+1)))
            trainY_accumulated = np.vstack(tuple(trainY_parts[j] for j in range(i+1)))
            tracker.start_task(f"fit_group_{i + 1}")
            model.fit(trainX_accumulated, trainY_accumulated)
            tracker.stop_task()

            # Predict and score
            predict_Y = model.predict(testX)
            acc = accuracy_score(testY, predict_Y) * 100
            log.info(f"Accuracy TEST incremental MLPC (group {i+1}): {acc:0.2f} %")
            acc_lst.append(acc)
    finally:
        _ = tracker.stop()

    return acc_lst


if __name__ == "__main__":
    # Configuring the TenSEAL context for the CKKS encryption scheme
    ts_ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ts_ctx.generate_galois_keys()
    ts_ctx.global_scale = 2 ** 40

    # [Experiments]
    exp_results = {}

    # Load entire simplified MNIST dataset from sklearn - 5620 samples
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_mnist_digits_full(f_test_size=0.3, b_preprocess=True, b_iid=True)

    exp_results["FEDHEONN_incremental_12_4"] = fedheonn(cc_project_name="FEDHEONN_incremental_12_4", ctx=ts_ctx, lam=10,
                                                        n_clients=12, n_groups=4)
    exp_results["FEDH_incremental_bagging_12_4"] = fedheonn(cc_project_name="FEDH_incremental_bagging_12_4", ctx=ts_ctx,
                                                            lam=0.1,bag=True, n_estimators=25, b_samples=True,
                                                            b_feat=False, p_samples=0.2, p_feat=1, n_clients=12,
                                                            n_groups=4)
    exp_results["MLPC_incremental_4"] = sklearn_mlpc(cc_project_name="MLPC_incremental_4", n_groups=3)

    # Export results
    pd.set_option("display.precision", 8)
    df = pd.DataFrame(exp_results.items(), columns=['Experiment', 'Accuracy'])
    df.to_csv("emissions_results/results.csv")
