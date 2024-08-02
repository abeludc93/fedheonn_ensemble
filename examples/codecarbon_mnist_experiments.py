#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
examples.codecarbon_mnist_experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Script that performs multiple experiments on both the simplified and original
MNIST dataset (loaded from sklearn and tensorflow datasets respectively),
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
from examples.utils import global_fit, incremental_fit, load_mnist_digits, load_mnist_digits_full, normalize_dataset, \
    shuffle_iid, one_hot_encoding
from codecarbon import EmissionsTracker
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tenseal as ts
import tensorflow as tf


def load_original_mnist(b_preprocess=True, b_iid=True):
    # Load original MNIST from TF repository
    mnist = tf.keras.datasets.mnist
    (train_X, train_t), (test_X, test_t) = mnist.load_data()
    # Flatten the images
    train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))
    # Normalize if necessary
    if b_preprocess:
        train_X, test_X = normalize_dataset(train_data=train_X, test_data=test_X)
    # One-hot encoding
    train_X, train_t = shuffle_iid(trainX=train_X, trainY=train_t, iid=b_iid)
    t_onehot = one_hot_encoding(trainY=train_t)

    return train_X, t_onehot, test_X, test_t, train_t


def fedheonn(cc_project_name: str,  # CodeCarbon project name
             ctx: ts.Context,       # TenSEAL context
             n_clients: int = 1,    # Number of clients
             n_groups: int = 0,     # Number of clients per group (used for partial/incremental fitting)
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

    # Incremental or global fit
    incremental = n_groups > 0

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
    lst_data = []
    for i in range(n_clients):
        # Split train equally data among clients
        rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
        client = FedHEONN_classifier(f=f_act, encrypted=enc, context=ctx, ensemble=ens_client, parallel=par_client)
        if ens_client:
            client.set_idx_feats(coordinator.send_idx_feats())
        lst_clients.append(client)
        lst_data.append((trainX[rang], trainY_onehot[rang]))

    # START TRACKING EMISSIONS - FIT MODEL AND PREDICT
    with EmissionsTracker(project_name=cc_project_name, measure_power_secs=1):
        # Fit client local data
        for i in range(n_clients):
            log.info(f"\t\tTraining client: {i + 1} of {n_clients}")
            client = lst_clients[i]
            train_x, train_y = lst_data[i]
            client.fit(train_x, train_y)

        # Aggregate matrix's & calculate optimal weights on coordinator and predict on test data:
        if incremental:
            acc, _ = incremental_fit(list_clients=lst_clients, ngroups=n_groups, coord=coordinator,
                                     testX=testX, testT=testY, regression=False, random_groups=False)
        else:
            acc, _ = global_fit(list_clients=lst_clients, coord=coordinator,
                                testX=testX, testT=testY, regression=False)

        # Print model's metrics
        log.info(f"Test accuracy {'incremental' if incremental else 'global'}: {acc:0.2f}")

    return acc


def sklearn_mlpc(cc_project_name: str, f_act: str = 'relu'):
    # Create MLPC Classifier
    model = MLPClassifier(hidden_layer_sizes=(128, 64),
                          activation=f_act,
                          max_iter=500,
                          tol=1e-4,
                          alpha=1e-4,
                          solver="sgd",
                          verbose=True,
                          random_state=42)
    with EmissionsTracker(project_name=cc_project_name, measure_power_secs=1):
        # Fit model
        model.fit(trainX, trainY)
        # Predict and score
        predict_Y = model.predict(testX)
        acc = accuracy_score(testY, predict_Y) * 100
        log.info(f"Test accuracy MLPC: {acc:0.2f} %")

    return acc


def keras_model(cc_project_name: str, f_act: str = 'relu'):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(128, activation=f_act),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    with EmissionsTracker(project_name=cc_project_name, measure_power_secs=1):
        model.fit(trainX, trainY, epochs=10)
        # Predict and score
        loss, acc = model.evaluate(testX, testY)
        log.info(f"Test accuracy KERAS: {acc * 100:0.2f} %")

    return acc * 100


if __name__ == "__main__":
    # Configuring the TenSEAL context for the CKKS encryption scheme
    ts_ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ts_ctx.generate_galois_keys()
    ts_ctx.global_scale = 2 ** 40

    # [Experiments]
    exp_results = {}

    # Load simplified MNIST dataset from sklearn (copy of the test set of the UCI ML digits dataset) - 1797 samples
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_mnist_digits(f_test_size=0.3, b_preprocess=True, b_iid=True)

    exp_results["MLPC_mnist"] = sklearn_mlpc(cc_project_name="MLPC_mnist")
    exp_results["FEDH_mnist"] = fedheonn(cc_project_name="FEDH_mnist", ctx=ts_ctx, lam=10)
    exp_results["FEDH_mnist_enc"] = fedheonn(cc_project_name="FEDH_mnist_enc", ctx=ts_ctx, lam=10, enc=True)
    exp_results["FEDH_mnist_bag_best"] = fedheonn(cc_project_name="FEDH_mnist_bag_best", ctx=ts_ctx, lam=0.01,
                                                  bag=True, n_estimators=75, b_samples=True, b_feat=False,
                                                  p_samples=0.2, p_feat=0.8)
    exp_results["FEDH_mnist_bag_alt"] = fedheonn(cc_project_name="FEDH_mnist_bag_alt", ctx=ts_ctx, lam=0.1,
                                                 bag=True, n_estimators=25, b_samples=True, b_feat=False,
                                                 p_samples=0.2, p_feat=1)


    # Load entire simplified MNIST dataset from sklearn - 5620 samples
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_mnist_digits_full(f_test_size=0.3, b_preprocess=True, b_iid=True)

    exp_results["MLPC_mnist_full"] = sklearn_mlpc(cc_project_name="MLPC_mnist_full")
    exp_results["FEDH_mnist_full"] = fedheonn(cc_project_name="FEDH_mnist_full", ctx=ts_ctx, lam=10)
    exp_results["FEDH_mnist_full_enc"] = fedheonn(cc_project_name="FEDH_mnist_full_enc", ctx=ts_ctx, lam=10, enc=True)
    exp_results["FEDH_mnist_full_bag_best"] = fedheonn(cc_project_name="FEDH_mnist_full_bag_best", ctx=ts_ctx, lam=0.01,
                                                       bag=True, n_estimators=75, b_samples=True, b_feat=False,
                                                       p_samples=0.2, p_feat=0.8)
    exp_results["FEDH_mnist_full_bag_alt"] = fedheonn(cc_project_name="FEDH_mnist_full_bag_alt", ctx=ts_ctx, lam=0.1,
                                                      bag=True, n_estimators=25, b_samples=True, b_feat=False,
                                                      p_samples=0.2, p_feat=1)

    # Load entire simplified MNIST dataset from sklearn - 70000 samples
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_original_mnist(b_preprocess=True, b_iid=True)

    exp_results["MLPC_mnist_orig"] = sklearn_mlpc(cc_project_name="MLPC_mnist_orig")
    exp_results["FEDH_mnist_orig"] = fedheonn(cc_project_name="FEDH_mnist_orig", ctx=ts_ctx, lam=10)
    exp_results["FEDH_mnist_orig_enc"] = fedheonn(cc_project_name="FEDH_mnist_orig_enc", ctx=ts_ctx, lam=10, enc=True)
    exp_results["FEDH_mnist_orig_bag_alt"] = fedheonn(cc_project_name="FEDH_mnist_orig_bag_alt", ctx=ts_ctx, lam=0.1,
                                                      bag=True, n_estimators=25, b_samples=True, b_feat=False,
                                                      p_samples=0.2, p_feat=1)
    exp_results["FEDH_mnist_orig_bag_alt_par"] = fedheonn(cc_project_name="FEDH_mnist_orig_bag_alt_par", ctx=ts_ctx, lam=0.1,
                                                          bag=True, n_estimators=25, b_samples=True, b_feat=False,
                                                          p_samples=0.2, p_feat=1, par_client=True, par_coord=True)
    exp_results["KERAS_mnist_orig"] = keras_model(cc_project_name="KERAS_mnist_orig")

    # Export results
    pd.set_option("display.precision", 8)
    df = pd.DataFrame(exp_results.items(), columns=['Experiment', 'Accuracy'])
    df.to_csv("emissions_results/results.csv")
