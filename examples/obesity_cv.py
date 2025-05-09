#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import global_fit, split_prepare_dataset
from auxiliary.logger import logger as log


def main():
    # Create the coordinator
    coordinator = FedHEONN_coordinator(f=f_act, lam=lam, encrypted=enc, ensemble=ens_coord)

    # Generate random indexes
    if ens_coord:
        n_attributes = trainX.shape[1]
        coordinator.calculate_idx_feats(n_estimators, n_attributes, p_feat, b_feat)

    # Cross-validation
    if kfold:
        cv = KFold(n_splits=split)
    else:
        cv = ShuffleSplit(n_splits=split, test_size=0.2, random_state=42)
    acc_glb_splits, w_glb_splits, acc_inc_splits, w_inc_splits = [], [], [], []

    for it, (train_index, test_index) in enumerate(cv.split(trainX, trainY_onehot)):

        # Get split indexes
        log.info(f"Cross validation split: {it+1}")
        trainX_data, trainT_data = trainX[train_index], trainY_onehot[train_index]
        testX_data, testT_data = trainX[test_index], trainY[test_index]
        n_split = trainT_data.shape[0]

        # Create a list of clients and fit clients with their local data
        lst_clients = []
        for i in range(0, n_clients):

            # Split train equally data among clients
            rang = range(int(i * n_split / n_clients), int(i * n_split / n_clients) + int(n_split / n_clients))
            client = FedHEONN_classifier(f=f_act, encrypted=enc, sparse=spr, context=ctx, ensemble=ens_client)
            log.info(f"\t\tTraining client: {i+1} of {n_clients} ({min(rang)}-{max(rang)})")
            if ens_client:
                client.set_idx_feats(coordinator.send_idx_feats())

            # Fit client local data
            client.fit(trainX_data[rang], trainT_data[rang])
            lst_clients.append(client)

        # Perform fit and predict validation split
        acc_glb, w_glb = global_fit(list_clients=lst_clients, coord=coordinator,
                                    testX=testX_data, testT=testT_data, regression=False)

        # Save results
        acc_glb_splits.append(acc_glb)
        w_glb_splits.append(w_glb)

        # Clean coordinator data for the next fold
        coordinator.clean_coordinator()

    log.info(f"CV ACCURACY GLOBAL: MEAN {np.array(acc_glb_splits).mean():.2f} % - STD: {np.array(acc_glb_splits).std():.2f}")


def load_obesity(f_test_size, b_preprocess, b_iid):
    # Fetch and split classification dataset
    skin_segmentation = fetch_ucirepo(id=544)
    X = skin_segmentation.data.features
    y = skin_segmentation.data.targets

    # Inputs and targets encoding
    cat_cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    col_transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop=None, sparse_output=False), cat_cols)],
        remainder='passthrough')
    # OneHotEncoding -> Test with drop='first'
    X_transformed = col_transformer.fit_transform(X)

    encoded_cols = col_transformer.named_transformers_['cat'].get_feature_names_out(cat_cols)
    X_transformed = pd.DataFrame(X_transformed,
                                 columns=list(encoded_cols)+[col for col in X.columns if col not in cat_cols])

    data = X_transformed.to_numpy()
    target = LabelEncoder().fit(y).transform(y)

    log.info(f"[*] OBESITY DATASET ({len(data)} samples, {data.shape[1]} features) [*]")

    # Split, preprocess and encode
    return split_prepare_dataset(X=data, y=target,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)


def alt():
    # Modelos
    logistic_model = LogisticRegression(max_iter=1000)  # Ajusta el max_iter si es necesario
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Validación cruzada (5 folds por defecto)
    logistic_scores = cross_val_score(logistic_model, trainX, trainY, cv=10)
    knn_scores = cross_val_score(knn_model, trainX, trainY, cv=10)

    # Resultados
    print("Logistic Regression CV Accuracy:", logistic_scores.mean())
    print("KNN CV Accuracy:", knn_scores.mean())

if __name__ == "__main__":
    # ----MODEL HYPERPARAMETERS----
    # Number of clients
    n_clients = 1
    # Number of clients per group
    n_groups = 1
    # Randomize number of clients per group in range (n_groups/2, groups*2)
    rnd = False
    # Encryption
    enc = False
    # Sparse matrices
    spr = True
    # Regularization
    lam = .1
    # Activation function
    f_act = 'relu'
    # IID or non-IID scenario (True or False)
    iid = True
    # Preprocess data
    pre = True
    # Ensemble
    bag = True
    # Random Patches bagging parameters
    n_estimators = 50
    p_samples = 0.25
    b_samples = False
    p_feat = 0.9
    b_feat = False
    # Cross-validation
    kfold = True
    split = 10
    # Parallelism
    par = False
    # --------

    # Configuring the TenSEAL context for the CKKS encryption scheme
    ctx = None
    if enc:
        import tenseal as ts

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40
    # Prepare ensemble dictionaries
    ens_client = {'bagging': n_estimators,
                  'bootstrap_samples': b_samples, 'p_samples': p_samples,
                  'bootstrap_features': b_feat, 'p_features': p_feat
                  } if bag else {}
    ens_coord = {'bagging'} if bag else {}

    # Load dataset
    np.random.seed(1)
    trainX, trainY_onehot, testX, testY, trainY = load_obesity(f_test_size=0.3, b_preprocess=pre, b_iid=iid)
    # CROSS VALIDATION MAIN FUNCTION
    #main()
    alt()