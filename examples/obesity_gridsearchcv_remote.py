#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ucimlrepo import fetch_ucirepo

# Application modules
from examples.utils import gridsearch_cv_classification, export_dataframe_results, split_prepare_dataset
from auxiliary.logger import logger as log


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


def main():

    # Load dataset
    trainX, trainY_onehot, testX, testY, trainY = load_obesity(f_test_size=0.3, b_preprocess=pre, b_iid=iid)

    # HYPER-PARAMETER SEARCH GRID
    # Classic execution
    dict_no_bag = gridsearch_cv_classification(f_activ=f_act, sparse=spr, encryption=enc, context=ctx,
                                               cv_type=kfold, n_splits=splits, bagging=False, train_X=trainX,
                                               train_Y_onehot=trainY_onehot, train_Y=trainY, clients=n_clients)
    # Ensemble execution
    dict_bag = gridsearch_cv_classification(f_activ=f_act, sparse=spr, encryption=enc, context=ctx,
                                            cv_type=kfold, n_splits=splits, bagging=True, train_X=trainX,
                                            train_Y_onehot=trainY_onehot, train_Y=trainY, clients=n_clients)

    # EXPORT RESULTS
    export_dataframe_results(dict_no_bag=dict_no_bag, dict_bag=dict_bag, dataset_name="Obesity", regression=False)


if __name__ == "__main__":
    # ----MODEL HYPERPARAMETERS----
    # Regression
    reg = False
    # Number of clients
    n_clients = 1
    # Encryption
    enc = False
    # Sparse matrices
    spr = True
    # Activation function
    f_act = 'logs'
    # IID or non-IID scenario (True or False)
    iid = True
    # Preprocess data
    pre = True
    # Cross-validation
    kfold = True
    splits = 10
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
    # --------
    # Extract two best models (from classic execution and ensemble execution), train and evaluate on test!
    main()
