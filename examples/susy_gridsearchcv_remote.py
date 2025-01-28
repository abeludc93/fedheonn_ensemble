#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
# Application modules
from examples.utils import gridsearch_cv_classification, export_dataframe_results, split_prepare_dataset
from auxiliary.logger import logger as log


def load_susy(f_test_size=0.3, b_preprocess=True, b_iid=True):

    # Load file
    Data = pd.read_csv('../datasets/SUSY.csv', delimiter=',')
    Inputs = Data.iloc[:, 1:].to_numpy()
    Targets = Data.iloc[:, 0].to_numpy()

    log.info(f"[*] SUSY ({len(Inputs)} samples) [*]")
    # Split, preprocess and encode
    return split_prepare_dataset(X=Inputs, y=Targets,
                                 test_size=f_test_size, preprocess=b_preprocess, iid=b_iid, regression=False)


def main():

    lmbda = [.001]
    n_est = [2, 10, 50, 100]
    betas = [False]

    # Load dataset
    trainX, trainY_onehot, testX, testY, trainY = load_susy(f_test_size=0.3, b_preprocess=pre, b_iid=iid)

    # HYPER-PARAMETER SEARCH GRID
    # Classic execution
    dict_no_bag = gridsearch_cv_classification(f_activ=f_act, sparse=spr, encryption=enc, context=ctx,
                                               cv_type=kfold, n_splits=splits, bagging=False, train_X=trainX,
                                               train_Y_onehot=trainY_onehot, train_Y=trainY, clients=n_clients,
                                               lambda_lst=lmbda, n_estimators_lst=n_est, b_lst=betas)
    # Ensemble execution
    dict_bag = gridsearch_cv_classification(f_activ=f_act, sparse=spr, encryption=enc, context=ctx,
                                            cv_type=kfold, n_splits=splits, bagging=True, train_X=trainX,
                                            train_Y_onehot=trainY_onehot, train_Y=trainY, clients=n_clients,
                                            lambda_lst=lmbda, n_estimators_lst=n_est, b_lst=betas)

    # EXPORT RESULTS
    export_dataframe_results(dict_no_bag=dict_no_bag, dict_bag=dict_bag, dataset_name="SUSY", regression=False)


if __name__ == "__main__":
    # ----MODEL HYPERPARAMETERS----
    # Regression
    reg = False
    # Number of clients
    n_clients = 100
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
