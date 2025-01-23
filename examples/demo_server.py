#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from api.client import Client
from api.utils import ServerError
from auxiliary.config_loader import load_config
import tenseal as ts


def setup_server():
    cfg = load_config()

    # IP/Port
    hostname, port = cfg['server']['hostname'], cfg['server']['port']
    client = Client(hostname=hostname, port=port)
    if not client.ping():
        raise ServerError('Server is not up or not reachable!')

    # Select and load dataset
    dataset = cfg['dataset']['selected_dataset']
    client.select_dataset(dataset)
    client.load_dataset()

    # Set up coordinator parameters
    activation_function = cfg['fedheonn']['activation_function']
    encrypted = cfg['fedheonn']['encrypted']
    sparse = cfg['fedheonn']['sparse']
    ensemble = cfg['fedheonn']['ensemble']
    regularization = cfg['fedheonn_coord']['regularization']
    parallel = cfg['fedheonn_coord']['parallel']
    client.update_coordinator_parameters(f_act=activation_function, lam=regularization, spr=sparse,
                                         enc=encrypted, par=parallel, bag=ensemble, ctx_str=None)

    # Randomize index features if necessary
    if ensemble:
        n_estimators = cfg['fedheonn_client']['n_estimators']
        p_features = cfg['fedheonn_client']['p_features']
        b_features = cfg['fedheonn_client']['b_features']
        # Hacky way of getting the feature length of the training data
        n_features = client.get_status().dataset_report.train_features
        client.calculate_index_features(n_estimators=n_estimators, n_features=n_features,
                                        p_features=p_features, b_features=b_features)

    # Upload context if necessary (TODO: test setting ctx_str on coordinator)
    if encrypted:
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40
        context_name = "demo_server_context"
        ctx_path = client.send_context(context_name=context_name, context=ctx)
        client.select_context(context_name=context_name)


# Run the main function
if __name__ == '__main__':
    setup_server()
