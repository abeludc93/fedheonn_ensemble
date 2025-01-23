#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from sklearn.metrics import accuracy_score, mean_squared_error
from algorithm.fedHEONN_clients import FedHEONN_regressor, FedHEONN_classifier, FedHEONN_client
from api.client import Client
from api.utils import ServerError, serialize_client_data
from auxiliary.config_loader import load_config


def setup_client():
    cfg = load_config()

    # IP/Port
    hostname, port = cfg['server']['hostname'], cfg['server']['port']
    client = Client(hostname=hostname, port=port)
    if not client.ping():
        raise ServerError('Server is not up or not reachable!')

    # FedHEONN client hyperparameters
    classification = cfg['fedheonn_client']['classification']
    activation_function = cfg['fedheonn']['activation_function']
    encrypted = cfg['fedheonn']['encrypted']
    sparse = cfg['fedheonn']['sparse']
    ensemble = cfg['fedheonn']['ensemble']
    n_estimators = cfg['fedheonn_client']['n_estimators']
    p_features = cfg['fedheonn_client']['p_features']
    b_features = cfg['fedheonn_client']['b_features']
    p_samples = cfg['fedheonn_client']['p_samples']
    b_samples = cfg['fedheonn_client']['b_samples']
    parallel = cfg['fedheonn_client']['parallel']

    # Random Patches (bagging) parameters if ensemble is enabled
    if ensemble:
        ensemble = FedHEONN_client.generate_ensemble_params(n_estimators=n_estimators,
                                                            p_samples=p_samples, b_samples=b_samples,
                                                            p_features=p_features, b_features=b_features)
    else:
        ensemble = {}

    # Download selected context from server if encryption is enabled
    ctx = None
    if encrypted:
        context_name = "demo_server_context" # TODO: make get_context return CURRENT_CONTEXT if no name is provided
        ctx = client.get_context(context_name=context_name)

    if classification:
        fed_client = FedHEONN_classifier(f=activation_function, encrypted=encrypted, sparse=sparse,
                                         context=ctx, ensemble=ensemble, parallel=parallel)
    else:
        fed_client = FedHEONN_regressor(f=activation_function, encrypted=encrypted, sparse=sparse,
                                        context=ctx, ensemble=ensemble, parallel=parallel)

    return client, fed_client


def fetch_train_data_client(srv_client):
    # Change accordingly on each embedded client
    client_train_size = 300
    return srv_client.fetch_dataset(client_train_size)


def fetch_test_data_client(srv_client):
    return srv_client.fetch_dataset_test()


def train_client(client, fed_client):
    train_x, train_y = fetch_train_data_client(client)
    input("\tStart training client?: ")
    print(f"Training client...")
    t_ini = time.perf_counter()
    fed_client.fit(train_x, train_y)
    M_c, US_c = fedHEONN_client.get_param()
    t_end = time.perf_counter()
    print(f"Elapsed time: {t_end-t_ini:.2f} s")
    print(f"Sending fitted train data...")
    data = serialize_client_data(m_data=M_c, US_data=US_c)
    response, data_id = client.aggregate_partial(data)
    print(f"{response}: {data_id}")
    return data_id


def check_status(client, data_id):
    while input("Check queue status of partial data? (y/n): ").lower().startswith("y"):
        print(client.check_aggregate_status(data_id))


def predict_test(client, fed_client):
    test_x, test_y = fetch_test_data_client(client)
    input("Receive weights and predict?: ")
    # Receive weights
    response = client.receive_weights()
    fed_client.set_weights(W=response, serialized=fed_client.encrypted)
    # Predict on test data
    test_predict = fed_client.predict(test_x)
    if type(fed_client) == FedHEONN_classifier:
        metric = 100 * accuracy_score(test_y, test_predict)
        print(f"Accuracy on whole TEST data: {metric:.2f}")
    else:
        metric = mean_squared_error(test_y, test_predict)
        print(f"Accuracy on whole TEST data: {metric:.2f}")


# Run the main function
if __name__ == '__main__':
    server_client, fedHEONN_client = setup_client()
    data_uuid = train_client(server_client, fedHEONN_client)
    check_status(server_client, data_uuid)
    predict_test(server_client, fedHEONN_client)
