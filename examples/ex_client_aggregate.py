#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from sklearn.metrics import accuracy_score
from algorithm.fedHEONN_clients import FedHEONN_classifier
from api.client import Client
import tenseal as ts
from api.utils import serialize_client_data
from auxiliary.logger import logger as log

#---------------------------------
# TenSEAL context
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.generate_galois_keys()
ctx.global_scale = 2 ** 40
# Server IP/Port
hostname = "localhost"
port = 8000
# Client
client = Client(hostname, port)
# Coordinator hyperparameters
f_act = 'logs'
lam = 0.01
enc = True
spr = True
bag = True
par = False
# Client parameters (bagging - random patches)
n_estimators = 8
p_samples = 1.0
b_samples = True
p_features = 1.0
b_features = False
#---------------------------------

# Initial conditions
log.info(f"Checking if server is up....")
log.info(f"\t[API]: {'UP' if client.ping() else 'DOWN'}")
log.info(f"Initial STATUS:")
log.info(f"{client.get_status()}")

# HYPERPARAMS
input("\tContinue setting up coordinator parameters?: ")
response = client.update_coordinator_parameters(f_act=f_act, lam=lam, spr=spr, enc=enc, par=par, bag=bag, ctx_str=None)
log.info(f"[update_coordinator_parameters] response: {response}")

# CONTEXT
input("\tContinue uploading tenSEAL context and selecting it?: ")
response = client.send_context(context_name="test_ts", context=ctx)
log.info(f"[send_context] response: {response}")
client.select_context("test_ts")

# DATASET
input("\tContinue selecting and loading dataset?: ")
response = client.select_dataset("mnist")
log.info(f"[select_dataset] response: {response}")
response = client.load_dataset()
length = int(response)
log.info(f"[load_dataset] response: {length}")
log.info(f"Current STATUS:")
log.info(f"{client.get_status()}")

# FETCH TRAIN AND TEST DATA
input("\tFetch train and test data?: ")
trainX, trainY = client.fetch_dataset(length//10)
testX, testY = client.fetch_dataset_test()
log.info("Train and test data received!")

input("\tTrain client?: ")
# Instantiate FedHEONN client
ens_client = FedHEONN_classifier.generate_ensemble_params(n_estimators=n_estimators,
                                                          p_samples=p_samples, b_samples=b_samples,
                                                          p_features=p_features, b_features=b_features) if bag else {}
fed_client = FedHEONN_classifier(f='logs', encrypted=enc, sparse=True, context=ctx, ensemble=ens_client)
# Bagging
if bag:
    n_features = trainX.shape[1]
    response = client.calculate_index_features(n_estimators=n_estimators, n_features=n_features,
                                               p_features=p_features, b_features=b_features)
    log.info(f"\tBagging enabled: {response}")
    idx_feats = client.get_index_features()
    log.debug(f"\t\tRandomly picked index features: {idx_feats}")
    fed_client.set_idx_feats(idx_feats)

# Fit client local data
log.info(f"[Training client...]")
fed_client.fit(trainX, trainY)
M_c, US_c = fed_client.get_param()
log.info(f"Client trained!")

# Aggregate partial data
input("\tAggregate client data?: ")
data = serialize_client_data(m_data=M_c, US_data=US_c)
for i in range(5):
    log.info(f"\t[aggregate_partial] ({i + 1}):")
    response, data_id = client.aggregate_partial(data)
    log.info(f"[{data_id}]: {response}")

# Receive weights
input("\tReceive weights and predict?: ")
response = client.receive_weights()
print(f"[receive_weights]: {len(response) if response is not None else 'NONE'}")
fed_client.set_weights(W=response, serialized=True)

# Predict on test data
test_predict = fed_client.predict(testX)
metric = 100 * accuracy_score(testY, test_predict)
log.info(f"Accuracy on whole TEST data: {metric:.2f}")
