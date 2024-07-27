#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from sklearn.metrics import accuracy_score
from algorithm.fedHEONN_clients import FedHEONN_classifier
from api.client import Client
import tenseal as ts
from api.utils import serialize_client_data

ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.generate_galois_keys()
ctx.global_scale = 2 ** 40

hostname = "localhost"
port = 8000
client = Client(hostname, port)

# Coordinator hyperparameters
f_act = 'logs'
lam = 0.01
enc = True
spr = True
bag = True
par = False


print(f"\tPING:\nAPI is {'UP' if client.ping() else 'DOWN'}")
print(f"\tSTATUS:\n{client.get_status()}")
# HYPERPARAMS
response = client.update_coordinator_parameters(f_act=f_act, lam=lam, spr=spr, enc=enc, par=par, bag=bag, ctx_str=None)
print(f"{response}\n\tSTATUS:\n{client.get_status()}")

# CONTEXT
client.send_context(context_name="test_ts", context=ctx)
print(f"\tSTATUS:\n{client.get_status()}")
client.select_context("test_ts")
print(f"\tSTATUS:\n{client.get_status()}")

# DATASET
response = client.select_dataset("mnist")
print(f"dataset select response: {response}")
response = client.load_dataset()
length = int(response)
print(f"dataset load response: {length}")
# FETCH TRAIN DATA
trainX, trainY = client.fetch_dataset(length//10)
# FETCH TEST DATA
testX, testY = client.fetch_dataset_test()

# FIT DATA
enc = True
bag = True
n_estimators = 8
p_samples = 1.0
b_samples = True
p_features = 1.0
b_features = False

print(f"\tTraining client...")
ens_client = FedHEONN_classifier.generate_ensemble_params(n_estimators=n_estimators,
                                                          p_samples=p_samples, b_samples=b_samples,
                                                          p_features=p_features, b_features=b_features) if bag else {}
fed_client = FedHEONN_classifier(f='logs', encrypted=enc, sparse=True, context=ctx, ensemble=ens_client)

if bag:
    n_features = trainX.shape[1]
    response = client.calculate_index_features(n_estimators=n_estimators, n_features=n_features,
                                    p_features=p_features, b_features=b_features)
    print(f"{response}")
    idx_feats = client.get_index_features()
    print(f"{type(idx_feats)}: {idx_feats}")
    fed_client.set_idx_feats(idx_feats)


# Fit client local data
fed_client.fit(trainX, trainY)
M_c, US_c = fed_client.get_param()

# Aggregate partial data

data = serialize_client_data(m_data=M_c, US_data=US_c)
for i in range(5):
    response, data_id = client.aggregate_partial(data)
    print(f"\tAGGREGATE PARTIAL ({i+1}):\n[{data_id}]: {response}")

# Receive weights
input("Receive weights and predict? ")
response = client.receive_weights()
print(f"WEIGHTS: {len(response)}")
fed_client.set_weights(W=response, serialized=True)

# PREDICT on TEST DATA
test_predict = fed_client.predict(testX)
metric = 100 * accuracy_score(testY, test_predict)
print(f"Accuracy on whole TEST data: {metric}")


"""
threads = list()
for index in range(20):
    print(f"Main : create and start thread {index}")
    x = threading.Thread(target=client.aggregate_partial, args=(data,))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    print(f"Main : before joining thread {index}")
    thread.join()
    print(f"Main : thread {index} done")
"""