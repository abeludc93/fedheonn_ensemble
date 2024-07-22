#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import threading

from algorithm.fedHEONN_clients import FedHEONN_classifier
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from api.client import Client
import tenseal as ts

ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.generate_galois_keys()
ctx.global_scale = 2 ** 40

hostname = "localhost"
port = 8000
client = Client(hostname, port)


print(f"\tPING:\nAPI is {'UP' if client.ping() else 'DOWN'}")
print(f"\tSTATUS:\n{client.get_status()}")
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

trainX, trainY = client.fetch_dataset(length)

# FIT DATA
bag = True
enc = True

print(f"\tTraining client...")
ens_client = FedHEONN_classifier.generate_ensemble_params(n_estimators=8,
                                                          p_samples=1.0, b_samples=True,
                                                          p_features=1.0, b_features=False) if bag else {}
fed_client = FedHEONN_classifier(f='logs', encrypted=enc, sparse=True, context=ctx, ensemble=ens_client)
# DEBUG
ens_coord = FedHEONN_coordinator.generate_ensemble_params() if bag else {}
debug_coord = FedHEONN_coordinator(f='logs', encrypted=enc, sparse=True, ensemble=ens_coord)
if bag:
    debug_coord.calculate_idx_feats(8, 64, 1.0, False)
    fed_client.set_idx_feats(debug_coord.send_idx_feats())

# Fit client local data
fed_client.fit(trainX, trainY)
M_c, US_c = fed_client.get_param()

# Aggregate partial data
#response = client.aggregate_partial(m_data=M_c, US_data=US_c)
data = Client.serialize_client_data(m_data=M_c, US_data=US_c)
response = client.aggregate_partial(data)
#for i in range(20):
#    response = client.aggregate_partial(data)
#    print(f"\tAGGREGATE PARTIAL ({i+1}):\n{response}")

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