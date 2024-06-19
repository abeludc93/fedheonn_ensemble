#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from eeval import Client
import tenseal as ts


hostname = "localhost"
port = 8000
client = Client(hostname, port)

# prepare the TenSEAL context
ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 60])
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
# we know that the model hosted on the server needs a vector of 16 as input
vec = [0.1] * 16
enc_vec = ts.ckks_vector(ctx, vec)

# print some info about the models hosted on the server
models = client.list_models()
print("============== Models ==============")
print("")
for i, model in enumerate(models):
    print(f"[{i + 1}] Model {model['model_name']}:")
    print(f"[*] Description: {model['description']}")
    print(f"[*] Versions: {model['versions']}")
    print(f"[*] Default version: {model['default_version']}")
    print("")

print("====================================")

# send the encrypted input and get encrypted output
result = client.evaluate(model_name="LinearLayer", context=ctx, ckks_vector=enc_vec)
print(f"decrypted result from the server: {result.decrypt()}")
