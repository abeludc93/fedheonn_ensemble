#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from api.client import Client
from api.utils import Answer418

hostname = "localhost"
port = 8000
client = Client(hostname, port)

ping = client.ping()
print(f"API is {'UP' if ping else 'DOWN'}")

status = client.get_status()
print(f"{status}")


import tenseal as ts
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=32768,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
ctx.generate_galois_keys()
ctx.global_scale = 2 ** 40

print(f"Uploaded context: PUBLIC - {ctx.is_public()} HAS_SK - {ctx.has_secret_key()}")

upload_ctx = client.send_context(context_name="test_ts", context=ctx)
print("Could NOT upload context to the server" if upload_ctx is None else f"Uploaded context to server: {upload_ctx}")

upload_ctx = client.send_context(context_name="test_ts2", context=ctx)

ctx_received = client.get_context(context_name="test_ts")
print(f"Downloaded context: PUBLIC - {ctx_received.is_public()} HAS_SK - {ctx_received.has_secret_key()}")
print(f"{client.get_status()}")

client.select_context(context_name="test_ts")
print(f"{client.get_status()}")

client.delete_context(context_name="test_ts")
print(f"{client.get_status()}")

client.delete_context(context_name="test_ts2")
print(f"{client.get_status()}")

# DATASET
response = client.select_dataset("mnist")
print(f"dataset select response: {response}")
print(f"{client.get_status()}")

response = client.load_dataset()
length = int(response)
print(f"dataset load response: {length}")

response = client.fetch_dataset(length//2)
print(f"{response}")

response = client.load_dataset()
length = int(response)
print(f"dataset load response: {length}")

response = client.fetch_dataset(length//3)
print(f"{response}")

response = client.fetch_dataset(0)
print(f"{response}")
try:
    response = client.fetch_dataset(length//2)
    print(f"{response}")
except Answer418 as err:
    print(f"Dataset DEPLETED: {err}")