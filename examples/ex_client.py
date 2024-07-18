#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from api.client import Client

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

upload_ctx = client.send_context(context_name="test_ts", context=ctx)
print("Could NOT upload context to the server" if upload_ctx is None else f"Uploaded context to server: {upload_ctx}")