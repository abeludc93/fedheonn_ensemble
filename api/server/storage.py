#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple storage for the server context and datasets

:author: youben11@github
:version: 0.0.1
"""
# Standard libraries
from typing import List, Tuple
from random import randint
from binascii import hexlify
# Third-party libraries
import tenseal as ts
import numpy as np

DATASETS = {
    # dataset_id: [ctx_id, X, Y, batch_size]
}

CONTEXTS = {
    # ctx_id: context
}

TOKEN_LENGTH = 32


def save_context(context: bytes) -> str:
    """Save a context into a permanent storage"""
    ctx_id = get_random_id()
    CONTEXTS[ctx_id] = context
    return ctx_id


def load_context(ctx_id: str) -> ts.Context:
    """Load a TenSEALContext"""
    context = get_raw_context(ctx_id)
    ctx = ts.context_from(context)
    return ctx


def get_raw_context(ctx_id: str) -> bytes:
    return CONTEXTS[ctx_id]


def save_dataset(X: np.ndarray, Y: np.ndarray, batch_size: int) -> str:
    """Save a dataset into a permanent storage"""
    dataset_id = get_random_id()
    DATASETS[dataset_id] = [X, Y, batch_size]
    return dataset_id


def load_dataset(dataset_id: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a dataset into CKKSVectors"""
    X, Y, batch_size = DATASETS[dataset_id]
    return X, Y, batch_size


def get_random_id():
    rand_bytes = [randint(0, 255) for i in range(TOKEN_LENGTH)]
    return hexlify(bytes(rand_bytes)).decode()
