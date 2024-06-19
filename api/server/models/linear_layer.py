#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.models.linear_layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Linear Layer to compute `out = encrypted_input.matmul(weight) + bias`

:author: youben11@github
:version: 0.0.1
"""
# Third-party libraries
import tenseal as ts
import numpy as np
# Application modules
from api.server.models.base import Model
from api.exceptions import EvaluationError


class LinearLayer(Model):
    """Linear Layer computing `out = encrypted_input.matmul(weight) + bias`
    input and output shapes depends on the parameters weight and bias
    The input should be encrypted as a tenseal.CKKSVector, the output will as well be encrypted.
    """

    def __init__(self, parameters):
        # parameters is the unpickled version file
        self.weight = parameters["weight"]
        self.bias = parameters["bias"]

    def evaluate_input(self, enc_x: ts.CKKSVector) -> np.ndarray:
        try:
            out = enc_x.mm(self.weight) + self.bias
        except Exception as e:
            raise EvaluationError(f"{e.__class__.__name__}: {str(e)}")
        return out
