#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.models.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract model that defines mandatory model methods

:author: youben11@github
:version: 0.0.1
"""
from typing import Dict
from abc import ABC, abstractmethod
import tenseal as ts
import numpy as np


class Model(ABC):

    @abstractmethod
    def __init__(self, parameters: Dict[str, list]):
        """Create self parameters from the parameters dictionary"""
        pass

    @abstractmethod
    def forward(self, enc_x: ts._ts_cpp.CKKSVector) -> ts._ts_cpp.CKKSVector:
        """Evaluate the model on the encrypted input `enc_x`

        Args:
            enc_x: encrypted input

        Returns:
            tenseal.CKKSVector: the evaluation output

        Raises:
            EvaluationError: if an issue arises during evaluation
        """
        pass

    @staticmethod
    def prepare_input(context: bytes, ckks_vector: bytes) -> ts._ts_cpp.CKKSVector:
        """Deserialize input and check if the parameters are appropriate for the model

        Args:
            context: TenSEALContext with keys required for the computation
            ckks_vector: CKKSVector representing the model input

        Returns:
            tenseal.CKKSVector: model input

        Raises:
            DeserializationError: if the parameter aren't appropriate for the model evaluation
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
