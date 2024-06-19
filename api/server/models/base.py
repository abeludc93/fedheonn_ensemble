#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.models.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base model that defines mandatory model methods

:author: youben11@github
:version: 0.0.1
"""
from abc import abstractmethod, ABC
import tenseal as ts
import numpy as np


class Model(ABC):

    @abstractmethod
    def __init__(self, parameters: dict[str, list]):
        """Create self parameters from the parameters dictionary"""
        pass

    @abstractmethod
    def evaluate_input(self, enc_x: ts.CKKSVector) -> np.ndarray:
        """Evaluate the model on the encrypted input `enc_x`

        Args:
            enc_x: encrypted input

        Returns:
            np.ndarray: the evaluation output

        Raises:
            EvaluationError: if an issue arises during evaluation
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluate_input(*args, **kwargs)
