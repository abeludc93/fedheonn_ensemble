#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.client.client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Client implementing the communication with the server.

:author: youben11@github
:version: 0.0.1
"""
# Standard libraries
from base64 import b64encode, b64decode
from typing import List, Union, Tuple
# Third-party libraries
import requests
import tenseal as ts
import numpy as np
# Application modules
from api.exceptions import *


class Client:
    """Client to communicate with server for aggregation and evaluation"""

    def __init__(self, hostname: str, port: int):
        self._base_url = f"http://{hostname}:{port}"

    def ping(self) -> bool:
        """Make sure the API is up

        Returns:
            bool: True if the API is up, False otherwise
        """
        url = self._base_url + "/ping"
        try:
            response = requests.get(url)
        except:
            return False
        if response.status_code != 200:
            return False
        elif response.json() != {"message": "pong"}:
            return False

        return True

    def model_info(self, model_name: str) -> dict:
        """Request information about a specific model `model_name`

        Args:
            model_name: the model name to request information about

        Returns:
            dict: information about the model

        Raises:
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """
        url = self._base_url + f"/models/{model_name}"
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        model_info = response.json()
        return model_info

    def evaluate(
        self,
        model_name: str,
        ckks_vector: Union[ts.CKKSVector, bytes],  # serialized or not
    ) -> np.ndarray:
        """Evaluate model `model_name` on the encrypted input data `ckks_vector`

        Args:
            model_name: the model name to use for evaluation
            ckks_vector: encrypted input to feed the model with

        Returns:
            np.ndarray: output of the evaluation

        Raises:
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/eval/{model_name}"

        if not isinstance(ckks_vector, bytes):
            ser_vec = ckks_vector.serialize()
        else:
            ser_vec = ckks_vector

        data = {"ckks_vector": b64encode(ser_vec).decode()}

        try:
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        ser_result = response.json()["output"]

        return ser_result

    def get_context(self,) -> ts.Context:
        """Get registered context on the server

        Returns:
            TenSEALContext

        Raises:
            ConnectionError: if a connection can't be established with the API
            ResourceNotFound: if the context identified with `ctx_id` can't be found
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/contexts/"

        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        ser_ctx = response.json()["context"]
        ctx = ts.context_from(b64decode(ser_ctx))

        return ctx

    def get_dataset(self, dataset_id: str, batch_size: int) -> Tuple[str, List[np.ndarray], int]:
        """Get a dataset using its id

        Args:
            dataset_id: id referring to the previously saved dataset
            batch_size: batch size, total elements.
        Returns:
            (dataset_id, [X, Y], batch_size)

        Raises:
            ConnectionError: if a connection can't be established with the API
            ResourceNotFound: if the dataset identified with `dataset_id` can't be found
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/datasets/"
        data = {"dataset_id": dataset_id, "batch_size": batch_size}

        try:
            response = requests.get(url, params=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        resp_json = response.json()
        dataset = resp_json["dataset_id"]
        batch = resp_json["batch_size"]
        X, Y = [], []
        for x_buff, y_buff in zip(resp_json["X"], resp_json["Y"]):
            X.append(x_buff)
            Y.append(y_buff)
        X = np.asarray(X)
        Y = np.asarray(Y)

        return dataset, [X, Y], batch

    @staticmethod
    def _handle_error_response(response: requests.Response):
        """Handle the responses that aren't a success (200)"""
        if response.status_code == 404:
            error_msg = response.json()["message"]
            raise ResourceNotFound(error_msg)
        elif response.status_code == 418:
            error_msg = response.json()["message"]
            raise Answer418(error_msg)
        elif response.status_code == 500:
            raise ServerError("Server error")
        else:
            raise RuntimeError(
                f"Unknown server error -> [status_code: {response.status_code}]: '{response.text}'"
            )
