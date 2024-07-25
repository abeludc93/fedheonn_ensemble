#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import requests
import json
# Third-party libraries
import tenseal as ts
import numpy as np
from pydantic import ValidationError
# Application modules
from api.utils import ServerStatus, handle_error_response, ParseError, ServerError, deserialize_coordinator_weights

class Client:
    """Client to communicate with server for aggregation and evaluation"""
    def __init__(self, hostname: str, port: int):
        self._base_url = f"http://{hostname}:{port}"

    def ping(self) -> bool:
        url = self._base_url + "/ping"

        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            return False

        if response.status_code != 200:
            return False
        elif response.json() != {"message": "pong"}:
            return False

        return True

    def get_status(self) -> ServerStatus:
        url = self._base_url + "/status"

        try:
            response = requests.get(url)
            status = ServerStatus(**response.json())
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))
        except ValidationError as v_err:
            raise ParseError(str(v_err))

        if response.status_code != 200:
            handle_error_response(response)

        return status

    """ CONTEXT METHODS """
    def send_context(self, context_name: str, context: ts.Context | bytes) -> str:
        url = self._base_url + "/context"

        # Serialize if not already
        if not isinstance(context, bytes):
            ser_ctx = context.serialize(save_public_key = True, save_secret_key = True,
                                        save_galois_keys = True, save_relin_keys = True)
        else:
            ser_ctx = context

        # Upload context
        try:
            response = requests.post(url, files={'file': (context_name, ser_ctx)})
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        ctx_result = response.json()
        return ctx_result['path']

    def get_context(self, context_name: str) -> ts.Context:
        url = self._base_url + f"/context/{context_name}"

        # Download context
        try:
            response = requests.get(url)
            ctx = ts.context_from(response.content)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return ctx

    def select_context(self, context_name: str) -> str:
        url = self._base_url + f"/context/{context_name}"

        # Select context on server-side
        try:
            response = requests.put(url)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    def delete_context(self, context_name: str) -> str:
        url = self._base_url + f"/context/{context_name}"

        # Delete tenSEAL context temp file
        try:
            response = requests.delete(url)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    """ DATASET METHODS """
    def select_dataset(self, dataset_name: str) -> str:
        url = self._base_url + f"/dataset/{dataset_name}"

        # Select dataset on server-side
        try:
            response = requests.put(url)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    def load_dataset(self) -> int:
        url = self._base_url + f"/dataset/load"

        # Load and split selected server-side dataset
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return int(response.text)

    def fetch_dataset(self, size=0) -> list[np.ndarray]:
        url = self._base_url + f"/dataset/fetch/?size={size}"

        # Fetch train dataset bit of given size
        train_array = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                train_array = [np.asarray(elem) for elem in json.loads(response.json())]
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return train_array

    def fetch_dataset_test(self) -> list[np.ndarray]:
        url = self._base_url + "/dataset/fetch/test"

        # Fetch test dataset
        test_array = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                test_array = [np.asarray(elem) for elem in json.loads(response.json())]
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return test_array

    """ FEDHEONN METHODS"""
    def aggregate_partial(self, data: list) -> str:
        url = self._base_url + "/aggregate/partial"

        # Send fitted data (ready for serialization) to the server for partial aggregation
        try:
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return response.json()["message"]

    def update_coordinator_parameters(self, f_act: str='logs', lam: float=0, spr: bool=True, enc: bool=False,
                                      bag: bool=False, par: bool=False, ctx_str: str=None):
        url = self._base_url + "/coordinator/parameters"

        # Updates FedHEONN coordinator with the given set of parameters below
        try:
            data = {"f": f_act, "lam": lam, "sparse": spr, "encrypted": enc,
                    "bagging": bag, "parallel": par, "ctx_str": ctx_str}
            response = requests.put(url, json=data)
        except requests.exceptions.ConnectionError as c_err:
            raise ServerError(str(c_err))

        if response.status_code != 200:
            handle_error_response(response)

        return response.json()["message"]

    def calculate_index_features(self, n_estimators: int, n_features: int, p_features:float, b_features:bool) -> str:
        url = self._base_url + "/coordinator/index_features"

        # Sends a signal to the FedHEONN coordinator to calculate a random set of features
        try:
            data = {"n_estimators": n_estimators, "n_features": n_features, "p_features": p_features, "b_features": b_features}
            response = requests.post(url, json=data)
            if response.status_code != 200:
                handle_error_response(response)
        except Exception as err:
            return f"CLIENT [calculate_index_features] error: {err}"

        return response.json()["message"]

    def get_index_features(self) -> list[int] | None:
        url = self._base_url + "/coordinator/index_features"

        # Receive the random features calculated from the FedHEONN coordinator (server-side)
        arr = None
        try:
            response = requests.get(url)
            if response.status_code == 200:
                arr = json.loads(response.json())
        except Exception as err:
            print(f"CLIENT [get_index_features] error: {err}")
            return None

        if response.status_code != 200:
            handle_error_response(response)

        return arr

    def receive_weights(self) -> list[np.ndarray | bytes] | None:
        url = self._base_url + "/coordinator/send_weights"

        # Receive weights calculated from the FedHEONN coordinator (server-side) and deserializes them (if necessary)
        W = None
        try:
            response = requests.get(url)
            if response.status_code == 200:
                msg = response.json()
                data = json.loads(msg["data"])
                encrypted = msg["message"] == "encrypted"
                W = deserialize_coordinator_weights(data, encrypted)
            else:
                handle_error_response(response)
        except Exception as err:
            print(f"CLIENT [send_weights] error: {err}")
            return None

        return W
