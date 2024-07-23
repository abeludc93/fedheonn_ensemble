#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
from base64 import b64encode, b64decode
import json
import requests
# Third-party libraries
import tenseal as ts
import numpy as np
from pydantic import ValidationError

# Application modules
from api.utils import *


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
        except requests.exceptions.ConnectionError:
            raise ConnectionError
        except ValidationError:
            raise ParseError

        if response.status_code != 200:
            handle_error_response(response)

        return status

    def send_context(self, context_name: str, context: ts.Context) -> str:
        url = self._base_url + "/context"

        # Serialize if not already
        if not isinstance(context, bytes):
            ser_ctx = context.serialize(save_public_key = True, save_secret_key = True,
                                        save_galois_keys = True, save_relin_keys = True)
        else:
            ser_ctx = context
        try:
            response = requests.post(url, files={'file': (context_name, ser_ctx)})
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        ctx_result = response.json()
        return ctx_result['path']

    def get_context(self, context_name: str) -> ts.Context:
        url = self._base_url + f"/context/{context_name}"

        try:
            response = requests.get(url)
            ctx = ts.context_from(response.content)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return ctx

    def select_context(self, context_name: str) -> str:
        url = self._base_url + f"/context/{context_name}"

        try:
            response = requests.put(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    def delete_context(self, context_name: str) -> str:
        url = self._base_url + f"/context/{context_name}"

        try:
            response = requests.delete(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    def select_dataset(self, dataset_name: str) -> str:
        url = self._base_url + f"/dataset/{dataset_name}"

        try:
            response = requests.put(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return response.text

    def load_dataset(self) -> int:
        url = self._base_url + f"/dataset/load"
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return int(response.text)

    def fetch_dataset(self, size=0) -> list[np.ndarray]:
        url = self._base_url + f"/dataset/fetch/?size={size}"

        train_array = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                train_array = [np.asarray(elem) for elem in json.loads(response.json())]
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return train_array

    def fetch_dataset_test(self) -> list[np.ndarray]:
        url = self._base_url + "/dataset/fetch/test"

        test_array = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                test_array = [np.asarray(elem) for elem in json.loads(response.json())]
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return test_array

    def aggregate_partial(self,
                          data
                          #m_data:  list[np.ndarray | ts.CKKSVector] | list[list[np.ndarray | ts.CKKSVector]],
                          #US_data: list[np.ndarray] | list[list[np.ndarray]]
                          ) -> str:
        url = self._base_url + "/aggregate/partial"

        try:
            #data = Client.serialize_client_data(m_data=m_data, US_data=US_data)
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return response.json()["message"]

    def update_coordinator_parameters(self, f_act: str='logs', lam: float=0, spr: bool=True, enc: bool=False,
                                      bag: bool=False, par: bool=False, ctx_str: str=None):
        url = self._base_url + "/coordinator/parameters"

        try:
            data = {"f": f_act, "lam": lam, "sparse": spr, "encrypted": enc, "bagging": bag, "parallel": par, "ctx_str": ctx_str}
            response = requests.put(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            handle_error_response(response)

        return response.json()["message"]

    @staticmethod
    def serialize_client_data(m_data:  list[np.ndarray | ts.CKKSVector] | list[list[np.ndarray | ts.CKKSVector]],
                              US_data: list[np.ndarray] | list[list[np.ndarray]]) -> list:

        bagging, encrypted = Client.check_bagging_encryption(m_data)
        data = []
        if bagging:
            if encrypted:
                # Bagging ensemble, encrypted M_e's
                for i in range(len(m_data)):
                    m_data[i] = [b64encode(arr.serialize()).decode('ascii') for arr in m_data[i]]
                    US_data[i] = [arr.tolist() for arr in US_data[i]]
            else:
                # Bagging ensemble, plain data
                for i in range(len(m_data)):
                    m_data[i] = [arr.tolist() for arr in m_data[i]]
                    US_data[i] = [arr.tolist() for arr in US_data[i]]
        else:
            if encrypted:
                # No bagging, encrypted M's
                m_data = [b64encode(arr.serialize()).decode('ascii') for arr in m_data]
                US_data = [arr.tolist() for arr in US_data]
            else:
                # No bagging, plain data
                m_data = [arr.tolist() for arr in m_data]
                US_data = [arr.tolist() for arr in US_data]

        data.append(m_data)
        data.append(US_data)

        return data

    @staticmethod
    def check_bagging_encryption(m_data):
        bagging     = type(m_data) == list and type(m_data[0]) == list
        if bagging:
            encrypted = type(m_data[0][0]) == ts.CKKSVector
        else:
            encrypted = type(m_data[0]) == ts.CKKSVector

        return bagging, encrypted
