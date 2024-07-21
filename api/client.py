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
            raise False

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