#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
from base64 import b64encode, b64decode
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
        except:
            return False

        if response.status_code != 200:
            return False
        elif response.json() != {"message": "pong"}:
            return False

        return True

    def get_status(self) -> ServerStatus | None:
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

    def send_context(self, context_name: str, context: ts.Context) -> str | None:
        url = self._base_url + "/context"

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
        return ctx_result['id'] if 'id' in ctx_result else None