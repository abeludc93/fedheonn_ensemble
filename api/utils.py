#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
# Standard libraries
import requests
# Third-party libraries
from fastapi.responses import JSONResponse
from pydantic import BaseModel

### BASE MODELS ###
class Context(BaseModel):
    context: str

class CKKSVector(BaseModel):
    ckks_vector: str

class NumpyArray(BaseModel):
    nd_array: str

class ServerStatus(BaseModel):
    contexts: list[str]
    datasets: list[str]
    selected_context: str | None
    selected_dataset: str | None
    status: str

### CUSTOM EXCEPTIONS ###
class Answer418(Exception):
    """Request specific error, the meaning mostly depends on the request"""
    pass


class ServerError(Exception):
    """When the server returns a status 500 response"""
    pass


class ResourceNotFound(Exception):
    """When a resource isn't found on the remote server"""
    pass

class ParseError(Exception):
    pass

class ModelNotFound(Exception):
    """When a model can't be found"""
    pass


class EvaluationError(Exception):
    """When a problem happens during evaluation"""
    pass


class DeserializationError(Exception):
    """When context or encrypted input can't be deserialized"""
    pass


class InvalidContext(Exception):
    """When the context isn't appropriate for a specific model"""
    pass


### HANDLE ERRORS ###
def answer_418(msg: str) -> JSONResponse:
    return JSONResponse(status_code=418, content={"message": f"Oops! Server says '{msg}'"})

def answer_404(msg: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"message": f"Resource not found. Server says '{msg}'"})

def handle_error_response(response: requests.Response):
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
        raise RuntimeError(f"Unknown server error -> [status_code: {response.status_code}]: '{response.text}'")


class DataSetLoader:
    def __init__(self, f_load=None, name=None):
        self.f_load = f_load
        self.dataset_name = name
        self.dataset_length = None
        self.dataset_index = None
        self.train_data = []
        self.test_data = []
        self.loaded = False

    def set_fload(self, f_load):
        self.f_load = f_load
    def set_dataset_name(self, name):
        self.dataset_name = name

    def get_name(self):
        return self.dataset_name

    def is_loaded(self):
        return self.loaded

    def clean_loader(self):
        self.f_load = None
        self.dataset_name = None
        self.dataset_length = None
        self.dataset_index = None
        self.train_data = []
        self.test_data = []
        self.loaded = False

    def load(self) -> int:
        if self.dataset_name is None:
            raise ValueError("No dataset selected")
        # Load
        tuple_data = self.f_load()
        # Arrange data
        self.train_data.append(tuple_data[0])
        self.train_data.append(tuple_data[1])
        self.test_data.append(tuple_data[2])
        self.test_data.append(tuple_data[3])
        # Training set shape
        self.dataset_index = 0
        self.dataset_length = tuple_data[0].shape[0]
        self.loaded = True

        return self.dataset_length

    def fetch_fragment(self, size: int) -> list[np.ndarray]:
        if self.dataset_name is None:
            raise ValueError("No dataset selected")
        elif not self.loaded:
            raise ValueError("Dataset not loaded")

        start = self.dataset_index
        end = min(start + size, self.dataset_length) if size > 0 else self.dataset_length
        self.dataset_index = end
        fragment = [data[start:end] for data in self.train_data]

        return fragment

    def fetch_test(self) -> list[np.ndarray]:
        if self.dataset_name is None:
            raise ValueError("No dataset selected")
        elif not self.loaded:
            raise ValueError("Dataset not loaded")

        return self.test_data

    def is_empty_dataset(self) -> bool:
        if self.dataset_name is None:
            return True
        return self.dataset_index >= self.dataset_length
