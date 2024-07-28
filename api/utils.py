#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import os
import requests
import tempfile
from base64 import b64encode, b64decode
# Third-party libraries
import numpy as np
import tenseal as ts
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from auxiliary.logger import logger as log

### BASE MODELS ###

class DataSetReport(BaseModel):
    name: str | None
    loaded: bool | None
    train_length: int | None
    train_index: int | None
    depleted: bool | None
    train_features: int | None

class ClientDataReport(BaseModel):
    client_data_total: int
    client_data_queued: int
    client_data_processing: int
    client_data_finished: int

class CoordinatorParams(BaseModel):
    f: str
    lam: float
    encrypted: bool
    sparse: bool
    bagging: bool
    parallel: bool
    ctx_str: str | None

class BaggingParams(BaseModel):
    n_estimators: int
    n_features: int
    p_features: float
    b_features: bool

class ServerStatus(BaseModel):
    contexts: list[str]
    selected_context: str | None
    datasets: list[str]
    selected_dataset: str | None
    dataset_report: DataSetReport
    client_data_report: ClientDataReport
    coord_params: CoordinatorParams | None

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

def answer_200(msg: str) -> JSONResponse:
    return JSONResponse(status_code=200, content={"message": f"Successful: '{msg}'"})

def answer_200_data(msg: str, data: str) -> JSONResponse:
    return JSONResponse(status_code=200, content={"message": msg, "data": data})

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

### CONTEXT AUXILIARY FUNCTIONS ###
def save_context(ctx_name: str, content: bytes, ctx_dict: dict) -> str:
    """Save a context into a permanent storage"""
    if ctx_name in ctx_dict:
        delete_context(ctx_dict[ctx_name])
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
    tmp_filename = tmp.name
    with open(tmp_filename, "wb") as ctx:
        ctx.write(content)
    ctx_dict[ctx_name] = tmp_filename

    return tmp_filename

def load_context(ctx_name: str, ctx_dict: dict) -> ts.Context | None:
    """Load a TenSEALContext"""
    if ctx_name not in ctx_dict:
        log.warn(f"{ctx_name} not found amongst contexts!")
        return None
    else:
        with open(ctx_dict[ctx_name], "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

def delete_context(ctx_filepath: str):
    """Deletes context temp file"""
    if ctx_filepath and os.path.isfile(ctx_filepath):
        os.remove(ctx_filepath)
    else:
        log.warn(f"\tCouldn't delete temporary file (empty or not found): {ctx_filepath}")

### CLIENT DATABASE AUXILIARY FUNCTION ###
def get_client_data_report(client_database: dict[str, str]) -> dict[str, int]:
    report = {"client_data_total": 0, "client_data_queued": 0, "client_data_processing": 0, "client_data_finished": 0}
    if not client_database:
        return report
    else:
        values = list(client_database.values())
        report["client_data_total"] = len(client_database)
        report["client_data_queued"] = values.count("ENQUEUED")
        report["client_data_processing"] = values.count("PROCESSING")
        report["client_data_finished"] = values.count("FINISHED")
        return report

### DATASET LOADER CLASS HELPER ###
class DataSetLoader:
    def __init__(self, f_load=None, name=None):
        self.f_load = f_load
        self.dataset_name = name
        self.dataset_length = None
        self.dataset_features = None
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
        self.dataset_features = tuple_data[0].shape[1]
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

    def get_report(self) -> dict:
        return {"name": self.get_name(),
                "loaded": self.is_loaded(),
                "train_length": self.dataset_length,
                "train_index": self.dataset_index,
                "depleted": self.is_empty_dataset(),
                "train_features": self.dataset_features}

### AUXILIARY FUNCTIONS FOR DATA SERIALIZATION ###
def deserialize_client_data(M, US, ctx):

    bagging, encrypted = check_bagging_encryption(M)
    if bagging:
        if encrypted:
            # Bagging ensemble, encrypted M_e's
            for i in range(len(M)):
                M[i] = [ts.ckks_vector_from(ctx, b64decode(arr)) for arr in M[i]]
                US[i] = [np.array(arr) for arr in US[i]]
        else:
            # Bagging ensemble, plain data
            for i in range(len(M)):
                M[i] = [arr.tolist() for arr in M[i]]
                US[i] = [np.array(arr) for arr in US[i]]
    else:
        if encrypted:
            # No bagging, encrypted M's
            M = [ts.ckks_vector_from(ctx, b64decode(arr)) for arr in M]
            US = [np.array(arr) for arr in US]
        else:
            # No bagging, plain data
            M = [arr.tolist() for arr in M]
            US = [np.array(arr) for arr in US]

    return M, US

def serialize_client_data(m_data:  list[np.ndarray | ts.CKKSVector] | list[list[np.ndarray | ts.CKKSVector]],
                          US_data: list[np.ndarray] | list[list[np.ndarray]]) -> list:

    bagging, encrypted = check_bagging_encryption(m_data)
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

def check_bagging_encryption(m_data: list[np.ndarray | ts.CKKSVector | str] |
                                     list[list[np.ndarray | ts.CKKSVector | str]]):

    bagging     = type(m_data) == list and type(m_data[0]) == list
    if bagging:
        encrypted = type(m_data[0][0]) == ts.CKKSVector or type(m_data[0][0]) == str
    else:
        encrypted = type(m_data[0]) == ts.CKKSVector or type(m_data[0]) == str

    return bagging, encrypted

def deserialize_coordinator_weights(data: list[np.ndarray | str] | list[list[np.ndarray | str]],
                                    encrypted: bool):
    W = []
    if encrypted:
        # Encrypted weights
        if type(data) == list and type(data[0]) == list:
            # Bagging
            for i in range(len(data)):
                W.append([b64decode(arr) for arr in data[i]])
        else:
            # No bagging, classic
            W = [b64decode(arr) for arr in data]
    else:
        # Plain data
        if type(data) == list and type(data[0]) == list and type(data[0][0]) == list:
            # Bagging
            for i in range(len(data)):
                W.append([np.asarray(arr) for arr in data[i]])
        else:
            # No bagging, classic
            W = [np.asarray(arr) for arr in data]

    return W

def serialize_coordinator_weights(W: list[np.ndarray | ts.CKKSVector] | list[list[np.ndarray | ts.CKKSVector]],
                                  encrypted: bool):
    W_length = len(W)
    if encrypted:
        # Encrypted W data (tenSEAL CKKS vectors)
        if type(W) == list and type(W[0]) == list:
            data = []
            for i in range(W_length):
                data.append([b64encode(arr.serialize()).decode('ascii') for arr in W[i]])
        else:
            data = [b64encode(arr.serialize()).decode('ascii') for arr in W]
        return data
    elif not encrypted:
        # Plain W data (optimal weights)
        data = []
        for i in range(W_length):
            data.append([arr.tolist() for arr in W[i]])
        return data
