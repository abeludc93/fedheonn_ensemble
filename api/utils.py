#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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
    has_context: bool
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
def answer_418(msg: str):
    return JSONResponse(status_code=418, content={"message": f"Oops! Server says '{msg}'"})

def answer_404(msg: str):
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
