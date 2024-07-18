#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import tempfile
# Standard libraries
from base64 import b64encode, b64decode
# Third-party libraries
import tenseal as ts
import uvicorn
from fastapi import FastAPI, Depends, UploadFile
# Application modules
from api.utils import *
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator

# Coordinator hyperparameters
f_act = 'logs'
lam = 0.01
enc = False
spr = True
ens = {}
par = False
# Server parameters
host = '0.0.0.0'
port = 8000

# TenSEAL database
CONTEXTS = {
    # context_name: path_to_file
}

def save_context(name: str, content: bytes) -> str:
    """Save a context into a permanent storage"""
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
    tmp_filename = tmp.name
    with open(tmp_filename, "wb") as ctx:
        ctx.write(content)
    CONTEXTS[name] = tmp_filename
    return tmp_filename
def load_context(ctx_name: str) -> ts.Context | None:
    """Load a TenSEALContext"""
    if ctx_name not in CONTEXTS:
        return None
    else:
        with open(CONTEXTS[ctx_name], "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

# Create FastAPI
app = FastAPI()

# Define singleton instance of coordinator
def singleton_coordinator():
    if not hasattr(singleton_coordinator, "coordinator"):
        singleton_coordinator.coordinator = FedHEONN_coordinator(f_act, lam, enc, spr, ens, par)
    return singleton_coordinator.coordinator


@app.get("/ping")
def ping() -> dict[str, str]:
    """Used to check if the API is up"""
    return {"message": "pong"}

@app.get("/status")
def status(coord: FedHEONN_coordinator = Depends(singleton_coordinator)) -> ServerStatus:
    """Used to check current server status"""
    test_status = {
        "has_context": coord.encrypted,
        "selected_dataset": "mnist",
        "status": "WAITING CLIENTS",
    }
    return ServerStatus(**test_status)

@app.post("/context")
def upload(file: UploadFile):
    try:
        contents = file.file.read()
        ctx_filepath = save_context(file.filename, contents)
    except:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"id": ctx_filepath}


def start():
    uvicorn.run(app, host=host, port=port)
