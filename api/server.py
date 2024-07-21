#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import tempfile
import os
import json
from base64 import b64encode, b64decode
# Third-party libraries
import tenseal as ts
import uvicorn
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import FileResponse, Response
# Application modules
from api.utils import *
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import load_skin_dataset, load_mini_boone, load_dry_bean, load_carbon_nanotube, load_mnist_digits

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
CURRENT_CONTEXT = None

DATASETS = {
    # dataset_name
    'carbon': load_carbon_nanotube,
    'dry_bean': load_dry_bean,
    'mini_boone': load_mini_boone,
    'mnist': load_mnist_digits,
    'skin': load_skin_dataset
}
CURRENT_DATASET = None

def save_context(ctx_name: str, content: bytes) -> str:
    """Save a context into a permanent storage"""
    global CURRENT_CONTEXT, CONTEXTS

    if ctx_name in CONTEXTS:
        delete_context(CONTEXTS[ctx_name])
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
    tmp_filename = tmp.name
    with open(tmp_filename, "wb") as ctx:
        ctx.write(content)
    CONTEXTS[ctx_name] = tmp_filename
    CURRENT_CONTEXT = ctx_name
    return tmp_filename

def load_context(ctx_name: str) -> ts.Context | None:
    """Load a TenSEALContext"""
    global CURRENT_CONTEXT, CONTEXTS

    if ctx_name not in CONTEXTS:
        return None
    else:
        with open(CONTEXTS[ctx_name], "rb") as f:
            loaded_context = ts.context_from(f.read())
        return loaded_context

def delete_context(ctx_filepath: str):
    if ctx_filepath and os.path.isfile(ctx_filepath):
        os.remove(ctx_filepath)
    else:
        print(f"\tCouldn't delete temporary file (empty or not found): {ctx_filepath}")

# Create FastAPI
app = FastAPI()

# Define singleton instance of coordinator
def singleton_coordinator():
    if not hasattr(singleton_coordinator, "coordinator"):
        singleton_coordinator.coordinator = FedHEONN_coordinator(f_act, lam, enc, spr, ens, par)
    return singleton_coordinator.coordinator

def singleton_dataset_loader():
    if not hasattr(singleton_dataset_loader, "loader"):
        singleton_dataset_loader.loader = DataSetLoader()
    return singleton_dataset_loader.loader

@app.get("/ping")
def ping() -> dict[str, str]:
    """Used to check if the API is up"""
    return {"message": "pong"}

@app.get("/status")
def status(coord: FedHEONN_coordinator = Depends(singleton_coordinator)) -> ServerStatus:
    """Used to check current server status"""
    test_status = {
        "contexts": list(CONTEXTS.keys()),
        "selected_context": CURRENT_CONTEXT,
        "datasets": list(DATASETS.keys()),
        "selected_dataset": CURRENT_DATASET,
        "status": "WAITING CLIENTS",
    }
    return ServerStatus(**test_status)

@app.post("/context", response_model=None)
def upload_context(file: UploadFile) -> dict[str, str] |  JSONResponse:
    try:
        contents = file.file.read()
        ctx_filepath = save_context(file.filename, contents)
    except Exception as e:
        return answer_418(f"{e}")
    finally:
        file.file.close()

    return {"ctx_name": file.filename, "path": ctx_filepath}

@app.get("/context/{ctx_name}")
def get_context(ctx_name: str) -> Response:
    try:
        if ctx_name in CONTEXTS:
            return FileResponse(path=CONTEXTS[ctx_name], media_type='application/octet-stream', filename=ctx_name)
        else:
            return answer_404(f'Context not found on server database: {ctx_name}')
    except Exception as e:
        return answer_418(e.__str__())

@app.put("/context/{ctx_name}", response_model=None)
def select_context(ctx_name: str) -> str | JSONResponse:
    global CURRENT_CONTEXT

    if ctx_name in CONTEXTS:
        CURRENT_CONTEXT = ctx_name
        return ctx_name
    else:
        return answer_404(f"Context not found on server database: {ctx_name}")

@app.delete("/context/{ctx_name}", response_model=None)
def delete_context(ctx_name: str) -> JSONResponse | None:
    global CURRENT_CONTEXT

    if ctx_name in CONTEXTS:
        delete_context(CONTEXTS[ctx_name])
        del CONTEXTS[ctx_name]
        if CURRENT_CONTEXT == ctx_name:
            CURRENT_CONTEXT = None
    else:
         return answer_404(f'Context not found on server database: {ctx_name}')

@app.put("/dataset/{dataset_name}", response_model=None)
def select_dataset(dataset_name: str) -> str | JSONResponse:
    global CURRENT_DATASET

    if dataset_name in DATASETS:
        CURRENT_DATASET = dataset_name
        return dataset_name
    else:
        return answer_404(f"Dataset not found on server database: {dataset_name}")

@app.get("/dataset/load", response_model=None)
def load_dataset(dataset_loader: DataSetLoader = Depends(singleton_dataset_loader)) -> int | JSONResponse:
    global CURRENT_DATASET

    if CURRENT_DATASET is not None:
        if dataset_loader.is_loaded():
            return dataset_loader.dataset_length - dataset_loader.dataset_index
        else:
            dataset_loader.clean_loader()
            dataset_loader.set_dataset_name(CURRENT_DATASET)
            dataset_loader.set_fload(DATASETS[CURRENT_DATASET])
            dataset_length = dataset_loader.load()
            print(f"\t\t Dataset: {dataset_loader.get_name()} loaded!")
            return dataset_length
    else:
        return answer_404("Dataset not selected!")

@app.get("/dataset/fetch", response_model=None)
def fetch_dataset(size:int, dataset_loader: DataSetLoader = Depends(singleton_dataset_loader)) -> str | JSONResponse:

    if dataset_loader.get_name() is None:
        return answer_404("Dataset not selected!")
    elif not dataset_loader.is_loaded():
        return answer_404("Current dataset is not loaded!")
    elif dataset_loader.is_empty_dataset():
        return answer_418(f"Current dataset depleted!: {dataset_loader.get_name()}")
    else:
        fragment = dataset_loader.fetch_fragment(size)
        return json.dumps([frag.tolist() for frag in fragment])

@app.get("/dataset/fetch/test", response_model=None)
def fetch_dataset_test(dataset_loader: DataSetLoader = Depends(singleton_dataset_loader)) -> str | JSONResponse:

    if dataset_loader.get_name() is None:
        return answer_404("Dataset not selected!")
    elif not dataset_loader.is_loaded():
        return answer_404("Current dataset is not loaded!")
    else:
        fragment = dataset_loader.fetch_test()
        return json.dumps([frag.tolist() for frag in fragment])



def start():
    uvicorn.run(app, host=host, port=port)
