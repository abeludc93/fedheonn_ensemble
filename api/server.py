#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import tempfile
import os
import json
import asyncio
from base64 import b64encode, b64decode

import numpy as np
# Third-party libraries
import tenseal as ts
import uvicorn
from fastapi import FastAPI, Request, Depends, UploadFile
from fastapi.responses import FileResponse, Response
# Application modules
from api.utils import *
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import load_skin_dataset, load_mini_boone, load_dry_bean, load_carbon_nanotube, load_mnist_digits

# Coordinator hyperparameters
f_act = 'logs'
lam = 0.01
enc = True
spr = True
ens = {'bagging'}
par = False
# Server parameters
host = '0.0.0.0'
port = 8000

# TenSEAL database
CONTEXTS = {
    # context_name: path_to_file
}
CURRENT_CONTEXT = (None, None)

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
    global CONTEXTS

    if ctx_name in CONTEXTS:
        delete_context(CONTEXTS[ctx_name])
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="FedHEONN")
    tmp_filename = tmp.name
    with open(tmp_filename, "wb") as ctx:
        ctx.write(content)
    CONTEXTS[ctx_name] = tmp_filename

    return tmp_filename

def load_context(ctx_name: str) -> ts.Context | None:
    """Load a TenSEALContext"""
    global CONTEXTS

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

class ServerCoordinator:
    def __init__(self):
        self.coordinator = FedHEONN_coordinator(f_act, lam, enc, spr, ens, par)
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def aggregate_partial_data(self, data: list[list]):
        async with self.lock:
            # Partial M & US lists
            await self.queue.put(data)

    async def process_aggregate_partial(self):
        while True:
            data = await self.queue.get()
            print(f"Processing data chunk from queue: {self.queue.qsize()}")
            assert len(data) == 2
            async with self.lock:
                self.coordinator.aggregate_partial(data[0], data[1])
                self.coordinator.calculate_weights()
                # Calculate optimal weights on last piece of data
                #if self.queue.empty():
                #    self.coordinator.calculate_weights()
            self.queue.task_done()

# Define singleton instance of coordinator
def singleton_coordinator():
    if not hasattr(singleton_coordinator, "server_coordinator"):
        singleton_coordinator.server_coordinator = ServerCoordinator()
    return singleton_coordinator.server_coordinator

def singleton_dataset_loader():
    if not hasattr(singleton_dataset_loader, "loader"):
        singleton_dataset_loader.loader = DataSetLoader()
    return singleton_dataset_loader.loader

@app.get("/ping")
def ping() -> dict[str, str]:
    """Used to check if the API is up"""
    return {"message": "pong"}

@app.get("/status")
def status() -> ServerStatus:
    """Used to check current server status"""
    test_status = {
        "contexts": list(CONTEXTS.keys()),
        "selected_context": CURRENT_CONTEXT[0],
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
        return answer_418(str(e))
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
        return answer_418(str(e))

@app.put("/context/{ctx_name}", response_model=None)
def select_context(ctx_name: str) -> str | JSONResponse:
    global CURRENT_CONTEXT

    if ctx_name in CONTEXTS:
        CURRENT_CONTEXT = (ctx_name, load_context(ctx_name))
        return ctx_name
    else:
        return answer_404(f"Context not found on server database: {ctx_name}")

@app.delete("/context/{ctx_name}", response_model=None)
def delete_context(ctx_name: str) -> JSONResponse | None:
    global CURRENT_CONTEXT

    if ctx_name in CONTEXTS:
        delete_context(CONTEXTS[ctx_name])
        del CONTEXTS[ctx_name]
        if CURRENT_CONTEXT[0] == ctx_name:
            CURRENT_CONTEXT = (None, None)
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

@app.post("/aggregate/partial")
async def aggregate_partial(request: Request,
                            server_coordinator: ServerCoordinator = Depends(singleton_coordinator)) -> JSONResponse:
    global CURRENT_CONTEXT
    try:
        # Status - ready for aggregation?
        if server_coordinator.coordinator.encrypted and CURRENT_CONTEXT[0] is None:
            return answer_418("Context not loaded/selected and encryption enabled!")
        # Load and parse data
        data = await request.json()
        assert len(data) == 2
        M_c, US_c = deserialize_data(data[0], data[1], CURRENT_CONTEXT[1])
        # Aggregate

        #await server_coordinator.aggregate_partial_data([M_c, US_c])
        asyncio.create_task(server_coordinator.aggregate_partial_data([M_c, US_c]))

        return answer_200(f"Data enqueued!")
    except Exception as e:
        return answer_404(str(e))
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(singleton_coordinator().process_aggregate_partial())

def deserialize_data(M, US, ctx):

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

def check_bagging_encryption(m_data):
    bagging = type(m_data) == list and type(m_data[0]) == list
    if bagging:
        try:
            np.asarray(m_data[0][0], dtype='float64')
            encrypted = False
        except ValueError:
            encrypted = True
    else:
        try:
            np.asarray(m_data[0], dtype='float64')
            encrypted = False
        except ValueError:
            encrypted = True
    return bagging, encrypted

def start():
    uvicorn.run(app, host=host, port=port)