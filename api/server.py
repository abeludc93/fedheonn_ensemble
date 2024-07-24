#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import tempfile
import os
import json
import asyncio
from base64 import b64encode, b64decode
from concurrent.futures import ThreadPoolExecutor
# Third-party libraries
import tenseal as ts
import uvicorn
from fastapi import FastAPI, Request, Depends, UploadFile
from fastapi.responses import FileResponse, Response
# Application modules
from api.utils import *
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from examples.utils import load_skin_dataset, load_mini_boone, load_dry_bean, load_carbon_nanotube, load_mnist_digits

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
        self.coordinator = FedHEONN_coordinator()
        self.queue = asyncio.Queue()
        #self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ServerCoord')

    async def aggregate_partial_data(self, data: list[list]):
        # Partial M & US lists
        await self.queue.put(data)

    async def process_aggregate_partial(self):
        while True:

            M_group, US_group = [], []
            data = await self.queue.get()
            print(f"Processing data chunk from queue")
            assert len(data) == 2
            M_group.append(data[0])
            US_group.append(data[1])

            asyncio.get_event_loop().run_in_executor(self.executor, self._aggregate_partial,M_group, US_group)

            self.queue.task_done()

    def _aggregate_partial(self, M_grp, US_grp):
        self.coordinator.aggregate_partial(M_grp, US_grp)
        self.coordinator.calculate_weights()

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
def status(sc: ServerCoordinator = Depends(singleton_coordinator)) -> ServerStatus:
    """Used to check current server status"""
    test_status = {
        "contexts": list(CONTEXTS.keys()),
        "selected_context": CURRENT_CONTEXT[0],
        "datasets": list(DATASETS.keys()),
        "selected_dataset": CURRENT_DATASET,
        "coord_params": CoordinatorParams(**sc.coordinator.get_parameters()),
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

@app.put("/coordinator/parameters")
def set_coordinator_parameters(coord_params: CoordinatorParams,
                               sc: ServerCoordinator = Depends(singleton_coordinator)) -> JSONResponse:
    try:
        sc.coordinator.set_activation_functions(coord_params.f)
        sc.coordinator.lam = coord_params.lam
        sc.coordinator.sparse = coord_params.sparse
        sc.coordinator.encrypted = coord_params.encrypted
        sc.coordinator.ensemble = FedHEONN_coordinator.generate_ensemble_params() if coord_params.bagging else None
        sc.coordinator.parallel = coord_params.parallel
        sc.coordinator.set_ctx_str(coord_params.ctx_str)
        return answer_200('Updated coordinator parameters!')
    except Exception as err:
        return answer_418(str(err))

@app.post("/coordinator/index_features")
def calculate_index_features(bagging_params: BaggingParams,
                             sc: ServerCoordinator = Depends(singleton_coordinator)) -> JSONResponse:
    try:
        sc.coordinator.calculate_idx_feats(n_estimators=bagging_params.n_estimators,
                                           n_features=bagging_params.n_features,
                                           p_features=bagging_params.p_features,
                                           b_features=bagging_params.b_features)
        return answer_200('Coordinator picked random patches for data features! ')
    except AssertionError as a_err:
        return answer_404(f"Check for n_estimators > 1 and p_features > 0!: {a_err}")
    except ValueError as v_err:
        return answer_404(f"Could not execute random choice, reason: {v_err}")
    except Exception as err:
        return answer_418(str(err))

@app.get("/coordinator/index_features")
def send_index_features(sc: ServerCoordinator = Depends(singleton_coordinator)) -> str:
    idx_feats = sc.coordinator.send_idx_feats()
    return json.dumps(idx_feats)

@app.get("/coordinator/send_weights")
def send_weights(sc: ServerCoordinator = Depends(singleton_coordinator)) -> JSONResponse:
    try:
        W_length = len(sc.coordinator.W)
        if sc.coordinator.encrypted and sc.coordinator.W:
            # Encrypted W data (tenSEAL CKKS vectors)
            if type(sc.coordinator.W) == list and type(sc.coordinator.W[0]) == list:
                data = []
                for i in range(W_length):
                    data.append([b64encode(arr.serialize()).decode('ascii') for arr in sc.coordinator.W[i]])
            else:
                data = [b64encode(arr.serialize()).decode('ascii') for arr in sc.coordinator.W]
            return answer_200_data(msg="encrypted", data=json.dumps(data))
        elif not sc.coordinator.encrypted and sc.coordinator.W:
            # Plain W data (optimal weights)
            data = []
            for i in range(W_length):
                data.append([arr.tolist() for arr in sc.coordinator.W[i]])
            return answer_200_data(msg="plain", data=json.dumps(data))
        else:
            # No optimal weights yet
            return answer_404("Empty optimal weights W, no data or not yet calculated!")
    except Exception as err:
        return answer_418(str(err))

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

        await server_coordinator.aggregate_partial_data([M_c, US_c])
        #await asyncio.create_task(server_coordinator.aggregate_partial_data([M_c, US_c]))

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
