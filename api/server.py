#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Standard libraries
import json
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
# Third-party libraries
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, Response, JSONResponse
from dataclasses import dataclass
# Application modules
from algorithm.fedHEONN_coordinators import FedHEONN_coordinator
from api.utils import load_context, save_context, delete_context
from api.utils import CoordinatorParams, ServerStatus, BaggingParams
from api.utils import answer_418, answer_404, answer_200, answer_200_data
from api.utils import DataSetLoader, deserialize_client_data, serialize_coordinator_weights
from examples.utils import load_skin_dataset, load_mini_boone, load_dry_bean, load_carbon_nanotube, load_mnist_digits

# Server parameters
host = '0.0.0.0'
port = 8000

# TenSEAL database
CONTEXTS = {
    # context_name: path_to_file
}
CURRENT_CONTEXT = (None, None)

# Datasets database
DATASETS = {
    # dataset_name: load_function
    'carbon': load_carbon_nanotube,
    'dry_bean': load_dry_bean,
    'mini_boone': load_mini_boone,
    'mnist': load_mnist_digits,
    'skin': load_skin_dataset
}
CURRENT_DATASET = None

# Client's fitted data database
CLIENT_DATA = {
    # uuid: process_status
}


@dataclass
class PartialData:
    id: str
    M_lst: list
    US_lst: list

# Computationally Intensive Task
def cpu_bound_task(partial_data: PartialData, sc: FedHEONN_coordinator):
    print(f"Aggregating partial data from: {partial_data.id}")
    sc.aggregate_partial([partial_data.M_lst], [partial_data.US_lst])
    sc.calculate_weights()
    time.sleep(15)

async def process_partial_aggregation(q: asyncio.Queue, pool: ThreadPoolExecutor):
    while True:
        # Get a request from the queue
        partial_data = await q.get()
        loop = asyncio.get_running_loop()
        print(f"Processing next client data, remaining: {q.qsize()}")
        CLIENT_DATA[partial_data.id] = "PROCESSING"
        await loop.run_in_executor(pool, cpu_bound_task, partial_data, singleton_coordinator())
        # Tell the queue that the processing on the task is completed
        q.task_done()
        CLIENT_DATA[partial_data.id] = "FINISHED"

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    q = asyncio.Queue()
    pool = ThreadPoolExecutor(max_workers=1)
    # Start the cpu intensive tasks
    asyncio.create_task(process_partial_aggregation(q, pool))
    yield {'q': q, 'pool': pool}
    # Free any resources that the pool is using when the currently pending futures are done executing
    pool.shutdown()

# Create FastAPI
app = FastAPI(lifespan=lifespan)
def start():
    uvicorn.run(app, host=host, port=port)

# Define singleton instance of coordinator
def singleton_coordinator() -> FedHEONN_coordinator:
    if not hasattr(singleton_coordinator, "server_coordinator"):
        singleton_coordinator.server_coordinator = FedHEONN_coordinator()
    return singleton_coordinator.server_coordinator

# Define singleton instance of the dataset loader
def singleton_dataset_loader():
    if not hasattr(singleton_dataset_loader, "loader"):
        singleton_dataset_loader.loader = DataSetLoader()
    return singleton_dataset_loader.loader

# REST API endpoints
@app.get("/ping")
def ping() -> dict[str, str]:
    """Used to check if the API is up"""
    return {"message": "pong"}

@app.get("/status")
def status() -> ServerStatus:
    """Used to check current server status"""
    sc = singleton_coordinator()

    test_status = {
        "contexts": list(CONTEXTS.keys()),
        "selected_context": CURRENT_CONTEXT[0],
        "datasets": list(DATASETS.keys()),
        "selected_dataset": CURRENT_DATASET,
        "coord_params": CoordinatorParams(**sc.get_parameters()),
        "status": "WAITING CLIENTS",
    }
    return ServerStatus(**test_status)

@app.post("/context", response_model=None)
def upload_context(file: UploadFile) -> dict[str, str] |  JSONResponse:
    global CONTEXTS

    try:
        contents = file.file.read()
        ctx_filepath = save_context(ctx_name=file.filename, content=contents, ctx_dict=CONTEXTS)
    except Exception as e:
        return answer_418(str(e))
    finally:
        file.file.close()

    return {"ctx_name": file.filename, "path": ctx_filepath}

@app.get("/context/{ctx_name}")
def get_context(ctx_name: str) -> Response:
    global CONTEXTS

    try:
        if ctx_name in CONTEXTS:
            return FileResponse(path=CONTEXTS[ctx_name], media_type='application/octet-stream', filename=ctx_name)
        else:
            return answer_404(f'Context not found on server database: {ctx_name}')
    except Exception as e:
        return answer_418(str(e))

@app.put("/context/{ctx_name}", response_model=None)
def select_context(ctx_name: str) -> str | JSONResponse:
    global CURRENT_CONTEXT, CONTEXTS

    if ctx_name in CONTEXTS:
        CURRENT_CONTEXT = (ctx_name, load_context(ctx_name=ctx_name, ctx_dict=CONTEXTS))
        return ctx_name
    else:
        return answer_404(f"Context not found on server database: {ctx_name}")

@app.delete("/context/{ctx_name}", response_model=None)
def erase_context(ctx_name: str) -> JSONResponse | None:
    global CURRENT_CONTEXT, CONTEXTS

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
def load_dataset() -> int | JSONResponse:
    global CURRENT_DATASET
    dataset_loader = singleton_dataset_loader()

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
def fetch_dataset(size:int) -> str | JSONResponse:
    dataset_loader = singleton_dataset_loader()

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
def fetch_dataset_test() -> str | JSONResponse:
    dataset_loader = singleton_dataset_loader()

    if dataset_loader.get_name() is None:
        return answer_404("Dataset not selected!")
    elif not dataset_loader.is_loaded():
        return answer_404("Current dataset is not loaded!")
    else:
        fragment = dataset_loader.fetch_test()
        return json.dumps([frag.tolist() for frag in fragment])

@app.put("/coordinator/parameters")
def set_coordinator_parameters(coord_params: CoordinatorParams) -> JSONResponse:
    sc = singleton_coordinator()

    try:
        sc.set_activation_functions(coord_params.f)
        sc.lam = coord_params.lam
        sc.sparse = coord_params.sparse
        sc.encrypted = coord_params.encrypted
        sc.ensemble = {'bagging'} if coord_params.bagging else None
        sc.parallel = coord_params.parallel
        sc.set_ctx_str(coord_params.ctx_str)
        return answer_200('Updated coordinator parameters!')
    except Exception as err:
        return answer_418(str(err))

@app.post("/coordinator/index_features")
def calculate_index_features(bagging_params: BaggingParams) -> JSONResponse:
    sc = singleton_coordinator()

    try:
        sc.calculate_idx_feats(n_estimators=bagging_params.n_estimators,
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
def send_index_features() -> str:
    sc = singleton_coordinator()

    idx_feats = sc.send_idx_feats()
    return json.dumps(idx_feats)

@app.get("/coordinator/send_weights")
def send_weights() -> JSONResponse:
    sc = singleton_coordinator()

    try:
        if not sc.W:
            # No optimal weights yet
            return answer_404("Empty optimal weights W, no data or not yet calculated!")
        else:
            data = serialize_coordinator_weights(sc.W, sc.encrypted)
            return answer_200_data(msg="encrypted" if sc.encrypted else "plain", data=json.dumps(data))

    except Exception as err:
        return answer_418(str(err))


@app.post("/aggregate/partial")
async def aggregate_partial(request: Request) -> JSONResponse:
    global CURRENT_CONTEXT

    try:
        sc = singleton_coordinator()
        # Status - ready for aggregation?
        if sc.encrypted and CURRENT_CONTEXT[0] is None:
            return answer_418("Context not loaded/selected and encryption enabled!")
        # Load and parse data
        data = await request.json()
        assert len(data) == 2
        M_c, US_c = deserialize_client_data(data[0], data[1], CURRENT_CONTEXT[1])
        # Add request to the queue
        partial_id = str(uuid.uuid4())
        partial_data = PartialData(partial_id, M_c, US_c)
        request.state.q.put_nowait(partial_data)
        # Add to client's database
        CLIENT_DATA[partial_id] = "ENQUEUED"

        return answer_200_data(msg=f"Data enqueued!", data=partial_id)

    except Exception as e:
        return answer_404(str(e))
