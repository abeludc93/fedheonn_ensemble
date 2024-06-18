#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESTful API providing the main evaluation service

:author: youben11@github
:version: 0.0.1
"""
# Standard libraries
from base64 import b64encode, b64decode
# Third-party libraries
import tenseal as ts
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
# Application modules
from eeval.server.models import get_model, get_all_model_def, get_model_def
from api.exceptions import *
from api.server import storage

CORS = True

app = FastAPI()

if CORS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def answer_418(msg: str):
    return JSONResponse(status_code=418, content={"message": f"Oops! Server says '{msg}'"})


def answer_404(msg: str):
    return JSONResponse(status_code=404, content={"message": f"Resource not found. Server says '{msg}'"})


@app.get("/models/{model_name}", response_model=dict, response_description="model info")
async def describe_model(model_name: str):
    """Describe model `model_name`"""
    try:
        model_def = get_model_def(model_name)
    except ModelNotFound as mnf:
        return answer_418(str(mnf))
    return model_def


@app.post("/eval/{model_name}", response_model=np.ndarray, response_description="output of the model")
async def evaluation(data: ts.CKKSVector, model_name: str):
    """
    Evaluate encrypted input data using the model `model_name`

    - ckks_vector: a serialized CKKSVector representing the input to the model
    - model_name: model being used for given evaluation
    """

    # fetch model
    try:
        model = get_model(model_name)
    except ModelNotFound as mnf:
        return answer_418(str(mnf))
    except:
        raise HTTPException(status_code=500)

    # decode data from client
    try:
        ctx = storage.load_context()
        ckks_vector = b64decode(data)
        ckks_vector = ts.ckks_vector_from(ctx, ckks_vector)
    except:
        return answer_418("bad base64 strings")

    # deserialize input and do the evaluation
    try:
        encrypted_out = model(ckks_vector)
        out = encrypted_out.decrypt()
    except (DeserializationError, EvaluationError, InvalidContext) as error:
        return answer_418(str(error))

    return {"output": b64encode(out.serialize())}


@app.get("/ping")
async def ping():
    """Used to check if the API is up"""
    return {"message": "pong"}


@app.get("/contexts/", response_model=ts.Context, response_description="Context referenced by `context_id`")
async def get_context(context_id: str):
    """Get a previously registered context"""
    try:
        ctx = storage.get_raw_context(context_id)
    except KeyError:
        return answer_404(f"No context with id {context_id}")
    return {"context": b64encode(ctx)}


@app.get("/datasets/", response_model=dict, response_description="registered dataset referenced by `dataset_id`")
async def get_dataset(dataset_id: str):
    """Get a previously registered dataset"""
    try:
        X, Y, batch_size = storage.load_dataset(dataset_id)
    except KeyError:
        return answer_404(f"No dataset with id {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "X": [b64encode(x) for x in X],
        "Y": [b64encode(y) for y in Y],
        "batch_size": batch_size,
    }


def start(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
