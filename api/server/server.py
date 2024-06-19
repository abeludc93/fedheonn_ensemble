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
from typing import Dict, List, Any
# Third-party libraries
import tenseal as ts
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
# Application modules
from api.server.models import get_model_info, get_model, get_all_model_info
from api.exceptions import *
from api.server import storage
from auxiliary.logger import logger as log

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


class ContextPK(BaseModel):
    context: str = Field(..., description="Context ID"),
    ctx_pk: str = Field(..., description="Serialized context PK")


class CKKSVector(BaseModel):
    ckks_vector: str = Field(..., description="Serialized CKKSVector representing the input to the model")


class Context(BaseModel):
    context: str = Field(..., description="Serialized TenSEALContext containing the keys needed for the evaluation")


class Dataset(BaseModel):
    X: List[str] = Field(..., description="Serialized CKKSVectors representing the data features")
    Y: List[str] = Field(..., description="Serialized CKKSVectors representing the data labels")
    batch_size: int = Field(1, min=1, description="Number of entries per CKKSVector")


class ModelDescription(BaseModel):
    name_model: str = Field(..., description="Name of the model. Used to query an evaluation")
    description: str = Field(..., description="The description of the model architecture, as well the input that "
                                              "should be passed to it")
    default_version: str = Field(..., description="The default version used during evaluation")
    versions: List[str] = Field(..., description="Available versions of the model")


class NumpyArray(BaseModel):
    nd_array: str = Field(..., description="Serialized numpy ndarray representing the output of the model")


def answer_418(msg: str):
    return JSONResponse(status_code=418, content={"message": f"Oops! Server says '{msg}'"})


def answer_404(msg: str):
    return JSONResponse(status_code=404, content={"message": f"Resource not found. Server says '{msg}'"})


@app.get("/models/", response_model=List[ModelDescription])
async def list_models():
    """List available models with their description"""
    return get_all_model_info()


@app.get("/models/{model_name}", response_model=ModelDescription)
async def describe_model(model_name: str):
    """Describe model `model_name`"""
    try:
        model_def = get_model_info(model_name)
    except ModelNotFound as mnf:
        return answer_418(str(mnf))
    return model_def


@app.post("/eval/{model_name}", response_model=NumpyArray)
async def evaluation(data: CKKSVector, model_name: str):
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


@app.get("/ping", response_model=None)
async def ping() -> dict[str, str]:
    """Used to check if the API is up"""
    return {"message": "pong"}


@app.get("/contexts/", response_model=Context)
async def get_context(context_id: str):
    """Get a previously registered context"""
    try:
        ctx = storage.get_raw_context(context_id)
    except KeyError:
        return answer_404(f"No context with id {context_id}")
    return {"context": b64encode(ctx)}


@app.get("/datasets/", response_model=Dataset)
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


@app.get("/create_context/", response_model=ContextPK)
def create_context(poly_modulus_degree: int, coef_mod_bit_sizes: list[int], global_scale: float,
                   gen_galois_keys: bool, save_secret_key: bool):
    """Create a TenSEAL context holding encryption keys and parameters:
        * poly_modulus_degree: polynomial modulus degree
        * coef_mod_bit_sizes: bit size of the coefficients' modulus
        * global_scale: scale to use by default for CKKS encoding
        * gen_galois_keys: generate galois keys
        * gen_relin_keys: generate relinearization keys
        * save_secret_key: save the secret key into the context
    """

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coef_mod_bit_sizes,
    )
    # set scale
    ctx.global_scale = global_scale
    log("context created")

    if gen_galois_keys:
        ctx.generate_galois_keys()
        log("galois keys generated")
    if not save_secret_key:
        # drop secret-key
        ctx_pk = ctx.secret_key()
        ctx_pk = ctx.serialize()
        ctx.make_context_public()
        log("secret key dropped")

    ctx_id = storage.save_context(ctx.serialize())
    log("context saved successfully!")

    return ctx_id if save_secret_key else ctx_id, ctx_pk


def start(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
