#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.server.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define models to process encrypted data

:author: youben11@github
:version: 0.0.1
"""
# Standard libraries
import os
import logging
import pickle
from typing import List
from inspect import isclass
from collections import namedtuple
# Application modules
from api.server.models.base import Model
from api.exceptions import ModelNotFound
from api.server.models.linear_layer import LinearLayer


_DEFAULT_DATA_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")

model_def = namedtuple("model_definition", ["constructor", "default_version", "versions", "data_dir"])

# contains definition of every registered model
_MODEL_DEFS = {
    # "model_x": model_def(
    #     constructor=model_x,
    #     default_version="0.1",
    #     versions=["0.1", "0.2"],
    #     data_dir="/some/path/here",
    # )
}

# cache model versions, set initially to one, then lazy loaded on demand
_MODELS_CACHE = {
    # model_name: {
    #     model_version: None for model_version in _MODEL_DEFS[model_name].versions
    # }
    # for model_name in _MODEL_DEFS.keys()
}


def register_model(
    constructor_class,
    versions: List[str],
    name_model: str = None,
    default_version: str = None,
    data_dir: str = None,
):
    """Register a model to serve in the API

    Args:
        constructor_class: class derived from eeval.server.models.abstract_model.Model
        versions: versions of the model, each version should contain a different parameter file
        name_model: name of the model to register it with. Default is the class name
        default_version: the default version to use. Set to the first element of versions if None
        data_dir: path to the directory to load parameters from, parameters files should be named
            {model_name}-{version}.pickle and can be unpickled. Use default folder if None

    Returns:
        str: the name of the model registered

    Raises:
        TypeError: constructor_class is not a subclass of models.abstract_model.Model
        ValueError: versions is empty or default_version not in versions
        FolderNotFound: folder not found
        FileNotFound: versions file not found
    """
    if not isclass(constructor_class):
        raise TypeError("constructor_class must be class")
    if not issubclass(constructor_class, Model):
        raise TypeError("constructor_class is not a subclass of models.abstract_model.Model")
    if len(versions) == 0:
        raise ValueError("versions can't be an empty list")

    if default_version is None:
        default_version = versions[0]
    elif default_version not in versions:
        raise ValueError("default_version must be in versions")

    if name_model is None:
        name_model = constructor_class.__name__

    # add the model to the definitions list
    _MODEL_DEFS[name_model] = model_def(
        constructor=constructor_class,
        default_version=default_version,
        versions=versions,
        data_dir=data_dir,
    )

    # add entries for each version in the cache
    _MODELS_CACHE[name_model] = {
        model_version: None for model_version in _MODEL_DEFS[name_model].versions
    }


def set_default_data_dir(path: str):
    """Set the default directory where to look at model's data such as parameters"""
    global _DEFAULT_DATA_DIR
    _DEFAULT_DATA_DIR = os.path.realpath(path)
    return _DEFAULT_DATA_DIR


def _load_parameters(name_model: str, version: str) -> dict:
    """Load parameters for `model_name`:`version` from the appropriate file.

    Args:
        name_model: the name of the model
        version: the version of the model

    Returns:
        dict: parameters loaded from the appropriate file

    Raises:
        OSError: if it can't open parameters' file
    """
    file_name = f"{name_model}-{version}.pickle"
    data_dir = _MODEL_DEFS[name_model].data_dir
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    file_path = os.path.realpath(os.path.join(data_dir, file_name))
    try:
        parameters = pickle.load(open(file_path, "rb"))
        print(f"Model `{name_model}` version `{version}` loaded from '{file_path}'")
    except OSError as ose:
        logging.error(f"Parameters file `{file_path}` not found for model `{name_model}`, version {version}")
        raise ose
    return parameters


def get_model(name_model: str, version: str = None) -> Model:
    """Get model `model_name`:`version` if it exists.

    Args:
        name_model: the name of the model to use
        version: version of the model to use, a default version will be used if not specified

    Returns:
        Model: loaded model

    Raises:
        ModelNotFound: if the `model_name` or `version` are incorrect
    """

    # check if (model_name, version) is available
    if name_model not in _MODEL_DEFS.keys():
        raise ModelNotFound(f"Model `{name_model}` can't be found in this server")
    if version is None:
        version = _MODEL_DEFS[name_model].default_version
    elif version not in _MODEL_DEFS[name_model].versions:
        raise ModelNotFound(f"Model `{name_model}` doesn't have version `{version}`")

    # lazy loading of models
    if _MODELS_CACHE[name_model][version] is None:
        parameters = _load_parameters(name_model, version)
        _MODELS_CACHE[name_model][version] = _MODEL_DEFS[name_model].constructor(parameters)

    return _MODELS_CACHE[name_model][version]


def get_model_info(name_model: str) -> dict:
    """Get descriptive attributes of model `model_name`.

    Args:
        name_model: the name of the model

    Returns:
        dict: containing descriptive model attributes

    Raises:
        ModelNotFound: if the model doesn't exist
    """
    try:
        model_def_tuple = _MODEL_DEFS[name_model]
    except KeyError:
        raise ModelNotFound(f"Model `{name_model}` can't be found in this server")

    return {
        "name_model": name_model,
        "description": model_def_tuple.constructor.__doc__,
        "default_version": model_def_tuple.default_version,
        "versions": model_def_tuple.versions,
    }


def get_all_model_info() -> List[dict]:
    """Get the description of all the available model"""
    model_definitions = [get_model_info(model_name) for model_name in _MODEL_DEFS]
    return model_definitions


__all__ = [
    "get_all_model_info",
    "get_model_info",
    "get_model",
    "Model",
    "register_model",
    "set_default_data_dir",
    "LinearLayer",
    # IMPLEMENTED MODELS HERE
]