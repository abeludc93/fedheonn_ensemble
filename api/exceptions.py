#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
api.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom exceptions. Basically just giving meaningful names.

:author: youben11@github
:version: 0.0.1
"""


class Answer418(Exception):
    """Request specific error, the meaning mostly depends on the request"""

    pass


class ServerError(Exception):
    """When the server returns a status 500 response"""

    pass


class ResourceNotFound(Exception):
    """When a resource isn't found on the remote server"""

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