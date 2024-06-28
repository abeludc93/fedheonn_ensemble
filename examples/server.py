#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from api.server import server as srv
from api.server import models as mdl


def start():
    # register the LinearLayer model
    mdl.register_model(mdl.LinearLayer, versions=["0.1"])
    srv.start(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()
