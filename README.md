# FEDHEONN_ENSEMBLE
Added ensemble capabilities (via the *Random Patches* bagging technique) to the original [FEDHEONN algorithm](https://github.com/ofontenla/FedHEONN) and a server/client REST API architecture (using [FastAPI](https://github.com/tiangolo/fastapi)) has been implemented for federated learning scenearios.

## Python
Developed using **Python_3.10.7**

## Virtual Environment
To create the virtual environment follow these steps:
```
mkdir -p  <path>
cd <path>
virtualenv .
```
To load and activate said environment:
```
. bin/activate
```

## Installing dependencies
Requirements:
```
pip3 install -r requirements.txt
```
