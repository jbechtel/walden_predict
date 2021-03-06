#! /bin/bash

PYTHON_VERSION=3.8.1
VENV_NAME=venv.walden-predict

pyenv install $PYTHON_VERSION
pyenv virtualenv $PYTHON_VERSION $VENV_NAME
pyenv local $VENV_NAME
pip install --upgrade pip
echo "INSTALL requirements.txt"
pip install -r requirements.txt
echo "INSTALL requirements-dev.txt"
pip install -r requirements-dev.txt
echo "INSTALL walden-predict"
pip install -e .

