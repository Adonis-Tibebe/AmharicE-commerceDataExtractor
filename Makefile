# Variables
VENV = new_venv
PYTHON = $(VENV)/Scripts/python

.PHONY: help venv install test lint clean

help:
    @echo "Common commands:"
    @echo "  make venv      - Create virtual environment"
    @echo "  make install   - Install dependencies"
    @echo "  make test      - Run all unit tests"
    @echo "  make lint      - Run code linter (flake8)"
    @echo "  make clean     - Remove Python cache and build files"
venv:
    python -m venv $(VENV)

install:
    $(PYTHON) -m pip install --upgrade pip
    $(PYTHON) -m pip install -r requirements.txt

test:
    $(PYTHON) -m pytest tests/unit

lint:
    $(PYTHON) -m pip install flake8
    $(PYTHON) -m flake8 src/ tests/

clean:
    rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache *.pyc *.pyo *.egg-info build dist