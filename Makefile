PYTHON ?= python
PACKAGE = copilot

.PHONY: setup data train evaluate repro mlflow-ui mlflow-smoke app test lint

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m copilot.data.pipeline all

train:
	$(PYTHON) -m copilot.models.train

evaluate:
	$(PYTHON) -m copilot.models.evaluate

repro:
	dvc repro

mlflow-ui:
	$(PYTHON) -m copilot.utils.mlflow_ui

mlflow-smoke:
	$(PYTHON) scripts/mlflow_smoke.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .
