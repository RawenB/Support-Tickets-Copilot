PYTHON ?= python
PACKAGE = copilot

.PHONY: setup data train evaluate repro mlflow-ui app test lint

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m copilot.data.ingest

train:
	$(PYTHON) -m copilot.models.train

evaluate:
	$(PYTHON) -m copilot.models.evaluate

repro:
	dvc repro

mlflow-ui:
	mlflow ui --backend-store-uri file://$$(pwd)/mlruns

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .
