PYTHON ?= python
TASK ?= queue
PACKAGE = copilot

.PHONY: setup data train train-queue train-priority evaluate rag stream drift retrain promote repro mlflow-ui mlflow-smoke app demo test lint

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m copilot.data.pipeline all

train: train-queue train-priority

train-queue:
	$(PYTHON) -m copilot.models.train --task queue

train-priority:
	$(PYTHON) -m copilot.models.train --task priority

evaluate:
	$(PYTHON) -m copilot.models.evaluate

rag:
	$(PYTHON) -m copilot.rag.faq_index
	$(PYTHON) -m copilot.rag.evaluate

stream:
	$(PYTHON) -m copilot.incidents.simulate_stream

drift:
	$(PYTHON) -m copilot.monitoring.drift

retrain:
	$(PYTHON) -m copilot.models.retrain --task $(TASK)

promote:
	$(PYTHON) -m copilot.registry.promote --task $(TASK)

repro:
	dvc repro

mlflow-ui:
	$(PYTHON) -m copilot.utils.mlflow_ui

mlflow-smoke:
	$(PYTHON) scripts/mlflow_smoke.py

app:
	streamlit run app/streamlit_app.py

demo:
	COPILOT_DEMO_MODE=1 streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .
