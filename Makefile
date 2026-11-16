PYTHON ?= python
TASK ?= queue
BASE ?= main
VERSION ?=

.PHONY: setup data validate train train-queue train-priority train-fast evaluate rag stream drift \
        retrain promote rollback gate repro status metrics params plots push pull release \
        mlflow-ui mlflow-smoke app demo test lint

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# ---- pipeline (DVC is the single entry point; every stage is cached and reproducible)
repro:            ## run every stale stage
	dvc repro

data:             ## ingest -> clean -> dedupe -> split -> validate
	dvc repro validate

validate:
	dvc repro validate

train:            ## fine-tune + select champions for both tasks (~40 min/task on Apple MPS)
	dvc repro train@queue train@priority

train-queue:
	dvc repro train@queue

train-priority:
	dvc repro train@priority

train-fast:       ## TF-IDF only experiment, outside DVC (no transformer)
	COPILOT_SKIP_TRANSFORMER=1 $(PYTHON) -m copilot.models.train --task $(TASK)

evaluate:
	dvc repro evaluate

rag:
	dvc repro rag_eval

stream:
	$(PYTHON) -m copilot.incidents.simulate_stream

drift:
	$(PYTHON) -m copilot.monitoring.drift

# ---- experiment comparison
status:
	dvc status

metrics:          ## metric deltas vs. $(BASE)
	dvc metrics diff $(BASE) --all

params:           ## hyper-parameter deltas vs. $(BASE)
	dvc params diff $(BASE)

plots:            ## confusion matrices + training curves vs. $(BASE) -> dvc_plots/index.html
	dvc plots diff $(BASE)

# ---- governance
gate:             ## absolute quality gates on the committed metrics
	$(PYTHON) scripts/check_model_gate.py

promote:          ## move @champion to @challenger if it passes the gates
	$(PYTHON) -m copilot.registry.promote --task $(TASK)

rollback:         ## move @champion back to @previous_champion
	$(PYTHON) -m copilot.registry.promote --task $(TASK) --rollback

retrain:
	$(PYTHON) -m copilot.models.retrain --task $(TASK)

release:          ## make release VERSION=0.2.0 -> annotated data/model/release git tags
	$(PYTHON) scripts/release.py --version $(VERSION) --apply

push:             ## publish data + model artifacts to the DVC remote
	dvc push

pull:
	dvc pull

# ---- tooling
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
