# MLOps guide

Code is versioned by **git**, data and model binaries by **DVC**, experiments and the
model registry by **MLflow**, and every release is pinned by **annotated git tags**.
CI enforces lint, tests, pipeline validity and the model quality gates.

```
raw CSV ──dvc──▶ ingest ▶ clean ▶ dedupe ▶ split ▶ validate ──▶ train@queue ─┐
                                                   (27 data checks)   train@priority ─┴▶ evaluate
                                                                        │
                                       MLflow: parent run + child run per candidate,
                                       per-epoch curves, confusion matrix, lineage tags
                                                                        │
                                       registry: new version ─▶ @challenger
                                                                        │  make promote (gates)
                                                                        ▼
                                                          @champion  (+ @previous_champion)
```

## 1. Reproduce everything

```bash
source .venv/bin/activate
dvc pull          # data + model artifacts from the DVC remote
dvc repro         # re-runs only the stages whose code, data or params changed
```

`dvc repro` is the single entry point (the Makefile targets are thin wrappers). Every stage
declares its code, data and params, so DVC knows exactly what is stale:

| Stage | Input | Output |
|---|---|---|
| `ingest → clean → dedupe → split` | raw CSV, `params.yaml: language, seed, split, min_queue_count` | `data/splits/*.csv`, `reports/data_audit.json` |
| `validate` | splits | `reports/data_validation.json` (fails the pipeline on any violation) |
| `train@queue`, `train@priority` | splits, `configs/train_<task>.yaml`, `params.yaml: seed, eval` | `models/<task>_model/`, `reports/metrics_<task>.json`, plots |
| `evaluate` | both models, test split | `reports/metrics_summary.json` |
| `build_faq_index`, `rag_eval` | `data/faq/*.md` | `data/faq_index/`, `reports/rag_eval.json` |

Metrics and plots are stored with `cache: false`, so they live in git and can be compared
across any two commits.

## 2. Run an experiment

1. Create a branch: `git switch -c exp/<idea>`.
2. Change a hyper-parameter in `configs/train_<task>.yaml` (or code).
3. `dvc repro` (or `make train-queue`).
4. Compare against `main`:
   ```bash
   make params     # dvc params diff main
   make metrics    # dvc metrics diff main --all
   make plots      # dvc plots diff main -> dvc_plots/index.html (confusion matrix, training curve)
   make mlflow-ui  # per-epoch curves, candidate comparison, artifacts
   ```
5. Commit `dvc.lock`, `reports/*.json`, configs; `dvc push`. The PR shows the metric delta
   in the *Model gate* workflow summary.

For quick TF-IDF-only iterations outside DVC: `make train-fast TASK=queue`.

## 3. What is tracked in MLflow

Each training run is a **parent run** (`<task>-train`) with **child runs** per candidate
(`tfidf_sgd`, `transformer`, `blend`).

- **Params**: every value from `configs/train_<task>.yaml`, seed, split sizes, blend weights.
- **Metrics**: per-epoch `train_loss`, `val_macro_f1`, `val_accuracy` (step = epoch);
  `val_*` for each candidate; `test_*` for the champion only, including
  `test_novel_macro_f1` (see §5).
- **Tags (lineage)**: `git_commit`, `git_branch`, `git_dirty`, `git_describe`,
  `data_fingerprint` + md5 of each split, Python version and platform.
- **Artifacts**: confusion matrix PNG, per-class report, label distribution, and the model
  as an MLflow **pyfunc** (`mlflow models serve -m models:/support-copilot-queue@champion`).

## 4. Registry and promotion (champion / challenger)

- Training registers a new version and moves the **`@challenger`** alias to it. It never
  touches `@champion`.
- `make promote TASK=queue` moves **`@champion`** only if the challenger
  1. clears `f1_gate` and `promotion.novel_f1_gate` in `params.yaml`, and
  2. does not lose more than `promotion.max_regression` macro F1 against the champion.
  The old champion keeps **`@previous_champion`**. The decision and its reasons go to
  `reports/promotion_<task>.json` and to a tag on the registry version.
- `make rollback TASK=queue` swaps `@champion` back to `@previous_champion`.
- The app serves `models/<task>_model/` (the DVC output of the checked-out commit) and falls
  back to `models:/support-copilot-<task>@champion`. It never falls back to a toy model.

## 5. Honest evaluation: the novel-ticket slice

About 45 % of test tickets are paraphrases of a training ticket (TF-IDF cosine ≥ 0.6),
and their labels agree 95 %+ of the time. A model that memorises therefore looks better
than it is. Every report also includes the macro F1 on **novel** tickets (closest
train/val ticket below `eval.novelty_threshold`). The champion has to clear
`promotion.novel_f1_gate` on that slice as well.

## 6. Versioning with git tags

| Tag | Meaning | Created |
|---|---|---|
| `data-v<N>` | a new DVC data version (split hashes in the tag message) | when the split hash changes |
| `model-<task>-v<R>` | commit whose `dvc.lock` produced MLflow registry version `R` | each release |
| `v<MAJOR.MINOR.PATCH>` | release bundling the data and model tags above | each release |

```bash
python scripts/release.py --version 0.2.0   # dry run
make release VERSION=0.2.0                  # create the annotated tags
git push origin --tags && dvc push          # publish code pointers + artifacts
git checkout model-queue-v7 && dvc pull     # restore exactly that model later
```

The release script refuses to tag a dirty tree or a commit where `dvc status` is not clean,
so every tag can be reproduced with `dvc pull && dvc repro`.

## 7. CI

- `ci.yml`: ruff, `dvc dag` (pipeline definition is valid), pytest. Tests that need local
  artifacts (models, MLflow store) skip on a fresh checkout.
- `model-gate.yml`: `scripts/check_model_gate.py` on the committed metrics, plus
  `dvc metrics diff` / `dvc params diff` against `main` in the job summary.

## 8. Human feedback loop

1. Agents correct routing or priority in the UI; corrections are stored in SQLite.
2. `make retrain TASK=<queue|priority>` appends them to the training split and retrains.
3. The result goes to `reports/retrain_<task>.json`; promotion still goes through the gates.

## Governance

- Support agents keep the final decision; low-confidence predictions go to human review.
- The product path never invents tickets or fallback predictions.
