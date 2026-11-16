"""Champion / challenger promotion in the MLflow model registry.

Training registers every new model as ``@challenger``. Promotion moves ``@champion`` to the
challenger only if it

1. clears the absolute quality gates in ``params.yaml`` (``f1_gate`` and
   ``promotion.novel_f1_gate``), and
2. does not regress against the current champion by more than
   ``promotion.max_regression`` macro F1.

The replaced champion keeps the ``@previous_champion`` alias, so ``--rollback`` is one call.
Every decision is written to ``reports/promotion_<task>.json`` and tagged on the version.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from copilot.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _params() -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")) or {}


def _gate() -> float:
    return float(_params().get("f1_gate", 0.70))


def _version_scores(client: Any, version: Any) -> dict[str, Any]:
    run = client.get_run(version.run_id)
    metrics = run.data.metrics
    return {
        "version": str(version.version),
        "run_id": version.run_id,
        "macro_f1": float(metrics.get("test_macro_f1", metrics.get("macro_f1", 0.0))),
        "novel_macro_f1": metrics.get("test_novel_macro_f1"),
        "data_fingerprint": run.data.tags.get("data_fingerprint"),
        "model_type": run.data.tags.get("champion_candidate") or run.data.params.get("model_type"),
    }


def _alias(client: Any, name: str, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:
        return None


def promote_model(
    task: str = "queue",
    *,
    run_id: str | None = None,
    alias: str = "champion",
    f1_gate: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Promote the challenger (or the version created from ``run_id``) if it passes the gates."""
    params = _params()
    promotion = params.get("promotion", {})
    gate = _gate() if f1_gate is None else float(f1_gate)
    novel_gate = float(promotion.get("novel_f1_gate", 0.0))
    max_regression = float(promotion.get("max_regression", 0.0))

    client = setup_mlflow(f"support-copilot-{task}")
    name = f"support-copilot-{task}"
    if run_id:
        matches = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
        if not matches:
            raise ValueError(f"No registered version of {name} comes from run {run_id}")
        candidate = max(matches, key=lambda v: int(v.version))
    else:
        candidate = _alias(client, name, "challenger")
        if candidate is None:
            raise ValueError(f"{name} has no @challenger. Train a model first.")
    challenger = _version_scores(client, candidate)
    current = _alias(client, name, alias)
    incumbent = _version_scores(client, current) if current is not None else None

    reasons: list[str] = []
    if challenger["macro_f1"] < gate:
        reasons.append(f"macro_f1 {challenger['macro_f1']:.4f} < gate {gate:.2f}")
    novel = challenger["novel_macro_f1"]
    if novel_gate and (novel is None or float(novel) < novel_gate):
        reasons.append(f"novel_macro_f1 {novel} < novel gate {novel_gate:.2f}")
    if incumbent and incumbent["version"] == challenger["version"]:
        reasons.append(f"version {challenger['version']} is already @{alias}")
    elif incumbent and challenger["macro_f1"] < incumbent["macro_f1"] - max_regression:
        reasons.append(
            f"regression vs @{alias} v{incumbent['version']}: "
            f"{challenger['macro_f1']:.4f} < {incumbent['macro_f1']:.4f} - {max_regression}"
        )
    same_data = incumbent is None or incumbent["data_fingerprint"] == challenger["data_fingerprint"]

    promoted = force or not reasons
    if promoted:
        if incumbent and incumbent["version"] != challenger["version"]:
            client.set_registered_model_alias(name, "previous_champion", incumbent["version"])
        client.set_registered_model_alias(name, alias, challenger["version"])
    decision = {
        "task": task,
        "model_name": name,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted": promoted,
        "forced": bool(force and reasons),
        "reasons": reasons,
        "alias": alias,
        "version": challenger["version"],
        "run_id": challenger["run_id"],
        "macro_f1": challenger["macro_f1"],
        "novel_macro_f1": challenger["novel_macro_f1"],
        "gate": gate,
        "novel_gate": novel_gate,
        "max_regression": max_regression,
        "challenger": challenger,
        "incumbent": incumbent,
        "same_test_data_as_incumbent": same_data,
    }
    client.set_model_version_tag(name, challenger["version"], "promotion_decision",
                                 "promoted" if promoted else "rejected: " + "; ".join(reasons))
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"promotion_{task}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def rollback(task: str, alias: str = "champion") -> dict[str, Any]:
    """Point ``@champion`` back at ``@previous_champion``."""
    client = setup_mlflow(f"support-copilot-{task}")
    name = f"support-copilot-{task}"
    previous = _alias(client, name, "previous_champion")
    if previous is None:
        raise ValueError(f"{name} has no @previous_champion to roll back to")
    current = _alias(client, name, alias)
    client.set_registered_model_alias(name, alias, previous.version)
    if current is not None:
        client.set_registered_model_alias(name, "previous_champion", current.version)
    return {"task": task, "alias": alias, "version": str(previous.version),
            "replaced": str(current.version) if current else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["queue", "priority"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--force", action="store_true", help="promote even if a gate fails (recorded)")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        print(json.dumps(rollback(args.task, args.alias), indent=2))
        return
    decision = promote_model(args.task, run_id=args.run_id, alias=args.alias, force=args.force)
    print(json.dumps(decision, indent=2))
    if not decision["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
