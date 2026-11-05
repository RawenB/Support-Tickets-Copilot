"""Drift detection and monitoring for model and data quality surveillance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
CURRENT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.json"
FEATURES = ("queue", "priority", "subject_length", "body_length")


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _current_frame() -> tuple[pd.DataFrame, str]:
    try:
        from copilot.db.repository import list_tickets

        tickets = list_tickets()
        if tickets:
            return pd.DataFrame(tickets), "sqlite"
    except Exception:
        pass
    return _load_frame(CURRENT_PATH), "test_split"


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FEATURES)
    result = pd.DataFrame(index=frame.index)
    result["queue"] = frame.get("queue", "unknown").fillna("unknown").astype(str).str.lower()
    result["priority"] = (
        frame.get("priority", "unknown").fillna("unknown").astype(str).str.lower()
    )
    result["subject_length"] = (
        frame.get("subject", "").fillna("").astype(str).str.len()
    )
    result["body_length"] = frame.get("body", "").fillna("").astype(str).str.len()
    return result


def _psi(reference: pd.Series, current: pd.Series, *, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        ref_counts = np.histogram(reference, bins=edges)[0]
        cur_counts = np.histogram(current, bins=edges)[0]
    else:
        categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
        ref_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_counts = current.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _fallback_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for feature in FEATURES:
        score = _psi(reference[feature], current[feature])
        columns[feature] = {
            "psi": round(score, 4),
            "drift_detected": bool(score >= 0.2),
        }
    drifted = sum(1 for value in columns.values() if value["drift_detected"])
    return {
        "dataset_drift": drifted > 0,
        "share_of_drifted_columns": round(drifted / len(FEATURES), 4),
        "drift_by_columns": columns,
        "method": "psi_fallback",
    }


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=reference,
        current_data=current,
    )
    payload = snapshot.dict()
    columns: dict[str, Any] = {}
    share = 0.0
    for metric in payload.get("metrics", []):
        metric_type = str(metric.get("config", {}).get("type", ""))
        value = metric.get("value")
        if metric_type.endswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value.get("share", 0.0))
        elif metric_type.endswith("ValueDrift"):
            column = str(metric.get("config", {}).get("column", "unknown"))
            threshold = float(metric.get("config", {}).get("threshold", 0.05))
            score = float(value) if isinstance(value, (int, float)) else 0.0
            columns[column] = {
                "score": round(score, 6),
                "threshold": threshold,
                "drift_detected": bool(score > threshold),
            }
    if not columns:
        return _fallback_report(reference, current)
    return {
        "dataset_drift": share >= 0.5,
        "share_of_drifted_columns": round(share, 4),
        "drift_by_columns": columns,
        "method": "evidently_data_drift_preset",
    }


def check_drift(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    """Measure drift between current and reference data distributions."""
    reference_frame = _features(reference if reference is not None else _load_frame(REFERENCE_PATH))
    if current is not None:
        current_frame = _features(current)
        source = "provided"
    else:
        raw_current, source = _current_frame()
        current_frame = _features(raw_current)
    if reference_frame.empty or current_frame.empty:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "n_reference": int(len(reference_frame)),
            "n_current": int(len(current_frame)),
            "dataset_drift": False,
            "share_of_drifted_columns": 0.0,
            "drift_by_columns": {},
            "method": "unavailable",
            "reason": "missing_reference_or_current_data",
        }
    else:
        try:
            result = _evidently_report(reference_frame, current_frame)
        except Exception:
            result = _fallback_report(reference_frame, current_frame)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "n_reference": int(len(reference_frame)),
            "n_current": int(len(current_frame)),
            **result,
        }
    if write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    print(json.dumps(check_drift(), indent=2))


if __name__ == "__main__":
    main()
