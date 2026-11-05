from __future__ import annotations

import pandas as pd

from copilot.monitoring.drift import check_drift


def _frame(queue: str, priority: str, size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "queue": [queue] * size,
            "priority": [priority] * size,
            "subject": ["Password reset request"] * size,
            "body": ["The reset link keeps returning to the login page."] * size,
        }
    )


def test_drift_report_is_stable_for_identical_distributions() -> None:
    reference = _frame("technical_support", "high", 80)
    current = _frame("technical_support", "high", 80)
    report = check_drift(reference, current, write_report=False)
    assert report["n_reference"] == 80
    assert report["n_current"] == 80
    assert report["method"] in {"evidently_data_drift_preset", "psi_fallback"}
    assert report["share_of_drifted_columns"] == 0.0


def test_drift_report_detects_a_real_distribution_shift() -> None:
    reference = _frame("technical_support", "high", 80)
    current = _frame("billing_and_payments", "low", 80)
    report = check_drift(reference, current, write_report=False)
    assert report["dataset_drift"] is True
    assert report["share_of_drifted_columns"] > 0.0
