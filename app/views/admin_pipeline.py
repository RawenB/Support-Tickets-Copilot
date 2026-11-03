"""Admin pipeline and release view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from app.ui.components import kpi_row, page_header, section_title

DVC_PATH = Path("dvc.yaml")
RELEASE_PATH = Path("reports") / "release.json"


def _load_stages() -> dict[str, Any]:
    if not DVC_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(DVC_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    stages = payload.get("stages")
    return stages if isinstance(stages, dict) else {}


def admin_pipeline_view() -> None:
    """Show pipeline DAG and release metadata for operational reviews."""
    stages = _load_stages()
    release_text = RELEASE_PATH.read_text(encoding="utf-8") if RELEASE_PATH.exists() else None

    page_header(
        "Pipeline",
        "Reproducible data and model stages tracked by DVC, with release metadata.",
        eyebrow="Admin",
        icon="pipeline",
    )

    release: dict[str, Any] = {}
    if release_text:
        try:
            release = json.loads(release_text)
        except json.JSONDecodeError:
            release = {}

    kpi_row(
        [
            {
                "label": "Pipeline stages",
                "value": len(stages),
                "delta": "Defined in dvc.yaml",
                "tone": "accent",
                "icon": "pipeline",
            },
            {
                "label": "Release status",
                "value": "Promoted" if release else "Pending",
                "delta": "Releases/release.json",
                "tone": "success" if release else "warning",
                "icon": "confirm",
            },
            {
                "label": "Reproducibility",
                "value": "DVC",
                "delta": "Data + pipeline versioned",
                "tone": "default",
                "icon": "refresh",
            },
            {
                "label": "Quality gate",
                "value": "Macro F1",
                "delta": "Primary promotion metric",
                "tone": "default",
                "icon": "shield",
            },
        ]
    )

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Pipeline stages", "Execution order", icon="pipeline")
            if not stages:
                st.caption("No DVC stages found.")
            for index, (name, stage) in enumerate(stages.items(), start=1):
                deps = stage.get("deps") or []
                outs = stage.get("outs") or []
                st.markdown(f"**{index}. {name.replace('_', ' ').title()}**")
                details = []
                if deps:
                    details.append(f"{len(deps)} dependencies")
                if outs:
                    details.append(f"{len(outs)} outputs")
                st.caption(" · ".join(details) or "No dependencies or outputs declared.")

        with st.expander("View dvc.yaml"):
            if DVC_PATH.exists():
                st.code(DVC_PATH.read_text(encoding="utf-8"), language="yaml")
            else:
                st.caption("dvc.yaml is not present.")

    with right:
        with st.container(border=True):
            section_title("Release record", icon="description")
            if release:
                st.json(release)
            else:
                st.info("No release record is present yet.")

        with st.container(border=True):
            section_title("Operational checklist", icon="focus")
            st.markdown(
                "- Run `dvc repro` after data or config changes\n"
                "- Log params, metrics, git SHA, and data hash to MLflow\n"
                "- Review the drift report before promotion\n"
                "- Complete human review before any outbound send"
            )
