from __future__ import annotations

from copilot.data.clean import clean
from copilot.data.dedupe import dedupe
from copilot.data.ingest import ingest
from copilot.data.split import split


def test_data_modules_importable() -> None:
    assert callable(ingest)
    assert callable(clean)
    assert callable(dedupe)
    assert callable(split)
