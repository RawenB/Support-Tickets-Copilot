from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
for path in (ROOT_DIR, SRC_ROOT):
    candidate = str(path)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
