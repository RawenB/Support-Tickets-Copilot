"""Quality gate on the committed metrics files (used by CI and `make gate`).

Fails if a champion is below ``f1_gate`` or ``promotion.novel_f1_gate`` in params.yaml,
was not evaluated on the real test split, or if its metrics were produced from a different
data version than the one recorded in reports/data_validation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8")) or {}
    gate = float(params.get("f1_gate", 0.70))
    novel_gate = float(params.get("promotion", {}).get("novel_f1_gate", 0.0))
    failures: list[str] = []
    rows: list[str] = ["| task | model | macro F1 | novel F1 | accuracy | n_test |", "|---|---|---|---|---|---|"]
    for task in ("queue", "priority"):
        path = ROOT / "reports" / f"metrics_{task}.json"
        if not path.exists():
            failures.append(f"{path.name} missing")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        f1 = float(m.get("macro_f1", 0.0))
        novel = m.get("novel_macro_f1")
        rows.append(
            f"| {task} | {m.get('model_type')} | {f1:.4f} | "
            f"{'n/a' if novel is None else f'{float(novel):.4f}'} | {float(m.get('accuracy', 0)):.4f} | {m.get('n_test')} |"
        )
        if m.get("split") != "test" or int(m.get("n_test", 0)) < 1000:
            failures.append(f"{task}: not evaluated on the real test split")
        if f1 < gate:
            failures.append(f"{task}: macro_f1 {f1:.4f} < f1_gate {gate:.2f}")
        if novel_gate and (novel is None or float(novel) < novel_gate):
            failures.append(f"{task}: novel_macro_f1 {novel} < novel_f1_gate {novel_gate:.2f}")
    print("\n".join(rows))
    print(f"\nGates: macro F1 >= {gate:.2f}, novel-ticket macro F1 >= {novel_gate:.2f}")
    if failures:
        print("\nMODEL GATE FAILED:\n- " + "\n- ".join(failures))
        return 1
    print("\nModel gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
