"""Create annotated git tags that pin code + data + model versions together.

    python scripts/release.py --version 0.2.0            # dry run: prints the tags
    python scripts/release.py --version 0.2.0 --apply    # creates them locally

Tags (see docs/MLOPS.md):
  data-v<N>            new data version (only when the DVC data hash changed since the last data tag)
  model-<task>-v<R>    R = MLflow registry version of the task's model in this commit
  v<semver>            the release that bundles all of the above

Refuses to tag when the work tree is dirty or `dvc status` reports stale stages, so a tag
always points at a commit from which `dvc pull && dvc repro` reproduces the tagged models.
Pushing is left to you: `git push origin --tags` and `dvc push`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sh(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"$ {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def data_hash() -> str:
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    outs = lock["stages"]["split"]["outs"]
    return ",".join(f"{Path(o['path']).name}:{o['md5'][:8]}" for o in outs)


def previous_data_tags() -> list[tuple[int, str]]:
    found = []
    for tag in sh("git", "tag", "--list", "data-v*").splitlines():
        match = re.fullmatch(r"data-v(\d+)", tag)
        if match:
            found.append((int(match.group(1)), tag))
    return sorted(found)


def tag_message_hash(tag: str) -> str | None:
    match = re.search(r"data_hash: (\S+)", sh("git", "tag", "-l", "--format=%(contents)", tag))
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="semantic version, e.g. 0.2.0")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        raise SystemExit("--version must look like 1.2.3")

    if not args.allow_dirty:
        if sh("git", "status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("work tree has uncommitted changes: commit first")
        status = sh("dvc", "status", "--quiet", check=False)
        if status:
            raise SystemExit(f"`dvc status` is not clean, run `dvc repro` and commit dvc.lock first:\n{status}")

    commit = sh("git", "rev-parse", "--short", "HEAD")
    tags: list[tuple[str, str]] = []
    current_hash = data_hash()
    validation = json.loads((ROOT / "reports" / "data_validation.json").read_text(encoding="utf-8"))
    previous = previous_data_tags()
    if not previous or tag_message_hash(previous[-1][1]) != current_hash:
        number = previous[-1][0] + 1 if previous else 1
        tags.append((f"data-v{number}", "\n".join([
            f"Data version {number}",
            f"data_hash: {current_hash}",
            f"rows: {json.dumps(validation.get('rows'))}",
            f"validation: {validation.get('n_checks', 0) - validation.get('n_failed', 0)}/{validation.get('n_checks')} checks passed",
        ])))

    for task in ("queue", "priority"):
        metrics = json.loads((ROOT / "reports" / f"metrics_{task}.json").read_text(encoding="utf-8"))
        version = metrics.get("registry_version")
        if not version:
            print(f"warning: {task} has no registry_version, skipping model tag", file=sys.stderr)
            continue
        name = f"model-{task}-v{version}"
        if sh("git", "tag", "--list", name):
            print(f"{name} already exists, skipping", file=sys.stderr)
            continue
        tags.append((name, "\n".join([
            f"{task} model, MLflow registry support-copilot-{task} v{version}",
            f"model_type: {metrics.get('model_type')}",
            f"test_macro_f1: {metrics.get('macro_f1'):.4f}",
            f"test_novel_macro_f1: {metrics.get('novel_macro_f1') or 0:.4f}",
            f"test_accuracy: {metrics.get('accuracy'):.4f}",
            f"mlflow_run_id: {metrics.get('mlflow_run_id')}",
            f"data_fingerprint: {metrics.get('data_fingerprint')}",
        ])))

    summary = [f"Release v{args.version}"] + [f"- {name}" for name, _ in tags]
    tags.append((f"v{args.version}", "\n".join(summary)))

    for name, message in tags:
        print(f"== {name} -> {commit}\n{message}\n")
        if args.apply:
            sh("git", "tag", "-a", name, "-m", message)
    if args.apply:
        print("Tags created locally. Publish with: git push origin --tags && dvc push")
    else:
        print("Dry run. Re-run with --apply to create the tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
