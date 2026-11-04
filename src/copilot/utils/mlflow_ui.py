"""Start the MLflow UI on the project store (safe for project paths that contain spaces)."""

from __future__ import annotations

import subprocess
import sys

from copilot.utils.mlflow_utils import resolve_artifact_root, resolve_tracking_uri

HOST = "127.0.0.1"
PORT = "5001"  # macOS uses port 5000 for AirPlay Receiver


def main() -> None:
    command = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", resolve_tracking_uri(),
        "--default-artifact-root", str(resolve_artifact_root()),
        "--host", HOST, "--port", PORT,
    ]
    print(f"MLflow UI: http://{HOST}:{PORT}  (Ctrl+C to stop)")
    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
