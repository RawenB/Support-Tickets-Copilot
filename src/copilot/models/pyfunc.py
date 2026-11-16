"""MLflow pyfunc wrapper so registered models load (or serve) without the local models/ folder.

    model = mlflow.pyfunc.load_model("models:/support-copilot-queue@champion")
    model.predict(pd.DataFrame({"text": ["my invoice is wrong"]}))

or ``mlflow models serve -m models:/support-copilot-queue@champion``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


class TicketClassifierPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from copilot.models.artifacts import load_model

        model_file = Path(context.artifacts["model_file"])
        # MLflow copies every artifact into the same folder, so the transformer weights
        # end up next to the pickle, which is exactly what ``bind`` expects.
        self.model = load_model(model_file)

    def predict(self, context: Any, model_input: Any, params: dict | None = None) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            column = "text" if "text" in model_input.columns else model_input.columns[0]
            texts = model_input[column].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        proba = self.model.predict_proba(texts)
        classes = np.asarray(self.model.classes_)
        return pd.DataFrame({"label": classes[proba.argmax(axis=1)], "confidence": proba.max(axis=1)})
