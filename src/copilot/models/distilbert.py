"""DistilBERT ticket classifier with a small, picklable inference wrapper."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_MAX_LENGTH = 128
logger = logging.getLogger("support_copilot.models.distilbert")


def _device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TicketDataset:
    def __init__(self, encodings: dict[str, Any], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class DistilBertTicketModel:
    """Picklable adapter that loads the saved transformer lazily."""

    def __init__(self, model_dir: str | Path, labels: Iterable[str], max_length: int = DEFAULT_MAX_LENGTH):
        self.model_dir = str(model_dir)
        self.labels = [str(label) for label in labels]
        self.max_length = int(max_length)
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.labels)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = _device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        self._load()
        text_list = [str(text) for text in texts]
        probabilities: list[np.ndarray] = []
        batch_size = 32
        with self._torch.no_grad():
            for start in range(0, len(text_list), batch_size):
                batch = text_list[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits
                probabilities.extend(
                    self._torch.softmax(logits, dim=-1).cpu().numpy()
                )
        return np.asarray(probabilities)

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        indices = probabilities.argmax(axis=1)
        return np.asarray([self.labels[index] for index in indices])


def _prepare_text(frame: pd.DataFrame) -> list[str]:
    if "text" in frame.columns:
        return frame["text"].fillna("").astype(str).tolist()
    return (
        frame["subject"].fillna("").astype(str).str.strip()
        + " "
        + frame["body"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).tolist()


def _evaluate_model(model: Any, tokenizer: Any, texts: list[str], labels: list[str], max_length: int) -> dict[str, Any]:
    import torch

    device = _device()
    model.to(device)
    model.eval()
    predictions: list[int] = []
    label_to_id = {label: index for index, label in enumerate(sorted(set(labels)))}
    target_ids = [label_to_id[label] for label in labels]
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        "macro_f1": float(f1_score(target_ids, predictions, average="macro")),
        "accuracy": float(accuracy_score(target_ids, predictions)),
        "n_test": int(len(labels)),
    }


def train_distilbert(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    task: str,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
    class_weights: bool = False,
) -> dict[str, Any]:
    """Fine-tune DistilBERT on the real training split and evaluate on the test split."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    labels = sorted(train_frame[task].astype(str).unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_texts = _prepare_text(train_frame)
    train_labels = [label_to_id[str(label)] for label in train_frame[task].astype(str)]
    test_texts = _prepare_text(test_frame)
    test_labels = test_frame[task].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    train_dataset = _TicketDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    )
    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = None
    if class_weights:
        counts = Counter(train_labels)
        weights = [
            len(train_labels) / (len(labels) * max(counts.get(index, 1), 1))
            for index in range(len(labels))
        ]
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    model.train()
    for epoch in range(max(1, int(epochs))):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if loss_function is not None:
                loss = loss_function(output.logits, batch["labels"])
                loss.backward()
            else:
                loss = output.loss
                loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            if step % 100 == 0 or step == len(train_loader):
                logger.info(
                    "epoch %d/%d batch %d/%d loss %.4f",
                    epoch + 1,
                    max(1, int(epochs)),
                    step,
                    len(train_loader),
                    running_loss / step,
                )

    val_texts = _prepare_text(val_frame)
    val_labels = val_frame[task].astype(str).tolist()
    val_metrics = _evaluate_model(model, tokenizer, val_texts, val_labels, max_length)
    metrics = _evaluate_model(model, tokenizer, test_texts, test_labels, max_length)
    model_dir = Path(output_dir) / "distilbert"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    wrapper = DistilBertTicketModel(model_dir, labels, max_length=max_length)
    wrapper_path = Path(output_dir) / f"{task}_model.joblib"
    joblib.dump(wrapper, wrapper_path)
    (Path(output_dir) / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return {
        "task": task,
        "model_type": "distilbert",
        "model_path": str(wrapper_path),
        "model_dir": str(model_dir),
        "labels": labels,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "n_test": metrics["n_test"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_size": int(len(val_frame)),
    }
