"""Fine-tune a pretrained transformer for ticket classification, plus a portable inference wrapper.

Training recipe (what fixed the earlier DistilBERT run that scored ~0.26 macro F1):
- dynamic per-batch padding instead of padding the whole dataset once;
- AdamW with weight decay, linear warmup and decay, gradient clipping;
- label smoothing and optional class-balanced loss for the long-tail queues;
- evaluation on the validation split after every epoch, early stopping on macro F1,
  and restore of the best checkpoint (the test split is never used for selection).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger("support_copilot.models.transformer")
TRANSFORMER_SUBDIR = "transformer"


def select_device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pad_multiple(max_length: int) -> int | None:
    """Pad to a few fixed shapes (keeps the MPS allocator cache small); must divide max_length."""
    for multiple in (64, 8):
        if max_length % multiple == 0:
            return multiple
    return None


def _empty_cache(device: Any) -> None:
    import torch

    if getattr(device, "type", "") == "mps":
        torch.mps.empty_cache()
    elif getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()


def _batched_proba(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    """Softmax probabilities in input order, batching by length to minimise padding."""
    import torch

    if len(texts) == 0:
        return np.zeros((0, model.config.num_labels))
    order = np.argsort([len(text) for text in texts])
    output = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            encoded = tokenizer(
                [texts[i] for i in index],
                truncation=True,
                padding=True,
                pad_to_multiple_of=_pad_multiple(max_length),
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            output[index] = torch.softmax(logits, dim=-1).cpu().numpy()
    return output


class TransformerTicketModel:
    """Picklable sklearn-style adapter around a saved Hugging Face classifier.

    The weights live in a directory next to the pickle. Only the directory *name* is
    stored, and :meth:`bind` resolves it, so artifacts survive being moved between
    machines, DVC caches, and MLflow downloads.
    """

    def __init__(
        self,
        labels: Iterable[str],
        *,
        subdir: str = TRANSFORMER_SUBDIR,
        max_length: int = 256,
        batch_size: int = 32,
        base_dir: str | Path | None = None,
    ):
        self.labels = [str(label) for label in labels]
        self.subdir = subdir
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._base_dir = str(base_dir) if base_dir is not None else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update({"_model": None, "_tokenizer": None, "_device": None, "_base_dir": None})
        return state

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.labels)

    def bind(self, base_dir: str | Path) -> "TransformerTicketModel":
        self._base_dir = str(base_dir)
        return self

    @property
    def model_dir(self) -> Path:
        if self._base_dir is None:
            raise RuntimeError("Load this model with copilot.models.artifacts.load_model().")
        return Path(self._base_dir) / self.subdir

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = select_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        self._load()
        return _batched_proba(
            self._model,
            self._tokenizer,
            [str(text) for text in texts],
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=self._device,
        )

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def _class_weights(label_ids: np.ndarray, n_labels: int, power: float) -> np.ndarray:
    counts = Counter(label_ids.tolist())
    weights = np.array(
        [(len(label_ids) / (n_labels * max(counts.get(i, 1), 1))) ** power for i in range(n_labels)],
        dtype=np.float32,
    )
    return weights / weights.mean()


def fine_tune(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    *,
    output_dir: str | Path,
    model_name: str = "roberta-base",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 256,
    label_smoothing: float = 0.05,
    class_weight_power: float = 0.0,
    patience: int = 2,
    seed: int = 42,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune ``model_name`` and save the best-on-validation checkpoint under ``output_dir``."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = sorted({str(label) for label in train_labels})
    label_to_id = {label: index for index, label in enumerate(labels)}
    y_train = np.array([label_to_id[str(label)] for label in train_labels])
    y_val = np.array([label_to_id.get(str(label), -1) for label in val_labels])

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
    ).to(device)
    encoded = tokenizer(list(train_texts), truncation=True, max_length=max_length)

    steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
    total_steps = steps_per_epoch * max(1, epochs)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )
    weight = None
    if class_weight_power > 0:
        weight = torch.tensor(
            _class_weights(y_train, len(labels), class_weight_power), device=device
        )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    history: list[dict[str, float]] = []
    best_f1, best_state, best_epoch, stale = -1.0, None, -1, 0
    started = time.time()
    for epoch in range(max(1, epochs)):
        model.train()
        permutation = rng.permutation(len(y_train))
        running = 0.0
        for step, start in enumerate(range(0, len(permutation), batch_size), start=1):
            index = permutation[start : start + batch_size]
            batch = tokenizer.pad(
                [{key: encoded[key][i] for key in encoded.keys()} for i in index],
                return_tensors="pt",
                pad_to_multiple_of=_pad_multiple(max_length),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, torch.as_tensor(y_train[index], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            if step % 50 == 0:
                _empty_cache(device)
            if step % 200 == 0:
                logger.info("epoch %d step %d/%d loss %.4f", epoch + 1, step, steps_per_epoch, running / step)

        val_proba = _batched_proba(
            model, tokenizer, list(val_texts), max_length=max_length, batch_size=64, device=device
        )
        val_pred = val_proba.argmax(axis=1)
        record = {
            "epoch": float(epoch + 1),
            "train_loss": running / steps_per_epoch,
            "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
            "val_accuracy": float(accuracy_score(y_val, val_pred)),
            "elapsed_s": time.time() - started,
        }
        history.append(record)
        logger.info("epoch %d: %s", epoch + 1, json.dumps(record))
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, record)
        if record["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = record["val_macro_f1"], epoch + 1, 0
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping after epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = Path(output_dir) / TRANSFORMER_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    wrapper = TransformerTicketModel(labels, max_length=max_length, base_dir=output_dir)
    return {
        "model": wrapper,
        "labels": labels,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "model_dir": str(model_dir),
        "config": copy.deepcopy(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "max_length": max_length,
                "label_smoothing": label_smoothing,
                "class_weight_power": class_weight_power,
                "patience": patience,
            }
        ),
    }
