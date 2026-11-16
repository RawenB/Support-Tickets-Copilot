"""Unit tests for the model building blocks (text, metrics, blending, transformer, pyfunc)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from copilot.data.text import build_text, frame_texts
from copilot.models.artifacts import load_model, save_model
from copilot.models.blend import BlendedTicketModel, aligned_proba, tune_weight
from copilot.models.metrics import classification_metrics, novelty_scores

TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"
TEXTS = ["cannot login to vpn", "invoice charged twice", "vpn down again", "refund my invoice"] * 3
LABELS = ["it", "billing", "it", "billing"] * 3


def test_build_text_matches_serving_format():
    assert build_text("  VPN  Down ", "Please\n help") == "vpn down please help"
    assert build_text(None, "body only") == "body only"
    assert build_text(float("nan"), "Body") == "body"
    frame = pd.DataFrame({"subject": ["A", ""], "body": ["b", "C"]})
    assert frame_texts(frame) == ["a b", "c"]


def test_novelty_scores_flag_paraphrases_and_new_text():
    train = ["my vpn connection keeps dropping", "invoice amount is wrong", "printer is jammed"]
    scores = novelty_scores(train * 2, ["my vpn connection keeps dropping", "completely unrelated words"])
    assert scores[0] > 0.9 and scores[1] < 0.1


def test_classification_metrics_reports_novel_slice():
    mask = np.array([True, False, True, False])
    metrics = classification_metrics(["a", "b", "a", "b"], ["a", "b", "b", "b"], labels=["a", "b"], novel_mask=mask)
    assert metrics["accuracy"] == 0.75
    assert metrics["novel_share"] == 0.5
    assert 0 <= metrics["novel_macro_f1"] <= 1


def _sk_model():
    return make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(TEXTS, LABELS)


def test_blend_aligns_columns_and_tunes_weight(tmp_path):
    model = _sk_model()
    labels = ["it", "billing"]  # deliberately not sklearn's sorted order
    proba = aligned_proba(model, TEXTS, labels)
    assert proba.shape == (len(TEXTS), 2)
    assert np.allclose(proba[:, 0], model.predict_proba(TEXTS)[:, list(model.classes_).index("it")])
    weight, f1 = tune_weight([proba, proba[:, ::-1]], LABELS, labels)
    assert weight > 0.5 and f1 == 1.0

    blend = BlendedTicketModel({"a": model, "b": _sk_model()}, {"a": 0.5, "b": 0.5}, labels)
    path = save_model(blend, tmp_path / "blend.joblib")
    loaded = load_model(path)
    assert list(loaded.predict(["vpn down"])) == ["it"]
    assert np.allclose(loaded.predict_proba(["vpn down"]).sum(axis=1), 1.0)


def _tiny_model_available() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(TINY_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tiny_model_available(), reason=f"{TINY_MODEL} not in the local HF cache")
def test_fine_tune_saves_portable_model_and_pyfunc_loads_it(tmp_path):
    from copilot.models.pyfunc import TicketClassifierPyfunc
    from copilot.models.transformer import fine_tune

    result = fine_tune(TEXTS, LABELS, TEXTS[:4], LABELS[:4], output_dir=tmp_path / "out",
                       model_name=TINY_MODEL, epochs=1, batch_size=4, max_length=16)
    assert result["best_epoch"] == 1 and len(result["history"]) == 1
    save_model(result["model"], tmp_path / "out" / "m.joblib")

    moved = tmp_path / "moved"
    (tmp_path / "out").rename(moved)  # artifacts must survive being moved (DVC / MLflow)
    model = load_model(moved / "m.joblib")
    proba = model.predict_proba(["vpn down"])
    assert proba.shape == (1, 2) and np.isclose(proba.sum(), 1.0, atol=1e-5)

    class _Ctx:
        artifacts = {"model_file": str(moved / "m.joblib")}

    pyfunc = TicketClassifierPyfunc()
    pyfunc.load_context(_Ctx())
    out = pyfunc.predict(None, pd.DataFrame({"text": ["vpn down", "refund"]}))
    assert list(out.columns) == ["label", "confidence"] and len(out) == 2


def test_knn_memory_finds_paraphrases_and_survives_pickle_and_clone(tmp_path):
    from sklearn.base import clone

    from copilot.models.knn import KNNTicketModel

    for representation in ("char", "word"):
        model = KNNTicketModel(representation, k=3, power=4).fit(TEXTS, LABELS)
        assert list(model.predict(["vpn is down again", "please refund the invoice"])) == ["it", "billing"]
        proba = model.predict_proba(["zzzz qqqq"])  # nothing similar: uniform, not a crash
        assert np.allclose(proba.sum(axis=1), 1.0)
        loaded = load_model(save_model(model, tmp_path / f"{representation}.joblib"))
        assert np.allclose(loaded.predict_proba(TEXTS), model.predict_proba(TEXTS))
        assert clone(model).get_params()["k"] == 3


def test_tune_weights_searches_the_simplex():
    from copilot.models.blend import tune_weights

    good = np.array([[1.0, 0.0], [0.0, 1.0]])
    bad = good[:, ::-1]
    weights, f1 = tune_weights({"good": good, "bad": bad, "noise": np.full((2, 2), 0.5)}, ["a", "b"], ["a", "b"])
    assert f1 == 1.0 and weights["good"] > weights["bad"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
