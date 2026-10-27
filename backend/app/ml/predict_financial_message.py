"""Load the experimental English/Bangla general scam-message model."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib

from app.config import ROOT

MODEL_PATH = ROOT / "models" / "financial_message_scam.joblib"
METADATA_PATH = ROOT / "models" / "financial_message_metadata.json"


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_financial_message(text: str) -> dict:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return {"available": False, "message": "General financial-message model is not trained."}
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    score = float(load_model().decision_function([text])[0])
    threshold = float(metadata["threshold"])
    return {
        "available": True,
        "signal": "higher_scam_text_signal" if score >= threshold else "lower_scam_text_signal",
        "decision_score": score,
        "threshold": threshold,
        "is_probability": False,
        "model": metadata["model"],
        "domain": metadata["domain"],
        "test_scam_precision": metadata["test_scam_precision"],
        "test_scam_recall": metadata["test_scam_recall"],
        "caution": "General scam-message triage only; not validated for scholarship authenticity. Always verify the publisher and terms independently.",
    }

