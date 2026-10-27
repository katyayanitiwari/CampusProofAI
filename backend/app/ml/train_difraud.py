"""Train and evaluate a conservative job-posting warning model on DIFrauD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from app.config import MODEL_PATH, ROOT

DATA_DIR = ROOT / "data" / "raw" / "difraud-job-scams"
REVISION = "26e18fecb5396f2eb9adb63d42cf916923043478"
DATASET_URL = "https://huggingface.co/datasets/difraud/difraud"


def read_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}, line {line_number}") from exc
        text = row.get("text") if isinstance(row, dict) else None
        label = row.get("label") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip() or label not in (0, 1):
            raise ValueError(f"Invalid text or label in {path.name}, line {line_number}")
        texts.append(" ".join(text.split()))
        labels.append(int(label))
    if not texts or len(set(labels)) != 2:
        raise ValueError(f"{path.name} must contain both classes and non-empty text")
    return texts, np.asarray(labels, dtype=np.int8)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=150_000,
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            ("model", classifier),
        ]
    )


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    if len(thresholds) == 0:
        return 0.0, {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_f1 = float(np.max(f1))
    best = np.flatnonzero(np.isclose(f1, best_f1))
    # If validation F1 ties, prefer the threshold with higher precision.
    index = int(max(best, key=lambda item: (precision[item], thresholds[item])))
    return float(thresholds[index]), {
        "f1": best_f1,
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def score_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "precision_recall_f1": {
            key: float(value)
            for key, value in zip(
                ("precision", "recall", "f1"),
                (
                    classification_report(labels, predicted, output_dict=True, zero_division=0)["1"]["precision"],
                    classification_report(labels, predicted, output_dict=True, zero_division=0)["1"]["recall"],
                    f1_score(labels, predicted, pos_label=1, zero_division=0),
                ),
            )
        },
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], target_names=["genuine", "fraudulent"],
            output_dict=True, zero_division=0,
        ),
        "positive_predictions": int(predicted.sum()),
        "actual_positive": int(labels.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    split_paths = {name: args.data_dir / f"{name}.jsonl" for name in ("train", "validation", "test")}
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "Dataset splits are missing. Run `python scripts/download_difraud_job_scams.py` first.\n"
            + "\n".join(missing)
        )

    splits = {name: read_split(path) for name, path in split_paths.items()}
    fingerprints = {name: {fingerprint(text) for text in pair[0]} for name, pair in splits.items()}
    overlaps = {
        f"{left}_{right}": len(fingerprints[left] & fingerprints[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise SystemExit(f"Exact text overlap between official splits; refusing possible leakage: {overlaps}")

    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    x_test, y_test = splits["test"]

    candidates = {
        "LogisticRegression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        ),
        "LinearSVC": make_pipeline(LinearSVC(class_weight="balanced", random_state=42)),
    }
    validation_results = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        validation_scores = model.decision_function(x_validation)
        threshold, metrics = choose_threshold(y_validation, validation_scores)
        validation_results[name] = {"threshold": threshold, **metrics}

    selected_name = max(
        candidates,
        key=lambda name: (
            validation_results[name]["f1"],
            validation_results[name]["precision"],
        ),
    )
    selected_model = candidates[selected_name]
    selected_threshold = validation_results[selected_name]["threshold"]
    test_scores = selected_model.decision_function(x_test)
    test_at_selected_threshold = score_metrics(y_test, test_scores, selected_threshold)
    test_at_zero = score_metrics(y_test, test_scores, 0.0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH, compress=3)
    positive_counts = {
        name: {"rows": len(labels), "genuine": int((labels == 0).sum()), "fraudulent": int((labels == 1).sum())}
        for name, (_, labels) in splits.items()
    }
    metrics = {
        "dataset": {
            "name": "DIFrauD Job Scams",
            "repository": DATASET_URL,
            "revision": REVISION,
            "license": "MIT (as declared by the dataset publisher)",
            "source_domain": "English employment/job postings; not scholarships",
        },
        "split_counts": positive_counts,
        "exact_text_overlap_across_splits": overlaps,
        "selection_method": "Best validation F1 after threshold selection; validation precision breaks ties.",
        "validation_candidates": validation_results,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "test_at_selected_threshold": test_at_selected_threshold,
        "test_at_zero_threshold": test_at_zero,
        "limitations": [
            "The dataset is about employment/job-posting scams, not scholarships or fellowships.",
            "The score is a linear-model decision score, not a calibrated probability.",
            "Metrics are an offline benchmark and do not establish live performance.",
        ],
    }

    metrics_dir = ROOT / "artifacts" / "metrics"
    reports_dir = ROOT / "artifacts" / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "job_scam_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "threshold": selected_threshold,
        "model": selected_name,
        "domain": "English job postings only; not validated for scholarships.",
        "is_probability": False,
        "training_dataset": metrics["dataset"],
        "validation_f1": validation_results[selected_name]["f1"],
        "test_f1": test_at_selected_threshold["precision_recall_f1"]["f1"],
    }
    (ROOT / "models" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports_dir / "job-scam-model-card.md").write_text(
        "# CampusProof job-posting warning model\n\n"
        f"- Model: `{selected_name}` with TF-IDF features and a validation-selected threshold.\n"
        f"- Training data: DIFrauD Job Scams at revision `{REVISION}`; MIT as declared by the publisher.\n"
        "- Intended use: a warning signal for English employment/job postings only.\n"
        "- Not intended for scholarship, fellowship, or award verification.\n"
        "- The model score is not a probability or a verdict.\n"
        "- See `artifacts/metrics/job_scam_model_metrics.json` for split sizes and held-out metrics.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

