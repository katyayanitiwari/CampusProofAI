"""Train an experimental general scam-message model on Mendeley Data."""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.config import ROOT

DATA_PATH = ROOT / "data" / "raw" / "mendeley-financial-scams" / "financial-scams-dataset.csv"
DATASET_URL = "https://data.mendeley.com/datasets/znsk27yk3h/2"
REVISION = 2
MODEL_PATH = ROOT / "models" / "financial_message_scam.joblib"


def clean(text: str) -> str:
    return " ".join(text.strip().split())


FEATURE_PROFILES = {
    "standard": {"word_features": 20_000, "char_features": 30_000},
    "compact": {"word_features": 10_000, "char_features": 15_000},
    "wordonly": {"word_features": 12_000, "char_features": None},
}


def make_pipeline(classifier: object, profile: str = "standard") -> Pipeline:
    limits = FEATURE_PROFILES[profile]
    components = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=limits["word_features"], sublinear_tf=True, dtype=np.float32)),
    ]
    if limits["char_features"] is not None:
        components.append(
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=limits["char_features"], sublinear_tf=True, dtype=np.float32))
        )
    features = FeatureUnion(components)
    return Pipeline([("features", features), ("model", classifier)])


def threshold_for_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, np.max(f1)))
    idx = int(max(best, key=lambda i: (precision[i], thresholds[i])))
    return float(thresholds[idx]), {
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int8)
    report = classification_report(labels, predicted, labels=[0, 1], target_names=["ham", "scam"], output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "scam_precision": float(report["scam"]["precision"]),
        "scam_recall": float(report["scam"]["recall"]),
        "scam_f1": float(report["scam"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix_labels_ham_then_scam": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit("Dataset missing; run `python scripts/download_mendeley_financial_scams.py` first.")
    frame = pd.read_csv(DATA_PATH)
    if not {"message", "label"}.issubset(frame.columns):
        raise SystemExit("Expected CSV columns `message` and `label`.")
    frame = frame[["message", "label"]].dropna().copy()
    frame["message"] = frame["message"].astype(str).map(clean)
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame[frame.message.str.len() > 0]
    invalid = sorted(set(frame.label) - {"ham", "scam"})
    if invalid:
        raise SystemExit(f"Unexpected labels: {invalid}")

    # Conflicting labels for identical text are annotation ambiguities; exclude them.
    frame["text_hash"] = frame.message.map(lambda text: hashlib.sha256(text.casefold().encode("utf-8")).hexdigest())
    conflict_hashes = frame.groupby("text_hash").label.nunique()
    ambiguous = set(conflict_hashes[conflict_hashes > 1].index)
    ambiguous_count = int(frame.text_hash.isin(ambiguous).sum())
    frame = frame[~frame.text_hash.isin(ambiguous)].drop_duplicates("text_hash").reset_index(drop=True)
    frame["target"] = (frame.label == "scam").astype(np.int8)
    x, y = frame.message.to_list(), frame.target.to_numpy()
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.1875, random_state=43, stratify=y_trainval
    )

    classifier_factories = {
        "LogisticRegression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "LinearSVC": lambda: LinearSVC(class_weight="balanced", random_state=42),
    }
    candidates = {
        f"{classifier_name}_{profile}": make_pipeline(factory(), profile)
        for profile in FEATURE_PROFILES
        for classifier_name, factory in classifier_factories.items()
    }
    selection = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        cutoff, stats = threshold_for_f1(y_val, model.decision_function(x_val))
        selection[name] = {"threshold": cutoff, **stats}
    best_f1 = max(result["f1"] for result in selection.values())
    feature_sizes = {
        name: int(candidate.named_steps["features"].transform(x_train[:1]).shape[1])
        for name, candidate in candidates.items()
    }
    best_standard_size = min(feature_sizes[name] for name in candidates if name.endswith("_standard"))
    smaller_candidates = [
        name for name in candidates
        if feature_sizes[name] < best_standard_size and selection[name]["f1"] >= best_f1 - 0.01
    ]
    if smaller_candidates:
        smallest_size = min(feature_sizes[name] for name in smaller_candidates)
        eligible = [name for name in smaller_candidates if feature_sizes[name] == smallest_size]
    else:
        eligible = [
            name for name in candidates
            if name.endswith("_standard") and selection[name]["f1"] == best_f1
        ]
    selected = max(eligible, key=lambda name: (selection[name]["f1"], selection[name]["precision"]))
    model, threshold = candidates[selected], selection[selected]["threshold"]
    standard_names = [name for name in candidates if name.endswith("_standard")]
    standard_selected = max(standard_names, key=lambda name: (selection[name]["f1"], selection[name]["precision"]))
    standard_model = candidates[standard_selected]
    test_scores = model.decision_function(x_test)
    test_metrics = metrics(y_test, test_scores, threshold)
    keyword_count = sum(
        bool(pd.Series([text]).str.contains(r"scholarship|fellowship|grant|tuition|education award", case=False, regex=True).iloc[0])
        for text in x
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    feature_count = int(model.named_steps["features"].transform(x_train[:1]).shape[1])
    def benchmark_single_message(candidate: Pipeline) -> dict[str, float | int]:
        for _ in range(5):
            candidate.decision_function([x_test[0]])
        samples = []
        for _ in range(50):
            started = time.perf_counter()
            candidate.decision_function([x_test[0]])
            samples.append((time.perf_counter() - started) * 1000)
        samples.sort()
        return {
            "median_ms_single_message": float(np.median(samples)),
            "p95_ms_single_message": float(samples[min(len(samples) - 1, int(len(samples) * 0.95))]),
            "runs": len(samples),
        }

    inference_benchmark = {
        "selected": benchmark_single_message(model),
        "standard_reference": benchmark_single_message(standard_model),
        "standard_reference_model": standard_selected,
    }
    standard_feature_count = best_standard_size
    feature_count = int(model.named_steps["features"].transform(x_train[:1]).shape[1])
    standard_buffer = io.BytesIO()
    joblib.dump(standard_model, standard_buffer, compress=3)
    standard_artifact_bytes = standard_buffer.tell()
    metadata = {
        "model": selected,
        "feature_profile": selected.rsplit("_", 1)[-1],
        "feature_count": feature_count,
        "standard_reference_feature_count": standard_feature_count,
        "standard_reference_artifact_bytes": standard_artifact_bytes,
        "inference_benchmark": inference_benchmark,
        "threshold": threshold,
        "is_probability": False,
        "domain": "English and Bangla general financial scam messages; not validated for scholarship offer fraud.",
        "dataset": "Mendeley Data Financial scams detection dataset v2",
        "dataset_url": DATASET_URL,
        "license": "CC BY 4.0",
        "rows_after_ambiguity_and_duplicate_removal": len(frame),
        "scholarship_related_keyword_rows": keyword_count,
        "test_scam_precision": test_metrics["scam_precision"],
        "test_scam_recall": test_metrics["scam_recall"],
    }
    vectorizer_description = "word-only TF-IDF" if metadata["feature_profile"] == "wordonly" else "word and character TF-IDF"
    (ROOT / "models" / "financial_message_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output = {
        "dataset": {"name": metadata["dataset"], "url": DATASET_URL, "version": REVISION, "license": "CC BY 4.0"},
        "rows": len(frame),
        "conflicting_duplicate_rows_excluded": ambiguous_count,
        "duplicate_text_rows_removed": int(len(pd.read_csv(DATA_PATH)) - ambiguous_count - len(frame)),
        "scholarship_related_keyword_rows": keyword_count,
        "counts": {"ham": int((y == 0).sum()), "scam": int((y == 1).sum())},
        "split_rows": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "validation_model_selection": selection,
        "selected_model": selected,
        "selected_feature_profile": selected.rsplit("_", 1)[-1],
        "feature_count": feature_count,
        "standard_reference_feature_count": standard_feature_count,
        "model_artifact_bytes": MODEL_PATH.stat().st_size,
        "standard_reference_artifact_bytes": standard_artifact_bytes,
        "inference_benchmark": inference_benchmark,
        "test_metrics": test_metrics,
        "limitations": [
            "The dataset contains broad English/Bangla financial scam messages, not scholarship-specific offer labels.",
            "Only a few rows match scholarship/grant/tuition keywords; these are not enough for scholarship evaluation.",
            "The small test set makes metrics uncertain; this is research triage, not a production decision model.",
            "Scores are linear decision scores, not probabilities.",
        ],
    }
    metric_dir = ROOT / "artifacts" / "metrics"
    report_dir = ROOT / "artifacts" / "reports"
    metric_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "financial_message_scam_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (report_dir / "financial-message-model-card.md").write_text(
        "# Student-facing financial scam message model\n\n"
        f"- Dataset: Mendeley Data v2, {len(frame)} rows after duplicate/conflict cleanup; CC BY 4.0.\n"
        f"- Scholarship-related keyword rows: {keyword_count}; insufficient for scholarship-specific validation.\n"
        f"- Model: `{selected}` with {vectorizer_description}; `{metadata['feature_profile']}` vocabulary profile ({feature_count:,} fitted features), threshold selected on validation data.\n"
        f"- Model footprint: {feature_count:,} fitted features and {MODEL_PATH.stat().st_size:,} bytes; standard reference has {standard_feature_count:,} features and {standard_artifact_bytes:,} bytes.\n"
        f"- One local single-message benchmark (50 runs): selected median {inference_benchmark['selected']['median_ms_single_message']:.3f} ms vs standard median {inference_benchmark['standard_reference']['median_ms_single_message']:.3f} ms; timing depends on hardware and system load.\n"
        f"- Held-out scam precision: {test_metrics['scam_precision']:.3f}; recall: {test_metrics['scam_recall']:.3f}.\n"
        "- Intended use: experimental general financial scam-message triage in English/Bangla.\n"
        "- Not intended for scholarship/fellowship authenticity decisions; score is not a probability or verdict.\n"
        "- Attribution: Ahmed, A. R. & Islam, G. F. (2025), Financial scams detection dataset, v2, doi:10.17632/znsk27yk3h.2.\n"
        "- Full metrics and limitations: `artifacts/metrics/financial_message_scam_metrics.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

