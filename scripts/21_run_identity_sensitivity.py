from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DIR = PROJECT_ROOT / "outputs" / "05_tfidf_features"
IDENTITY_DIR = PROJECT_ROOT / "outputs" / "19_identity_tfidf"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "20_identity_sensitivity"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
MODEL_DIR = OUTPUT_DIR / "models"

SEEDS = [42, 123, 2024, 3407, 7777]
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3]
MAX_ITER = 30
TOLERANCE = 1e-3

TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def exact_macro_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)

    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    total_positive = int(sorted_labels.sum())
    total_negative = int(len(sorted_labels) - total_positive)

    boundaries = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )

    tp = cumulative_tp[boundaries].astype(np.float64)
    fp = cumulative_fp[boundaries].astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp

    positive_f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2 * tp + fp + fn) != 0,
    )
    negative_f1 = np.divide(
        2 * tn,
        2 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=(2 * tn + fp + fn) != 0,
    )
    macro_f1 = (positive_f1 + negative_f1) / 2

    all_negative_macro_f1 = (
        2 * total_negative
        / (2 * total_negative + total_positive)
    ) / 2

    best_index = int(np.argmax(macro_f1))
    if all_negative_macro_f1 > macro_f1[best_index]:
        return float(sorted_scores[0] + 1e-12), float(
            all_negative_macro_f1
        )

    boundary = int(boundaries[best_index])
    if boundary == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-12)
    else:
        threshold = float(
            (
                sorted_scores[boundary]
                + sorted_scores[boundary + 1]
            )
            / 2
        )

    return threshold, float(macro_f1[best_index])


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
        "precision_bot": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall_bot": recall_score(
            labels, predictions, zero_division=0
        ),
        "specificity_human": tn / (tn + fp),
        "f1_bot": f1_score(
            labels, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, predictions


def build_classifier(
    alpha: float,
    seed: int,
) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        shuffle=True,
        random_state=seed,
        n_jobs=-1,
        average=True,
    )


def tune_alpha(
    model_name: str,
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for alpha in ALPHAS:
        started = time.time()
        classifier = build_classifier(alpha, SEEDS[0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(
                matrix[train_mask], labels[train_mask]
            )
            converged = not any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )

        val_scores = classifier.decision_function(
            matrix[val_mask]
        )
        threshold, val_macro_f1 = (
            exact_macro_f1_threshold(
                labels[val_mask], val_scores
            )
        )
        val_metrics, _ = calculate_metrics(
            labels[val_mask],
            val_scores,
            threshold,
        )

        rows.append(
            {
                "model": model_name,
                "alpha": alpha,
                "seed": SEEDS[0],
                "threshold": threshold,
                "converged": converged,
                "epochs": int(classifier.n_iter_),
                "validation_macro_f1": val_macro_f1,
                "validation_auroc": val_metrics["auroc"],
                "validation_auprc": val_metrics["auprc"],
                "runtime_seconds": time.time() - started,
            }
        )

        print(
            f"TUNE {model_name}: alpha={alpha:g}, "
            f"macro-F1={val_macro_f1:.6f}, "
            f"AUROC={val_metrics['auroc']:.6f}"
        )

    tuning = pd.DataFrame(rows).sort_values(
        [
            "validation_macro_f1",
            "validation_auprc",
            "validation_auroc",
        ],
        ascending=False,
    )
    return float(tuning.iloc[0]["alpha"]), tuning


def run_model(
    model_name: str,
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    node_ids: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    best_alpha: float,
) -> list[dict]:
    rows = []

    for seed in SEEDS:
        started = time.time()
        classifier = build_classifier(best_alpha, seed)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(
                matrix[train_mask], labels[train_mask]
            )
            converged = not any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )

        val_scores = classifier.decision_function(
            matrix[val_mask]
        )
        threshold, val_macro_f1 = (
            exact_macro_f1_threshold(
                labels[val_mask], val_scores
            )
        )

        # Test is accessed after alpha and threshold selection.
        test_scores = classifier.decision_function(
            matrix[test_mask]
        )
        test_metrics, predictions = calculate_metrics(
            labels[test_mask],
            test_scores,
            threshold,
        )

        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "alpha": best_alpha,
                "threshold": threshold,
                "converged": converged,
                "epochs": int(classifier.n_iter_),
                "validation_macro_f1": val_macro_f1,
                "runtime_seconds": time.time() - started,
                **test_metrics,
            }
        )

        joblib.dump(
            {
                "classifier": classifier,
                "threshold": threshold,
                "alpha": best_alpha,
                "seed": seed,
                "model": model_name,
            },
            MODEL_DIR / f"{model_name}_seed_{seed}.joblib",
            compress=3,
        )

        pd.DataFrame(
            {
                "node_id": node_ids[test_mask],
                "label": labels[test_mask],
                "score": test_scores.astype(np.float32),
                "prediction": predictions,
            }
        ).to_parquet(
            PREDICTION_DIR
            / f"{model_name}_seed_{seed}.parquet",
            index=False,
        )

        print(
            f"TEST {model_name}, seed={seed}: "
            f"macro-F1={test_metrics['macro_f1']:.6f}, "
            f"AUROC={test_metrics['auroc']:.6f}, "
            f"AUPRC={test_metrics['auprc']:.6f}"
        )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    semantic = sparse.load_npz(
        SEMANTIC_DIR / "semantic_tfidf_features.npz"
    ).tocsr()
    identity = sparse.load_npz(
        IDENTITY_DIR / "identity_tfidf_features.npz"
    ).tocsr()
    metadata = np.load(
        METADATA_DIR / "metadata_features.npy",
        mmap_mode="r",
    )

    node_ids = np.load(
        IDENTITY_DIR / "node_ids.npy",
        mmap_mode="r",
    )
    labels = np.load(
        IDENTITY_DIR / "labels.npy",
        mmap_mode="r",
    ).astype(np.int8)
    split_ids = np.load(
        IDENTITY_DIR / "split_ids.npy",
        mmap_mode="r",
    )

    if not (
        semantic.shape[0]
        == identity.shape[0]
        == metadata.shape[0]
        == len(labels)
    ):
        raise RuntimeError("Feature row counts differ.")

    metadata_sparse = sparse.csr_matrix(
        np.asarray(metadata, dtype=np.float32)
    )

    semantic_identity = sparse.hstack(
        [semantic, identity],
        format="csr",
        dtype=np.float32,
    )
    semantic_identity_metadata = sparse.hstack(
        [semantic, identity, metadata_sparse],
        format="csr",
        dtype=np.float32,
    )

    matrices = {
        "identity_only": identity,
        "semantic_identity": semantic_identity,
        "semantic_identity_metadata": (
            semantic_identity_metadata
        ),
    }

    train_mask = split_ids == TRAIN_ID
    val_mask = split_ids == VAL_ID
    test_mask = split_ids == TEST_ID

    tuning_frames = []
    selected_alphas = {}
    result_rows = []

    for model_name, matrix in matrices.items():
        print(f"\nSelecting alpha for {model_name}...")
        best_alpha, tuning = tune_alpha(
            model_name,
            matrix,
            labels,
            train_mask,
            val_mask,
        )
        tuning_frames.append(tuning)
        selected_alphas[model_name] = best_alpha

        print(
            f"Selected alpha for {model_name}: "
            f"{best_alpha:g}"
        )
        result_rows.extend(
            run_model(
                model_name,
                matrix,
                labels,
                node_ids,
                train_mask,
                val_mask,
                test_mask,
                best_alpha,
            )
        )

    pd.concat(
        tuning_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "validation_hyperparameter_search.csv",
        index=False,
    )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / "identity_seed_results.csv",
        index=False,
    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_bot",
        "recall_bot",
        "specificity_human",
        "f1_bot",
        "macro_f1",
        "mcc",
        "auroc",
        "auprc",
    ]
    summary_rows = []

    for model_name, group in results.groupby("model"):
        row = {
            "model": model_name,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=1)
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    summary.to_csv(
        OUTPUT_DIR / "identity_sensitivity_summary.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "purpose": (
            "Quantify the incremental performance and shortcut risk "
            "of display-name and username lexical identity cues"
        ),
        "seeds": SEEDS,
        "training_split": "official train only",
        "hyperparameter_and_threshold_selection": (
            "official validation only"
        ),
        "test_policy": (
            "test accessed after alpha and threshold selection"
        ),
        "models": list(matrices),
        "selected_alphas": selected_alphas,
        "identity_fields": ["name_text", "username_text"],
        "semantic_fields": [
            "description_text",
            "location_text",
        ],
        "classifier": {
            "type": "averaged SGD logistic regression",
            "class_weight": "balanced",
            "penalty": "l2",
        },
        "input_hashes": {
            "semantic_tfidf": sha256_file(
                SEMANTIC_DIR / "semantic_tfidf_features.npz"
            ),
            "identity_tfidf": sha256_file(
                IDENTITY_DIR / "identity_tfidf_features.npz"
            ),
            "metadata": sha256_file(
                METADATA_DIR / "metadata_features.npy"
            ),
        },
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nIDENTITY SENSITIVITY: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()