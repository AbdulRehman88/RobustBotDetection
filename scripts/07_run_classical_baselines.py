from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

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
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
TFIDF_DIR = PROJECT_ROOT / "outputs" / "05_tfidf_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "06_classical_baselines"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

SEEDS = [42, 123, 2024, 3407, 7777]
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3]
MAX_ITER = 30
TOLERANCE = 1e-3

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)

    if unique_scores.size <= 1000:
        if unique_scores.size == 1:
            return unique_scores
        midpoints = (
            unique_scores[:-1] + unique_scores[1:]
        ) / 2.0
        return np.unique(
            np.concatenate(
                [
                    [unique_scores[0] - 1e-12],
                    midpoints,
                    [unique_scores[-1] + 1e-12],
                ]
            )
        )

    quantiles = np.linspace(0.001, 0.999, 999)
    thresholds = np.quantile(scores, quantiles)
    thresholds = np.concatenate(
        [
            [scores.min() - 1e-12],
            thresholds,
            [scores.max() + 1e-12],
        ]
    )
    return np.unique(thresholds)


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.0
    best_macro_f1 = -np.inf

    for threshold in candidate_thresholds(scores):
        predictions = (scores >= threshold).astype(np.int8)
        score = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        if score > best_macro_f1:
            best_macro_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_macro_f1


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
        "precision_bot": precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_bot": recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "specificity_human": specificity,
        "f1_bot": f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
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
    }
    return metrics, predictions


def build_classifier(alpha: float, seed: int) -> SGDClassifier:
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
    matrix,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for alpha in ALPHAS:
        started = time.time()
        classifier = build_classifier(alpha=alpha, seed=SEEDS[0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(matrix[train_mask], labels[train_mask])
            converged = not any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )

        validation_scores = classifier.decision_function(
            matrix[val_mask]
        )
        threshold, validation_macro_f1 = select_threshold(
            labels[val_mask],
            validation_scores,
        )

        validation_metrics, _ = calculate_metrics(
            labels[val_mask],
            validation_scores,
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
                "validation_macro_f1": validation_macro_f1,
                "validation_auroc": validation_metrics["auroc"],
                "validation_auprc": validation_metrics["auprc"],
                "runtime_seconds": time.time() - started,
            }
        )

        print(
            f"TUNE {model_name}: alpha={alpha:g}, "
            f"macro-F1={validation_macro_f1:.6f}, "
            f"AUROC={validation_metrics['auroc']:.6f}"
        )

    tuning = pd.DataFrame(rows).sort_values(
        [
            "validation_macro_f1",
            "validation_auprc",
            "validation_auroc",
            "alpha",
        ],
        ascending=[False, False, False, False],
    )
    best_alpha = float(tuning.iloc[0]["alpha"])
    return best_alpha, tuning


def run_seeded_model(
    model_name: str,
    matrix,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    test_node_ids: np.ndarray,
    best_alpha: float,
    text_available_test: np.ndarray,
    graph_available_test: np.ndarray,
) -> list[dict]:
    rows = []

    for seed in SEEDS:
        started = time.time()
        classifier = build_classifier(best_alpha, seed)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(matrix[train_mask], labels[train_mask])
            converged = not any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )

        validation_scores = classifier.decision_function(
            matrix[val_mask]
        )
        threshold, validation_macro_f1 = select_threshold(
            labels[val_mask],
            validation_scores,
        )

        # Test data are accessed only after model, alpha, and threshold
        # have been fixed using training and validation data.
        test_scores = classifier.decision_function(matrix[test_mask])
        test_metrics, test_predictions = calculate_metrics(
            labels[test_mask],
            test_scores,
            threshold,
        )

        row = {
            "model": model_name,
            "seed": seed,
            "alpha": best_alpha,
            "threshold": threshold,
            "validation_macro_f1": validation_macro_f1,
            "converged": converged,
            "epochs": int(classifier.n_iter_),
            "runtime_seconds": time.time() - started,
            **test_metrics,
        }
        rows.append(row)

        prediction_frame = pd.DataFrame(
            {
                "node_id": test_node_ids,
                "label": labels[test_mask],
                "score": test_scores.astype(np.float32),
                "prediction": test_predictions,
                "text_available": text_available_test,
                "graph_available": graph_available_test,
            }
        )
        prediction_frame.to_parquet(
            PREDICTION_DIR
            / f"{model_name}_seed_{seed}_test_predictions.parquet",
            index=False,
        )

        joblib_path = (
            OUTPUT_DIR / f"{model_name}_seed_{seed}.joblib"
        )
        import joblib

        joblib.dump(
            {
                "classifier": classifier,
                "threshold": threshold,
                "alpha": best_alpha,
                "seed": seed,
                "model_name": model_name,
            },
            joblib_path,
            compress=3,
        )

        print(
            f"TEST {model_name}, seed={seed}: "
            f"macro-F1={test_metrics['macro_f1']:.6f}, "
            f"AUROC={test_metrics['auroc']:.6f}, "
            f"AUPRC={test_metrics['auprc']:.6f}"
        )

    return rows


def run_majority_baseline(
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    test_node_ids: np.ndarray,
    text_available_test: np.ndarray,
    graph_available_test: np.ndarray,
) -> list[dict]:
    train_prevalence = float(labels[train_mask].mean())
    # The official training distribution has a human majority.
    constant_prediction = int(train_prevalence >= 0.5)
    test_labels = labels[test_mask]
    predictions = np.full(
        test_labels.shape,
        constant_prediction,
        dtype=np.int8,
    )
    scores = np.full(
        test_labels.shape,
        train_prevalence,
        dtype=np.float64,
    )

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    base = {
        "model": "majority",
        "alpha": np.nan,
        "threshold": 0.5,
        "validation_macro_f1": np.nan,
        "converged": True,
        "epochs": 0,
        "runtime_seconds": 0.0,
        "accuracy": accuracy_score(test_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            test_labels, predictions
        ),
        "precision_bot": precision_score(
            test_labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_bot": recall_score(
            test_labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "specificity_human": (
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "f1_bot": f1_score(
            test_labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            test_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(test_labels, predictions),
        "auroc": 0.5,
        "auprc": float(test_labels.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    prediction_frame = pd.DataFrame(
        {
            "node_id": test_node_ids,
            "label": test_labels,
            "score": scores.astype(np.float32),
            "prediction": predictions,
            "text_available": text_available_test,
            "graph_available": graph_available_test,
        }
    )

    rows = []
    for seed in SEEDS:
        row = dict(base)
        row["seed"] = seed
        rows.append(row)
        prediction_frame.to_parquet(
            PREDICTION_DIR
            / f"majority_seed_{seed}_test_predictions.parquet",
            index=False,
        )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(
        CANONICAL_DIR / "users.parquet",
        columns=[
            "node_id",
            "label",
            "split",
            "graph_available",
        ],
    ).sort_values("node_id").reset_index(drop=True)

    node_ids = users["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        node_ids, np.arange(len(users), dtype=np.int64)
    ):
        raise RuntimeError("Canonical node order is invalid.")

    # Explicitly define bot as the positive class, independent of any
    # historical numeric label encoding.
    normalized_labels = users["label"].astype(str).str.lower()
    if set(normalized_labels.unique()) != {"bot", "human"}:
        raise RuntimeError(
            f"Unexpected labels: {sorted(normalized_labels.unique())}"
        )
    labels = normalized_labels.eq("bot").to_numpy(dtype=np.int8)

    split_names = users["split"].astype(str).to_numpy()
    train_mask = split_names == "train"
    val_mask = split_names == "val"
    test_mask = split_names == "test"

    metadata = np.load(
        METADATA_DIR / "metadata_features.npy",
        mmap_mode="r",
    )
    tfidf = sparse.load_npz(
        TFIDF_DIR / "semantic_tfidf_features.npz"
    ).tocsr()
    text_available = np.load(
        TFIDF_DIR / "text_available.npy"
    ).astype(bool)
    graph_available = users["graph_available"].to_numpy(dtype=bool)

    if metadata.shape != (len(users), 10):
        raise RuntimeError(
            f"Unexpected metadata shape: {metadata.shape}"
        )
    if tfidf.shape[0] != len(users):
        raise RuntimeError("TF-IDF row count does not match users.")
    if len(text_available) != len(users):
        raise RuntimeError("Text availability length mismatch.")

    metadata_sparse = sparse.csr_matrix(
        np.asarray(metadata, dtype=np.float32)
    )
    availability_matrix = sparse.csr_matrix(
        np.column_stack(
            [
                text_available.astype(np.float32),
                graph_available.astype(np.float32),
            ]
        )
    )
    fused_matrix = sparse.hstack(
        [tfidf, metadata_sparse],
        format="csr",
        dtype=np.float32,
    )

    matrices = {
        "mask_only": availability_matrix,
        "metadata_only": metadata_sparse,
        "tfidf_only": tfidf,
        "tfidf_metadata_concat": fused_matrix,
    }

    test_node_ids = node_ids[test_mask]
    text_available_test = text_available[test_mask]
    graph_available_test = graph_available[test_mask]

    result_rows = run_majority_baseline(
        labels,
        train_mask,
        test_mask,
        test_node_ids,
        text_available_test,
        graph_available_test,
    )
    tuning_frames = []
    selected_alphas = {}

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
            f"Selected alpha for {model_name}: {best_alpha:g}"
        )
        result_rows.extend(
            run_seeded_model(
                model_name,
                matrix,
                labels,
                train_mask,
                val_mask,
                test_mask,
                test_node_ids,
                best_alpha,
                text_available_test,
                graph_available_test,
            )
        )

    tuning_results = pd.concat(
        tuning_frames, ignore_index=True
    )
    tuning_results.to_csv(
        OUTPUT_DIR / "validation_hyperparameter_search.csv",
        index=False,
    )

    seed_results = pd.DataFrame(result_rows)
    seed_results.to_csv(
        OUTPUT_DIR / "classical_baseline_seed_results.csv",
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
    for model_name, group in seed_results.groupby("model"):
        row = {
            "model": model_name,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=1)
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1_mean",
        ascending=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "classical_baseline_summary.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "protocol": {
            "training_data": "official train split only",
            "hyperparameter_selection": (
                "alpha selected on official validation split using seed 42"
            ),
            "threshold_selection": (
                "per-seed threshold selected on validation macro-F1"
            ),
            "test_access": (
                "test evaluated only after alpha and threshold were fixed"
            ),
            "seeds": SEEDS,
            "positive_class": "bot",
        },
        "models": {
            "majority": "constant training-majority prediction",
            "mask_only": (
                "semantic-text and graph availability indicators only"
            ),
            "metadata_only": "10 leakage-safe account metadata features",
            "tfidf_only": (
                "80,000-dimensional train-fitted semantic TF-IDF"
            ),
            "tfidf_metadata_concat": (
                "sparse concatenation of TF-IDF and metadata"
            ),
        },
        "selected_alphas": selected_alphas,
        "classifier": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "penalty": "l2",
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "tolerance": TOLERANCE,
        },
        "input_hashes": {
            "users": sha256_file(
                CANONICAL_DIR / "users.parquet"
            ),
            "metadata": sha256_file(
                METADATA_DIR / "metadata_features.npy"
            ),
            "tfidf": sha256_file(
                TFIDF_DIR / "semantic_tfidf_features.npz"
            ),
        },
    }

    with (OUTPUT_DIR / "classical_baseline_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nCLASSICAL BASELINES: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()