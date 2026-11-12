from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
FEATURE_DIR = (
    PROJECT_ROOT / "outputs" / "16_heterogeneous_graph_features"
)
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "17_heterogeneous_graph_baseline"
)
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

SEEDS = [42, 123, 2024, 3407, 7777]
MAX_DEPTHS = [4, 6, 8]
LEARNING_RATES = [0.05, 0.10]

N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 5
REG_LAMBDA = 1.0
REG_ALPHA = 0.0

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
        return float(sorted_scores[0] + 1e-7), float(
            all_negative_macro_f1
        )

    boundary = int(boundaries[best_index])
    if boundary == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-7)
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


def build_model(
    max_depth: int,
    learning_rate: float,
    seed: int,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        reg_alpha=REG_ALPHA,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    features = np.load(
        FEATURE_DIR / "heterogeneous_graph_features.npy",
        mmap_mode="r",
    )
    split_ids = np.load(
        FEATURE_DIR / "split_ids.npy",
        mmap_mode="r",
    )

    users = pd.read_parquet(
        CANONICAL_DIR / "users.parquet",
        columns=["node_id", "label", "split"],
    ).sort_values("node_id").reset_index(drop=True)

    node_ids = users["node_id"].to_numpy(dtype=np.int64)
    normalized_labels = users["label"].astype(str).str.lower()
    if set(normalized_labels.unique()) != {"bot", "human"}:
        raise RuntimeError("Unexpected label vocabulary.")
    labels = normalized_labels.eq("bot").to_numpy(dtype=np.int8)

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    x_train = np.asarray(
        features[train_indices], dtype=np.float32
    )
    x_val = np.asarray(
        features[val_indices], dtype=np.float32
    )
    x_test = np.asarray(
        features[test_indices], dtype=np.float32
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives

    tuning_rows = []
    print("Selecting XGBoost hyperparameters on validation data...")

    for max_depth, learning_rate in itertools.product(
        MAX_DEPTHS, LEARNING_RATES
    ):
        started = time.time()
        model = build_model(
            max_depth,
            learning_rate,
            SEEDS[0],
            scale_pos_weight,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        val_scores = model.predict_proba(x_val)[:, 1]
        threshold, val_macro_f1 = exact_macro_f1_threshold(
            y_val, val_scores
        )
        val_metrics, _ = calculate_metrics(
            y_val, val_scores, threshold
        )

        row = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "seed": SEEDS[0],
            "best_iteration": int(model.best_iteration),
            "threshold": threshold,
            "validation_macro_f1": val_macro_f1,
            "validation_auroc": val_metrics["auroc"],
            "validation_auprc": val_metrics["auprc"],
            "runtime_seconds": time.time() - started,
        }
        tuning_rows.append(row)

        print(
            f"depth={max_depth}, lr={learning_rate:.2f}: "
            f"macro-F1={val_macro_f1:.6f}, "
            f"AUROC={val_metrics['auroc']:.6f}, "
            f"iteration={model.best_iteration}"
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        [
            "validation_macro_f1",
            "validation_auprc",
            "validation_auroc",
        ],
        ascending=False,
    )
    tuning.to_csv(
        OUTPUT_DIR / "validation_hyperparameter_search.csv",
        index=False,
    )

    best_depth = int(tuning.iloc[0]["max_depth"])
    best_learning_rate = float(
        tuning.iloc[0]["learning_rate"]
    )

    print(
        f"Selected: depth={best_depth}, "
        f"learning_rate={best_learning_rate:.2f}"
    )

    result_rows = []

    for seed in SEEDS:
        started = time.time()
        model = build_model(
            best_depth,
            best_learning_rate,
            seed,
            scale_pos_weight,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        val_scores = model.predict_proba(x_val)[:, 1]
        threshold, val_macro_f1 = exact_macro_f1_threshold(
            y_val, val_scores
        )

        # Test is evaluated only after model iteration and threshold
        # are fixed through training and validation data.
        test_scores = model.predict_proba(x_test)[:, 1]
        test_metrics, predictions = calculate_metrics(
            y_test, test_scores, threshold
        )

        result_rows.append(
            {
                "model": "xgboost_heterogeneous_graph",
                "seed": seed,
                "max_depth": best_depth,
                "learning_rate": best_learning_rate,
                "best_iteration": int(model.best_iteration),
                "threshold": threshold,
                "validation_macro_f1": val_macro_f1,
                "runtime_seconds": time.time() - started,
                **test_metrics,
            }
        )

        model_path = (
            MODEL_DIR
            / f"xgboost_heterogeneous_graph_seed_{seed}.json"
        )
        model.save_model(model_path)

        pd.DataFrame(
            {
                "node_id": node_ids[test_indices],
                "label": y_test,
                "score": test_scores.astype(np.float32),
                "prediction": predictions,
            }
        ).to_parquet(
            PREDICTION_DIR
            / f"seed_{seed}_test_predictions.parquet",
            index=False,
        )

        print(
            f"TEST seed={seed}: "
            f"macro-F1={test_metrics['macro_f1']:.6f}, "
            f"AUROC={test_metrics['auroc']:.6f}, "
            f"AUPRC={test_metrics['auprc']:.6f}"
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR
        / "xgboost_heterogeneous_graph_seed_results.csv",
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
    summary = {
        "model": "xgboost_heterogeneous_graph",
        "seeds": len(SEEDS),
    }
    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(
            results[metric].mean()
        )
        summary[f"{metric}_std"] = float(
            results[metric].std(ddof=1)
        )

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "xgboost_heterogeneous_graph_summary.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "model": "XGBoost heterogeneous graph expert",
        "xgboost_version": xgb.__version__,
        "features": (
            "29 leakage-free relation-direction topology features "
            "standardized using training users only"
        ),
        "training_split": "official train only",
        "hyperparameter_selection": "official validation only",
        "test_policy": (
            "test accessed after hyperparameters, boosting iteration, "
            "and threshold selection"
        ),
        "seeds": SEEDS,
        "positive_class": "bot",
        "selected_hyperparameters": {
            "max_depth": best_depth,
            "learning_rate": best_learning_rate,
            "n_estimators_upper_bound": N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "min_child_weight": MIN_CHILD_WEIGHT,
            "scale_pos_weight": scale_pos_weight,
            "device": "cuda",
        },
        "input_hash": sha256_file(
            FEATURE_DIR / "heterogeneous_graph_features.npy"
        ),
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nXGBOOST METADATA: PASS")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
