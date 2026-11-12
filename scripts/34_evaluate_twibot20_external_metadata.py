from __future__ import annotations

import hashlib
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
FEATURE_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"
SOURCE_ROOT = PROJECT_ROOT / "outputs" / "15_xgboost_metadata"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "33_twibot20_external_metadata"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
MODEL_ROOT = OUTPUT_ROOT / "replication_models"

SEEDS = [42, 123, 2024, 3407, 7777]
TRAIN_ID, VAL_ID, TEST_ID = 0, 1, 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exact_macro_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    total_positive = int(sorted_labels.sum())
    total_negative = int(len(sorted_labels) - total_positive)
    boundaries = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = cumulative_tp[boundaries].astype(np.float64)
    fp = cumulative_fp[boundaries].astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp
    positive_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    negative_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) != 0)
    macro_f1 = (positive_f1 + negative_f1) / 2
    all_negative_f1 = (2 * total_negative / (2 * total_negative + total_positive)) / 2
    best = int(np.argmax(macro_f1))
    if all_negative_f1 > macro_f1[best]:
        return float(sorted_scores[0] + 1e-6), float(all_negative_f1)
    last_positive = int(boundaries[best])
    threshold = (
        float(sorted_scores[-1] - 1e-6)
        if last_positive == len(sorted_scores) - 1
        else float((sorted_scores[last_positive] + sorted_scores[last_positive + 1]) / 2)
    )
    return threshold, float(macro_f1[best])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision_bot": precision_score(labels, predictions, zero_division=0),
        "recall_bot": recall_score(labels, predictions, zero_division=0),
        "specificity_human": tn / (tn + fp),
        "f1_bot": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, predictions


def save_predictions(
    filename: str,
    external_ids: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> None:
    pd.DataFrame({
        "external_id": external_ids[indices],
        "label": labels[indices],
        "score": scores.astype(np.float32),
        "prediction": predictions,
        "threshold": threshold,
    }).to_parquet(PREDICTION_ROOT / filename, index=False)


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    source_features = np.load(FEATURE_ROOT / "metadata_twibot22_scaled.npy", mmap_mode="r")
    replication_features = np.load(
        FEATURE_ROOT / "metadata_twibot20_train_scaled.npy", mmap_mode="r"
    )
    labels = np.load(FEATURE_ROOT / "labels.npy", mmap_mode="r").astype(np.int8)
    split_ids = np.load(FEATURE_ROOT / "split_ids.npy", mmap_mode="r").astype(np.int8)
    external_ids = np.load(FEATURE_ROOT / "external_ids.npy", mmap_mode="r")
    if source_features.shape != replication_features.shape or source_features.shape != (len(labels), 10):
        raise RuntimeError("External metadata arrays are not aligned 10-feature matrices.")

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    source_results = pd.read_csv(SOURCE_ROOT / "xgboost_metadata_seed_results.csv")
    source_thresholds = dict(zip(source_results["seed"].astype(int), source_results["threshold"].astype(float)))

    rows = []
    for seed in SEEDS:
        model_path = SOURCE_ROOT / "models" / f"xgboost_metadata_seed_{seed}.json"
        if not model_path.is_file() or seed not in source_thresholds:
            raise FileNotFoundError(f"Missing source model/threshold for seed {seed}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        val_scores = model.predict_proba(np.asarray(source_features[val_indices]))[:, 1]
        adapted_threshold, adapted_val_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
        test_scores = model.predict_proba(np.asarray(source_features[test_indices]))[:, 1]

        for protocol, threshold in [
            ("strict_zero_shot_source_threshold", source_thresholds[seed]),
            ("validation_threshold_adaptation", adapted_threshold),
        ]:
            result, predictions = metrics(labels[test_indices], test_scores, threshold)
            rows.append({
                "model": "xgboost_metadata",
                "protocol": protocol,
                "seed": seed,
                "threshold": threshold,
                "source_validation_threshold": source_thresholds[seed],
                "external_validation_selected_threshold": adapted_threshold,
                "external_validation_macro_f1_at_adapted_threshold": adapted_val_f1,
                "model_sha256": sha256(model_path),
                **result,
            })
            save_predictions(
                f"xgboost_metadata_{protocol}_seed_{seed}.parquet",
                external_ids, test_indices, labels, test_scores, predictions, threshold,
            )
            print(
                f"EXTERNAL metadata, {protocol}, seed={seed}: "
                f"macro-F1={result['macro_f1']:.6f}, AUROC={result['auroc']:.6f}, "
                f"AUPRC={result['auprc']:.6f}"
            )

    positive = int(labels[train_indices].sum())
    negative = int(len(train_indices) - positive)
    external_scale_pos_weight = negative / positive
    for seed in SEEDS:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=8,
            learning_rate=0.05,
            n_estimators=2000,
            early_stopping_rounds=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=external_scale_pos_weight,
            random_state=seed,
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
        )
        model.fit(
            np.asarray(replication_features[train_indices]),
            labels[train_indices],
            eval_set=[(np.asarray(replication_features[val_indices]), labels[val_indices])],
            verbose=False,
        )
        val_scores = model.predict_proba(np.asarray(replication_features[val_indices]))[:, 1]
        threshold, val_macro_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
        test_scores = model.predict_proba(np.asarray(replication_features[test_indices]))[:, 1]
        result, predictions = metrics(labels[test_indices], test_scores, threshold)
        model_path = MODEL_ROOT / f"locked_replication_seed_{seed}.json"
        model.save_model(model_path)
        rows.append({
            "model": "xgboost_metadata",
            "protocol": "locked_within_twibot20_replication",
            "seed": seed,
            "threshold": threshold,
            "source_validation_threshold": np.nan,
            "external_validation_selected_threshold": threshold,
            "external_validation_macro_f1_at_adapted_threshold": val_macro_f1,
            "best_iteration": int(model.best_iteration),
            "model_sha256": sha256(model_path),
            **result,
        })
        save_predictions(
            f"xgboost_metadata_locked_replication_seed_{seed}.parquet",
            external_ids, test_indices, labels, test_scores, predictions, threshold,
        )
        print(
            f"EXTERNAL metadata locked replication, seed={seed}: "
            f"macro-F1={result['macro_f1']:.6f}, AUROC={result['auroc']:.6f}, "
            f"AUPRC={result['auprc']:.6f}"
        )

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(OUTPUT_ROOT / "external_metadata_seed_results.csv", index=False)
    metric_names = [
        "accuracy", "balanced_accuracy", "precision_bot", "recall_bot",
        "specificity_human", "f1_bot", "macro_f1", "mcc", "auroc", "auprc",
    ]
    summary_rows = []
    for (model_name, protocol), group in seed_results.groupby(["model", "protocol"], sort=True):
        row = {"model": model_name, "protocol": protocol, "seeds": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_ROOT / "external_metadata_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "model": "XGBoost metadata-only control",
        "primary_external_protocol": "strict_zero_shot_source_threshold",
        "zero_shot_source_preprocessing_and_models_frozen": True,
        "replication_hyperparameters_locked_from_twibot22": True,
        "replication_scale_pos_weight": external_scale_pos_weight,
        "external_test_never_used_for_training_selection_or_thresholding": True,
        "seeds": SEEDS,
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "external_metadata_seed_results.csv": sha256(OUTPUT_ROOT / "external_metadata_seed_results.csv"),
            "external_metadata_summary.csv": sha256(OUTPUT_ROOT / "external_metadata_summary.csv"),
        },
    }
    (OUTPUT_ROOT / "external_metadata_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL METADATA CONTROL: PASS")
    print(summary[["model", "protocol", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auprc_mean"]].to_string(index=False))
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
