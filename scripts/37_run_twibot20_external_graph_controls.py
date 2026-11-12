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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "TwiBot-20-Format22" / "Twibot-20"
FEATURE_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "36_twibot20_external_graph_controls"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
MODEL_ROOT = OUTPUT_ROOT / "models"

SEEDS = [42, 123, 2024, 3407, 7777]
TRAIN_ID, VAL_ID, TEST_ID = 0, 1, 2
CHUNK_SIZE = 1_000_000
RELATIONS = ("follow", "friend")
RAW_FEATURE_NAMES = [
    "follow_out", "follow_in", "friend_out", "friend_in",
    "total_out", "total_in", "total_degree", "relation_diversity",
    "graph_available",
]


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
    last = int(boundaries[best])
    threshold = (
        float(sorted_scores[-1] - 1e-6)
        if last == len(sorted_scores) - 1
        else float((sorted_scores[last] + sorted_scores[last + 1]) / 2)
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


def build_topology_features(external_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    row_map = {user_id: index for index, user_id in enumerate(external_ids.tolist())}
    n = len(external_ids)
    counts = {name: np.zeros(n, dtype=np.float64) for name in ["follow_out", "follow_in", "friend_out", "friend_in"]}
    relation_edge_counts = {relation: 0 for relation in RELATIONS}
    total_edges_streamed = 0

    print("Streaming TwiBot-20 user-user topology from edge.csv...")
    for chunk in pd.read_csv(
        DATA_ROOT / "edge.csv",
        dtype=str,
        usecols=["source_id", "relation", "target_id"],
        chunksize=CHUNK_SIZE,
    ):
        total_edges_streamed += len(chunk)
        chunk["relation"] = chunk["relation"].str.strip().str.lower()
        selected = chunk[chunk["relation"].isin(RELATIONS)]
        for relation in RELATIONS:
            relation_chunk = selected[selected["relation"].eq(relation)]
            relation_edge_counts[relation] += len(relation_chunk)
            if relation_chunk.empty:
                continue
            source_rows = relation_chunk["source_id"].map(row_map).dropna().astype(np.int64).to_numpy()
            target_rows = relation_chunk["target_id"].map(row_map).dropna().astype(np.int64).to_numpy()
            if source_rows.size:
                np.add.at(counts[f"{relation}_out"], source_rows, 1.0)
            if target_rows.size:
                np.add.at(counts[f"{relation}_in"], target_rows, 1.0)
        if total_edges_streamed % 5_000_000 < CHUNK_SIZE:
            print(f"  streamed {total_edges_streamed:,} edges")

    total_out = counts["follow_out"] + counts["friend_out"]
    total_in = counts["follow_in"] + counts["friend_in"]
    total_degree = total_out + total_in
    relation_diversity = (
        (counts["follow_out"] > 0).astype(np.float64)
        + (counts["follow_in"] > 0).astype(np.float64)
        + (counts["friend_out"] > 0).astype(np.float64)
        + (counts["friend_in"] > 0).astype(np.float64)
    )
    graph_available = (total_degree > 0).astype(np.float64)
    raw = np.column_stack([
        counts["follow_out"], counts["follow_in"], counts["friend_out"], counts["friend_in"],
        total_out, total_in, total_degree, relation_diversity, graph_available,
    ])
    transformed = raw.copy()
    transformed[:, :7] = np.log1p(transformed[:, :7])
    audit = {
        "total_edges_streamed": total_edges_streamed,
        "relation_edge_counts": relation_edge_counts,
        "labeled_graph_users": int(graph_available.sum()),
        "labeled_graph_availability_rate": float(graph_available.mean()),
        "topology_only": True,
    }
    return transformed, audit


def train_model(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    external_ids: np.ndarray,
) -> list[dict]:
    positive = int(labels[train_indices].sum())
    negative = int(len(train_indices) - positive)
    scale_pos_weight = negative / positive
    rows = []
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
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
        )
        model.fit(
            features[train_indices], labels[train_indices],
            eval_set=[(features[val_indices], labels[val_indices])], verbose=False,
        )
        val_scores = model.predict_proba(features[val_indices])[:, 1]
        threshold, val_macro_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
        test_scores = model.predict_proba(features[test_indices])[:, 1]
        result, predictions = metrics(labels[test_indices], test_scores, threshold)
        model_path = MODEL_ROOT / f"{model_name}_seed_{seed}.json"
        model.save_model(model_path)
        prediction_path = PREDICTION_ROOT / f"{model_name}_seed_{seed}.parquet"
        pd.DataFrame({
            "external_id": external_ids[test_indices],
            "label": labels[test_indices],
            "score": test_scores.astype(np.float32),
            "prediction": predictions,
            "threshold": threshold,
        }).to_parquet(prediction_path, index=False)
        rows.append({
            "model": model_name,
            "protocol": "locked_within_twibot20_replication",
            "seed": seed,
            "threshold": threshold,
            "validation_macro_f1": val_macro_f1,
            "validation_auroc": roc_auc_score(labels[val_indices], val_scores),
            "validation_auprc": average_precision_score(labels[val_indices], val_scores),
            "best_iteration": int(model.best_iteration),
            "model_sha256": sha256(model_path),
            **result,
        })
        print(
            f"TEST {model_name}, seed={seed}: macro-F1={result['macro_f1']:.6f}, "
            f"AUROC={result['auroc']:.6f}, AUPRC={result['auprc']:.6f}"
        )
    return rows


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    external_ids = np.load(FEATURE_ROOT / "external_ids.npy")
    labels = np.load(FEATURE_ROOT / "labels.npy").astype(np.int8)
    split_ids = np.load(FEATURE_ROOT / "split_ids.npy").astype(np.int8)
    metadata = np.load(FEATURE_ROOT / "metadata_twibot20_train_scaled.npy").astype(np.float32)
    if len(external_ids) != len(labels) or metadata.shape != (len(labels), 10):
        raise RuntimeError("External arrays are not aligned.")
    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    raw_graph, graph_audit = build_topology_features(external_ids)
    if graph_audit["labeled_graph_users"] != 11814:
        raise RuntimeError(
            f"Graph availability disagrees with Script 29: {graph_audit['labeled_graph_users']}"
        )

    lower = np.quantile(raw_graph[train_indices], 0.001, axis=0)
    upper = np.quantile(raw_graph[train_indices], 0.999, axis=0)
    scaler = StandardScaler().fit(np.clip(raw_graph[train_indices], lower, upper))
    graph_features = scaler.transform(np.clip(raw_graph, lower, upper)).astype(np.float32)
    if not np.isfinite(graph_features).all():
        raise RuntimeError("Non-finite graph features detected.")
    metadata_graph_features = np.concatenate([metadata, graph_features], axis=1).astype(np.float32)

    np.save(OUTPUT_ROOT / "graph_features.npy", graph_features)
    np.save(OUTPUT_ROOT / "metadata_graph_features.npy", metadata_graph_features)
    pd.DataFrame({
        "feature": RAW_FEATURE_NAMES,
        "training_lower_bound": lower,
        "training_upper_bound": upper,
        "training_mean_after_clipping": scaler.mean_,
        "training_scale_after_clipping": scaler.scale_,
    }).to_csv(OUTPUT_ROOT / "graph_preprocessing_parameters.csv", index=False)

    rows = []
    rows += train_model(
        "xgboost_graph_topology", graph_features, labels,
        train_indices, val_indices, test_indices, external_ids,
    )
    rows += train_model(
        "xgboost_metadata_graph_topology", metadata_graph_features, labels,
        train_indices, val_indices, test_indices, external_ids,
    )
    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(OUTPUT_ROOT / "external_graph_control_seed_results.csv", index=False)

    metric_names = [
        "accuracy", "balanced_accuracy", "precision_bot", "recall_bot",
        "specificity_human", "f1_bot", "macro_f1", "mcc", "auroc", "auprc",
    ]
    summary_rows = []
    for model_name, group in seed_results.groupby("model", sort=True):
        row = {"model": model_name, "protocol": "locked_within_twibot20_replication", "seeds": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(OUTPUT_ROOT / "external_graph_control_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "protocol": "locked within-dataset replication only",
        "strict_zero_shot_graph_transfer": False,
        "zero_shot_exclusion_reason": "TwiBot-20 follow/friend relation vocabulary is not feature-compatible with the 14-relation TwiBot-22 heterogeneous graph expert",
        "features": RAW_FEATURE_NAMES,
        "feature_source": "topology only; no text, labels, or profile metadata in graph-only representation",
        "preprocessing_fit_scope": "TwiBot-20 official training split only",
        "hyperparameters": "locked XGBoost family/settings from TwiBot-22; class weight recomputed from external training prevalence",
        "external_test_used_for_selection": False,
        "graph_audit": graph_audit,
        "seeds": SEEDS,
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "graph_features.npy": sha256(OUTPUT_ROOT / "graph_features.npy"),
            "metadata_graph_features.npy": sha256(OUTPUT_ROOT / "metadata_graph_features.npy"),
            "external_graph_control_seed_results.csv": sha256(OUTPUT_ROOT / "external_graph_control_seed_results.csv"),
            "external_graph_control_summary.csv": sha256(OUTPUT_ROOT / "external_graph_control_summary.csv"),
        },
    }
    (OUTPUT_ROOT / "external_graph_control_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL GRAPH CONTROLS: PASS")
    print(f"Graph-available labeled users: {graph_audit['labeled_graph_users']:,}/{len(labels):,}")
    print(summary[["model", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auprc_mean"]].to_string(index=False))
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
