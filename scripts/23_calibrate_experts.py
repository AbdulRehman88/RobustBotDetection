from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "08_frozen_bertweet"
METADATA_FEATURE_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
GRAPH_FEATURE_DIR = PROJECT_ROOT / "outputs" / "16_heterogeneous_graph_features"
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
BERT_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
METADATA_DIR = PROJECT_ROOT / "outputs" / "15_xgboost_metadata"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "17_heterogeneous_graph_baseline"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "22_expert_calibration"
CALIBRATED_DIR = OUTPUT_DIR / "calibrated_test_predictions"

SEEDS = [42, 123, 2024, 3407, 7777]
VAL_ID = 1
TEST_ID = 2
EVAL_BATCH_SIZE = 8192
HIDDEN_DIM = 256
BOTTLENECK_DIM = 64
EPSILON = 1e-7
ECE_BINS = 15


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
            nn.LayerNorm(BOTTLENECK_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(BOTTLENECK_DIM, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(
        np.digitize(probabilities, edges[1:-1], right=True),
        0,
        bins - 1,
    )
    total = len(labels)
    value = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count:
            accuracy = float(labels[mask].mean())
            confidence = float(probabilities[mask].mean())
            value += (count / total) * abs(accuracy - confidence)
    return float(value)


def adaptive_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, bins)
    value = 0.0
    for group in groups:
        if len(group):
            value += (len(group) / len(labels)) * abs(
                float(labels[group].mean())
                - float(probabilities[group].mean())
            )
    return float(value)


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece_15": expected_calibration_error(labels, probabilities),
        "adaptive_ece_15": adaptive_calibration_error(labels, probabilities),
        "mean_probability": float(probabilities.mean()),
        "observed_prevalence": float(labels.mean()),
        "calibration_in_the_large": float(
            probabilities.mean() - labels.mean()
        ),
    }


def fit_platt(
    labels: np.ndarray,
    raw_values: np.ndarray,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    calibrator.fit(raw_values.reshape(-1, 1), labels)
    return calibrator


def prepare_bert_batch(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    include_metadata: bool,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, np.maximum(norms, 1e-12), out=values)
    parts = [values, availability[indices, None].astype(np.float32)]
    if include_metadata:
        parts.append(np.asarray(metadata[indices], dtype=np.float32))
    return torch.from_numpy(np.concatenate(parts, axis=1)).to(
        device, non_blocking=True
    )


def bert_logits(
    checkpoint_path: Path,
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    include_metadata: bool,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint["state_dict"]
    input_dim = int(state["network.0.weight"].shape[1])
    model = MLPClassifier(input_dim, float(checkpoint["dropout"]))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH_SIZE):
            batch_indices = indices[start : start + EVAL_BATCH_SIZE]
            batch = prepare_bert_batch(
                embeddings,
                metadata,
                availability,
                batch_indices,
                include_metadata,
                device,
            )
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def load_xgboost_scores(
    model_path: Path,
    features: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(
        np.asarray(features[indices], dtype=np.float32)
    )[:, 1]
    return probabilities.astype(np.float64)


def aligned_test_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values("node_id").reset_index(drop=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for BERTweet checkpoint inference.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    node_ids = np.load(EMBEDDING_DIR / "node_ids.npy", mmap_mode="r")
    labels = np.load(EMBEDDING_DIR / "labels.npy", mmap_mode="r").astype(np.int8)
    split_ids = np.load(EMBEDDING_DIR / "split_ids.npy", mmap_mode="r")
    availability = np.load(
        EMBEDDING_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    metadata = np.load(
        METADATA_FEATURE_DIR / "metadata_features.npy", mmap_mode="r"
    )
    mean_embeddings = np.load(
        EMBEDDING_DIR / "mean_pooled_embeddings.npy", mmap_mode="r"
    )
    graph_features = np.load(
        GRAPH_FEATURE_DIR / "heterogeneous_graph_features.npy", mmap_mode="r"
    )
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    y_val = np.asarray(labels[val_indices], dtype=np.int8)
    y_test = np.asarray(labels[test_indices], dtype=np.int8)

    canonical = pd.read_parquet(
        CANONICAL_DIR / "users.parquet", columns=["node_id", "label"]
    ).sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(
        canonical["node_id"].to_numpy(dtype=np.int64),
        np.asarray(node_ids, dtype=np.int64),
    ):
        raise RuntimeError("Canonical/BERTweet node order mismatch.")

    rows = []
    parameter_rows = []

    for seed in SEEDS:
        expert_data: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}

        for expert, model_name, include_metadata in [
            ("text", "bertweet_mean", False),
            ("text_metadata", "bertweet_mean_metadata", True),
        ]:
            val_raw = bert_logits(
                BERT_DIR / "checkpoints" / f"{model_name}_seed_{seed}.pt",
                mean_embeddings,
                metadata,
                availability,
                val_indices,
                include_metadata,
                device,
            )
            test_frame = aligned_test_frame(
                BERT_DIR
                / "test_predictions"
                / f"{model_name}_seed_{seed}_test_predictions.parquet"
            )
            if not np.array_equal(test_frame["label"].to_numpy(np.int8), y_test):
                raise RuntimeError(f"Test-label mismatch: {expert}, seed {seed}.")
            expert_data[expert] = (
                val_raw,
                test_frame["score"].to_numpy(dtype=np.float64),
                test_frame,
            )

        for expert, base_dir, model_stem, features in [
            ("metadata", METADATA_DIR, "xgboost_metadata", metadata),
            (
                "graph",
                GRAPH_DIR,
                "xgboost_heterogeneous_graph",
                graph_features,
            ),
        ]:
            val_probability = load_xgboost_scores(
                base_dir / "models" / f"{model_stem}_seed_{seed}.json",
                features,
                val_indices,
            )
            test_frame = aligned_test_frame(
                base_dir
                / "test_predictions"
                / f"seed_{seed}_test_predictions.parquet"
            )
            if not np.array_equal(test_frame["label"].to_numpy(np.int8), y_test):
                raise RuntimeError(f"Test-label mismatch: {expert}, seed {seed}.")
            expert_data[expert] = (
                logit(np.clip(val_probability, EPSILON, 1.0 - EPSILON)),
                logit(
                    np.clip(
                        test_frame["score"].to_numpy(dtype=np.float64),
                        EPSILON,
                        1.0 - EPSILON,
                    )
                ),
                test_frame,
            )

        for expert, (val_raw, test_raw, test_frame) in expert_data.items():
            calibrator = fit_platt(y_val, val_raw)
            val_before = expit(val_raw)
            test_before = expit(test_raw)
            val_after = calibrator.predict_proba(val_raw.reshape(-1, 1))[:, 1]
            test_after = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]

            parameter_rows.append(
                {
                    "expert": expert,
                    "seed": seed,
                    "platt_slope": float(calibrator.coef_[0, 0]),
                    "platt_intercept": float(calibrator.intercept_[0]),
                    "fit_split": "official_validation",
                    "fit_users": int(len(y_val)),
                }
            )

            for split_name, split_labels, before, after in [
                ("validation", y_val, val_before, val_after),
                ("test", y_test, test_before, test_after),
            ]:
                for status, probabilities in [
                    ("uncalibrated", before),
                    ("platt_calibrated", after),
                ]:
                    rows.append(
                        {
                            "expert": expert,
                            "seed": seed,
                            "split": split_name,
                            "calibration": status,
                            **calibration_metrics(split_labels, probabilities),
                        }
                    )

            pd.DataFrame(
                {
                    "node_id": test_frame["node_id"].to_numpy(dtype=np.int64),
                    "label": y_test,
                    "uncalibrated_probability": test_before.astype(np.float32),
                    "calibrated_probability": test_after.astype(np.float32),
                    "original_prediction": test_frame["prediction"].to_numpy(
                        dtype=np.int8
                    ),
                }
            ).to_parquet(
                CALIBRATED_DIR / f"{expert}_seed_{seed}_test.parquet",
                index=False,
            )

            print(
                f"{expert}, seed={seed}: test ECE "
                f"{expected_calibration_error(y_test, test_before):.6f} -> "
                f"{expected_calibration_error(y_test, test_after):.6f}; "
                f"Brier {brier_score_loss(y_test, test_before):.6f} -> "
                f"{brier_score_loss(y_test, test_after):.6f}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "calibration_seed_results.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        OUTPUT_DIR / "platt_parameters.csv", index=False
    )

    test_results = results.loc[results["split"].eq("test")]
    summary = (
        test_results.groupby(["expert", "calibration"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            nll_mean=("nll", "mean"),
            nll_std=("nll", "std"),
            ece_15_mean=("ece_15", "mean"),
            ece_15_std=("ece_15", "std"),
            adaptive_ece_15_mean=("adaptive_ece_15", "mean"),
            adaptive_ece_15_std=("adaptive_ece_15", "std"),
            mean_probability_mean=("mean_probability", "mean"),
            observed_prevalence=("observed_prevalence", "mean"),
        )
        .sort_values(["expert", "calibration"])
    )
    summary.to_csv(OUTPUT_DIR / "test_calibration_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "method": "seed-specific Platt scaling",
        "calibration_fit_split": "official validation only",
        "evaluation_split": "official test, evaluated once",
        "experts": ["text", "text_metadata", "metadata", "graph"],
        "seeds": SEEDS,
        "metrics": [
            "Brier score",
            "negative log-likelihood",
            "15-bin ECE",
            "15-bin adaptive ECE",
            "calibration-in-the-large",
        ],
        "classification_policy": (
            "Original validation-selected thresholds and predictions are retained; "
            "calibration changes reported probabilities only."
        ),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nEXPERT CALIBRATION: PASS")
    print(
        summary[
            [
                "expert",
                "calibration",
                "brier_mean",
                "nll_mean",
                "ece_15_mean",
                "adaptive_ece_15_mean",
            ]
        ].to_string(index=False)
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
