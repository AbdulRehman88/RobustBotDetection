from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT23 = PROJECT_ROOT / "scripts" / "23_calibrate_experts.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "23_robustness_stress_tests"

SEEDS = [42, 123, 2024, 3407, 7777]
DROP_RATES = [0.10, 0.30, 0.50, 0.70]
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30]
MISSINGNESS_REPETITIONS = 10
TEST_ID = 2
TRAIN_BOT_PREVALENCE = 54_586 / 700_000
EVAL_BATCH_SIZE = 8192

ROUTING = {
    "T": "text",
    "G": "graph",
    "M": "metadata",
    "TG": "text",
    "TM": "text_metadata",
    "GM": "metadata",
    "TGM": "text_metadata",
}


def load_script23():
    specification = importlib.util.spec_from_file_location(
        "expert_calibration_module", SCRIPT23
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT23}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float((labels == predictions).mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_bot": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def load_expert_test_predictions(calibration_module, seed: int):
    sources = {
        expert: (
            calibration_module.CALIBRATED_DIR
            / f"{expert}_seed_{seed}_test.parquet"
        )
        for expert in ["text", "text_metadata", "metadata", "graph"]
    }
    frames = {}
    for expert, path in sources.items():
        frame = pd.read_parquet(path).sort_values("node_id").reset_index(drop=True)
        frames[expert] = {
            "node_id": frame["node_id"].to_numpy(dtype=np.int64),
            "label": frame["label"].to_numpy(dtype=np.int8),
            "score": frame["calibrated_probability"].to_numpy(dtype=np.float32),
            "prediction": frame["original_prediction"].to_numpy(dtype=np.int8),
        }
    reference = frames["text"]
    for expert, frame in frames.items():
        if not np.array_equal(frame["node_id"], reference["node_id"]):
            raise RuntimeError(f"Node mismatch for {expert}, seed {seed}.")
        if not np.array_equal(frame["label"], reference["label"]):
            raise RuntimeError(f"Label mismatch for {expert}, seed {seed}.")
    return frames


def dropout_probabilities(
    labels: np.ndarray,
    rate: float,
    mechanism: str,
) -> np.ndarray:
    if mechanism == "mcar":
        return np.full(len(labels), rate, dtype=np.float64)
    if mechanism == "bot_heavier":
        return np.where(labels == 1, rate, rate / 2.0)
    if mechanism == "human_heavier":
        return np.where(labels == 0, rate, rate / 2.0)
    raise ValueError(mechanism)


def run_missingness_stress(calibration_module) -> pd.DataFrame:
    rows = []
    mechanisms = ["mcar", "bot_heavier", "human_heavier"]
    modality_names = np.asarray(["T", "G", "M"])

    for seed in SEEDS:
        experts = load_expert_test_predictions(calibration_module, seed)
        labels = experts["text"]["label"]
        node_ids = experts["text"]["node_id"]

        for mechanism_index, mechanism in enumerate(mechanisms):
            for rate_index, rate in enumerate(DROP_RATES):
                probability = dropout_probabilities(labels, rate, mechanism)
                for repetition in range(MISSINGNESS_REPETITIONS):
                    rng = np.random.default_rng(
                        seed * 1_000_000
                        + mechanism_index * 100_000
                        + rate_index * 1_000
                        + repetition
                    )
                    dropped = rng.random((len(labels), 3)) < probability[:, None]
                    available = ~dropped
                    scores = np.full(
                        len(labels), TRAIN_BOT_PREVALENCE, dtype=np.float32
                    )
                    predictions = np.zeros(len(labels), dtype=np.int8)
                    fallback_count = 0

                    for mask_value in range(8):
                        bit_mask = np.asarray(
                            [
                                bool(mask_value & 4),
                                bool(mask_value & 2),
                                bool(mask_value & 1),
                            ]
                        )
                        selected = np.all(available == bit_mask, axis=1)
                        if not selected.any():
                            continue
                        pattern = "".join(modality_names[bit_mask])
                        if not pattern:
                            fallback_count += int(selected.sum())
                            continue
                        expert = ROUTING[pattern]
                        scores[selected] = experts[expert]["score"][selected]
                        predictions[selected] = experts[expert]["prediction"][selected]

                    row_metrics = metrics(labels, scores, predictions)
                    rows.append(
                        {
                            "seed": seed,
                            "mechanism": mechanism,
                            "nominal_drop_rate": rate,
                            "repetition": repetition,
                            "users": len(labels),
                            "fallback_users": fallback_count,
                            "fallback_rate": fallback_count / len(labels),
                            "realized_text_drop_rate": float(dropped[:, 0].mean()),
                            "realized_graph_drop_rate": float(dropped[:, 1].mean()),
                            "realized_metadata_drop_rate": float(dropped[:, 2].mean()),
                            **row_metrics,
                        }
                    )

                    if repetition == 0:
                        print(
                            f"MISSINGNESS seed={seed}, {mechanism}, rate={rate:.2f}: "
                            f"macro-F1={row_metrics['macro_f1']:.6f}, "
                            f"fallback={fallback_count / len(labels):.4f}"
                        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "missingness_stress_results.csv", index=False)
    return result


def load_bert_model(calibration_module, model_name: str, seed: int, device):
    checkpoint = torch.load(
        calibration_module.BERT_DIR
        / "checkpoints"
        / f"{model_name}_seed_{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    input_dim = int(state["network.0.weight"].shape[1])
    model = calibration_module.MLPClassifier(
        input_dim, float(checkpoint["dropout"])
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), float(checkpoint["selection"]["threshold"])


def noisy_bert_scores(
    model,
    embeddings,
    metadata,
    availability,
    test_indices,
    include_metadata: bool,
    text_noise: float,
    metadata_noise: float,
    random_seed: int,
    device,
) -> np.ndarray:
    model.eval()
    outputs = []
    rng = np.random.default_rng(random_seed)
    with torch.inference_mode():
        for start in range(0, len(test_indices), EVAL_BATCH_SIZE):
            batch_indices = test_indices[start : start + EVAL_BATCH_SIZE]
            text = np.asarray(embeddings[batch_indices], dtype=np.float32)
            norms = np.linalg.norm(text, axis=1, keepdims=True)
            np.divide(text, np.maximum(norms, 1e-12), out=text)
            if text_noise > 0:
                # Use a controlled relative-L2 perturbation. Because each
                # embedding has unit norm, text_noise=0.30 means a noise
                # vector with norm 0.30, not per-dimension sigma=0.30
                # (which would have expected norm sqrt(768)*0.30).
                perturbation = rng.normal(0.0, 1.0, text.shape).astype(np.float32)
                perturbation_norms = np.linalg.norm(
                    perturbation, axis=1, keepdims=True
                )
                np.divide(
                    perturbation,
                    np.maximum(perturbation_norms, 1e-12),
                    out=perturbation,
                )
                text += text_noise * perturbation
                norms = np.linalg.norm(text, axis=1, keepdims=True)
                np.divide(text, np.maximum(norms, 1e-12), out=text)
            parts = [
                text,
                availability[batch_indices, None].astype(np.float32),
            ]
            if include_metadata:
                meta = np.asarray(metadata[batch_indices], dtype=np.float32)
                if metadata_noise > 0:
                    meta += rng.normal(
                        0.0, metadata_noise, meta.shape
                    ).astype(np.float32)
                parts.append(meta)
            batch = torch.from_numpy(np.concatenate(parts, axis=1)).to(
                device, non_blocking=True
            )
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def load_xgb_model(path: Path):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def noisy_xgb_scores(
    model,
    features: np.ndarray,
    test_indices: np.ndarray,
    noise_level: float,
    random_seed: int,
) -> np.ndarray:
    values = np.asarray(features[test_indices], dtype=np.float32)
    if noise_level > 0:
        rng = np.random.default_rng(random_seed)
        values += rng.normal(0.0, noise_level, values.shape).astype(np.float32)
    return model.predict_proba(values)[:, 1].astype(np.float64)


def threshold_lookup(path: Path, model_name: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if model_name is not None:
        frame = frame.loc[frame["model"].eq(model_name)]
    return frame.set_index("seed")


def run_noise_stress(calibration_module) -> pd.DataFrame:
    device = torch.device("cuda")
    labels = np.load(
        calibration_module.EMBEDDING_DIR / "labels.npy", mmap_mode="r"
    ).astype(np.int8)
    split_ids = np.load(
        calibration_module.EMBEDDING_DIR / "split_ids.npy", mmap_mode="r"
    )
    availability = np.load(
        calibration_module.EMBEDDING_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    embeddings = np.load(
        calibration_module.EMBEDDING_DIR / "mean_pooled_embeddings.npy",
        mmap_mode="r",
    )
    metadata = np.load(
        calibration_module.METADATA_FEATURE_DIR / "metadata_features.npy",
        mmap_mode="r",
    )
    graph = np.load(
        calibration_module.GRAPH_FEATURE_DIR
        / "heterogeneous_graph_features.npy",
        mmap_mode="r",
    )
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    y_test = np.asarray(labels[test_indices], dtype=np.int8)

    bert_results = threshold_lookup(
        calibration_module.BERT_DIR / "bertweet_baseline_seed_results.csv"
    )
    metadata_results = threshold_lookup(
        calibration_module.METADATA_DIR / "xgboost_metadata_seed_results.csv"
    )
    graph_results = threshold_lookup(
        calibration_module.GRAPH_DIR
        / "xgboost_heterogeneous_graph_seed_results.csv"
    )

    rows = []
    for seed in SEEDS:
        text_model, text_threshold = load_bert_model(
            calibration_module, "bertweet_mean", seed, device
        )
        tm_model, tm_threshold = load_bert_model(
            calibration_module, "bertweet_mean_metadata", seed, device
        )
        metadata_model = load_xgb_model(
            calibration_module.METADATA_DIR
            / "models"
            / f"xgboost_metadata_seed_{seed}.json"
        )
        graph_model = load_xgb_model(
            calibration_module.GRAPH_DIR
            / "models"
            / f"xgboost_heterogeneous_graph_seed_{seed}.json"
        )
        metadata_threshold = float(metadata_results.loc[seed, "threshold"])
        graph_threshold = float(graph_results.loc[seed, "threshold"])

        # Checkpoint thresholds are authoritative; CSV consistency is audited.
        for model_name, checkpoint_threshold in [
            ("bertweet_mean", text_threshold),
            ("bertweet_mean_metadata", tm_threshold),
        ]:
            csv_value = float(
                bert_results.loc[
                    (bert_results.index == seed)
                    & bert_results["model"].eq(model_name),
                    "threshold",
                ].iloc[0]
            )
            if not np.isclose(checkpoint_threshold, csv_value, atol=1e-8):
                raise RuntimeError(
                    f"Threshold mismatch for {model_name}, seed {seed}."
                )

        for level_index, level in enumerate(NOISE_LEVELS):
            base_random_seed = seed * 10_000 + level_index * 100
            experiments = []

            text_logits = noisy_bert_scores(
                text_model,
                embeddings,
                metadata,
                availability,
                test_indices,
                False,
                level,
                0.0,
                base_random_seed + 1,
                device,
            )
            experiments.append(
                ("text", "text_embedding_gaussian", text_logits, text_threshold, "logit")
            )

            tm_text_logits = noisy_bert_scores(
                tm_model,
                embeddings,
                metadata,
                availability,
                test_indices,
                True,
                level,
                0.0,
                base_random_seed + 2,
                device,
            )
            experiments.append(
                (
                    "text_metadata",
                    "text_embedding_gaussian",
                    tm_text_logits,
                    tm_threshold,
                    "logit",
                )
            )

            tm_meta_logits = noisy_bert_scores(
                tm_model,
                embeddings,
                metadata,
                availability,
                test_indices,
                True,
                0.0,
                level,
                base_random_seed + 3,
                device,
            )
            experiments.append(
                (
                    "text_metadata",
                    "metadata_gaussian",
                    tm_meta_logits,
                    tm_threshold,
                    "logit",
                )
            )

            tm_joint_logits = noisy_bert_scores(
                tm_model,
                embeddings,
                metadata,
                availability,
                test_indices,
                True,
                level,
                level,
                base_random_seed + 4,
                device,
            )
            experiments.append(
                (
                    "text_metadata",
                    "joint_text_metadata_gaussian",
                    tm_joint_logits,
                    tm_threshold,
                    "logit",
                )
            )

            metadata_scores = noisy_xgb_scores(
                metadata_model,
                metadata,
                test_indices,
                level,
                base_random_seed + 5,
            )
            experiments.append(
                (
                    "metadata",
                    "metadata_gaussian",
                    metadata_scores,
                    metadata_threshold,
                    "probability",
                )
            )

            graph_scores = noisy_xgb_scores(
                graph_model,
                graph,
                test_indices,
                level,
                base_random_seed + 6,
            )
            experiments.append(
                (
                    "graph",
                    "graph_feature_gaussian",
                    graph_scores,
                    graph_threshold,
                    "probability",
                )
            )

            for expert, corruption, raw_scores, threshold, score_type in experiments:
                probabilities = expit(raw_scores) if score_type == "logit" else raw_scores
                predictions = (raw_scores >= threshold).astype(np.int8)
                row_metrics = metrics(y_test, probabilities, predictions)
                rows.append(
                    {
                        "seed": seed,
                        "expert": expert,
                        "corruption": corruption,
                        "noise_std": level,
                        "noise_space": (
                            "L2-normalized text embedding and training-standardized metadata"
                            if corruption == "joint_text_metadata_gaussian"
                            else (
                                "L2-normalized frozen text embedding"
                                if "text" in corruption
                                else "training-standardized feature"
                            )
                        ),
                        **row_metrics,
                    }
                )
                print(
                    f"NOISE seed={seed}, {expert}, {corruption}, std={level:.2f}: "
                    f"macro-F1={row_metrics['macro_f1']:.6f}"
                )

        del text_model, tm_model, metadata_model, graph_model
        torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "feature_noise_stress_results.csv", index=False)
    return result


def summarize_missingness(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["mechanism", "nominal_drop_rate"], as_index=False)
        .agg(
            runs=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            fallback_rate_mean=("fallback_rate", "mean"),
        )
        .sort_values(["mechanism", "nominal_drop_rate"])
    )


def summarize_noise(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(
            ["expert", "corruption", "noise_std", "noise_space"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
        )
        .sort_values(["expert", "corruption", "noise_std"])
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if not SCRIPT23.exists():
        raise FileNotFoundError(SCRIPT23)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_module = load_script23()

    missingness_results = run_missingness_stress(calibration_module)
    missingness_summary = summarize_missingness(missingness_results)
    missingness_summary.to_csv(
        OUTPUT_DIR / "missingness_stress_summary.csv", index=False
    )

    noise_results = run_noise_stress(calibration_module)
    noise_summary = summarize_noise(noise_results)
    noise_summary.to_csv(OUTPUT_DIR / "feature_noise_stress_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "seeds": SEEDS,
        "missingness": {
            "rates": DROP_RATES,
            "repetitions_per_seed_rate_mechanism": MISSINGNESS_REPETITIONS,
            "mechanisms": {
                "mcar": "equal independent per-user dropout for T, G, and M",
                "bot_heavier": "bot dropout rate p; human dropout rate p/2",
                "human_heavier": "human dropout rate p; bot dropout rate p/2",
            },
            "empty_pattern_policy": (
                "training-prevalence score and human prediction; no test-label use"
            ),
            "routing": ROUTING,
        },
        "noise": {
            "levels": NOISE_LEVELS,
            "text": (
                "random isotropic direction scaled to the stated fraction of the "
                "unit-normalized frozen BERTweet embedding L2 norm, followed by "
                "renormalization"
            ),
            "metadata_and_graph": (
                "isotropic Gaussian perturbation in training-standardized feature space"
            ),
            "interpretation_limit": (
                "feature-space sensitivity, not raw-text or raw-network corruption"
            ),
            "thresholds": "locked validation-selected thresholds",
        },
        "test_policy": (
            "all rates and mechanisms fixed before execution; no test-driven selection"
        ),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nROBUSTNESS STRESS TESTS: PASS")
    print("\nMissingness summary:")
    print(missingness_summary.to_string(index=False))
    print("\nFeature-noise summary:")
    print(noise_summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
