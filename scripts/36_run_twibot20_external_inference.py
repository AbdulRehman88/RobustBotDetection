from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PREDICTIONS = PROJECT_ROOT / "outputs" / "32_twibot20_external_transfer" / "predictions"
METADATA_PREDICTIONS = PROJECT_ROOT / "outputs" / "33_twibot20_external_metadata" / "predictions"
LENGTH128_PREDICTIONS = PROJECT_ROOT / "outputs" / "34_twibot20_length_sensitivity" / "predictions_128"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "35_twibot20_external_inference"

SEEDS = [42, 123, 2024, 3407, 7777]
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20261030


COMPARISONS = [
    {
        "name": "debiased_vs_original_strict_zero_shot",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "debiased_text_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "comparator_root": TRANSFER_PREDICTIONS,
        "comparator_pattern": "bertweet_mean_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "family": "strict_zero_shot",
    },
    {
        "name": "debiased_vs_metadata_strict_zero_shot",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "debiased_text_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "comparator_root": METADATA_PREDICTIONS,
        "comparator_pattern": "xgboost_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "family": "strict_zero_shot",
    },
    {
        "name": "original_vs_metadata_strict_zero_shot",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "bertweet_mean_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "comparator_root": METADATA_PREDICTIONS,
        "comparator_pattern": "xgboost_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "family": "strict_zero_shot",
    },
    {
        "name": "debiased_vs_metadata_threshold_adapted",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "debiased_text_metadata_validation_threshold_adaptation_seed_{seed}.parquet",
        "comparator_root": METADATA_PREDICTIONS,
        "comparator_pattern": "xgboost_metadata_validation_threshold_adaptation_seed_{seed}.parquet",
        "family": "threshold_adapted",
    },
    {
        "name": "original_vs_metadata_threshold_adapted",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "bertweet_mean_metadata_validation_threshold_adaptation_seed_{seed}.parquet",
        "comparator_root": METADATA_PREDICTIONS,
        "comparator_pattern": "xgboost_metadata_validation_threshold_adaptation_seed_{seed}.parquet",
        "family": "threshold_adapted",
    },
    {
        "name": "multimodal_vs_metadata_locked_replication",
        "primary_root": TRANSFER_PREDICTIONS,
        "primary_pattern": "bertweet_mean_metadata_locked_replication_seed_{seed}.parquet",
        "comparator_root": METADATA_PREDICTIONS,
        "comparator_pattern": "xgboost_metadata_locked_replication_seed_{seed}.parquet",
        "family": "locked_replication",
    },
    {
        "name": "original_length128_vs_length64_strict",
        "primary_root": LENGTH128_PREDICTIONS,
        "primary_pattern": "bertweet_mean_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "comparator_root": TRANSFER_PREDICTIONS,
        "comparator_pattern": "bertweet_mean_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "family": "length_sensitivity",
    },
    {
        "name": "debiased_length128_vs_length64_strict",
        "primary_root": LENGTH128_PREDICTIONS,
        "primary_pattern": "debiased_text_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "comparator_root": TRANSFER_PREDICTIONS,
        "comparator_pattern": "debiased_text_metadata_strict_zero_shot_source_threshold_seed_{seed}.parquet",
        "family": "length_sensitivity",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def macro_f1_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    number = len(values)
    for rank, index in enumerate(order):
        candidate = (number - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def load_aligned_pair(comparison: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels_by_seed = []
    primary_by_seed = []
    comparator_by_seed = []
    ids_reference = None
    source_hashes = []

    for seed in SEEDS:
        primary_path = comparison["primary_root"] / comparison["primary_pattern"].format(seed=seed)
        comparator_path = comparison["comparator_root"] / comparison["comparator_pattern"].format(seed=seed)
        if not primary_path.is_file() or not comparator_path.is_file():
            raise FileNotFoundError(f"Missing paired prediction file: {primary_path} or {comparator_path}")
        primary = pd.read_parquet(primary_path).sort_values("external_id").reset_index(drop=True)
        comparator = pd.read_parquet(comparator_path).sort_values("external_id").reset_index(drop=True)
        required = {"external_id", "label", "prediction"}
        if not required.issubset(primary.columns) or not required.issubset(comparator.columns):
            raise ValueError("Prediction file lacks required aligned columns.")
        if not np.array_equal(primary["external_id"].to_numpy(), comparator["external_id"].to_numpy()):
            raise ValueError(f"External IDs are not paired for seed {seed}: {comparison['name']}")
        if not np.array_equal(primary["label"].to_numpy(), comparator["label"].to_numpy()):
            raise ValueError(f"Labels are not paired for seed {seed}: {comparison['name']}")
        ids = primary["external_id"].astype(str).to_numpy()
        if ids_reference is None:
            ids_reference = ids
        elif not np.array_equal(ids_reference, ids):
            raise ValueError(f"User ordering changes across seeds: {comparison['name']}")
        labels_by_seed.append(primary["label"].to_numpy(dtype=np.int8))
        primary_by_seed.append(primary["prediction"].to_numpy(dtype=np.int8))
        comparator_by_seed.append(comparator["prediction"].to_numpy(dtype=np.int8))
        source_hashes.extend([sha256(primary_path), sha256(comparator_path)])

    labels_matrix = np.stack(labels_by_seed)
    if not np.all(labels_matrix == labels_matrix[0]):
        raise ValueError(f"Labels change across seeds: {comparison['name']}")
    return (
        labels_matrix[0],
        np.stack(primary_by_seed),
        np.stack(comparator_by_seed),
        ids_reference,
        source_hashes,
    )


def stratified_hierarchical_bootstrap(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    differences = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    primary_means = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    number_of_seeds = primary.shape[0]

    for repetition in range(BOOTSTRAP_REPETITIONS):
        sampled_seeds = rng.integers(0, number_of_seeds, size=number_of_seeds)
        sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
        sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
        sampled_users = np.concatenate([sampled_positive, sampled_negative])
        sampled_labels = labels[sampled_users]
        primary_values = []
        comparator_values = []
        for seed_index in sampled_seeds:
            primary_values.append(
                macro_f1_from_predictions(sampled_labels, primary[seed_index, sampled_users])
            )
            comparator_values.append(
                macro_f1_from_predictions(sampled_labels, comparator[seed_index, sampled_users])
            )
        primary_mean = float(np.mean(primary_values))
        comparator_mean = float(np.mean(comparator_values))
        primary_means[repetition] = primary_mean
        differences[repetition] = primary_mean - comparator_mean
    return differences, primary_means


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    all_bootstrap_differences = {}

    for comparison in COMPARISONS:
        print(f"Bootstrapping {comparison['name']}...")
        labels, primary, comparator, external_ids, source_hashes = load_aligned_pair(comparison)
        primary_seed_f1 = np.array([
            macro_f1_from_predictions(labels, primary[index]) for index in range(len(SEEDS))
        ])
        comparator_seed_f1 = np.array([
            macro_f1_from_predictions(labels, comparator[index]) for index in range(len(SEEDS))
        ])
        observed_seed_differences = primary_seed_f1 - comparator_seed_f1
        observed_difference = float(observed_seed_differences.mean())
        bootstrap_differences, _ = stratified_hierarchical_bootstrap(
            labels, primary, comparator, rng
        )
        all_bootstrap_differences[comparison["name"]] = bootstrap_differences
        lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])
        probability_superior = float(np.mean(bootstrap_differences > 0))
        p_lower = (np.sum(bootstrap_differences <= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
        p_upper = (np.sum(bootstrap_differences >= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
        bootstrap_p = min(1.0, 2 * min(p_lower, p_upper))
        try:
            wilcoxon_p = float(
                wilcoxon(
                    observed_seed_differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="exact",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0

        rows.append({
            "comparison": comparison["name"],
            "family": comparison["family"],
            "metric": "macro_f1",
            "test_users": len(labels),
            "bot_users": int(labels.sum()),
            "human_users": int(len(labels) - labels.sum()),
            "seeds": len(SEEDS),
            "primary_mean": float(primary_seed_f1.mean()),
            "comparator_mean": float(comparator_seed_f1.mean()),
            "mean_observed_difference": observed_difference,
            "bootstrap_ci_lower_95": float(lower),
            "bootstrap_ci_upper_95": float(upper),
            "bootstrap_p": bootstrap_p,
            "probability_primary_superior": probability_superior,
            "wilcoxon_p": wilcoxon_p,
            "source_prediction_hashes": "|".join(source_hashes),
        })

    results = pd.DataFrame(rows)
    inferential_mask = ~results["family"].eq("length_sensitivity")
    adjusted_bootstrap = holm_adjust(results.loc[inferential_mask, "bootstrap_p"].tolist())
    adjusted_wilcoxon = holm_adjust(results.loc[inferential_mask, "wilcoxon_p"].tolist())
    results["bootstrap_p_holm"] = results["bootstrap_p"]
    results["wilcoxon_p_holm"] = results["wilcoxon_p"]
    results.loc[inferential_mask, "bootstrap_p_holm"] = adjusted_bootstrap
    results.loc[inferential_mask, "wilcoxon_p_holm"] = adjusted_wilcoxon
    results.to_csv(OUTPUT_ROOT / "external_paired_inference.csv", index=False)

    bootstrap_frame = pd.DataFrame(all_bootstrap_differences)
    bootstrap_frame.to_parquet(OUTPUT_ROOT / "external_bootstrap_distributions.parquet", index=False)

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22 official test split",
        "method": "paired hierarchical class-stratified bootstrap over seeds and users",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "multiple_comparison_control": "Holm adjustment within six inferential model comparisons; length sensitivities descriptive",
        "wilcoxon_role": "seed-level sensitivity; exact two-sided test has minimum p=0.0625 with five all-positive nonzero differences",
        "external_test_used_for_model_selection": False,
        "comparisons": [comparison["name"] for comparison in COMPARISONS],
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "external_paired_inference.csv": sha256(OUTPUT_ROOT / "external_paired_inference.csv"),
            "external_bootstrap_distributions.parquet": sha256(
                OUTPUT_ROOT / "external_bootstrap_distributions.parquet"
            ),
        },
    }
    (OUTPUT_ROOT / "external_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL STATISTICAL INFERENCE: PASS")
    print(
        results[[
            "comparison", "primary_mean", "comparator_mean", "mean_observed_difference",
            "bootstrap_ci_lower_95", "bootstrap_ci_upper_95", "bootstrap_p_holm",
            "probability_primary_superior", "wilcoxon_p_holm",
        ]].to_string(index=False)
    )
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
