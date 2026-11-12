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
TRANSFER_ROOT = PROJECT_ROOT / "outputs" / "32_twibot20_external_transfer" / "predictions"
METADATA_ROOT = PROJECT_ROOT / "outputs" / "33_twibot20_external_metadata" / "predictions"
GRAPH_ROOT = PROJECT_ROOT / "outputs" / "36_twibot20_external_graph_controls" / "predictions"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "37_twibot20_graph_inference"

SEEDS = [42, 123, 2024, 3407, 7777]
REPETITIONS = 10_000
RNG_SEED = 20261030

COMPARISONS = [
    (
        "metadata_graph_vs_metadata",
        GRAPH_ROOT, "xgboost_metadata_graph_topology_seed_{seed}.parquet",
        METADATA_ROOT, "xgboost_metadata_locked_replication_seed_{seed}.parquet",
    ),
    (
        "multimodal_vs_metadata_graph",
        TRANSFER_ROOT, "bertweet_mean_metadata_locked_replication_seed_{seed}.parquet",
        GRAPH_ROOT, "xgboost_metadata_graph_topology_seed_{seed}.parquet",
    ),
    (
        "metadata_vs_graph_only",
        METADATA_ROOT, "xgboost_metadata_locked_replication_seed_{seed}.parquet",
        GRAPH_ROOT, "xgboost_graph_topology_seed_{seed}.parquet",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def load_pair(primary_root: Path, primary_pattern: str, comparator_root: Path, comparator_pattern: str):
    primary_rows, comparator_rows = [], []
    labels_reference = None
    ids_reference = None
    hashes = []
    for seed in SEEDS:
        primary_path = primary_root / primary_pattern.format(seed=seed)
        comparator_path = comparator_root / comparator_pattern.format(seed=seed)
        primary = pd.read_parquet(primary_path).sort_values("external_id").reset_index(drop=True)
        comparator = pd.read_parquet(comparator_path).sort_values("external_id").reset_index(drop=True)
        ids = primary["external_id"].astype(str).to_numpy()
        labels = primary["label"].to_numpy(dtype=np.int8)
        if not np.array_equal(ids, comparator["external_id"].astype(str).to_numpy()):
            raise ValueError(f"Unaligned user IDs for seed {seed}")
        if not np.array_equal(labels, comparator["label"].to_numpy(dtype=np.int8)):
            raise ValueError(f"Unaligned labels for seed {seed}")
        if ids_reference is None:
            ids_reference, labels_reference = ids, labels
        elif not np.array_equal(ids_reference, ids) or not np.array_equal(labels_reference, labels):
            raise ValueError("User ordering or labels change across seeds")
        primary_rows.append(primary["prediction"].to_numpy(dtype=np.int8))
        comparator_rows.append(comparator["prediction"].to_numpy(dtype=np.int8))
        hashes.extend([sha256(primary_path), sha256(comparator_path)])
    return labels_reference, np.stack(primary_rows), np.stack(comparator_rows), hashes


def bootstrap(labels: np.ndarray, primary: np.ndarray, comparator: np.ndarray, rng: np.random.Generator):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    distribution = np.empty(REPETITIONS, dtype=np.float64)
    for repetition in range(REPETITIONS):
        seed_sample = rng.integers(0, len(SEEDS), size=len(SEEDS))
        user_sample = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        sampled_labels = labels[user_sample]
        differences = [
            macro_f1(sampled_labels, primary[seed, user_sample])
            - macro_f1(sampled_labels, comparator[seed, user_sample])
            for seed in seed_sample
        ]
        distribution[repetition] = np.mean(differences)
    return distribution


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    distributions = {}

    for name, primary_root, primary_pattern, comparator_root, comparator_pattern in COMPARISONS:
        print(f"Bootstrapping {name}...")
        labels, primary, comparator, hashes = load_pair(
            primary_root, primary_pattern, comparator_root, comparator_pattern
        )
        primary_seed = np.array([macro_f1(labels, row) for row in primary])
        comparator_seed = np.array([macro_f1(labels, row) for row in comparator])
        seed_differences = primary_seed - comparator_seed
        distribution = bootstrap(labels, primary, comparator, rng)
        distributions[name] = distribution
        lower, upper = np.quantile(distribution, [0.025, 0.975])
        p_lower = (np.sum(distribution <= 0) + 1) / (REPETITIONS + 1)
        p_upper = (np.sum(distribution >= 0) + 1) / (REPETITIONS + 1)
        bootstrap_p = min(1.0, 2 * min(p_lower, p_upper))
        try:
            wilcoxon_p = float(wilcoxon(seed_differences, method="exact").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append({
            "comparison": name,
            "metric": "macro_f1",
            "test_users": len(labels),
            "seeds": len(SEEDS),
            "primary_mean": primary_seed.mean(),
            "comparator_mean": comparator_seed.mean(),
            "mean_observed_difference": seed_differences.mean(),
            "bootstrap_ci_lower_95": lower,
            "bootstrap_ci_upper_95": upper,
            "bootstrap_p": bootstrap_p,
            "probability_primary_superior": np.mean(distribution > 0),
            "wilcoxon_p": wilcoxon_p,
            "source_hashes": "|".join(hashes),
        })

    results = pd.DataFrame(rows)
    results["bootstrap_p_holm"] = holm(results["bootstrap_p"].tolist())
    results["wilcoxon_p_holm"] = holm(results["wilcoxon_p"].tolist())
    results.to_csv(OUTPUT_ROOT / "external_graph_paired_inference.csv", index=False)
    pd.DataFrame(distributions).to_parquet(
        OUTPUT_ROOT / "external_graph_bootstrap_distributions.parquet", index=False
    )

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22 official test split",
        "protocol": "locked within-dataset replication",
        "method": "paired hierarchical class-stratified bootstrap over users and seeds",
        "bootstrap_repetitions": REPETITIONS,
        "bootstrap_seed": RNG_SEED,
        "multiple_comparison_control": "Holm across three graph-control comparisons",
        "external_test_used_for_selection": False,
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "external_graph_paired_inference.csv": sha256(
                OUTPUT_ROOT / "external_graph_paired_inference.csv"
            ),
            "external_graph_bootstrap_distributions.parquet": sha256(
                OUTPUT_ROOT / "external_graph_bootstrap_distributions.parquet"
            ),
        },
    }
    (OUTPUT_ROOT / "external_graph_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 GRAPH-CONTROL INFERENCE: PASS")
    print(results[[
        "comparison", "primary_mean", "comparator_mean", "mean_observed_difference",
        "bootstrap_ci_lower_95", "bootstrap_ci_upper_95", "bootstrap_p_holm",
        "probability_primary_superior", "wilcoxon_p_holm",
    ]].to_string(index=False))
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
