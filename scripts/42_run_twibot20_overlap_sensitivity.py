from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
DATA_ROOT = PROJECT_ROOT.parent
TWIBOT22_LABEL = DATA_ROOT / "label.csv"
TWIBOT22_SPLIT = DATA_ROOT / "split.csv"
TWIBOT20_ROOT = DATA_ROOT / "TwiBot-20-Format22" / "Twibot-20"

TRANSFER_DIR = PROJECT_ROOT / "outputs" / "32_twibot20_external_transfer" / "predictions"
METADATA_DIR = PROJECT_ROOT / "outputs" / "33_twibot20_external_metadata" / "predictions"
LENGTH128_DIR = PROJECT_ROOT / "outputs" / "34_twibot20_length_sensitivity" / "predictions_128"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "36_twibot20_external_graph_controls" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "39_twibot20_overlap_sensitivity"

SEEDS = (42, 123, 2024, 3407, 7777)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261113
PROTOCOLS = (
    "strict_zero_shot_source_threshold",
    "validation_threshold_adaptation",
    "locked_replication",
)


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    scores = frame["score"].to_numpy(dtype=np.float64)
    predictions = frame["prediction"].to_numpy(dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_bot": float(precision_score(labels, predictions, zero_division=0)),
        "recall_bot": float(recall_score(labels, predictions, zero_division=0)),
        "specificity_human": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_bot": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def fast_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(np.int8, copy=False)
    predictions = predictions.astype(np.int8, copy=False)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    bot_denominator = 2 * tp + fp + fn
    human_denominator = 2 * tn + fp + fn
    bot_f1 = 2 * tp / bot_denominator if bot_denominator else 0.0
    human_f1 = 2 * tn / human_denominator if human_denominator else 0.0
    return float((bot_f1 + human_f1) / 2.0)


def parse_standard_prediction(path: Path, source: str) -> tuple[str, str, int, int]:
    stem = path.stem
    match = re.search(r"_seed_(\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot parse seed from {path}")
    seed = int(match.group(1))
    key = stem[: match.start()]

    if source == "graph":
        return key, "locked_within_twibot20_replication", seed, 64

    for protocol in PROTOCOLS:
        suffix = f"_{protocol}"
        if key.endswith(suffix):
            model = key[: -len(suffix)]
            normalized_protocol = (
                "locked_within_twibot20_replication"
                if protocol == "locked_replication"
                else protocol
            )
            max_length = 128 if source == "length128" else 64
            return model, normalized_protocol, seed, max_length
    raise ValueError(f"Cannot parse model/protocol from {path}")


def load_audited_ids() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(TWIBOT22_LABEL, dtype={"id": "string"}).merge(
        pd.read_csv(TWIBOT22_SPLIT, dtype={"id": "string"}),
        on="id",
        validate="one_to_one",
    )
    external = pd.read_csv(TWIBOT20_ROOT / "label.csv", dtype={"id": "string"}).merge(
        pd.read_csv(TWIBOT20_ROOT / "split.csv", dtype={"id": "string"}),
        on="id",
        validate="one_to_one",
    )
    overlap = external.merge(source, on="id", suffixes=("_20", "_22"), validate="one_to_one")
    return source, external, overlap


def load_filtered_predictions(
    source_ids: set[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str, int, int], pd.DataFrame]]:
    specifications = (
        (TRANSFER_DIR, "transfer"),
        (METADATA_DIR, "metadata"),
        (LENGTH128_DIR, "length128"),
        (GRAPH_DIR, "graph"),
    )
    rows: list[dict] = []
    frames: dict[tuple[str, str, int, int], pd.DataFrame] = {}

    for directory, source in specifications:
        if not directory.exists():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.parquet")):
            model, protocol, seed, max_length = parse_standard_prediction(path, source)
            frame = pd.read_parquet(path)
            required = {"external_id", "label", "score", "prediction"}
            missing = required.difference(frame.columns)
            if missing:
                raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

            frame = frame.copy()
            frame["external_id"] = frame["external_id"].astype("string")
            if frame["external_id"].duplicated().any():
                raise RuntimeError(f"Duplicate external identifiers in {path}")
            filtered = frame.loc[~frame["external_id"].isin(source_ids)].copy()
            filtered = filtered.sort_values("external_id").reset_index(drop=True)
            if len(filtered) != 885:
                raise RuntimeError(
                    f"Expected 885 disjoint users in {path}, observed {len(filtered)}"
                )

            key = (model, protocol, seed, max_length)
            if key in frames:
                raise RuntimeError(f"Duplicate prediction key: {key}")
            frames[key] = filtered
            rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "seed": seed,
                    "max_length": max_length,
                    "test_users": len(filtered),
                    "bots": int(filtered["label"].sum()),
                    "humans": int(len(filtered) - filtered["label"].sum()),
                    **classification_metrics(filtered),
                }
            )

    return pd.DataFrame(rows), frames


def summarize(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = (
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
    )
    rows: list[dict] = []
    group_columns = ["model", "protocol", "max_length"]
    for keys, group in seed_results.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "seeds": int(group["seed"].nunique()),
                "test_users": int(group["test_users"].iloc[0]),
                "bots": int(group["bots"].iloc[0]),
                "humans": int(group["humans"].iloc[0]),
            }
        )
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def aligned_arrays(
    frames: dict[tuple[str, str, int, int], pd.DataFrame],
    model: str,
    protocol: str,
    max_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = frames[(model, protocol, seed, max_length)]
    return (
        frame["external_id"].astype(str).to_numpy(),
        frame["label"].to_numpy(dtype=np.int8),
        frame["prediction"].to_numpy(dtype=np.int8),
    )


def paired_bootstrap(
    frames: dict[tuple[str, str, int, int], pd.DataFrame],
    primary: tuple[str, str, int],
    comparator: tuple[str, str, int],
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    seed_payload = []
    observed_seed_differences = []
    for seed in SEEDS:
        primary_ids, labels, primary_predictions = aligned_arrays(
            frames, primary[0], primary[1], primary[2], seed
        )
        comparator_ids, comparator_labels, comparator_predictions = aligned_arrays(
            frames, comparator[0], comparator[1], comparator[2], seed
        )
        if not np.array_equal(primary_ids, comparator_ids):
            raise RuntimeError(f"Identifier mismatch for seed {seed}: {primary} vs {comparator}")
        if not np.array_equal(labels, comparator_labels):
            raise RuntimeError(f"Label mismatch for seed {seed}: {primary} vs {comparator}")
        human = np.flatnonzero(labels == 0)
        bot = np.flatnonzero(labels == 1)
        seed_payload.append((labels, primary_predictions, comparator_predictions, human, bot))
        observed_seed_differences.append(
            fast_macro_f1(labels, primary_predictions)
            - fast_macro_f1(labels, comparator_predictions)
        )

    bootstrap_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seed_positions = rng.integers(0, len(SEEDS), size=len(SEEDS))
        differences = []
        for position in sampled_seed_positions:
            labels, primary_predictions, comparator_predictions, human, bot = seed_payload[position]
            sampled = np.concatenate(
                [
                    rng.choice(human, size=len(human), replace=True),
                    rng.choice(bot, size=len(bot), replace=True),
                ]
            )
            sampled_labels = labels[sampled]
            differences.append(
                fast_macro_f1(sampled_labels, primary_predictions[sampled])
                - fast_macro_f1(sampled_labels, comparator_predictions[sampled])
            )
        bootstrap_differences[replicate] = float(np.mean(differences))

    lower, upper = np.percentile(bootstrap_differences, [2.5, 97.5])
    probability = float(
        np.mean(bootstrap_differences > 0)
        + 0.5 * np.mean(bootstrap_differences == 0)
    )
    lower_tail = (np.sum(bootstrap_differences <= 0) + 1) / (BOOTSTRAP_REPLICATES + 1)
    upper_tail = (np.sum(bootstrap_differences >= 0) + 1) / (BOOTSTRAP_REPLICATES + 1)
    p_value = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
    try:
        wilcoxon_p = float(
            wilcoxon(observed_seed_differences, alternative="two-sided").pvalue
        )
    except ValueError:
        wilcoxon_p = 1.0

    return {
        "mean_observed_difference": float(np.mean(observed_seed_differences)),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "bootstrap_p": p_value,
        "probability_primary_superior": probability,
        "wilcoxon_p": wilcoxon_p,
        "seed_differences": [float(value) for value in observed_seed_differences],
    }


def holm_adjust(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty(len(raw), dtype=float)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def run_inference(
    frames: dict[tuple[str, str, int, int], pd.DataFrame]
) -> pd.DataFrame:
    comparisons = (
        (
            "debiased_vs_original_strict_disjoint",
            ("debiased_text_metadata", "strict_zero_shot_source_threshold", 64),
            ("bertweet_mean_metadata", "strict_zero_shot_source_threshold", 64),
        ),
        (
            "debiased_vs_metadata_strict_disjoint",
            ("debiased_text_metadata", "strict_zero_shot_source_threshold", 64),
            ("xgboost_metadata", "strict_zero_shot_source_threshold", 64),
        ),
        (
            "original_vs_metadata_strict_disjoint",
            ("bertweet_mean_metadata", "strict_zero_shot_source_threshold", 64),
            ("xgboost_metadata", "strict_zero_shot_source_threshold", 64),
        ),
        (
            "debiased_vs_metadata_adapted_disjoint",
            ("debiased_text_metadata", "validation_threshold_adaptation", 64),
            ("xgboost_metadata", "validation_threshold_adaptation", 64),
        ),
        (
            "original_vs_metadata_adapted_disjoint",
            ("bertweet_mean_metadata", "validation_threshold_adaptation", 64),
            ("xgboost_metadata", "validation_threshold_adaptation", 64),
        ),
        (
            "multimodal_vs_metadata_replication_disjoint",
            ("bertweet_mean_metadata", "locked_within_twibot20_replication", 64),
            ("xgboost_metadata", "locked_within_twibot20_replication", 64),
        ),
        (
            "multimodal_vs_metadata_graph_replication_disjoint",
            ("bertweet_mean_metadata", "locked_within_twibot20_replication", 64),
            ("xgboost_metadata_graph_topology", "locked_within_twibot20_replication", 64),
        ),
        (
            "metadata_graph_vs_metadata_replication_disjoint",
            ("xgboost_metadata_graph_topology", "locked_within_twibot20_replication", 64),
            ("xgboost_metadata", "locked_within_twibot20_replication", 64),
        ),
        (
            "metadata_vs_graph_replication_disjoint",
            ("xgboost_metadata", "locked_within_twibot20_replication", 64),
            ("xgboost_graph_topology", "locked_within_twibot20_replication", 64),
        ),
        (
            "original_length128_vs_length64_strict_disjoint",
            ("bertweet_mean_metadata", "strict_zero_shot_source_threshold", 128),
            ("bertweet_mean_metadata", "strict_zero_shot_source_threshold", 64),
        ),
        (
            "debiased_length128_vs_length64_strict_disjoint",
            ("debiased_text_metadata", "strict_zero_shot_source_threshold", 128),
            ("debiased_text_metadata", "strict_zero_shot_source_threshold", 64),
        ),
    )

    rows = []
    for number, (name, primary, comparator) in enumerate(comparisons):
        print(f"Bootstrapping {name}...")
        rng = np.random.default_rng(BOOTSTRAP_SEED + number)
        result = paired_bootstrap(frames, primary, comparator, rng)
        rows.append(
            {
                "comparison": name,
                "primary_model": primary[0],
                "primary_protocol": primary[1],
                "primary_max_length": primary[2],
                "comparator_model": comparator[0],
                "comparator_protocol": comparator[1],
                "comparator_max_length": comparator[2],
                **result,
            }
        )
    inference = pd.DataFrame(rows)
    inference["bootstrap_p_holm"] = holm_adjust(inference["bootstrap_p"])
    inference["wilcoxon_p_holm"] = holm_adjust(inference["wilcoxon_p"])
    return inference


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source, external, overlap = load_audited_ids()
    source_ids = set(source["id"].dropna().astype(str))
    external_test = external.loc[external["split"].eq("test")].copy()
    test_overlap = overlap.loc[overlap["split_20"].eq("test")].copy()
    disjoint_test = external_test.loc[~external_test["id"].isin(source_ids)].copy()

    overlap_by_split = pd.crosstab(
        overlap["split_20"], overlap["split_22"], margins=True
    )
    overlap_by_split.to_csv(OUTPUT_DIR / "split_overlap_crosstab.csv")
    test_overlap[["id", "label_20", "split_20", "label_22", "split_22"]].to_csv(
        OUTPUT_DIR / "twibot20_test_overlap_records.csv", index=False
    )

    audit = {
        "status": "PASS",
        "twibot22_labeled_ids": int(source["id"].nunique()),
        "twibot20_labeled_ids": int(external["id"].nunique()),
        "all_labeled_overlap": int(len(overlap)),
        "all_overlap_label_agreement": float(
            (overlap["label_20"] == overlap["label_22"]).mean()
        ),
        "twibot20_test_users": int(len(external_test)),
        "twibot20_test_overlap": int(len(test_overlap)),
        "twibot20_test_disjoint": int(len(disjoint_test)),
        "twibot20_test_overlap_percent": float(100 * len(test_overlap) / len(external_test)),
        "twibot20_test_overlap_label_agreement": float(
            (test_overlap["label_20"] == test_overlap["label_22"]).mean()
        ),
        "exclusion_rule": "Remove every TwiBot-20 test identifier appearing anywhere in the labeled TwiBot-22 release.",
        "model_policy": "Reuse locked predictions; do not retrain models or reselect thresholds.",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (OUTPUT_DIR / "overlap_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    seed_results, frames = load_filtered_predictions(source_ids)
    seed_results.to_csv(OUTPUT_DIR / "disjoint_seed_results.csv", index=False)
    summary = summarize(seed_results)
    summary.to_csv(OUTPUT_DIR / "disjoint_summary.csv", index=False)
    inference = run_inference(frames)
    inference.to_csv(OUTPUT_DIR / "disjoint_paired_inference.csv", index=False)

    print("\nTWIBOT-20 OVERLAP-EXCLUDED SENSITIVITY: PASS")
    print(json.dumps(audit, indent=2))
    display = summary[
        [
            "model",
            "protocol",
            "max_length",
            "seeds",
            "test_users",
            "macro_f1_mean",
            "macro_f1_std",
            "auroc_mean",
            "auprc_mean",
        ]
    ]
    print("\nDisjoint-subset performance:")
    print(display.to_string(index=False))
    print("\nDisjoint-subset paired inference:")
    print(
        inference[
            [
                "comparison",
                "mean_observed_difference",
                "ci_lower_95",
                "ci_upper_95",
                "bootstrap_p_holm",
                "probability_primary_superior",
                "wilcoxon_p_holm",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
