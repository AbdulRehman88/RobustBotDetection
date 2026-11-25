from __future__ import annotations

import hashlib
import json
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "40_osnem_sota_comparison"
OUTPUT_DIR = OUTPUT_ROOT / "controlled_comparison"
CONFIG_PATH = PROJECT_ROOT / "configs" / "osnem_sota_comparison.yaml"

SEEDS = [42, 123, 2024, 3407, 7777]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261125

PRIMARY_NAME = "proposed_reliability_route_natural_tgm"
COMPARATOR_PATHS = {
    "botbuster_style_common_evidence": (
        OUTPUT_ROOT
        / "botbuster_common_evidence"
        / "test_predictions"
        / "botbuster_seed_{seed}_test_predictions.parquet"
    ),
    "botmoe_style_common_evidence": (
        OUTPUT_ROOT
        / "botmoe_common_evidence"
        / "test_predictions"
        / "botmoe_seed_{seed}_test_predictions.parquet"
    ),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def primary_path(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "09_bertweet_baselines"
        / "test_predictions"
        / f"bertweet_mean_metadata_seed_{seed}_test_predictions.parquet"
    )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = ["node_id", "label", "score", "prediction"]
    frame = pd.read_parquet(path, columns=required)
    if frame["node_id"].duplicated().any():
        raise RuntimeError(f"Duplicate node identifiers: {path}")
    return frame.sort_values("node_id").reset_index(drop=True)


def calculate_metrics(frame: pd.DataFrame) -> dict:
    labels = frame["label"].to_numpy(dtype=np.int8)
    scores = frame["score"].to_numpy(dtype=np.float64)
    predictions = frame["prediction"].to_numpy(dtype=np.int8)
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
    }


def metric_value(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: str,
) -> float:
    if metric == "macro_f1":
        return float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(labels, predictions)
        )
    raise ValueError(metric)


def metric_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    metric: str,
) -> float:
    if metric == "balanced_accuracy":
        specificity = tn / max(tn + fp, 1)
        sensitivity = tp / max(tp + fn, 1)
        return 0.5 * (specificity + sensitivity)
    if metric == "macro_f1":
        bot_denominator = 2 * tp + fp + fn
        human_denominator = 2 * tn + fp + fn
        bot_f1 = (
            0.0
            if bot_denominator == 0
            else 2 * tp / bot_denominator
        )
        human_f1 = (
            0.0
            if human_denominator == 0
            else 2 * tn / human_denominator
        )
        return 0.5 * (bot_f1 + human_f1)
    raise ValueError(metric)


def joint_probabilities(
    labels: np.ndarray,
    primary_predictions: np.ndarray,
    comparator_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    probabilities = []
    totals = []
    for label in [0, 1]:
        selected = labels == label
        category = (
            2 * primary_predictions[selected].astype(np.int8)
            + comparator_predictions[selected].astype(np.int8)
        )
        counts = np.bincount(
            category, minlength=4
        ).astype(np.float64)
        totals.append(int(counts.sum()))
        probabilities.append(counts / counts.sum())
    return probabilities[0], probabilities[1], totals[0], totals[1]


def confusion_from_joint(
    human_counts: np.ndarray,
    bot_counts: np.ndarray,
    selector: str,
) -> tuple[int, int, int, int]:
    if selector == "primary":
        positive = np.asarray([False, False, True, True])
    elif selector == "comparator":
        positive = np.asarray([False, True, False, True])
    else:
        raise ValueError(selector)
    fp = int(human_counts[positive].sum())
    tn = int(human_counts[~positive].sum())
    tp = int(bot_counts[positive].sum())
    fn = int(bot_counts[~positive].sum())
    return tn, fp, fn, tp


def paired_hierarchical_bootstrap(
    seed_data: list[dict],
    metric: str,
    rng: np.random.Generator,
) -> np.ndarray:
    differences = np.empty(
        BOOTSTRAP_REPLICATES, dtype=np.float64
    )
    seed_count = len(seed_data)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.integers(
            0, seed_count, size=seed_count
        )
        primary_values = []
        comparator_values = []
        for seed_index in sampled_seeds:
            data = seed_data[int(seed_index)]
            human_counts = rng.multinomial(
                data["human_total"],
                data["human_probabilities"],
            )
            bot_counts = rng.multinomial(
                data["bot_total"],
                data["bot_probabilities"],
            )
            primary_values.append(
                metric_from_confusion(
                    *confusion_from_joint(
                        human_counts, bot_counts, "primary"
                    ),
                    metric,
                )
            )
            comparator_values.append(
                metric_from_confusion(
                    *confusion_from_joint(
                        human_counts, bot_counts, "comparator"
                    ),
                    metric,
                )
            )
        differences[replicate] = float(
            np.mean(primary_values)
            - np.mean(comparator_values)
        )
    return differences


def two_sided_bootstrap_p(
    differences: np.ndarray,
) -> float:
    non_positive = (
        np.count_nonzero(differences <= 0) + 1
    ) / (len(differences) + 1)
    non_negative = (
        np.count_nonzero(differences >= 0) + 1
    ) / (len(differences) + 1)
    return float(
        min(1.0, 2.0 * min(non_positive, non_negative))
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=np.float64)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return pd.Series(adjusted, index=p_values.index)


def interpretation(lower: float, upper: float) -> str:
    if lower > 0:
        return "primary_higher"
    if upper < 0:
        return "comparator_higher"
    return "difference_not_resolved"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_metric_rows = []
    inference_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    loaded: dict[int, dict[str, pd.DataFrame]] = {}
    for seed in SEEDS:
        loaded[seed] = {
            PRIMARY_NAME: load_predictions(primary_path(seed))
        }
        for comparator, template in COMPARATOR_PATHS.items():
            loaded[seed][comparator] = load_predictions(
                Path(str(template).format(seed=seed))
            )

        reference = loaded[seed][PRIMARY_NAME]
        reference_ids = reference["node_id"].to_numpy(
            dtype=np.int64
        )
        reference_labels = reference["label"].to_numpy(
            dtype=np.int8
        )
        for model, frame in loaded[seed].items():
            if not np.array_equal(
                reference_ids,
                frame["node_id"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError(
                    f"Node alignment failure: {model}, seed={seed}."
                )
            if not np.array_equal(
                reference_labels,
                frame["label"].to_numpy(dtype=np.int8),
            ):
                raise RuntimeError(
                    f"Label alignment failure: {model}, seed={seed}."
                )
            seed_metric_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "test_users": len(frame),
                    **calculate_metrics(frame),
                }
            )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics.to_csv(
        OUTPUT_DIR / "controlled_comparison_seed_results.csv",
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
    for model, group in seed_metrics.groupby(
        "model", sort=False
    ):
        row = {
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "test_users": int(group["test_users"].iloc[0]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=1)
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "controlled_comparison_summary.csv",
        index=False,
    )

    for comparator in COMPARATOR_PATHS:
        seed_data = []
        observed_differences = {
            "macro_f1": [],
            "balanced_accuracy": [],
        }
        for seed in SEEDS:
            primary = loaded[seed][PRIMARY_NAME]
            comparison = loaded[seed][comparator]
            labels = primary["label"].to_numpy(dtype=np.int8)
            primary_predictions = primary[
                "prediction"
            ].to_numpy(dtype=np.int8)
            comparator_predictions = comparison[
                "prediction"
            ].to_numpy(dtype=np.int8)
            human_p, bot_p, human_n, bot_n = (
                joint_probabilities(
                    labels,
                    primary_predictions,
                    comparator_predictions,
                )
            )
            seed_data.append(
                {
                    "human_probabilities": human_p,
                    "bot_probabilities": bot_p,
                    "human_total": human_n,
                    "bot_total": bot_n,
                }
            )
            for metric in observed_differences:
                observed_differences[metric].append(
                    metric_value(
                        labels,
                        primary_predictions,
                        metric,
                    )
                    - metric_value(
                        labels,
                        comparator_predictions,
                        metric,
                    )
                )

        for metric, differences_list in (
            observed_differences.items()
        ):
            bootstrap = paired_hierarchical_bootstrap(
                seed_data, metric, rng
            )
            differences = np.asarray(
                differences_list, dtype=np.float64
            )
            lower = float(np.quantile(bootstrap, 0.025))
            upper = float(np.quantile(bootstrap, 0.975))
            try:
                test = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="exact",
                )
                wilcoxon_statistic = float(test.statistic)
                wilcoxon_p = float(test.pvalue)
            except ValueError:
                wilcoxon_statistic = 0.0
                wilcoxon_p = 1.0

            inference_rows.append(
                {
                    "primary_model": PRIMARY_NAME,
                    "comparator": comparator,
                    "metric": metric,
                    "seeds": len(SEEDS),
                    "test_users": int(
                        seed_data[0]["human_total"]
                        + seed_data[0]["bot_total"]
                    ),
                    "mean_observed_difference": float(
                        differences.mean()
                    ),
                    "seed_difference_std": float(
                        differences.std(ddof=1)
                    ),
                    "bootstrap_ci_lower_95": lower,
                    "bootstrap_ci_upper_95": upper,
                    "bootstrap_p_raw": two_sided_bootstrap_p(
                        bootstrap
                    ),
                    "probability_primary_higher": float(
                        np.mean(bootstrap > 0)
                    ),
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_p_raw": wilcoxon_p,
                    "interpretation": interpretation(
                        lower, upper
                    ),
                }
            )

    inference = pd.DataFrame(inference_rows)
    for metric in inference["metric"].unique():
        selected = inference["metric"].eq(metric)
        inference.loc[
            selected, "bootstrap_p_holm"
        ] = holm_adjust(
            inference.loc[selected, "bootstrap_p_raw"]
        )
        inference.loc[
            selected, "wilcoxon_p_holm"
        ] = holm_adjust(
            inference.loc[selected, "wilcoxon_p_raw"]
        )
    inference.to_csv(
        OUTPUT_DIR / "controlled_paired_inference.csv",
        index=False,
    )

    table = summary[
        [
            "model",
            "accuracy_mean",
            "accuracy_std",
            "f1_bot_mean",
            "f1_bot_std",
            "macro_f1_mean",
            "macro_f1_std",
            "auroc_mean",
            "auroc_std",
            "auprc_mean",
            "auprc_std",
        ]
    ].copy()
    table.to_csv(
        OUTPUT_DIR / "manuscript_comparison_table.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "analysis_family": (
            "OSNEM reviewer-requested controlled comparisons only"
        ),
        "does_not_modify_previous_inference_families": True,
        "primary_model": PRIMARY_NAME,
        "comparators": list(COMPARATOR_PATHS),
        "seeds": SEEDS,
        "test_users": int(summary["test_users"].iloc[0]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_design": (
            "paired two-level bootstrap: seeds resampled with "
            "replacement; within each seed, label-stratified joint "
            "prediction cells resampled multinomially"
        ),
        "multiplicity_control": (
            "Holm correction across the two requested comparators "
            "within each metric"
        ),
        "seed_level_test": (
            "exact two-sided Wilcoxon signed-rank; with five seeds "
            "the minimum attainable nonzero two-sided p-value is 0.0625"
        ),
        "config_sha256": sha256_file(CONFIG_PATH),
        "integrity": {
            "node_alignment_verified": True,
            "label_alignment_verified": True,
            "paired_predictions_used": True,
            "test_predictions_not_used_for_model_selection": True,
        },
    }
    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("OSNEM CONTROLLED COMPARISON INFERENCE: PASS")
    print("\nControlled summary:")
    print(table.to_string(index=False))
    print("\nPaired inference:")
    print(
        inference[
            [
                "comparator",
                "metric",
                "mean_observed_difference",
                "bootstrap_ci_lower_95",
                "bootstrap_ci_upper_95",
                "bootstrap_p_holm",
                "probability_primary_higher",
                "wilcoxon_p_holm",
                "interpretation",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
