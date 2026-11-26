from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "40_osnem_sota_comparison"
OUTPUT_DIR = OUTPUT_ROOT / "forced_availability_comparison"
ROUTER_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "21_reliability_router"
    / "test_predictions"
)
BOTMOE_DIR = (
    OUTPUT_ROOT
    / "botmoe_forced_availability"
    / "test_predictions"
)
BOTMOE_MANIFEST = (
    OUTPUT_ROOT
    / "botmoe_forced_availability"
    / "manifest.json"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "osnem_sota_comparison.yaml"

SEEDS = [42, 123, 2024, 3407, 7777]
SCENARIOS = ["T", "G", "M", "TG", "TM", "GM", "TGM"]
METRICS = ["macro_f1", "balanced_accuracy"]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261125
PRIMARY_NAME = "validation_selected_reliability_router"
COMPARATOR_NAME = "botmoe_style_common_evidence"


def sha256_file(
    path: Path, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def router_path(seed: int, scenario: str) -> Path:
    return ROUTER_DIR / f"seed_{seed}_{scenario}.parquet"


def botmoe_path(seed: int, scenario: str) -> Path:
    return (
        BOTMOE_DIR / f"botmoe_seed_{seed}_{scenario}.parquet"
    )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = [
        "node_id",
        "label",
        "prediction",
        "text_available",
        "graph_available",
        "metadata_available",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame["node_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate node identifiers in {path}."
        )
    if not frame["prediction"].isin([0, 1]).all():
        raise RuntimeError(
            f"Nonbinary predictions in {path}."
        )
    return frame.sort_values("node_id").reset_index(drop=True)


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
) -> dict:
    record: dict[str, np.ndarray | int] = {}
    for label, prefix in [(0, "human"), (1, "bot")]:
        selected = labels == label
        category = (
            2
            * primary_predictions[selected].astype(np.int8)
            + comparator_predictions[selected].astype(np.int8)
        )
        counts = np.bincount(
            category, minlength=4
        ).astype(np.float64)
        total = int(counts.sum())
        if total == 0:
            raise RuntimeError(
                f"No observations for label {label}."
            )
        record[f"{prefix}_probabilities"] = counts / total
        record[f"{prefix}_total"] = total
    return record


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


def paired_scenario_bootstrap(
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
                int(data["human_total"]),
                data["human_probabilities"],
            )
            bot_counts = rng.multinomial(
                int(data["bot_total"]),
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


def encode_joint_patterns(
    primary: np.ndarray,
    comparator: np.ndarray,
) -> np.ndarray:
    if primary.shape != comparator.shape:
        raise RuntimeError(
            "Robust pattern matrices have unequal shapes."
        )
    if (
        primary.ndim != 2
        or primary.shape[1] < 1
        or primary.shape[1] > 8
    ):
        raise RuntimeError(
            "Robust pattern matrices require 1--8 scenarios."
        )
    codes = np.zeros(primary.shape[0], dtype=np.uint16)
    for index in range(primary.shape[1]):
        codes |= (
            primary[:, index].astype(np.uint16)
            << np.uint16(2 * index)
        )
        codes |= (
            comparator[:, index].astype(np.uint16)
            << np.uint16(2 * index + 1)
        )
    return codes


def build_robust_seed_data(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
) -> dict:
    codes = encode_joint_patterns(primary, comparator)
    record: dict[str, np.ndarray | int] = {
        "scenario_count": int(primary.shape[1]),
    }
    for label, prefix in [(0, "human"), (1, "bot")]:
        values, counts = np.unique(
            codes[labels == label], return_counts=True
        )
        counts = counts.astype(np.float64)
        total = int(counts.sum())
        record[f"{prefix}_codes"] = values.astype(
            np.uint16, copy=False
        )
        record[f"{prefix}_probabilities"] = counts / total
        record[f"{prefix}_total"] = total
    return record


def confusion_from_pattern_counts(
    counts: np.ndarray,
    codes: np.ndarray,
    label: int,
    scenario_index: int,
    selector: str,
) -> tuple[int, int]:
    selector_offset = 0 if selector == "primary" else 1
    bit = 2 * scenario_index + selector_offset
    positive = ((codes >> np.uint16(bit)) & 1).astype(
        bool
    )
    positive_count = int(counts[positive].sum())
    negative_count = int(counts[~positive].sum())
    if label == 0:
        return negative_count, positive_count
    return negative_count, positive_count


def pattern_confusion(
    human_counts: np.ndarray,
    bot_counts: np.ndarray,
    data: dict,
    scenario_index: int,
    selector: str,
) -> tuple[int, int, int, int]:
    tn, fp = confusion_from_pattern_counts(
        human_counts,
        data["human_codes"],
        0,
        scenario_index,
        selector,
    )
    fn, tp = confusion_from_pattern_counts(
        bot_counts,
        data["bot_codes"],
        1,
        scenario_index,
        selector,
    )
    return tn, fp, fn, tp


def paired_robust_bootstrap(
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
        primary_seed_values = []
        comparator_seed_values = []
        for seed_index in sampled_seeds:
            data = seed_data[int(seed_index)]
            human_counts = rng.multinomial(
                int(data["human_total"]),
                data["human_probabilities"],
            )
            bot_counts = rng.multinomial(
                int(data["bot_total"]),
                data["bot_probabilities"],
            )
            primary_scenarios = []
            comparator_scenarios = []
            for scenario_index in range(
                int(data["scenario_count"])
            ):
                primary_scenarios.append(
                    metric_from_confusion(
                        *pattern_confusion(
                            human_counts,
                            bot_counts,
                            data,
                            scenario_index,
                            "primary",
                        ),
                        metric,
                    )
                )
                comparator_scenarios.append(
                    metric_from_confusion(
                        *pattern_confusion(
                            human_counts,
                            bot_counts,
                            data,
                            scenario_index,
                            "comparator",
                        ),
                        metric,
                    )
                )
            primary_seed_values.append(
                float(np.mean(primary_scenarios))
            )
            comparator_seed_values.append(
                float(np.mean(comparator_scenarios))
            )
        differences[replicate] = float(
            np.mean(primary_seed_values)
            - np.mean(comparator_seed_values)
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


def exact_wilcoxon(
    differences: np.ndarray,
) -> tuple[float, float]:
    try:
        test = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
        return float(test.statistic), float(test.pvalue)
    except ValueError:
        return 0.0, 1.0


def interpretation(lower: float, upper: float) -> str:
    if lower > 0:
        return "primary_higher"
    if upper < 0:
        return "comparator_higher"
    return "difference_not_resolved"


def inference_record(
    comparison_scope: str,
    metric: str,
    observed_differences: np.ndarray,
    bootstrap: np.ndarray,
) -> dict:
    lower = float(np.quantile(bootstrap, 0.025))
    upper = float(np.quantile(bootstrap, 0.975))
    statistic, wilcoxon_p = exact_wilcoxon(
        observed_differences
    )
    return {
        "primary_model": PRIMARY_NAME,
        "comparator": COMPARATOR_NAME,
        "scope": comparison_scope,
        "metric": metric,
        "seeds": len(SEEDS),
        "test_users": 100_000,
        "mean_observed_difference": float(
            observed_differences.mean()
        ),
        "seed_difference_std": float(
            observed_differences.std(ddof=1)
        ),
        "bootstrap_ci_lower_95": lower,
        "bootstrap_ci_upper_95": upper,
        "bootstrap_p_raw": two_sided_bootstrap_p(
            bootstrap
        ),
        "probability_primary_higher": float(
            np.mean(bootstrap > 0)
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_raw": wilcoxon_p,
        "interpretation": interpretation(lower, upper),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BOTMOE_MANIFEST.is_file():
        raise RuntimeError(
            "Run scripts/48_evaluate_osnem_forced_availability.py "
            "first."
        )
    with BOTMOE_MANIFEST.open(
        "r", encoding="utf-8"
    ) as handle:
        forced_manifest = json.load(handle)
    if forced_manifest.get("status") != "PASS":
        raise RuntimeError(
            "BotMoE forced-availability manifest did not pass."
        )
    if forced_manifest.get("scenarios") != SCENARIOS:
        raise RuntimeError(
            "Forced-availability scenario registry differs."
        )

    loaded: dict[
        int, dict[str, dict[str, pd.DataFrame]]
    ] = {}
    descriptive_rows: list[dict] = []
    robust_seed_rows: list[dict] = []
    graph_availability_rows: list[dict] = []

    for seed in SEEDS:
        loaded[seed] = {}
        reference_ids = None
        reference_labels = None
        primary_matrix = np.empty(
            (100_000, len(SCENARIOS)), dtype=np.int8
        )
        comparator_matrix = np.empty_like(primary_matrix)

        for scenario_index, scenario in enumerate(SCENARIOS):
            primary = load_predictions(
                router_path(seed, scenario)
            )
            comparator = load_predictions(
                botmoe_path(seed, scenario)
            )
            if len(primary) != 100_000 or len(comparator) != 100_000:
                raise RuntimeError(
                    f"Unexpected test size for seed={seed}, "
                    f"scenario={scenario}."
                )
            primary_ids = primary["node_id"].to_numpy(
                dtype=np.int64
            )
            primary_labels = primary["label"].to_numpy(
                dtype=np.int8
            )
            if reference_ids is None:
                reference_ids = primary_ids
                reference_labels = primary_labels
            if not np.array_equal(reference_ids, primary_ids):
                raise RuntimeError(
                    f"Router node alignment failure: seed={seed}, "
                    f"scenario={scenario}."
                )
            if not np.array_equal(
                reference_labels, primary_labels
            ):
                raise RuntimeError(
                    f"Router label alignment failure: seed={seed}, "
                    f"scenario={scenario}."
                )
            if not np.array_equal(
                reference_ids,
                comparator["node_id"].to_numpy(
                    dtype=np.int64
                ),
            ):
                raise RuntimeError(
                    f"Comparator node alignment failure: "
                    f"seed={seed}, scenario={scenario}."
                )
            if not np.array_equal(
                reference_labels,
                comparator["label"].to_numpy(
                    dtype=np.int8
                ),
            ):
                raise RuntimeError(
                    f"Comparator label alignment failure: "
                    f"seed={seed}, scenario={scenario}."
                )
            # Text and metadata eligibility are common evidence and
            # must align exactly. Graph eligibility is representation-
            # specific by construction: the router uses the 14-relation
            # heterogeneous topology mask, whereas the BotMoE comparator
            # uses the locked follower/following R-GCN mask. Preserve both
            # native masks and audit their overlap instead of silently
            # redefining either evaluated system.
            for availability_column in [
                "text_available",
                "metadata_available",
            ]:
                if not np.array_equal(
                    primary[availability_column].to_numpy(
                        dtype=bool
                    ),
                    comparator[
                        availability_column
                    ].to_numpy(dtype=bool),
                ):
                    raise RuntimeError(
                        "Common-evidence availability failure: "
                        f"{availability_column}, seed={seed}, "
                        f"scenario={scenario}."
                    )

            primary_graph = primary[
                "graph_available"
            ].to_numpy(dtype=bool)
            comparator_graph = comparator[
                "graph_available"
            ].to_numpy(dtype=bool)
            graph_availability_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "test_users": len(primary_graph),
                    "router_graph_available": int(
                        primary_graph.sum()
                    ),
                    "botmoe_graph_available": int(
                        comparator_graph.sum()
                    ),
                    "both_graph_available": int(
                        (primary_graph & comparator_graph).sum()
                    ),
                    "router_only_graph_available": int(
                        (primary_graph & ~comparator_graph).sum()
                    ),
                    "botmoe_only_graph_available": int(
                        (~primary_graph & comparator_graph).sum()
                    ),
                    "neither_graph_available": int(
                        (~primary_graph & ~comparator_graph).sum()
                    ),
                    "graph_mask_mismatches": int(
                        np.count_nonzero(
                            primary_graph != comparator_graph
                        )
                    ),
                    "graph_mask_mismatch_rate": float(
                        np.mean(
                            primary_graph != comparator_graph
                        )
                    ),
                }
            )

            loaded[seed][scenario] = {
                "primary": primary,
                "comparator": comparator,
            }
            primary_predictions = primary[
                "prediction"
            ].to_numpy(dtype=np.int8)
            comparator_predictions = comparator[
                "prediction"
            ].to_numpy(dtype=np.int8)
            primary_matrix[:, scenario_index] = (
                primary_predictions
            )
            comparator_matrix[:, scenario_index] = (
                comparator_predictions
            )
            for model, predictions in [
                (PRIMARY_NAME, primary_predictions),
                (COMPARATOR_NAME, comparator_predictions),
            ]:
                descriptive_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "scenario": scenario,
                        "test_users": len(primary_labels),
                        "macro_f1": metric_value(
                            primary_labels,
                            predictions,
                            "macro_f1",
                        ),
                        "balanced_accuracy": metric_value(
                            primary_labels,
                            predictions,
                            "balanced_accuracy",
                        ),
                    }
                )

        for model, matrix in [
            (PRIMARY_NAME, primary_matrix),
            (COMPARATOR_NAME, comparator_matrix),
        ]:
            row = {
                "model": model,
                "seed": seed,
                "scenarios": len(SCENARIOS),
            }
            for metric in METRICS:
                row[metric] = float(
                    np.mean(
                        [
                            metric_value(
                                reference_labels,
                                matrix[:, index],
                                metric,
                            )
                            for index in range(len(SCENARIOS))
                        ]
                    )
                )
            robust_seed_rows.append(row)

    graph_availability_audit = pd.DataFrame(
        graph_availability_rows
    )
    graph_availability_audit.to_csv(
        OUTPUT_DIR / "graph_availability_audit.csv",
        index=False,
    )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(
        OUTPUT_DIR / "forced_comparison_seed_results.csv",
        index=False,
    )
    robust_seeds = pd.DataFrame(robust_seed_rows)
    robust_seeds.to_csv(
        OUTPUT_DIR / "robust_mean_seed_results.csv",
        index=False,
    )

    summary_rows: list[dict] = []
    for scenario in SCENARIOS:
        for model in [PRIMARY_NAME, COMPARATOR_NAME]:
            subset = descriptive.loc[
                descriptive["scenario"].eq(scenario)
                & descriptive["model"].eq(model)
            ]
            row = {
                "model": model,
                "scenario": scenario,
                "seeds": int(subset["seed"].nunique()),
                "test_users": int(
                    subset["test_users"].iloc[0]
                ),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(
                    subset[metric].mean()
                )
                row[f"{metric}_std"] = float(
                    subset[metric].std(ddof=1)
                )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "forced_comparison_summary.csv",
        index=False,
    )

    inference_rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for scenario in SCENARIOS:
        scenario_seed_data = []
        observed: dict[str, list[float]] = {
            metric: [] for metric in METRICS
        }
        for seed in SEEDS:
            primary = loaded[seed][scenario]["primary"]
            comparator = loaded[seed][scenario][
                "comparator"
            ]
            labels = primary["label"].to_numpy(
                dtype=np.int8
            )
            primary_predictions = primary[
                "prediction"
            ].to_numpy(dtype=np.int8)
            comparator_predictions = comparator[
                "prediction"
            ].to_numpy(dtype=np.int8)
            scenario_seed_data.append(
                joint_probabilities(
                    labels,
                    primary_predictions,
                    comparator_predictions,
                )
            )
            for metric in METRICS:
                observed[metric].append(
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
        for metric in METRICS:
            bootstrap = paired_scenario_bootstrap(
                scenario_seed_data, metric, rng
            )
            inference_rows.append(
                inference_record(
                    scenario,
                    metric,
                    np.asarray(
                        observed[metric], dtype=np.float64
                    ),
                    bootstrap,
                )
            )

    robust_pattern_data = []
    robust_observed: dict[str, list[float]] = {
        metric: [] for metric in METRICS
    }
    for seed in SEEDS:
        labels = loaded[seed][SCENARIOS[0]]["primary"][
            "label"
        ].to_numpy(dtype=np.int8)
        primary_matrix = np.column_stack(
            [
                loaded[seed][scenario]["primary"][
                    "prediction"
                ].to_numpy(dtype=np.int8)
                for scenario in SCENARIOS
            ]
        )
        comparator_matrix = np.column_stack(
            [
                loaded[seed][scenario]["comparator"][
                    "prediction"
                ].to_numpy(dtype=np.int8)
                for scenario in SCENARIOS
            ]
        )
        robust_pattern_data.append(
            build_robust_seed_data(
                labels, primary_matrix, comparator_matrix
            )
        )
        for metric in METRICS:
            primary_mean = np.mean(
                [
                    metric_value(
                        labels,
                        primary_matrix[:, index],
                        metric,
                    )
                    for index in range(len(SCENARIOS))
                ]
            )
            comparator_mean = np.mean(
                [
                    metric_value(
                        labels,
                        comparator_matrix[:, index],
                        metric,
                    )
                    for index in range(len(SCENARIOS))
                ]
            )
            robust_observed[metric].append(
                float(primary_mean - comparator_mean)
            )

    robust_inference_rows = []
    for metric in METRICS:
        bootstrap = paired_robust_bootstrap(
            robust_pattern_data, metric, rng
        )
        robust_inference_rows.append(
            inference_record(
                "robust_mean_across_7_scenarios",
                metric,
                np.asarray(
                    robust_observed[metric],
                    dtype=np.float64,
                ),
                bootstrap,
            )
        )

    # Exact common-availability sensitivity. These three
    # conditions exclude the graph channel, so text and metadata
    # eligibility are identical for both systems on every test user.
    aligned_scenarios = ["T", "M", "TM"]
    aligned_pattern_data = []
    aligned_observed: dict[str, list[float]] = {
        metric: [] for metric in METRICS
    }
    for seed in SEEDS:
        labels = loaded[seed][aligned_scenarios[0]][
            "primary"
        ]["label"].to_numpy(dtype=np.int8)
        primary_matrix = np.column_stack(
            [
                loaded[seed][scenario]["primary"][
                    "prediction"
                ].to_numpy(dtype=np.int8)
                for scenario in aligned_scenarios
            ]
        )
        comparator_matrix = np.column_stack(
            [
                loaded[seed][scenario]["comparator"][
                    "prediction"
                ].to_numpy(dtype=np.int8)
                for scenario in aligned_scenarios
            ]
        )
        aligned_pattern_data.append(
            build_robust_seed_data(
                labels, primary_matrix, comparator_matrix
            )
        )
        for metric in METRICS:
            primary_mean = np.mean(
                [
                    metric_value(
                        labels,
                        primary_matrix[:, index],
                        metric,
                    )
                    for index in range(len(aligned_scenarios))
                ]
            )
            comparator_mean = np.mean(
                [
                    metric_value(
                        labels,
                        comparator_matrix[:, index],
                        metric,
                    )
                    for index in range(len(aligned_scenarios))
                ]
            )
            aligned_observed[metric].append(
                float(primary_mean - comparator_mean)
            )

    for metric in METRICS:
        bootstrap = paired_robust_bootstrap(
            aligned_pattern_data, metric, rng
        )
        robust_inference_rows.append(
            inference_record(
                "aligned_non_graph_mean_across_T_M_TM",
                metric,
                np.asarray(
                    aligned_observed[metric],
                    dtype=np.float64,
                ),
                bootstrap,
            )
        )

    inference = pd.DataFrame(inference_rows)
    for metric in METRICS:
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
        OUTPUT_DIR / "forced_scenario_paired_inference.csv",
        index=False,
    )

    robust_inference = pd.DataFrame(
        robust_inference_rows
    )
    robust_inference["bootstrap_p_holm"] = holm_adjust(
        robust_inference["bootstrap_p_raw"]
    )
    robust_inference["wilcoxon_p_holm"] = holm_adjust(
        robust_inference["wilcoxon_p_raw"]
    )
    robust_inference.to_csv(
        OUTPUT_DIR / "robust_mean_paired_inference.csv",
        index=False,
    )

    manuscript_table = summary.pivot(
        index="scenario",
        columns="model",
        values=["macro_f1_mean", "macro_f1_std"],
    ).reindex(SCENARIOS)
    manuscript_table.columns = [
        "_".join(column).strip()
        for column in manuscript_table.columns
    ]
    manuscript_table = manuscript_table.reset_index()
    manuscript_table["difference_router_minus_botmoe"] = (
        manuscript_table[
            f"macro_f1_mean_{PRIMARY_NAME}"
        ]
        - manuscript_table[
            f"macro_f1_mean_{COMPARATOR_NAME}"
        ]
    )
    manuscript_table.to_csv(
        OUTPUT_DIR / "manuscript_forced_comparison_table.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "analysis_family": (
            "OSNEM reviewer-requested forced-availability "
            "comparison"
        ),
        "primary_model": PRIMARY_NAME,
        "comparator": COMPARATOR_NAME,
        "benchmark": "TwiBot-22",
        "split": "official_test",
        "seeds": SEEDS,
        "scenarios": SCENARIOS,
        "test_users": 100_000,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "scenario_bootstrap_design": (
            "Paired two-level bootstrap: seeds resampled with "
            "replacement; within each seed, label-stratified joint "
            "prediction cells resampled multinomially."
        ),
        "robust_bootstrap_design": (
            "Paired two-level bootstrap preserving cross-scenario "
            "dependence through label-stratified resampling of joint "
            "14-bit prediction patterns."
        ),
        "graph_availability_policy": (
            "Preserve and audit native graph eligibility: the router "
            "uses 14-relation heterogeneous topology, whereas the "
            "BotMoE comparator uses follower/following R-GCN evidence. "
            "Text and metadata eligibility remain exactly aligned."
        ),
        "comparison_interpretation": (
            "All-user paired system comparison under each method's "
            "declared native graph representation; it is not an "
            "identical-graph-encoder ablation."
        ),
        "scenario_multiplicity_control": (
            "Holm correction across seven forced scenarios within "
            "each metric."
        ),
        "aligned_common_availability_sensitivity": {
            "scenarios": ["T", "M", "TM"],
            "rationale": (
                "These conditions exclude graph evidence; text and "
                "metadata eligibility are identical across systems."
            ),
        },
        "robust_multiplicity_control": (
            "Holm correction across two inference scopes and two "
            "prespecified robust metrics."
        ),
        "seed_level_test": (
            "Exact two-sided Wilcoxon signed-rank; with five seeds "
            "its attainable resolution is explicitly recognized."
        ),
        "config_sha256": sha256_file(CONFIG_PATH),
        "botmoe_forced_manifest_sha256": sha256_file(
            BOTMOE_MANIFEST
        ),
        "evaluation_script_sha256": sha256_file(
            Path(__file__)
        ),
        "integrity": {
            "node_alignment_verified": True,
            "label_alignment_verified": True,
            "text_availability_alignment_verified": True,
            "metadata_availability_alignment_verified": True,
            "native_graph_availability_audited": True,
            "graph_availability_is_representation_specific": True,
            "paired_predictions_used": True,
            "test_predictions_not_used_for_model_selection": True,
            "locked_thresholds_preserved": True,
        },
    }
    with (
        OUTPUT_DIR / "manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("OSNEM FORCED-AVAILABILITY INFERENCE: PASS")
    print("\nNative graph-availability audit:")
    print(
        graph_availability_audit.groupby(
            "scenario", sort=False
        )[
            [
                "router_graph_available",
                "botmoe_graph_available",
                "both_graph_available",
                "graph_mask_mismatches",
                "graph_mask_mismatch_rate",
            ]
        ].first().to_string()
    )
    print("\nScenario macro-F1 comparison:")
    print(manuscript_table.to_string(index=False))
    print("\nScenario paired inference:")
    print(
        inference.loc[
            inference["metric"].eq("macro_f1"),
            [
                "scope",
                "mean_observed_difference",
                "bootstrap_ci_lower_95",
                "bootstrap_ci_upper_95",
                "bootstrap_p_holm",
                "probability_primary_higher",
                "wilcoxon_p_holm",
                "interpretation",
            ],
        ].to_string(index=False)
    )
    print("\nRobust-mean paired inference:")
    print(
        robust_inference[
            [
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
