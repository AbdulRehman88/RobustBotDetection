from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "24_statistical_inference"
BERT_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
CLASSICAL_DIR = PROJECT_ROOT / "outputs" / "06_classical_baselines"
METADATA_DIR = PROJECT_ROOT / "outputs" / "15_xgboost_metadata"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "17_heterogeneous_graph_baseline"
STRESS_DIR = PROJECT_ROOT / "outputs" / "23_robustness_stress_tests"
ROUTER_DIR = PROJECT_ROOT / "outputs" / "21_reliability_router"
GATED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "13_multimodal_fusion"
    / "gated_full_transductive_moddrop_on"
)

SEEDS = [42, 123, 2024, 3407, 7777]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 260_829
PRIMARY_MODEL = "bertweet_mean_metadata"
ROUTER_SCENARIOS = ["natural", "T", "G", "M", "TG", "TM", "GM", "TGM"]
FORCED_SCENARIOS = ["T", "G", "M", "TG", "TM", "GM", "TGM"]

COMPARATORS = {
    "tfidf_metadata_concat": (
        CLASSICAL_DIR
        / "test_predictions"
        / "tfidf_metadata_concat_seed_{seed}_test_predictions.parquet"
    ),
    "bertweet_mean": (
        BERT_DIR
        / "test_predictions"
        / "bertweet_mean_seed_{seed}_test_predictions.parquet"
    ),
    "xgboost_metadata": (
        METADATA_DIR
        / "test_predictions"
        / "seed_{seed}_test_predictions.parquet"
    ),
    "xgboost_heterogeneous_graph": (
        GRAPH_DIR
        / "test_predictions"
        / "seed_{seed}_test_predictions.parquet"
    ),
    "mask_only": (
        CLASSICAL_DIR
        / "test_predictions"
        / "mask_only_seed_{seed}_test_predictions.parquet"
    ),
}


def primary_path(seed: int) -> Path:
    return (
        BERT_DIR
        / "test_predictions"
        / f"bertweet_mean_metadata_seed_{seed}_test_predictions.parquet"
    )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    required = ["node_id", "label", "prediction"]
    frame = pd.read_parquet(path, columns=required)
    return frame.sort_values("node_id").reset_index(drop=True)


def metric_value(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: str,
) -> float:
    if metric == "macro_f1":
        return float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, predictions))
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
        bot_f1 = 0.0 if bot_denominator == 0 else 2 * tp / bot_denominator
        human_f1 = (
            0.0 if human_denominator == 0 else 2 * tn / human_denominator
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
        counts = np.bincount(category, minlength=4).astype(np.float64)
        totals.append(int(counts.sum()))
        probabilities.append(counts / counts.sum())
    return probabilities[0], probabilities[1], totals[0], totals[1]


def confusion_from_joint(
    human_counts: np.ndarray,
    bot_counts: np.ndarray,
    prediction_selector: str,
) -> tuple[int, int, int, int]:
    # Category = 2*primary_prediction + comparator_prediction.
    if prediction_selector == "primary":
        predicted_positive = np.asarray([False, False, True, True])
    elif prediction_selector == "comparator":
        predicted_positive = np.asarray([False, True, False, True])
    else:
        raise ValueError(prediction_selector)
    fp = int(human_counts[predicted_positive].sum())
    tn = int(human_counts[~predicted_positive].sum())
    tp = int(bot_counts[predicted_positive].sum())
    fn = int(bot_counts[~predicted_positive].sum())
    return tn, fp, fn, tp


def paired_hierarchical_bootstrap(
    seed_data: list[dict],
    metric: str,
    rng: np.random.Generator,
) -> np.ndarray:
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    seed_count = len(seed_data)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        primary_values = []
        comparator_values = []
        for seed_index in sampled_seeds:
            data = seed_data[int(seed_index)]
            human_counts = rng.multinomial(
                data["human_total"], data["human_probabilities"]
            )
            bot_counts = rng.multinomial(
                data["bot_total"], data["bot_probabilities"]
            )
            primary_confusion = confusion_from_joint(
                human_counts, bot_counts, "primary"
            )
            comparator_confusion = confusion_from_joint(
                human_counts, bot_counts, "comparator"
            )
            primary_values.append(
                metric_from_confusion(*primary_confusion, metric)
            )
            comparator_values.append(
                metric_from_confusion(*comparator_confusion, metric)
            )
        differences[replicate] = float(
            np.mean(primary_values) - np.mean(comparator_values)
        )
    return differences


def two_sided_bootstrap_p(differences: np.ndarray) -> float:
    non_positive = (np.count_nonzero(differences <= 0) + 1) / (
        len(differences) + 1
    )
    non_negative = (np.count_nonzero(differences >= 0) + 1) / (
        len(differences) + 1
    )
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


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


def run_model_comparisons() -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    seed_metric_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for comparator_name, comparator_template in COMPARATORS.items():
        seed_data = []
        per_seed_differences: dict[str, list[float]] = {
            "macro_f1": [],
            "balanced_accuracy": [],
        }
        for seed in SEEDS:
            primary = load_predictions(primary_path(seed))
            comparator = load_predictions(
                Path(str(comparator_template).format(seed=seed))
            )
            if not np.array_equal(
                primary["node_id"].to_numpy(dtype=np.int64),
                comparator["node_id"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError(
                    f"Node alignment failure: {comparator_name}, seed {seed}."
                )
            labels = primary["label"].to_numpy(dtype=np.int8)
            if not np.array_equal(
                labels, comparator["label"].to_numpy(dtype=np.int8)
            ):
                raise RuntimeError(
                    f"Label alignment failure: {comparator_name}, seed {seed}."
                )
            primary_predictions = primary["prediction"].to_numpy(dtype=np.int8)
            comparator_predictions = comparator["prediction"].to_numpy(
                dtype=np.int8
            )
            human_p, bot_p, human_n, bot_n = joint_probabilities(
                labels, primary_predictions, comparator_predictions
            )
            seed_data.append(
                {
                    "human_probabilities": human_p,
                    "bot_probabilities": bot_p,
                    "human_total": human_n,
                    "bot_total": bot_n,
                }
            )
            for metric in ["macro_f1", "balanced_accuracy"]:
                primary_value = metric_value(labels, primary_predictions, metric)
                comparator_value = metric_value(
                    labels, comparator_predictions, metric
                )
                difference = primary_value - comparator_value
                per_seed_differences[metric].append(difference)
                seed_metric_rows.append(
                    {
                        "primary_model": PRIMARY_MODEL,
                        "comparator": comparator_name,
                        "seed": seed,
                        "metric": metric,
                        "primary_value": primary_value,
                        "comparator_value": comparator_value,
                        "difference": difference,
                    }
                )

        for metric in ["macro_f1", "balanced_accuracy"]:
            bootstrap = paired_hierarchical_bootstrap(
                seed_data, metric, rng
            )
            seed_differences = np.asarray(
                per_seed_differences[metric], dtype=np.float64
            )
            try:
                wilcoxon_result = wilcoxon(
                    seed_differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                wilcoxon_statistic = float(wilcoxon_result.statistic)
                wilcoxon_p = float(wilcoxon_result.pvalue)
            except ValueError:
                wilcoxon_statistic = 0.0
                wilcoxon_p = 1.0
            comparison_rows.append(
                {
                    "primary_model": PRIMARY_MODEL,
                    "comparator": comparator_name,
                    "metric": metric,
                    "seeds": len(SEEDS),
                    "test_users": int(
                        seed_data[0]["human_total"]
                        + seed_data[0]["bot_total"]
                    ),
                    "mean_observed_difference": float(seed_differences.mean()),
                    "bootstrap_ci_lower_95": float(
                        np.quantile(bootstrap, 0.025)
                    ),
                    "bootstrap_ci_upper_95": float(
                        np.quantile(bootstrap, 0.975)
                    ),
                    "bootstrap_p_raw": two_sided_bootstrap_p(bootstrap),
                    "probability_primary_superior": float(
                        np.mean(bootstrap > 0)
                    ),
                    "seed_difference_std": float(
                        seed_differences.std(ddof=1)
                    ),
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_p_raw": wilcoxon_p,
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    for metric in comparisons["metric"].unique():
        selected = comparisons["metric"].eq(metric)
        comparisons.loc[selected, "bootstrap_p_holm"] = holm_adjust(
            comparisons.loc[selected, "bootstrap_p_raw"]
        )
        comparisons.loc[selected, "wilcoxon_p_holm"] = holm_adjust(
            comparisons.loc[selected, "wilcoxon_p_raw"]
        )
    return comparisons, pd.DataFrame(seed_metric_rows)


def router_path(seed: int, scenario: str) -> Path:
    return ROUTER_DIR / "test_predictions" / f"seed_{seed}_{scenario}.parquet"


def gated_path(seed: int, scenario: str) -> Path:
    return GATED_DIR / "test_predictions" / f"seed_{seed}_{scenario}.parquet"


def run_router_gated_scenario_inference() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10)
    for scenario in ROUTER_SCENARIOS:
        seed_data = []
        observed = {"macro_f1": [], "balanced_accuracy": []}
        for seed in SEEDS:
            router = load_predictions(router_path(seed, scenario))
            gated = load_predictions(gated_path(seed, scenario))
            if not np.array_equal(
                router["node_id"].to_numpy(dtype=np.int64),
                gated["node_id"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError(
                    f"Router/gated node mismatch: {scenario}, seed {seed}."
                )
            labels = router["label"].to_numpy(dtype=np.int8)
            if not np.array_equal(
                labels, gated["label"].to_numpy(dtype=np.int8)
            ):
                raise RuntimeError(
                    f"Router/gated label mismatch: {scenario}, seed {seed}."
                )
            router_prediction = router["prediction"].to_numpy(dtype=np.int8)
            gated_prediction = gated["prediction"].to_numpy(dtype=np.int8)
            human_p, bot_p, human_n, bot_n = joint_probabilities(
                labels, router_prediction, gated_prediction
            )
            seed_data.append(
                {
                    "human_probabilities": human_p,
                    "bot_probabilities": bot_p,
                    "human_total": human_n,
                    "bot_total": bot_n,
                }
            )
            for metric in observed:
                observed[metric].append(
                    metric_value(labels, router_prediction, metric)
                    - metric_value(labels, gated_prediction, metric)
                )

        for metric in ["macro_f1", "balanced_accuracy"]:
            bootstrap = paired_hierarchical_bootstrap(seed_data, metric, rng)
            seed_differences = np.asarray(observed[metric], dtype=np.float64)
            try:
                test = wilcoxon(
                    seed_differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                wilcoxon_p = float(test.pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "primary_model": "reliability_router",
                    "comparator": "gated_modality_dropout",
                    "scenario": scenario,
                    "metric": metric,
                    "seeds": len(SEEDS),
                    "test_users": int(
                        seed_data[0]["human_total"] + seed_data[0]["bot_total"]
                    ),
                    "mean_observed_difference": float(seed_differences.mean()),
                    "bootstrap_ci_lower_95": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci_upper_95": float(np.quantile(bootstrap, 0.975)),
                    "bootstrap_p_raw": two_sided_bootstrap_p(bootstrap),
                    "probability_router_superior": float(np.mean(bootstrap > 0)),
                    "wilcoxon_p_raw": wilcoxon_p,
                }
            )
    results = pd.DataFrame(rows)
    for metric in results["metric"].unique():
        selected = results["metric"].eq(metric)
        results.loc[selected, "bootstrap_p_holm"] = holm_adjust(
            results.loc[selected, "bootstrap_p_raw"]
        )
        results.loc[selected, "wilcoxon_p_holm"] = holm_adjust(
            results.loc[selected, "wilcoxon_p_raw"]
        )
    return results


def build_multiscenario_seed_data(seed: int) -> dict:
    labels = None
    node_ids = None
    router_predictions = []
    gated_predictions = []
    for scenario in FORCED_SCENARIOS:
        router = load_predictions(router_path(seed, scenario))
        gated = load_predictions(gated_path(seed, scenario))
        current_nodes = router["node_id"].to_numpy(dtype=np.int64)
        current_labels = router["label"].to_numpy(dtype=np.int8)
        if not np.array_equal(
            current_nodes, gated["node_id"].to_numpy(dtype=np.int64)
        ) or not np.array_equal(
            current_labels, gated["label"].to_numpy(dtype=np.int8)
        ):
            raise RuntimeError(
                f"Router/gated alignment failure: {scenario}, seed {seed}."
            )
        if labels is None:
            labels = current_labels
            node_ids = current_nodes
        elif not np.array_equal(labels, current_labels) or not np.array_equal(
            node_ids, current_nodes
        ):
            raise RuntimeError(f"Cross-scenario alignment failure, seed {seed}.")
        router_predictions.append(router["prediction"].to_numpy(dtype=np.uint8))
        gated_predictions.append(gated["prediction"].to_numpy(dtype=np.uint8))

    router_matrix = np.column_stack(router_predictions)
    gated_matrix = np.column_stack(gated_predictions)
    category_codes = np.zeros(len(labels), dtype=np.uint16)
    for scenario_index in range(len(FORCED_SCENARIOS)):
        category_codes |= (
            router_matrix[:, scenario_index].astype(np.uint16)
            << (2 * scenario_index)
        )
        category_codes |= (
            gated_matrix[:, scenario_index].astype(np.uint16)
            << (2 * scenario_index + 1)
        )

    strata = {}
    for label in [0, 1]:
        selected_codes = category_codes[labels == label]
        unique_codes, counts = np.unique(selected_codes, return_counts=True)
        router_by_category = np.column_stack(
            [
                ((unique_codes >> (2 * index)) & 1).astype(bool)
                for index in range(len(FORCED_SCENARIOS))
            ]
        )
        gated_by_category = np.column_stack(
            [
                ((unique_codes >> (2 * index + 1)) & 1).astype(bool)
                for index in range(len(FORCED_SCENARIOS))
            ]
        )
        strata[label] = {
            "total": int(counts.sum()),
            "probabilities": counts.astype(np.float64) / counts.sum(),
            "router": router_by_category,
            "gated": gated_by_category,
        }

    observed_router = []
    observed_gated = []
    for scenario_index in range(len(FORCED_SCENARIOS)):
        observed_router.append(
            metric_value(labels, router_matrix[:, scenario_index], "macro_f1")
        )
        observed_gated.append(
            metric_value(labels, gated_matrix[:, scenario_index], "macro_f1")
        )
    return {
        "strata": strata,
        "observed_router_robust_mean": float(np.mean(observed_router)),
        "observed_gated_robust_mean": float(np.mean(observed_gated)),
    }


def robust_mean_from_multiscenario_counts(
    human_counts: np.ndarray,
    bot_counts: np.ndarray,
    data: dict,
    model: str,
) -> float:
    values = []
    human_predictions = data["strata"][0][model]
    bot_predictions = data["strata"][1][model]
    for scenario_index in range(len(FORCED_SCENARIOS)):
        human_positive = human_predictions[:, scenario_index]
        bot_positive = bot_predictions[:, scenario_index]
        fp = int(human_counts[human_positive].sum())
        tn = int(human_counts[~human_positive].sum())
        tp = int(bot_counts[bot_positive].sum())
        fn = int(bot_counts[~bot_positive].sum())
        values.append(metric_from_confusion(tn, fp, fn, tp, "macro_f1"))
    return float(np.mean(values))


def run_router_gated_robust_inference() -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 11)
    seed_data = [build_multiscenario_seed_data(seed) for seed in SEEDS]
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        router_values = []
        gated_values = []
        for seed_index in sampled_seeds:
            data = seed_data[int(seed_index)]
            human_counts = rng.multinomial(
                data["strata"][0]["total"],
                data["strata"][0]["probabilities"],
            )
            bot_counts = rng.multinomial(
                data["strata"][1]["total"],
                data["strata"][1]["probabilities"],
            )
            router_values.append(
                robust_mean_from_multiscenario_counts(
                    human_counts, bot_counts, data, "router"
                )
            )
            gated_values.append(
                robust_mean_from_multiscenario_counts(
                    human_counts, bot_counts, data, "gated"
                )
            )
        bootstrap[replicate] = np.mean(router_values) - np.mean(gated_values)

    seed_differences = np.asarray(
        [
            data["observed_router_robust_mean"]
            - data["observed_gated_robust_mean"]
            for data in seed_data
        ]
    )
    try:
        test = wilcoxon(seed_differences, alternative="two-sided", method="auto")
        wilcoxon_p = float(test.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return pd.DataFrame(
        [
            {
                "primary_model": "reliability_router",
                "comparator": "gated_modality_dropout",
                "estimand": "mean_macro_f1_across_seven_forced_scenarios",
                "scenarios": "|".join(FORCED_SCENARIOS),
                "seeds": len(SEEDS),
                "router_mean": float(
                    np.mean(
                        [d["observed_router_robust_mean"] for d in seed_data]
                    )
                ),
                "gated_mean": float(
                    np.mean(
                        [d["observed_gated_robust_mean"] for d in seed_data]
                    )
                ),
                "mean_observed_difference": float(seed_differences.mean()),
                "bootstrap_ci_lower_95": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_upper_95": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_p": two_sided_bootstrap_p(bootstrap),
                "probability_router_superior": float(np.mean(bootstrap > 0)),
                "wilcoxon_p": wilcoxon_p,
            }
        ]
    )


def clustered_mean_ci(
    group: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    seed_values = sorted(group["seed"].unique())
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        selected_values = []
        for seed in sampled_seeds:
            candidates = group.loc[group["seed"].eq(seed), value_column]
            selected_values.append(float(rng.choice(candidates.to_numpy())))
        bootstrap[replicate] = np.mean(selected_values)
    return (
        float(group[value_column].mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def run_stress_inference() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    missingness = pd.read_csv(
        STRESS_DIR / "missingness_stress_results.csv"
    )
    missingness_rows = []
    for (mechanism, rate), group in missingness.groupby(
        ["mechanism", "nominal_drop_rate"]
    ):
        for metric in ["macro_f1", "auroc", "auprc"]:
            mean, lower, upper = clustered_mean_ci(group, metric, rng)
            missingness_rows.append(
                {
                    "mechanism": mechanism,
                    "nominal_drop_rate": rate,
                    "metric": metric,
                    "seeds": int(group["seed"].nunique()),
                    "repetitions_per_seed": int(
                        group["repetition"].nunique()
                    ),
                    "mean": mean,
                    "clustered_ci_lower_95": lower,
                    "clustered_ci_upper_95": upper,
                }
            )

    noise = pd.read_csv(STRESS_DIR / "feature_noise_stress_results.csv")
    noise_rows = []
    for (expert, corruption, level), group in noise.groupby(
        ["expert", "corruption", "noise_std"]
    ):
        for metric in ["macro_f1", "auroc", "auprc"]:
            mean, lower, upper = clustered_mean_ci(group, metric, rng)
            noise_rows.append(
                {
                    "expert": expert,
                    "corruption": corruption,
                    "noise_std": level,
                    "metric": metric,
                    "seeds": int(group["seed"].nunique()),
                    "mean": mean,
                    "seed_bootstrap_ci_lower_95": lower,
                    "seed_bootstrap_ci_upper_95": upper,
                }
            )
    return pd.DataFrame(missingness_rows), pd.DataFrame(noise_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons, seed_metrics = run_model_comparisons()
    comparisons.to_csv(
        OUTPUT_DIR / "paired_model_comparisons.csv", index=False
    )
    seed_metrics.to_csv(
        OUTPUT_DIR / "paired_model_seed_metrics.csv", index=False
    )

    router_gated_scenarios = run_router_gated_scenario_inference()
    router_gated_scenarios.to_csv(
        OUTPUT_DIR / "router_vs_gated_scenario_comparisons.csv",
        index=False,
    )
    router_gated_robust = run_router_gated_robust_inference()
    router_gated_robust.to_csv(
        OUTPUT_DIR / "router_vs_gated_robust_aggregate.csv",
        index=False,
    )

    missingness_ci, noise_ci = run_stress_inference()
    missingness_ci.to_csv(
        OUTPUT_DIR / "missingness_clustered_confidence_intervals.csv",
        index=False,
    )
    noise_ci.to_csv(
        OUTPUT_DIR / "noise_seed_confidence_intervals.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "primary_metric": "macro-F1",
        "secondary_classification_metric": "balanced accuracy",
        "primary_model": PRIMARY_MODEL,
        "comparators": list(COMPARATORS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_design": (
            "paired two-level bootstrap: seeds resampled with replacement; "
            "within each seed, users resampled by class from joint primary/comparator "
            "prediction categories"
        ),
        "multiplicity": (
            "Holm adjustment separately across comparators for each metric"
        ),
        "seed_sensitivity": (
            "paired Wilcoxon signed-rank test reported, interpreted cautiously at n=5"
        ),
        "stress_intervals": (
            "seed-clustered bootstrap; one repetition sampled within each sampled seed"
        ),
        "gated_fusion_inference": (
            "paired scenario-wise inference plus a joint seven-scenario robust-mean "
            "bootstrap preserving cross-scenario prediction patterns within users"
        ),
        "random_seed": BOOTSTRAP_SEED,
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("STATISTICAL INFERENCE: PASS")
    print("\nPaired macro-F1 comparisons:")
    print(
        comparisons.loc[
            comparisons["metric"].eq("macro_f1"),
            [
                "comparator",
                "mean_observed_difference",
                "bootstrap_ci_lower_95",
                "bootstrap_ci_upper_95",
                "bootstrap_p_holm",
                "probability_primary_superior",
                "wilcoxon_p_holm",
            ],
        ].to_string(index=False)
    )
    print("\nSelected 70% missingness macro-F1 intervals:")
    print(
        missingness_ci.loc[
            missingness_ci["nominal_drop_rate"].eq(0.70)
            & missingness_ci["metric"].eq("macro_f1")
        ].to_string(index=False)
    )
    print("\nRouter versus gated fusion, robust aggregate:")
    print(router_gated_robust.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
