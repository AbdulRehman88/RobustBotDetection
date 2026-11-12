from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "26_missingness_bias_audit"
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
TFIDF_DIR = PROJECT_ROOT / "outputs" / "05_tfidf_features"
CLASSICAL_DIR = PROJECT_ROOT / "outputs" / "06_classical_baselines"
BERT_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
METADATA_DIR = PROJECT_ROOT / "outputs" / "15_xgboost_metadata"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "17_heterogeneous_graph_baseline"

SEEDS = [42, 123, 2024, 3407, 7777]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 271_029
PATTERN_ORDER = ["M", "GM", "TM", "TGM"]

MODELS = {
    "mask_only": (
        CLASSICAL_DIR
        / "test_predictions"
        / "mask_only_seed_{seed}_test_predictions.parquet"
    ),
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
    "bertweet_mean_metadata": (
        BERT_DIR
        / "test_predictions"
        / "bertweet_mean_metadata_seed_{seed}_test_predictions.parquet"
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
}

PRIMARY_MODEL = "bertweet_mean_metadata"
PRIMARY_COMPARATORS = [
    "mask_only",
    "tfidf_metadata_concat",
    "xgboost_metadata",
]


def availability_pattern(
    text_available: np.ndarray,
    graph_available: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            ("T" if text else "")
            + ("G" if graph else "")
            + "M"
            for text, graph in zip(text_available, graph_available)
        ],
        dtype="U3",
    )


def load_population() -> pd.DataFrame:
    users = pd.read_parquet(
        CANONICAL_DIR / "users.parquet",
        columns=["node_id", "label", "split", "graph_available"],
    ).sort_values("node_id").reset_index(drop=True)
    text_available = np.load(
        TFIDF_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    if len(text_available) != len(users):
        raise RuntimeError("Text availability length mismatch.")
    labels = users["label"].astype(str).str.lower().eq("bot").to_numpy(np.int8)
    users["label_id"] = labels
    users["text_available"] = text_available
    users["availability_pattern"] = availability_pattern(
        text_available,
        users["graph_available"].to_numpy(dtype=bool),
    )
    return users


def pattern_distribution(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_frame in users.groupby("split", sort=False):
        for pattern in PATTERN_ORDER:
            selected = split_frame.loc[
                split_frame["availability_pattern"].eq(pattern)
            ]
            if selected.empty:
                continue
            bots = int(selected["label_id"].sum())
            total = len(selected)
            rows.append(
                {
                    "split": split,
                    "availability_pattern": pattern,
                    "users": total,
                    "bots": bots,
                    "humans": total - bots,
                    "bot_prevalence": bots / total,
                    "split_fraction": total / len(split_frame),
                }
            )
    return pd.DataFrame(rows)


def association_audit(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in users.groupby("split", sort=False):
        table = pd.crosstab(frame["availability_pattern"], frame["label_id"])
        table = table.reindex(index=PATTERN_ORDER, columns=[0, 1], fill_value=0)
        chi2, p_value, _, _ = chi2_contingency(table.to_numpy())
        n = int(table.to_numpy().sum())
        minimum_dimension = min(table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * minimum_dimension))
        mutual_information = mutual_info_score(
            frame["label_id"], frame["availability_pattern"]
        )
        rows.append(
            {
                "split": split,
                "users": n,
                "chi_square": float(chi2),
                "degrees_of_freedom": int(
                    (table.shape[0] - 1) * (table.shape[1] - 1)
                ),
                "p_value": float(p_value),
                "cramers_v": float(cramers_v),
                "mutual_information_nats": float(mutual_information),
            }
        )
    return pd.DataFrame(rows)


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"node_id", "label", "score", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns {sorted(missing)} in {path}.")
    return frame.sort_values("node_id").reset_index(drop=True)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    result = {
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1_bot": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["auprc"] = float(average_precision_score(labels, scores))
    else:
        result["auroc"] = np.nan
        result["auprc"] = np.nan
    return result


def weighted_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    patterns: np.ndarray,
) -> dict[str, float]:
    observed_patterns = [p for p in PATTERN_ORDER if np.any(patterns == p)]
    strata = [
        (pattern, label)
        for pattern in observed_patterns
        for label in [0, 1]
        if np.any((patterns == pattern) & (labels == label))
    ]
    weights = np.zeros(len(labels), dtype=np.float64)
    for pattern, label in strata:
        selected = (patterns == pattern) & (labels == label)
        weights[selected] = 1.0 / (len(strata) * int(selected.sum()))
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
        sample_weight=weights,
    ).ravel()
    bot_f1 = 2 * tp / max(2 * tp + fp + fn, 1e-15)
    human_f1 = 2 * tn / max(2 * tn + fp + fn, 1e-15)
    sensitivity = tp / max(tp + fn, 1e-15)
    specificity = tn / max(tn + fp, 1e-15)
    return {
        "standardized_macro_f1": float(0.5 * (bot_f1 + human_f1)),
        "standardized_balanced_accuracy": float(
            0.5 * (sensitivity + specificity)
        ),
        "standardization_strata": len(strata),
    }


def evaluate_models(users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_users = users.loc[users["split"].eq("test")].sort_values("node_id")
    test_node_ids = test_users["node_id"].to_numpy(dtype=np.int64)
    test_labels = test_users["label_id"].to_numpy(dtype=np.int8)
    test_patterns = test_users["availability_pattern"].to_numpy(dtype="U3")
    pattern_rows = []
    standardized_rows = []

    for model_name, template in MODELS.items():
        for seed in SEEDS:
            frame = load_prediction(Path(str(template).format(seed=seed)))
            if not np.array_equal(
                frame["node_id"].to_numpy(dtype=np.int64), test_node_ids
            ):
                raise RuntimeError(f"Node mismatch: {model_name}, seed {seed}.")
            if not np.array_equal(
                frame["label"].to_numpy(dtype=np.int8), test_labels
            ):
                raise RuntimeError(f"Label mismatch: {model_name}, seed {seed}.")
            scores = frame["score"].to_numpy(dtype=np.float64)
            predictions = frame["prediction"].to_numpy(dtype=np.int8)
            natural = classification_metrics(test_labels, scores, predictions)
            standardized = weighted_metrics(
                test_labels, predictions, test_patterns
            )
            standardized_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "natural_macro_f1": natural["macro_f1"],
                    "natural_balanced_accuracy": natural["balanced_accuracy"],
                    **standardized,
                    "natural_minus_standardized_macro_f1": (
                        natural["macro_f1"]
                        - standardized["standardized_macro_f1"]
                    ),
                }
            )
            for pattern in PATTERN_ORDER:
                selected = test_patterns == pattern
                if not selected.any():
                    continue
                pattern_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "availability_pattern": pattern,
                        "users": int(selected.sum()),
                        "bots": int(test_labels[selected].sum()),
                        "humans": int(selected.sum() - test_labels[selected].sum()),
                        "bot_prevalence": float(test_labels[selected].mean()),
                        **classification_metrics(
                            test_labels[selected],
                            scores[selected],
                            predictions[selected],
                        ),
                    }
                )
    return pd.DataFrame(pattern_rows), pd.DataFrame(standardized_rows)


def bootstrap_standardized_comparison(
    users: pd.DataFrame,
    comparator_name: str,
) -> dict[str, float | str | int]:
    test_users = users.loc[users["split"].eq("test")].sort_values("node_id")
    labels = test_users["label_id"].to_numpy(dtype=np.int8)
    patterns = test_users["availability_pattern"].to_numpy(dtype="U3")
    rng = np.random.default_rng(
        BOOTSTRAP_SEED + PRIMARY_COMPARATORS.index(comparator_name)
    )
    seed_data = []
    observed_differences = []

    for seed in SEEDS:
        primary = load_prediction(
            Path(str(MODELS[PRIMARY_MODEL]).format(seed=seed))
        )["prediction"].to_numpy(dtype=np.int8)
        comparator = load_prediction(
            Path(str(MODELS[comparator_name]).format(seed=seed))
        )["prediction"].to_numpy(dtype=np.int8)
        observed_differences.append(
            weighted_metrics(labels, primary, patterns)[
                "standardized_macro_f1"
            ]
            - weighted_metrics(labels, comparator, patterns)[
                "standardized_macro_f1"
            ]
        )
        strata = []
        for pattern in PATTERN_ORDER:
            for label in [0, 1]:
                selected = (patterns == pattern) & (labels == label)
                if not selected.any():
                    continue
                category = 2 * primary[selected] + comparator[selected]
                counts = np.bincount(category, minlength=4).astype(np.float64)
                strata.append(
                    {
                        "label": label,
                        "total": int(counts.sum()),
                        "probabilities": counts / counts.sum(),
                    }
                )
        seed_data.append(strata)

    def standardized_from_counts(sampled_strata, selector: str) -> float:
        tn = fp = fn = tp = 0.0
        stratum_weight = 1.0 / len(sampled_strata)
        if selector == "primary":
            positive = np.asarray([False, False, True, True])
        else:
            positive = np.asarray([False, True, False, True])
        for stratum, counts in sampled_strata:
            fractions = counts / max(counts.sum(), 1)
            positive_mass = float(fractions[positive].sum()) * stratum_weight
            negative_mass = stratum_weight - positive_mass
            if stratum["label"] == 0:
                fp += positive_mass
                tn += negative_mass
            else:
                tp += positive_mass
                fn += negative_mass
        bot_f1 = 2 * tp / max(2 * tp + fp + fn, 1e-15)
        human_f1 = 2 * tn / max(2 * tn + fp + fn, 1e-15)
        return 0.5 * (bot_f1 + human_f1)

    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        primary_values = []
        comparator_values = []
        for seed_index in sampled_seeds:
            sampled_strata = []
            for stratum in seed_data[int(seed_index)]:
                counts = rng.multinomial(
                    stratum["total"], stratum["probabilities"]
                )
                sampled_strata.append((stratum, counts))
            primary_values.append(
                standardized_from_counts(sampled_strata, "primary")
            )
            comparator_values.append(
                standardized_from_counts(sampled_strata, "comparator")
            )
        bootstrap[replicate] = np.mean(primary_values) - np.mean(
            comparator_values
        )

    p_value = min(
        1.0,
        2.0
        * min(
            (np.count_nonzero(bootstrap <= 0) + 1)
            / (BOOTSTRAP_REPLICATES + 1),
            (np.count_nonzero(bootstrap >= 0) + 1)
            / (BOOTSTRAP_REPLICATES + 1),
        ),
    )
    return {
        "primary_model": PRIMARY_MODEL,
        "comparator": comparator_name,
        "estimand": "pattern-and-label-standardized macro-F1",
        "seeds": len(SEEDS),
        "mean_observed_difference": float(np.mean(observed_differences)),
        "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
        "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
        "two_sided_bootstrap_p": float(p_value),
        "probability_primary_superior": float(np.mean(bootstrap > 0)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    users = load_population()
    distribution = pattern_distribution(users)
    distribution.to_csv(OUTPUT_DIR / "availability_pattern_distribution.csv", index=False)
    associations = association_audit(users)
    associations.to_csv(OUTPUT_DIR / "pattern_label_association.csv", index=False)

    pattern_results, standardized_results = evaluate_models(users)
    pattern_results.to_csv(OUTPUT_DIR / "within_pattern_seed_results.csv", index=False)
    standardized_results.to_csv(
        OUTPUT_DIR / "pattern_label_standardized_seed_results.csv", index=False
    )
    pattern_summary = (
        pattern_results.groupby(["model", "availability_pattern"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            users=("users", "first"),
            bot_prevalence=("bot_prevalence", "first"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
        )
        .sort_values(["model", "availability_pattern"])
    )
    pattern_summary.to_csv(OUTPUT_DIR / "within_pattern_summary.csv", index=False)
    standardized_summary = (
        standardized_results.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            natural_macro_f1_mean=("natural_macro_f1", "mean"),
            standardized_macro_f1_mean=("standardized_macro_f1", "mean"),
            standardized_macro_f1_std=("standardized_macro_f1", "std"),
            standardized_balanced_accuracy_mean=(
                "standardized_balanced_accuracy",
                "mean",
            ),
            natural_minus_standardized_mean=(
                "natural_minus_standardized_macro_f1",
                "mean",
            ),
        )
        .sort_values("standardized_macro_f1_mean", ascending=False)
    )
    standardized_summary.to_csv(
        OUTPUT_DIR / "pattern_label_standardized_summary.csv", index=False
    )

    inference = pd.DataFrame(
        [
            bootstrap_standardized_comparison(users, comparator)
            for comparator in PRIMARY_COMPARATORS
        ]
    )
    inference.to_csv(
        OUTPUT_DIR / "standardized_paired_comparisons.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "availability_definition": {
            "T": "non-empty semantic description/location text",
            "G": "at least one canonical user-user follower/following edge",
            "M": "complete account metadata; available for every user",
        },
        "patterns": PATTERN_ORDER,
        "bias_audit": [
            "pattern-label association in train, validation, and test",
            "within-pattern model performance",
            "pattern-and-label-standardized performance",
            "paired standardized bootstrap comparisons",
        ],
        "standardization": (
            "equal weight to each observed availability-pattern by label stratum, "
            "removing performance benefit from the empirical pattern-label mixture"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seeds": SEEDS,
        "interpretation": (
            "standardization is a diagnostic estimand, not a replacement for natural "
            "test performance and not a causal intervention"
        ),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("MISSINGNESS BIAS AUDIT: PASS")
    print("\nPattern-label association:")
    print(associations.to_string(index=False))
    print("\nPattern-and-label-standardized performance:")
    print(standardized_summary.to_string(index=False))
    print("\nPaired standardized comparisons:")
    print(inference.to_string(index=False))
    print("\nPrimary model within-pattern performance:")
    print(
        pattern_summary.loc[
            pattern_summary["model"].eq(PRIMARY_MODEL)
        ].to_string(index=False)
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
