from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
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

BERT_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
METADATA_DIR = PROJECT_ROOT / "outputs" / "15_xgboost_metadata"
GRAPH_DIR = (
    PROJECT_ROOT / "outputs" / "17_heterogeneous_graph_baseline"
)
GRAPH_FEATURE_DIR = (
    PROJECT_ROOT / "outputs" / "16_heterogeneous_graph_features"
)
GATED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "13_multimodal_fusion"
    / "gated_full_transductive_moddrop_on"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "21_reliability_router"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

SEEDS = [42, 123, 2024, 3407, 7777]
MODALITY_ORDER = ("T", "G", "M")
SCENARIOS = ["T", "G", "M", "TG", "TM", "GM", "TGM"]

TRAIN_BOT_PREVALENCE = 54_586 / 700_000


EXPERTS = {
    "text": {
        "required_modalities": frozenset({"T"}),
        "score_type": "logit",
    },
    "graph": {
        "required_modalities": frozenset({"G"}),
        "score_type": "probability",
    },
    "metadata": {
        "required_modalities": frozenset({"M"}),
        "score_type": "probability",
    },
    "text_metadata": {
        "required_modalities": frozenset({"T", "M"}),
        "score_type": "logit",
    },
}


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
        "precision_bot": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall_bot": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "specificity_human": tn / (tn + fp),
        "f1_bot": f1_score(
            labels,
            predictions,
            zero_division=0,
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


def validation_evidence() -> pd.DataFrame:
    bert = pd.read_csv(
        BERT_DIR / "bertweet_baseline_seed_results.csv"
    )
    metadata = pd.read_csv(
        METADATA_DIR / "xgboost_metadata_seed_results.csv"
    )
    graph = pd.read_csv(
        GRAPH_DIR
        / "xgboost_heterogeneous_graph_seed_results.csv"
    )

    rows = []

    for expert_name, model_name in [
        ("text", "bertweet_mean"),
        ("text_metadata", "bertweet_mean_metadata"),
    ]:
        selected = bert.loc[bert["model"].eq(model_name)]
        rows.append(
            {
                "expert": expert_name,
                "required_modalities": "".join(
                    sorted(
                        EXPERTS[expert_name][
                            "required_modalities"
                        ],
                        key=MODALITY_ORDER.index,
                    )
                ),
                "validation_macro_f1_mean": float(
                    selected[
                        "validation_macro_f1"
                    ].mean()
                ),
                "validation_macro_f1_std": float(
                    selected[
                        "validation_macro_f1"
                    ].std(ddof=1)
                ),
                "seeds": int(selected["seed"].nunique()),
            }
        )

    rows.append(
        {
            "expert": "metadata",
            "required_modalities": "M",
            "validation_macro_f1_mean": float(
                metadata["validation_macro_f1"].mean()
            ),
            "validation_macro_f1_std": float(
                metadata["validation_macro_f1"].std(ddof=1)
            ),
            "seeds": int(metadata["seed"].nunique()),
        }
    )
    rows.append(
        {
            "expert": "graph",
            "required_modalities": "G",
            "validation_macro_f1_mean": float(
                graph["validation_macro_f1"].mean()
            ),
            "validation_macro_f1_std": float(
                graph["validation_macro_f1"].std(ddof=1)
            ),
            "seeds": int(graph["seed"].nunique()),
        }
    )

    return pd.DataFrame(rows).sort_values(
        "validation_macro_f1_mean",
        ascending=False,
    )


def select_expert(
    available_modalities: frozenset[str],
    evidence: pd.DataFrame,
) -> str | None:
    candidates = []

    for expert_name, specification in EXPERTS.items():
        required = specification["required_modalities"]
        if required.issubset(available_modalities):
            validation_score = float(
                evidence.loc[
                    evidence["expert"].eq(expert_name),
                    "validation_macro_f1_mean",
                ].iloc[0]
            )
            candidates.append(
                (
                    validation_score,
                    len(required),
                    expert_name,
                )
            )

    if not candidates:
        return None

    # Validation performance is primary; modality utilization breaks
    # exact ties, followed by a deterministic expert name.
    candidates.sort(reverse=True)
    return candidates[0][2]


def load_prediction_frame(
    expert: str,
    seed: int,
) -> pd.DataFrame:
    if expert == "text":
        path = (
            BERT_DIR
            / "test_predictions"
            / f"bertweet_mean_seed_{seed}_test_predictions.parquet"
        )
    elif expert == "text_metadata":
        path = (
            BERT_DIR
            / "test_predictions"
            / (
                f"bertweet_mean_metadata_seed_{seed}"
                f"_test_predictions.parquet"
            )
        )
    elif expert == "metadata":
        path = (
            METADATA_DIR
            / "test_predictions"
            / f"seed_{seed}_test_predictions.parquet"
        )
    elif expert == "graph":
        path = (
            GRAPH_DIR
            / "test_predictions"
            / f"seed_{seed}_test_predictions.parquet"
        )
    else:
        raise ValueError(expert)

    frame = pd.read_parquet(path).sort_values(
        "node_id"
    ).reset_index(drop=True)

    if EXPERTS[expert]["score_type"] == "logit":
        frame["score"] = expit(
            frame["score"].to_numpy(dtype=np.float64)
        ).astype(np.float32)

    return frame[
        ["node_id", "label", "score", "prediction"]
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    evidence = validation_evidence()
    evidence.to_csv(
        OUTPUT_DIR / "expert_validation_evidence.csv",
        index=False,
    )

    selections = []
    for scenario in SCENARIOS:
        expert = select_expert(
            frozenset(scenario),
            evidence,
        )
        selections.append(
            {
                "available_pattern": scenario,
                "selected_expert": expert,
                "selection_basis": (
                    "highest mean validation macro-F1 among "
                    "compatible experts"
                ),
            }
        )

    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(
        OUTPUT_DIR / "validation_selected_router.csv",
        index=False,
    )

    print("Validation-selected routing:")
    print(selection_table.to_string(index=False))

    graph_available = np.load(
        GRAPH_FEATURE_DIR / "graph_available.npy",
        mmap_mode="r",
    ).astype(bool)
    split_ids = np.load(
        GRAPH_FEATURE_DIR / "split_ids.npy",
        mmap_mode="r",
    )
    test_indices = np.flatnonzero(split_ids == 2)
    test_graph_available = graph_available[test_indices]

    result_rows = []

    for seed in SEEDS:
        expert_frames = {
            expert: load_prediction_frame(expert, seed)
            for expert in EXPERTS
        }

        reference = expert_frames["text"]
        node_ids = reference["node_id"].to_numpy(
            dtype=np.int64
        )
        labels = reference["label"].to_numpy(
            dtype=np.int8
        )

        for expert, frame in expert_frames.items():
            if not np.array_equal(
                frame["node_id"].to_numpy(dtype=np.int64),
                node_ids,
            ):
                raise RuntimeError(
                    f"Node-order mismatch for {expert}, seed {seed}."
                )
            if not np.array_equal(
                frame["label"].to_numpy(dtype=np.int8),
                labels,
            ):
                raise RuntimeError(
                    f"Label mismatch for {expert}, seed {seed}."
                )

        if len(node_ids) != len(test_graph_available):
            raise RuntimeError(
                "Graph availability/test prediction length mismatch."
            )

        text_available = pd.read_parquet(
            BERT_DIR
            / "test_predictions"
            / f"bertweet_mean_seed_{seed}_test_predictions.parquet",
            columns=["node_id", "text_available"],
        ).sort_values("node_id")[
            "text_available"
        ].to_numpy(dtype=bool)

        natural_availability = np.column_stack(
            [
                text_available,
                test_graph_available,
                np.ones(len(node_ids), dtype=bool),
            ]
        )

        evaluation_scenarios = {
            "natural": np.ones(3, dtype=bool),
            **{
                scenario: np.asarray(
                    [
                        "T" in scenario,
                        "G" in scenario,
                        "M" in scenario,
                    ],
                    dtype=bool,
                )
                for scenario in SCENARIOS
            },
        }

        for scenario_name, forced_mask in (
            evaluation_scenarios.items()
        ):
            actual = natural_availability & forced_mask

            # Experts were trained with their modality-specific natural
            # missingness and explicit availability indicators. Select one
            # compatible expert for the requested deployment scenario and
            # allow that expert to handle its own naturally missing rows.
            # Per-row fallback routing would discard this learned behavior
            # and unfairly penalize text and text+metadata experts.
            routing_pattern = (
                "TGM" if scenario_name == "natural" else scenario_name
            )
            selected_expert = select_expert(
                frozenset(routing_pattern),
                evidence,
            )
            if selected_expert is None:
                raise RuntimeError(
                    f"No compatible expert for scenario {scenario_name}."
                )

            expert_frame = expert_frames[selected_expert]
            routed_scores = expert_frame["score"].to_numpy(
                dtype=np.float32
            )
            routed_predictions = expert_frame[
                "prediction"
            ].to_numpy(dtype=np.int8)
            assigned_experts = np.full(
                len(node_ids),
                selected_expert,
                dtype=object,
            )

            scenario_metrics = classification_metrics(
                labels,
                routed_scores,
                routed_predictions,
            )

            assignment_counts = pd.Series(
                assigned_experts
            ).value_counts()

            result_rows.append(
                {
                    "model": "validation_selected_reliability_router",
                    "seed": seed,
                    "scenario": scenario_name,
                    **scenario_metrics,
                    "text_assignments": int(
                        assignment_counts.get("text", 0)
                    ),
                    "graph_assignments": int(
                        assignment_counts.get("graph", 0)
                    ),
                    "metadata_assignments": int(
                        assignment_counts.get("metadata", 0)
                    ),
                    "text_metadata_assignments": int(
                        assignment_counts.get(
                            "text_metadata", 0
                        )
                    ),
                    "fallback_assignments": int(
                        assignment_counts.get(
                            "fallback_human", 0
                        )
                    ),
                }
            )

            pd.DataFrame(
                {
                    "node_id": node_ids,
                    "label": labels,
                    "score": routed_scores,
                    "prediction": routed_predictions,
                    "selected_expert": assigned_experts,
                    "text_available": actual[:, 0],
                    "graph_available": actual[:, 1],
                    "metadata_available": actual[:, 2],
                }
            ).to_parquet(
                PREDICTION_DIR
                / f"seed_{seed}_{scenario_name}.parquet",
                index=False,
            )

            print(
                f"TEST seed={seed}, scenario={scenario_name}: "
                f"macro-F1={scenario_metrics['macro_f1']:.6f}, "
                f"AUROC={scenario_metrics['auroc']:.6f}"
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / "router_seed_scenario_results.csv",
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

    for scenario, group in results.groupby("scenario"):
        row = {
            "model": (
                "validation_selected_reliability_router"
            ),
            "scenario": scenario,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=1)
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "scenario"
    )
    summary.to_csv(
        OUTPUT_DIR / "router_scenario_summary.csv",
        index=False,
    )

    forced_summary = summary.loc[
        summary["scenario"].isin(SCENARIOS)
    ]
    robust_macro_f1_mean = float(
        forced_summary["macro_f1_mean"].mean()
    )

    comparison = {
        "router_robust_macro_f1_mean": (
            robust_macro_f1_mean
        )
    }

    if (
        GATED_DIR / "scenario_summary.csv"
    ).exists():
        gated = pd.read_csv(
            GATED_DIR / "scenario_summary.csv"
        )
        gated_forced = gated.loc[
            gated["scenario"].isin(SCENARIOS)
        ]
        comparison[
            "gated_moddrop_robust_macro_f1_mean"
        ] = float(
            gated_forced["macro_f1_mean"].mean()
        )
        comparison["absolute_improvement"] = (
            comparison["router_robust_macro_f1_mean"]
            - comparison[
                "gated_moddrop_robust_macro_f1_mean"
            ]
        )

    with (OUTPUT_DIR / "robustness_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(comparison, handle, indent=2)

    manifest = {
        "status": "PASS",
        "method": (
            "availability-conditioned reliability router"
        ),
        "selection_data": (
            "official validation results only"
        ),
        "selection_rule": (
            "choose the compatible expert with highest mean "
            "validation macro-F1 for each deployment scenario; the "
            "selected expert retains its trained handling of natural "
            "within-modality missingness and suppresses unsupported "
            "or harmful modalities"
        ),
        "experts": {
            "text": "frozen BERTweet mean-pooled MLP",
            "graph": (
                "XGBoost heterogeneous interaction-topology expert"
            ),
            "metadata": "XGBoost account-metadata expert",
            "text_metadata": (
                "frozen BERTweet plus metadata MLP"
            ),
        },
        "seeds": SEEDS,
        "test_policy": (
            "router fixed from validation evidence before test "
            "aggregation"
        ),
        "score_note": (
            "BERT logits converted to sigmoid probabilities; "
            "formal calibration will be evaluated separately"
        ),
        "robustness_comparison": comparison,
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nRELIABILITY ROUTER: PASS")
    print(
        summary[
            [
                "scenario",
                "macro_f1_mean",
                "macro_f1_std",
                "auroc_mean",
                "auprc_mean",
            ]
        ].to_string(index=False)
    )
    print(
        f"Mean macro-F1 across forced modality scenarios: "
        f"{robust_macro_f1_mean:.6f}"
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
