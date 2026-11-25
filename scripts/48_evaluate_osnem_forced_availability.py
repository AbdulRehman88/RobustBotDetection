from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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
CONFIG_PATH = PROJECT_ROOT / "configs" / "osnem_sota_comparison.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "40_osnem_sota_comparison"
CONTRACT_DIR = OUTPUT_ROOT / "input_contract"
MODEL_DIR = OUTPUT_ROOT / "botmoe_common_evidence" / "models"
OUTPUT_DIR = OUTPUT_ROOT / "botmoe_forced_availability"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
TRAINING_SCRIPT = (
    PROJECT_ROOT / "scripts" / "46_train_botmoe_comparator.py"
)

TRAIN_ID = 0
TEST_ID = 2
MODALITY_ORDER = ("T", "G", "M")
SCENARIOS = {
    "T": np.array([True, False, False], dtype=bool),
    "G": np.array([False, True, False], dtype=bool),
    "M": np.array([False, False, True], dtype=bool),
    "TG": np.array([True, True, False], dtype=bool),
    "TM": np.array([True, False, True], dtype=bool),
    "GM": np.array([False, True, True], dtype=bool),
    "TGM": np.array([True, True, True], dtype=bool),
}
METRIC_COLUMNS = [
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


def sha256_file(
    path: Path, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_module():
    specification = importlib.util.spec_from_file_location(
        "osnem_botmoe_training", TRAINING_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"Unable to import training code from {TRAINING_SCRIPT}."
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {
        "model_state_dict",
        "seed",
        "threshold",
        "model_designation",
    }
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Checkpoint {path.name} lacks fields: {sorted(missing)}."
        )
    if (
        checkpoint["model_designation"]
        != "botmoe_style_common_evidence"
    ):
        raise RuntimeError(
            f"Unexpected model designation in {path.name}."
        )
    return checkpoint


def build_model(
    module,
    parameters: dict,
    device: torch.device,
) -> torch.nn.Module:
    return module.BotMoECommonEvidence(
        text_dimension=int(
            parameters["modalities"]["text"]["input_dimension"]
        ),
        graph_dimension=int(
            parameters["modalities"]["graph"]["input_dimension"]
        ),
        metadata_dimension=int(
            parameters["modalities"]["metadata"]["input_dimension"]
        ),
        hidden_dimension=int(parameters["hidden_dimension"]),
        text_experts=int(
            parameters["modalities"]["text"]["experts"]
        ),
        graph_experts=int(
            parameters["modalities"]["graph"]["experts"]
        ),
        metadata_experts=int(
            parameters["modalities"]["metadata"]["experts"]
        ),
        dropout=float(parameters["dropout"]),
    ).to(device)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    if not (
        len(labels) == len(scores) == len(predictions)
    ):
        raise RuntimeError("Metric inputs have unequal lengths.")
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite scores reached evaluation.")

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    specificity = (
        float(tn / (tn + fp)) if (tn + fp) else float("nan")
    )
    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision_bot": float(
            precision_score(
                labels, predictions, zero_division=0
            )
        ),
        "recall_bot": float(
            recall_score(
                labels, predictions, zero_division=0
            )
        ),
        "specificity_human": specificity,
        "f1_bot": float(
            f1_score(
                labels, predictions, pos_label=1, zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(labels, predictions)
        ),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(
            average_precision_score(labels, scores)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        subset = seed_results.loc[
            seed_results["scenario"] == scenario
        ]
        if len(subset) != 5:
            raise RuntimeError(
                f"Scenario {scenario} has {len(subset)} seed rows."
            )
        row: dict[str, float | int | str] = {
            "model": "botmoe_style_common_evidence",
            "scenario": scenario,
            "seeds": int(len(subset)),
            "test_users": int(subset["test_users"].iloc[0]),
            "no_modality_users": int(
                subset["no_modality_users"].iloc[0]
            ),
            "no_modality_rate": float(
                subset["no_modality_rate"].iloc[0]
            ),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(
                subset[metric].mean()
            )
            row[f"{metric}_std"] = float(
                subset[metric].std(ddof=1)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(TRAINING_SCRIPT)
    contract_path = CONTRACT_DIR / "manifest.json"
    if not contract_path.is_file():
        raise RuntimeError(
            "Run scripts/44_prepare_osnem_comparators.py first."
        )
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("status") != "PASS":
        raise RuntimeError("Comparator input contract did not pass.")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    inputs = config["canonical_inputs"]
    parameters = config["botmoe_common_evidence"]
    training = config["training"]
    seeds = [
        int(value)
        for value in config["comparison_policy"]["seeds"]
    ]
    if seeds != [42, 123, 2024, 3407, 7777]:
        raise RuntimeError("Unexpected comparison seed registry.")

    def input_path(key: str) -> Path:
        return PROJECT_ROOT / inputs[key]

    node_ids = np.load(
        input_path("node_ids"), mmap_mode="r"
    )
    labels = np.load(
        input_path("labels"), mmap_mode="r"
    ).astype(np.int8, copy=False)
    split_ids = np.load(
        input_path("split_ids"), mmap_mode="r"
    )
    text_embeddings = np.load(
        input_path("semantic_text_embeddings"), mmap_mode="r"
    )
    metadata = np.load(
        input_path("metadata_features"), mmap_mode="r"
    )
    availability_archive = np.load(
        CONTRACT_DIR / "availability_masks.npz"
    )
    natural_availability = np.column_stack(
        [
            availability_archive["semantic_text"],
            availability_archive["graph"],
            availability_archive["metadata"],
        ]
    ).astype(bool)

    if not (
        len(node_ids)
        == len(labels)
        == len(split_ids)
        == text_embeddings.shape[0]
        == metadata.shape[0]
        == natural_availability.shape[0]
    ):
        raise RuntimeError("Forced-evaluation inputs are misaligned.")
    if not np.array_equal(
        node_ids, np.arange(len(node_ids), dtype=node_ids.dtype)
    ):
        raise RuntimeError("Canonical node identifiers are not aligned.")

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    if len(test_indices) != 100_000:
        raise RuntimeError(
            f"Expected 100,000 test users; found {len(test_indices)}."
        )
    source_bot_prevalence = float(labels[train_indices].mean())

    device_name = str(training["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    batch_size = int(training["evaluation_batch_size"])
    module = load_training_module()

    result_rows: list[dict] = []
    checkpoint_records: list[dict] = []

    for seed in seeds:
        module.set_seed(seed)
        graph_path = PROJECT_ROOT / inputs[
            "graph_embedding_pattern"
        ].format(seed=seed)
        graph_embeddings = np.load(
            graph_path, mmap_mode="r"
        )
        if graph_embeddings.shape != (len(labels), 64):
            raise RuntimeError(
                f"Seed {seed} graph shape={graph_embeddings.shape}."
            )

        checkpoint_path = (
            MODEL_DIR / f"botmoe_seed_{seed}.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing locked checkpoint: {checkpoint_path}"
            )
        checkpoint = load_checkpoint(checkpoint_path, device)
        if int(checkpoint["seed"]) != seed:
            raise RuntimeError(
                f"Checkpoint seed mismatch for {seed}."
            )
        threshold = float(checkpoint["threshold"])
        if not np.isfinite(threshold):
            raise RuntimeError(
                f"Non-finite threshold for seed {seed}."
            )

        model = build_model(module, parameters, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        checkpoint_records.append(
            {
                "seed": seed,
                "path": str(
                    checkpoint_path.relative_to(PROJECT_ROOT)
                ),
                "sha256": sha256_file(checkpoint_path),
                "threshold": threshold,
            }
        )

        for scenario, permitted in SCENARIOS.items():
            forced_availability = (
                natural_availability & permitted.reshape(1, 3)
            )
            test_availability = forced_availability[
                test_indices
            ]
            usable = test_availability.any(axis=1)
            usable_indices = test_indices[usable]
            no_modality_count = int((~usable).sum())

            scores = np.full(
                len(test_indices),
                source_bot_prevalence,
                dtype=np.float32,
            )
            if len(usable_indices):
                usable_scores, _ = module.predict(
                    model,
                    usable_indices,
                    text_embeddings,
                    graph_embeddings,
                    metadata,
                    forced_availability,
                    labels,
                    device,
                    batch_size,
                )
                scores[usable] = usable_scores

            predictions = (
                scores >= threshold
            ).astype(np.int8)
            # A row with no usable evidence is assigned the locked
            # conservative human fallback; its score remains the source
            # training prevalence for threshold-free metrics.
            predictions[~usable] = 0

            test_labels = np.asarray(
                labels[test_indices], dtype=np.int8
            )
            metrics = classification_metrics(
                test_labels, scores, predictions
            )
            result_rows.append(
                {
                    "model": "botmoe_style_common_evidence",
                    "seed": seed,
                    "scenario": scenario,
                    "threshold": threshold,
                    "test_users": len(test_indices),
                    "usable_users": int(usable.sum()),
                    "no_modality_users": no_modality_count,
                    "no_modality_rate": float(
                        no_modality_count / len(test_indices)
                    ),
                    "fallback_score": source_bot_prevalence,
                    **metrics,
                }
            )

            pd.DataFrame(
                {
                    "node_id": np.asarray(
                        node_ids[test_indices],
                        dtype=np.int64,
                    ),
                    "label": test_labels,
                    "score": scores,
                    "prediction": predictions,
                    "text_available": test_availability[:, 0],
                    "graph_available": test_availability[:, 1],
                    "metadata_available": test_availability[:, 2],
                    "no_modality_fallback": ~usable,
                }
            ).to_parquet(
                PREDICTION_DIR
                / f"botmoe_seed_{seed}_{scenario}.parquet",
                index=False,
            )
            print(
                f"TEST seed={seed}, scenario={scenario}: "
                f"macro-F1={metrics['macro_f1']:.6f}, "
                f"balanced-accuracy="
                f"{metrics['balanced_accuracy']:.6f}, "
                f"fallback={no_modality_count:,}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_results = pd.DataFrame(result_rows)
    expected_rows = len(seeds) * len(SCENARIOS)
    if len(seed_results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows; got {len(seed_results)}."
        )
    seed_results.to_csv(
        OUTPUT_DIR / "forced_seed_scenario_results.csv",
        index=False,
    )

    summary = summarize(seed_results)
    summary.to_csv(
        OUTPUT_DIR / "forced_scenario_summary.csv",
        index=False,
    )

    robust_by_seed = (
        seed_results.groupby("seed", as_index=False)[
            METRIC_COLUMNS
        ]
        .mean()
        .rename(
            columns={
                metric: f"mean_across_scenarios_{metric}"
                for metric in METRIC_COLUMNS
            }
        )
    )
    robust_by_seed.to_csv(
        OUTPUT_DIR / "robust_mean_seed_results.csv",
        index=False,
    )
    robust_summary = {
        "model": "botmoe_style_common_evidence",
        "estimand": (
            "unweighted mean across the seven prespecified forced "
            "availability scenarios"
        ),
        "seeds": len(seeds),
    }
    for metric in METRIC_COLUMNS:
        column = f"mean_across_scenarios_{metric}"
        robust_summary[f"{metric}_mean"] = float(
            robust_by_seed[column].mean()
        )
        robust_summary[f"{metric}_std"] = float(
            robust_by_seed[column].std(ddof=1)
        )

    with (
        OUTPUT_DIR / "robust_mean_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(robust_summary, handle, indent=2)

    manifest = {
        "status": "PASS",
        "model": "botmoe_style_common_evidence",
        "evaluation": "forced_modality_availability",
        "scenarios": list(SCENARIOS),
        "modality_order": list(MODALITY_ORDER),
        "seeds": seeds,
        "benchmark": "TwiBot-22",
        "split": "official_test",
        "test_users": len(test_indices),
        "checkpoint_policy": (
            "Frozen natural-availability checkpoints; no retraining."
        ),
        "threshold_policy": (
            "Reuse each seed's locked natural-validation threshold; "
            "no scenario-specific threshold selection."
        ),
        "fallback_policy": (
            "When forced and natural availability leave no usable "
            "modality, assign a hard human prediction and the source "
            "training bot prevalence as the probability score."
        ),
        "source_training_bot_prevalence": (
            source_bot_prevalence
        ),
        "checkpoint_records": checkpoint_records,
        "config_sha256": sha256_file(CONFIG_PATH),
        "training_script_sha256": sha256_file(
            TRAINING_SCRIPT
        ),
        "evaluation_script_sha256": sha256_file(
            Path(__file__)
        ),
    }
    with (
        OUTPUT_DIR / "manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nBOTMOE FORCED-AVAILABILITY EVALUATION: PASS")
    print("\nScenario summary:")
    print(
        summary[
            [
                "scenario",
                "no_modality_rate",
                "macro_f1_mean",
                "macro_f1_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
            ]
        ].to_string(index=False)
    )
    print("\nRobust mean:")
    print(json.dumps(robust_summary, indent=2))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
