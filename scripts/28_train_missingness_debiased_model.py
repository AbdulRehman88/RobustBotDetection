from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "08_frozen_bertweet"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
ORIGINAL_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "27_missingness_debiased_model"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

SEEDS = [42, 123, 2024, 3407, 7777]
TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2
BATCH_SIZE = 4096
MAX_EPOCHS = 20
PATIENCE = 4
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 281_029


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import {path}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def stratum_weights(
    labels: np.ndarray,
    text_available: np.ndarray,
    graph_available: np.ndarray,
) -> np.ndarray:
    pattern = (
        2 * text_available.astype(np.int8)
        + graph_available.astype(np.int8)
    )
    stratum = 2 * pattern + labels.astype(np.int8)
    weights = np.zeros(len(labels), dtype=np.float64)
    observed = np.unique(stratum)
    for value in observed:
        selected = stratum == value
        weights[selected] = len(labels) / (
            len(observed) * int(selected.sum())
        )
    if not np.isclose(weights.mean(), 1.0, atol=1e-10):
        raise RuntimeError("Stratum weights do not average to one.")
    return weights.astype(np.float32)


def weighted_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
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
        "standardized_auroc": float(
            roc_auc_score(labels, scores, sample_weight=weights)
        ),
        "standardized_auprc": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
    }


def natural_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def exact_weighted_macro_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order].astype(np.float64)
    positive_weight = sorted_weights * (sorted_labels == 1)
    negative_weight = sorted_weights * (sorted_labels == 0)
    cumulative_tp = np.cumsum(positive_weight)
    cumulative_fp = np.cumsum(negative_weight)
    total_positive = float(positive_weight.sum())
    total_negative = float(negative_weight.sum())
    boundary = np.r_[
        sorted_scores[:-1] != sorted_scores[1:], True
    ]
    indices = np.flatnonzero(boundary)
    tp = cumulative_tp[indices]
    fp = cumulative_fp[indices]
    fn = total_positive - tp
    tn = total_negative - fp
    bot_f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-15)
    human_f1 = 2 * tn / np.maximum(2 * tn + fp + fn, 1e-15)
    macro_f1 = 0.5 * (bot_f1 + human_f1)

    all_negative = 0.5 * (
        2 * total_negative
        / max(2 * total_negative + total_positive, 1e-15)
    )
    best_index = int(np.argmax(macro_f1))
    if all_negative > macro_f1[best_index]:
        return float(sorted_scores[0] + 1e-6), float(all_negative)
    last_positive = int(indices[best_index])
    if last_positive == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-6)
    else:
        threshold = float(
            0.5
            * (
                sorted_scores[last_positive]
                + sorted_scores[last_positive + 1]
            )
        )
    return threshold, float(macro_f1[best_index])


def locked_hyperparameters() -> tuple[float, float]:
    results = pd.read_csv(
        ORIGINAL_DIR / "bertweet_baseline_seed_results.csv"
    )
    selected = results.loc[
        results["model"].eq("bertweet_mean_metadata")
    ]
    rates = selected["learning_rate"].unique()
    dropouts = selected["dropout"].unique()
    if len(rates) != 1 or len(dropouts) != 1:
        raise RuntimeError("Original hyperparameters are not uniquely locked.")
    return float(rates[0]), float(dropouts[0])


def fit_debiased(
    baseline_module,
    embeddings,
    metadata,
    availability,
    labels,
    train_indices,
    val_indices,
    train_weights,
    val_weights,
    learning_rate,
    dropout,
    seed,
    device,
):
    set_seed(seed)
    input_dim = embeddings.shape[1] + 1 + metadata.shape[1]
    model = baseline_module.MLPClassifier(input_dim, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_record = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(
            len(train_indices), generator=generator
        ).numpy()
        total_weighted_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            positions = permutation[start : start + BATCH_SIZE]
            batch_indices = train_indices[positions]
            features = baseline_module.prepare_batch(
                embeddings,
                metadata,
                availability,
                batch_indices,
                True,
                device,
            )
            targets = torch.from_numpy(
                labels[batch_indices].astype(np.float32)
            ).to(device, non_blocking=True)
            batch_weights = torch.from_numpy(
                train_weights[positions].astype(np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            losses = criterion(logits, targets)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()
            total_weighted_loss += float(
                (losses.detach() * batch_weights).sum().item()
            )
            total_weight += float(batch_weights.sum().item())

        val_scores = baseline_module.predict_scores(
            model,
            embeddings,
            metadata,
            availability,
            val_indices,
            True,
            device,
        )
        threshold, standardized_macro_f1 = (
            exact_weighted_macro_f1_threshold(
                labels[val_indices], val_scores, val_weights
            )
        )
        predictions = (val_scores >= threshold).astype(np.int8)
        standardized = weighted_metrics(
            labels[val_indices], val_scores, predictions, val_weights
        )
        natural = natural_metrics(
            labels[val_indices], val_scores, predictions
        )
        record = {
            "epoch": epoch,
            "train_weighted_loss": total_weighted_loss / total_weight,
            "threshold": threshold,
            "validation_standardized_macro_f1": standardized_macro_f1,
            "validation_standardized_balanced_accuracy": standardized[
                "standardized_balanced_accuracy"
            ],
            "validation_natural_macro_f1": natural["macro_f1"],
        }
        history.append(record)
        improved = best_record is None or (
            record["validation_standardized_macro_f1"],
            record["validation_standardized_balanced_accuracy"],
            record["validation_natural_macro_f1"],
        ) > (
            best_record["validation_standardized_macro_f1"],
            best_record["validation_standardized_balanced_accuracy"],
            best_record["validation_natural_macro_f1"],
        )
        if improved:
            best_record = record
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"debiased seed={seed}, epoch={epoch}: "
            f"loss={record['train_weighted_loss']:.6f}, "
            f"val standardized macro-F1={standardized_macro_f1:.6f}"
        )
        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None or best_record is None:
        raise RuntimeError("No debiased checkpoint selected.")
    model.load_state_dict(best_state)
    best_record = dict(best_record)
    best_record["epochs_run"] = len(history)
    best_record["history"] = history
    return model, best_record


def load_original_model(
    baseline_module, seed: int, device
):
    checkpoint = torch.load(
        ORIGINAL_DIR
        / "checkpoints"
        / f"bertweet_mean_metadata_seed_{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    input_dim = int(state["network.0.weight"].shape[1])
    model = baseline_module.MLPClassifier(
        input_dim, float(checkpoint["dropout"])
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate_variant(
    name,
    seed,
    scores,
    labels,
    weights,
    threshold,
    node_ids,
):
    predictions = (scores >= threshold).astype(np.int8)
    row = {
        "model": name,
        "seed": seed,
        "threshold": threshold,
        **natural_metrics(labels, scores, predictions),
        **weighted_metrics(labels, scores, predictions, weights),
    }
    pd.DataFrame(
        {
            "node_id": node_ids,
            "label": labels,
            "score": scores.astype(np.float32),
            "prediction": predictions,
        }
    ).to_parquet(
        PREDICTION_DIR / f"{name}_seed_{seed}_test_predictions.parquet",
        index=False,
    )
    return row


def paired_standardized_bootstrap(
    labels,
    text_available,
    graph_available,
    primary_predictions,
    comparator_predictions,
):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_strata = []
    observed = []
    weights = stratum_weights(labels, text_available, graph_available)
    for primary, comparator in zip(
        primary_predictions, comparator_predictions
    ):
        primary_metrics = weighted_metrics(
            labels, primary.astype(float), primary, weights
        )
        comparator_metrics = weighted_metrics(
            labels, comparator.astype(float), comparator, weights
        )
        observed.append(
            primary_metrics["standardized_macro_f1"]
            - comparator_metrics["standardized_macro_f1"]
        )
        pattern = 2 * text_available.astype(np.int8) + graph_available.astype(np.int8)
        strata = []
        for pattern_value in np.unique(pattern):
            for label in [0, 1]:
                selected = (pattern == pattern_value) & (labels == label)
                category = 2 * primary[selected] + comparator[selected]
                counts = np.bincount(category, minlength=4).astype(float)
                strata.append((label, int(counts.sum()), counts / counts.sum()))
        seed_strata.append(strata)

    def score(sampled, selector):
        positive = (
            np.asarray([False, False, True, True])
            if selector == "primary"
            else np.asarray([False, True, False, True])
        )
        tn = fp = fn = tp = 0.0
        mass = 1.0 / len(sampled)
        for label, counts in sampled:
            fractions = counts / counts.sum()
            positive_mass = fractions[positive].sum() * mass
            negative_mass = mass - positive_mass
            if label == 0:
                fp += positive_mass
                tn += negative_mass
            else:
                tp += positive_mass
                fn += negative_mass
        bot_f1 = 2 * tp / (2 * tp + fp + fn)
        human_f1 = 2 * tn / (2 * tn + fp + fn)
        return 0.5 * (bot_f1 + human_f1)

    differences = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seed_indices = rng.integers(0, len(SEEDS), len(SEEDS))
        primary_values = []
        comparator_values = []
        for seed_index in sampled_seed_indices:
            sampled = []
            for label, total, probabilities in seed_strata[int(seed_index)]:
                sampled.append(
                    (label, rng.multinomial(total, probabilities))
                )
            primary_values.append(score(sampled, "primary"))
            comparator_values.append(score(sampled, "comparator"))
        differences[replicate] = np.mean(primary_values) - np.mean(
            comparator_values
        )
    p_value = min(
        1.0,
        2
        * min(
            (np.count_nonzero(differences <= 0) + 1)
            / (BOOTSTRAP_REPLICATES + 1),
            (np.count_nonzero(differences >= 0) + 1)
            / (BOOTSTRAP_REPLICATES + 1),
        ),
    )
    return {
        "comparison": "debiased_minus_original_rethresholded",
        "metric": "pattern-and-label-standardized macro-F1",
        "mean_difference": float(np.mean(observed)),
        "ci_lower_95": float(np.quantile(differences, 0.025)),
        "ci_upper_95": float(np.quantile(differences, 0.975)),
        "two_sided_bootstrap_p": float(p_value),
        "probability_debiased_superior": float(np.mean(differences > 0)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_module = load_module(
        "bertweet_baseline_debiased",
        SCRIPTS_DIR / "10_run_bertweet_baselines.py",
    )
    device = torch.device("cuda")
    embeddings = np.load(
        EMBEDDING_DIR / "mean_pooled_embeddings.npy", mmap_mode="r"
    )
    metadata = np.load(
        METADATA_DIR / "metadata_features.npy", mmap_mode="r"
    )
    node_ids = np.load(EMBEDDING_DIR / "node_ids.npy", mmap_mode="r")
    labels = np.load(EMBEDDING_DIR / "labels.npy", mmap_mode="r").astype(np.int8)
    split_ids = np.load(EMBEDDING_DIR / "split_ids.npy", mmap_mode="r")
    availability = np.load(
        EMBEDDING_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    users = pd.read_parquet(
        CANONICAL_DIR / "users.parquet", columns=["node_id", "graph_available"]
    ).sort_values("node_id")
    if not np.array_equal(
        users["node_id"].to_numpy(dtype=np.int64), np.asarray(node_ids)
    ):
        raise RuntimeError("Canonical node alignment failure.")
    graph_available = users["graph_available"].to_numpy(dtype=bool)
    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    train_weights = stratum_weights(
        labels[train_indices],
        availability[train_indices],
        graph_available[train_indices],
    )
    val_weights = stratum_weights(
        labels[val_indices],
        availability[val_indices],
        graph_available[val_indices],
    )
    test_weights = stratum_weights(
        labels[test_indices],
        availability[test_indices],
        graph_available[test_indices],
    )
    learning_rate, dropout = locked_hyperparameters()
    rows = []
    debiased_predictions = []
    original_predictions = []

    for seed in SEEDS:
        model, selection = fit_debiased(
            baseline_module,
            embeddings,
            metadata,
            availability,
            labels,
            train_indices,
            val_indices,
            train_weights,
            val_weights,
            learning_rate,
            dropout,
            seed,
            device,
        )
        test_scores = baseline_module.predict_scores(
            model,
            embeddings,
            metadata,
            availability,
            test_indices,
            True,
            device,
        )
        debiased_row = evaluate_variant(
            "debiased_pattern_label_weighted",
            seed,
            test_scores,
            labels[test_indices],
            test_weights,
            selection["threshold"],
            node_ids[test_indices],
        )
        debiased_row.update(
            {
                "best_epoch": selection["epoch"],
                "epochs_run": selection["epochs_run"],
                "validation_standardized_macro_f1": selection[
                    "validation_standardized_macro_f1"
                ],
            }
        )
        rows.append(debiased_row)
        debiased_predictions.append(
            (test_scores >= selection["threshold"]).astype(np.int8)
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "selection": selection,
                "learning_rate": learning_rate,
                "dropout": dropout,
            },
            CHECKPOINT_DIR / f"debiased_seed_{seed}.pt",
        )

        original_model = load_original_model(baseline_module, seed, device)
        original_val_scores = baseline_module.predict_scores(
            original_model,
            embeddings,
            metadata,
            availability,
            val_indices,
            True,
            device,
        )
        original_threshold, _ = exact_weighted_macro_f1_threshold(
            labels[val_indices], original_val_scores, val_weights
        )
        original_test_scores = baseline_module.predict_scores(
            original_model,
            embeddings,
            metadata,
            availability,
            test_indices,
            True,
            device,
        )
        original_row = evaluate_variant(
            "original_validation_standardized_threshold",
            seed,
            original_test_scores,
            labels[test_indices],
            test_weights,
            original_threshold,
            node_ids[test_indices],
        )
        rows.append(original_row)
        original_predictions.append(
            (original_test_scores >= original_threshold).astype(np.int8)
        )
        print(
            f"TEST seed={seed}: debiased standardized macro-F1="
            f"{debiased_row['standardized_macro_f1']:.6f}; original rethresholded="
            f"{original_row['standardized_macro_f1']:.6f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "seed_results.csv", index=False)
    summary = (
        results.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            natural_macro_f1_mean=("macro_f1", "mean"),
            natural_macro_f1_std=("macro_f1", "std"),
            standardized_macro_f1_mean=("standardized_macro_f1", "mean"),
            standardized_macro_f1_std=("standardized_macro_f1", "std"),
            standardized_balanced_accuracy_mean=(
                "standardized_balanced_accuracy", "mean"
            ),
            natural_auroc_mean=("auroc", "mean"),
            standardized_auroc_mean=("standardized_auroc", "mean"),
        )
        .sort_values("standardized_macro_f1_mean", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    inference = paired_standardized_bootstrap(
        labels[test_indices],
        availability[test_indices],
        graph_available[test_indices],
        debiased_predictions,
        original_predictions,
    )
    pd.DataFrame([inference]).to_csv(
        OUTPUT_DIR / "paired_inference.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "role": "missingness-debiasing sensitivity; not replacement of primary model",
        "training_weights": (
            "equal total weight for each observed semantic-text availability by "
            "user-user graph availability by label stratum in training"
        ),
        "selection": (
            "epoch and threshold selected on official validation pattern-and-label-"
            "standardized macro-F1"
        ),
        "locked_components": {
            "encoder": "frozen vinai/bertweet-base, 64 tokens, semantic mean pooling",
            "metadata": "same 10 training-fitted features",
            "architecture": "same MLP",
            "learning_rate": learning_rate,
            "dropout": dropout,
            "seeds": SEEDS,
        },
        "fair_comparator": (
            "original checkpoints re-thresholded on the identical standardized "
            "validation estimand"
        ),
        "development_note": (
            "motivated by the prespecified validation-side missingness-bias diagnostic; "
            "reported as sensitivity and requires external confirmation"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nMISSINGNESS-DEBIASED MODEL: PASS")
    print(summary.to_string(index=False))
    print("\nPaired inference:")
    print(pd.DataFrame([inference]).to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
