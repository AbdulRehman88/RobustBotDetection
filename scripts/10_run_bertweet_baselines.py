from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "08_frozen_bertweet"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

SEEDS = [42, 123, 2024, 3407, 7777]
LEARNING_RATES = [1e-3, 3e-4]
DROPOUTS = [0.2, 0.5]
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
MAX_EPOCHS = 20
PATIENCE = 4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
BOTTLENECK_DIM = 64

TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
            nn.LayerNorm(BOTTLENECK_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(BOTTLENECK_DIM, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def exact_macro_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)

    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)

    total_positive = int(sorted_labels.sum())
    total_negative = int(len(sorted_labels) - total_positive)

    boundary_indices = np.flatnonzero(
        np.r_[
            sorted_scores[:-1] != sorted_scores[1:],
            True,
        ]
    )

    tp = cumulative_tp[boundary_indices].astype(np.float64)
    fp = cumulative_fp[boundary_indices].astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp

    positive_f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2 * tp + fp + fn) != 0,
    )
    negative_f1 = np.divide(
        2 * tn,
        2 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=(2 * tn + fp + fn) != 0,
    )
    macro_f1 = (positive_f1 + negative_f1) / 2

    all_negative_f1 = (
            2 * total_negative
            / (2 * total_negative + total_positive)
    ) / 2
    best_index = int(np.argmax(macro_f1))

    if all_negative_f1 > macro_f1[best_index]:
        return float(sorted_scores[0] + 1e-6), float(all_negative_f1)

    last_positive_index = int(boundary_indices[best_index])
    if last_positive_index == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-6)
    else:
        threshold = float(
            (
                sorted_scores[last_positive_index]
                + sorted_scores[last_positive_index + 1]
            )
            / 2
        )

    return threshold, float(macro_f1[best_index])


def metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int8)
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
    }, predictions


def prepare_batch(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    include_metadata: bool,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(
        embeddings[indices], dtype=np.float32
    )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(
        values,
        np.maximum(norms, 1e-12),
        out=values,
    )

    parts = [
        values,
        availability[indices, None].astype(np.float32),
    ]
    if include_metadata:
        parts.append(
            np.asarray(metadata[indices], dtype=np.float32)
        )

    features = np.concatenate(parts, axis=1)
    return torch.from_numpy(features).to(
        device, non_blocking=True
    )


def predict_scores(
    model: nn.Module,
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    include_metadata: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []

    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH_SIZE):
            batch_indices = indices[start : start + EVAL_BATCH_SIZE]
            features = prepare_batch(
                embeddings,
                metadata,
                availability,
                batch_indices,
                include_metadata,
                device,
            )
            outputs.append(
                model(features).float().cpu().numpy()
            )

    return np.concatenate(outputs)


def fit_model(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    include_metadata: bool,
    learning_rate: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    set_seed(seed)

    input_dim = embeddings.shape[1] + 1
    if include_metadata:
        input_dim += metadata.shape[1]

    model = MLPClassifier(input_dim, dropout).to(device)

    train_labels = labels[train_indices]
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    positive_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    best_state = None
    best_record = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(
            len(train_indices),
            generator=generator,
        ).numpy()

        total_loss = 0.0
        total_examples = 0

        for start in range(0, len(permutation), BATCH_SIZE):
            positions = permutation[start : start + BATCH_SIZE]
            batch_indices = train_indices[positions]

            features = prepare_batch(
                embeddings,
                metadata,
                availability,
                batch_indices,
                include_metadata,
                device,
            )
            targets = torch.from_numpy(
                labels[batch_indices].astype(np.float32)
            ).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_indices)
            total_examples += len(batch_indices)

        val_scores = predict_scores(
            model,
            embeddings,
            metadata,
            availability,
            val_indices,
            include_metadata,
            device,
        )
        threshold, val_macro_f1 = exact_macro_f1_threshold(
            labels[val_indices], val_scores
        )
        val_auroc = roc_auc_score(
            labels[val_indices], val_scores
        )
        val_auprc = average_precision_score(
            labels[val_indices], val_scores
        )

        record = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "threshold": threshold,
            "validation_macro_f1": val_macro_f1,
            "validation_auroc": val_auroc,
            "validation_auprc": val_auprc,
        }
        history.append(record)

        improved = (
            best_record is None
            or (
                val_macro_f1,
                val_auprc,
                val_auroc,
            )
            > (
                best_record["validation_macro_f1"],
                best_record["validation_auprc"],
                best_record["validation_auroc"],
            )
        )

        if improved:
            best_record = record
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None or best_record is None:
        raise RuntimeError("No valid model checkpoint was selected.")

    model.load_state_dict(best_state)
    best_record = dict(best_record)
    best_record["epochs_run"] = len(history)
    best_record["history"] = history
    return model, best_record


def run_configuration(
    model_name: str,
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    node_ids: np.ndarray,
    include_metadata: bool,
    learning_rate: float,
    dropout: float,
    seed: int,
    device: torch.device,
    save_test: bool,
) -> dict:
    started = time.time()
    model, selection = fit_model(
        embeddings,
        metadata,
        availability,
        labels,
        train_indices,
        val_indices,
        include_metadata,
        learning_rate,
        dropout,
        seed,
        device,
    )

    result = {
        "model": model_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "best_epoch": selection["epoch"],
        "epochs_run": selection["epochs_run"],
        "threshold": selection["threshold"],
        "validation_macro_f1": selection[
            "validation_macro_f1"
        ],
        "validation_auroc": selection["validation_auroc"],
        "validation_auprc": selection["validation_auprc"],
        "runtime_seconds": time.time() - started,
    }

    if save_test:
        test_scores = predict_scores(
            model,
            embeddings,
            metadata,
            availability,
            test_indices,
            include_metadata,
            device,
        )
        test_metrics, predictions = metrics(
            labels[test_indices],
            test_scores,
            selection["threshold"],
        )
        result.update(test_metrics)

        pd.DataFrame(
            {
                "node_id": node_ids[test_indices],
                "label": labels[test_indices],
                "score": test_scores.astype(np.float32),
                "prediction": predictions,
                "text_available": availability[test_indices],
            }
        ).to_parquet(
            PREDICTION_DIR
            / f"{model_name}_seed_{seed}_test_predictions.parquet",
            index=False,
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "seed": seed,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "selection": selection,
                "include_metadata": include_metadata,
            },
            CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt",
        )

        print(
            f"TEST {model_name}, seed={seed}: "
            f"macro-F1={result['macro_f1']:.6f}, "
            f"AUROC={result['auroc']:.6f}, "
            f"AUPRC={result['auprc']:.6f}"
        )

    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    node_ids = np.load(
        EMBEDDING_DIR / "node_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        EMBEDDING_DIR / "labels.npy", mmap_mode="r"
    ).astype(np.int8)
    split_ids = np.load(
        EMBEDDING_DIR / "split_ids.npy", mmap_mode="r"
    )
    availability = np.load(
        EMBEDDING_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    metadata = np.load(
        METADATA_DIR / "metadata_features.npy", mmap_mode="r"
    )

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    configurations = {
        "bertweet_mean": {
            "path": EMBEDDING_DIR
            / "mean_pooled_embeddings.npy",
            "include_metadata": False,
        },
        "bertweet_cls": {
            "path": EMBEDDING_DIR / "cls_embeddings.npy",
            "include_metadata": False,
        },
        "bertweet_mean_metadata": {
            "path": EMBEDDING_DIR
            / "mean_pooled_embeddings.npy",
            "include_metadata": True,
        },
    }

    tuning_rows = []
    selected = {}
    final_rows = []

    for model_name, configuration in configurations.items():
        embeddings = np.load(
            configuration["path"], mmap_mode="r"
        )
        include_metadata = configuration["include_metadata"]

        print(f"\nTuning {model_name}...")
        model_tuning = []

        for learning_rate in LEARNING_RATES:
            for dropout in DROPOUTS:
                result = run_configuration(
                    model_name,
                    embeddings,
                    metadata,
                    availability,
                    labels,
                    train_indices,
                    val_indices,
                    test_indices,
                    node_ids,
                    include_metadata,
                    learning_rate,
                    dropout,
                    SEEDS[0],
                    device,
                    save_test=False,
                )
                model_tuning.append(result)
                tuning_rows.append(result)
                print(
                    f"TUNE {model_name}: lr={learning_rate:g}, "
                    f"dropout={dropout:.1f}, "
                    f"macro-F1="
                    f"{result['validation_macro_f1']:.6f}"
                )

        best = max(
            model_tuning,
            key=lambda row: (
                row["validation_macro_f1"],
                row["validation_auprc"],
                row["validation_auroc"],
            ),
        )
        selected[model_name] = {
            "learning_rate": best["learning_rate"],
            "dropout": best["dropout"],
        }

        print(
            f"Selected {model_name}: "
            f"lr={best['learning_rate']:g}, "
            f"dropout={best['dropout']:.1f}"
        )

        for seed in SEEDS:
            final_rows.append(
                run_configuration(
                    model_name,
                    embeddings,
                    metadata,
                    availability,
                    labels,
                    train_indices,
                    val_indices,
                    test_indices,
                    node_ids,
                    include_metadata,
                    best["learning_rate"],
                    best["dropout"],
                    seed,
                    device,
                    save_test=True,
                )
            )

        del embeddings
        torch.cuda.empty_cache()

    pd.DataFrame(tuning_rows).to_csv(
        OUTPUT_DIR / "validation_hyperparameter_search.csv",
        index=False,
    )

    results = pd.DataFrame(final_rows)
    results.to_csv(
        OUTPUT_DIR / "bertweet_baseline_seed_results.csv",
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

    for model_name, group in results.groupby("model"):
        row = {
            "model": model_name,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=1)
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    summary.to_csv(
        OUTPUT_DIR / "bertweet_baseline_summary.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "encoder": "frozen vinai/bertweet-base",
        "pooling_comparison": ["semantic_mean", "CLS"],
        "seeds": SEEDS,
        "positive_class": "bot",
        "training_split": "official train only",
        "model_selection_split": "official validation only",
        "test_policy": (
            "accessed only after configuration, epoch, and threshold "
            "selection"
        ),
        "embedding_normalization": (
            "per-row L2 normalization; missing text remains zero"
        ),
        "availability_indicator_appended": True,
        "selected_hyperparameters": selected,
        "architecture": {
            "hidden_dim": HIDDEN_DIM,
            "bottleneck_dim": BOTTLENECK_DIM,
            "activation": "GELU",
            "normalization": "LayerNorm",
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
        "input_hashes": {
            "mean_embeddings": sha256_file(
                EMBEDDING_DIR / "mean_pooled_embeddings.npy"
            ),
            "cls_embeddings": sha256_file(
                EMBEDDING_DIR / "cls_embeddings.npy"
            ),
            "metadata": sha256_file(
                METADATA_DIR / "metadata_features.npy"
            ),
        },
    }

    with (OUTPUT_DIR / "bertweet_baseline_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nBERTWEET BASELINES: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()