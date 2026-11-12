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
FEATURE_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"
EMBEDDING_ROOT = PROJECT_ROOT / "outputs" / "31_twibot20_external_bertweet"
SOURCE_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "09_bertweet_baselines" / "checkpoints"
DEBIASED_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "27_missingness_debiased_model" / "checkpoints"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "32_twibot20_external_transfer"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
REPLICATION_CHECKPOINT_ROOT = OUTPUT_ROOT / "replication_checkpoints"

SEEDS = [42, 123, 2024, 3407, 7777]
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
MAX_EPOCHS = 20
PATIENCE = 4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
BOTTLENECK_DIM = 64
TRAIN_ID, VAL_ID, TEST_ID = 0, 1, 2
LOCKED_REPLICATION_LR = 3e-4
LOCKED_REPLICATION_DROPOUT = 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
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


def exact_macro_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    total_positive = int(sorted_labels.sum())
    total_negative = int(len(sorted_labels) - total_positive)
    boundaries = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = cumulative_tp[boundaries].astype(np.float64)
    fp = cumulative_fp[boundaries].astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp
    positive_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    negative_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) != 0)
    macro_f1 = (positive_f1 + negative_f1) / 2
    all_negative_f1 = (2 * total_negative / (2 * total_negative + total_positive)) / 2
    best = int(np.argmax(macro_f1))
    if all_negative_f1 > macro_f1[best]:
        return float(sorted_scores[0] + 1e-6), float(all_negative_f1)
    last_positive = int(boundaries[best])
    if last_positive == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-6)
    else:
        threshold = float((sorted_scores[last_positive] + sorted_scores[last_positive + 1]) / 2)
    return threshold, float(macro_f1[best])


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision_bot": precision_score(labels, predictions, zero_division=0),
        "recall_bot": recall_score(labels, predictions, zero_division=0),
        "specificity_human": tn / (tn + fp),
        "f1_bot": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, predictions


def prepare_batch(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    include_metadata: bool,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, np.maximum(norms, 1e-12), out=values)
    parts = [values, availability[indices, None].astype(np.float32)]
    if include_metadata:
        parts.append(np.asarray(metadata[indices], dtype=np.float32))
    return torch.from_numpy(np.concatenate(parts, axis=1)).to(device, non_blocking=True)


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
            selected = indices[start:start + EVAL_BATCH_SIZE]
            features = prepare_batch(
                embeddings, metadata, availability, selected, include_metadata, device
            )
            outputs.append(model(features).float().cpu().numpy())
    return np.concatenate(outputs)


def load_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dropout = float(checkpoint["dropout"])
    model = MLPClassifier(input_dim, dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def save_predictions(
    filename: str,
    external_ids: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    availability: np.ndarray,
    threshold: float,
) -> None:
    pd.DataFrame({
        "external_id": external_ids[indices],
        "label": labels[indices],
        "score": scores.astype(np.float32),
        "prediction": predictions,
        "text_available": availability[indices],
        "threshold": threshold,
    }).to_parquet(PREDICTION_ROOT / filename, index=False)


def evaluate_source_family(
    family: str,
    checkpoint_pattern: str,
    include_metadata: bool,
    embeddings: np.ndarray,
    source_metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    split_ids: np.ndarray,
    external_ids: np.ndarray,
    device: torch.device,
) -> list[dict]:
    rows = []
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    input_dim = embeddings.shape[1] + 1 + (source_metadata.shape[1] if include_metadata else 0)

    for seed in SEEDS:
        if family == "debiased_text_metadata":
            checkpoint_path = DEBIASED_CHECKPOINT_ROOT / checkpoint_pattern.format(seed=seed)
        else:
            checkpoint_path = SOURCE_CHECKPOINT_ROOT / checkpoint_pattern.format(seed=seed)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, checkpoint = load_model(checkpoint_path, input_dim, device)
        selection = checkpoint["selection"]
        source_threshold = float(selection["threshold"])
        val_scores = predict_scores(
            model, embeddings, source_metadata, availability, val_indices, include_metadata, device
        )
        adapted_threshold, adapted_val_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
        test_scores = predict_scores(
            model, embeddings, source_metadata, availability, test_indices, include_metadata, device
        )

        for protocol, threshold in [
            ("strict_zero_shot_source_threshold", source_threshold),
            ("validation_threshold_adaptation", adapted_threshold),
        ]:
            result, predictions = compute_metrics(labels[test_indices], test_scores, threshold)
            row = {
                "model": family,
                "protocol": protocol,
                "seed": seed,
                "threshold": threshold,
                "source_validation_threshold": source_threshold,
                "external_validation_selected_threshold": adapted_threshold,
                "external_validation_macro_f1_at_adapted_threshold": adapted_val_f1,
                "checkpoint_sha256": sha256(checkpoint_path),
                **result,
            }
            rows.append(row)
            save_predictions(
                f"{family}_{protocol}_seed_{seed}.parquet",
                external_ids, test_indices, labels, test_scores, predictions,
                availability, threshold,
            )
            print(
                f"EXTERNAL {family}, {protocol}, seed={seed}: "
                f"macro-F1={result['macro_f1']:.6f}, AUROC={result['auroc']:.6f}, "
                f"AUPRC={result['auprc']:.6f}"
            )
        del model
    return rows


def fit_locked_replication(
    seed: int,
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = MLPClassifier(embeddings.shape[1] + 1 + metadata.shape[1], LOCKED_REPLICATION_DROPOUT).to(device)
    train_labels = labels[train_indices]
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LOCKED_REPLICATION_LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_record = None
    no_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator).numpy()
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            selected = train_indices[permutation[start:start + BATCH_SIZE]]
            features = prepare_batch(
                embeddings, metadata, availability, selected, True, device
            )
            targets = torch.from_numpy(labels[selected].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
            total_examples += len(selected)

        val_scores = predict_scores(
            model, embeddings, metadata, availability, val_indices, True, device
        )
        threshold, val_macro_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "threshold": threshold,
            "validation_macro_f1": val_macro_f1,
            "validation_auroc": roc_auc_score(labels[val_indices], val_scores),
            "validation_auprc": average_precision_score(labels[val_indices], val_scores),
        }
        history.append(record)
        improved = best_record is None or (
            record["validation_macro_f1"], record["validation_auprc"], record["validation_auroc"]
        ) > (
            best_record["validation_macro_f1"], best_record["validation_auprc"], best_record["validation_auroc"]
        )
        if improved:
            best_record = record
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        print(
            f"replication seed={seed}, epoch={epoch}: loss={record['train_loss']:.6f}, "
            f"val macro-F1={val_macro_f1:.6f}"
        )
        if no_improvement >= PATIENCE:
            break

    if best_state is None or best_record is None:
        raise RuntimeError("No locked replication checkpoint selected.")
    model.load_state_dict(best_state)
    selection = dict(best_record)
    selection["epochs_run"] = len(history)
    selection["history"] = history
    return model, selection


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    REPLICATION_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    embeddings = np.load(EMBEDDING_ROOT / "mean_pooled_embeddings.npy", mmap_mode="r")
    source_metadata = np.load(FEATURE_ROOT / "metadata_twibot22_scaled.npy", mmap_mode="r")
    replication_metadata = np.load(
        FEATURE_ROOT / "metadata_twibot20_train_scaled.npy", mmap_mode="r"
    )
    availability = np.load(FEATURE_ROOT / "text_available.npy", mmap_mode="r").astype(bool)
    labels = np.load(FEATURE_ROOT / "labels.npy", mmap_mode="r").astype(np.int8)
    split_ids = np.load(FEATURE_ROOT / "split_ids.npy", mmap_mode="r").astype(np.int8)
    external_ids = np.load(FEATURE_ROOT / "external_ids.npy", mmap_mode="r")

    n = len(labels)
    if any(len(array) != n for array in [embeddings, source_metadata, replication_metadata, availability, split_ids, external_ids]):
        raise RuntimeError("External feature arrays are not row aligned.")
    if embeddings.shape[1] != 768 or source_metadata.shape[1] != 10:
        raise RuntimeError("Unexpected external feature dimensions.")
    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    if (len(train_indices), len(val_indices), len(test_indices)) != (8278, 2365, 1183):
        raise RuntimeError(
            f"Unexpected TwiBot-20 split sizes: {(len(train_indices), len(val_indices), len(test_indices))}"
        )

    rows = []
    rows += evaluate_source_family(
        "bertweet_mean_metadata", "bertweet_mean_metadata_seed_{seed}.pt", True,
        embeddings, source_metadata, availability, labels, split_ids, external_ids, device,
    )
    rows += evaluate_source_family(
        "bertweet_mean", "bertweet_mean_seed_{seed}.pt", False,
        embeddings, source_metadata, availability, labels, split_ids, external_ids, device,
    )
    rows += evaluate_source_family(
        "debiased_text_metadata", "debiased_seed_{seed}.pt", True,
        embeddings, source_metadata, availability, labels, split_ids, external_ids, device,
    )

    for seed in SEEDS:
        model, selection = fit_locked_replication(
            seed, embeddings, replication_metadata, availability, labels,
            train_indices, val_indices, device,
        )
        test_scores = predict_scores(
            model, embeddings, replication_metadata, availability, test_indices, True, device
        )
        result, predictions = compute_metrics(labels[test_indices], test_scores, selection["threshold"])
        row = {
            "model": "bertweet_mean_metadata",
            "protocol": "locked_within_twibot20_replication",
            "seed": seed,
            "threshold": selection["threshold"],
            "source_validation_threshold": np.nan,
            "external_validation_selected_threshold": selection["threshold"],
            "external_validation_macro_f1_at_adapted_threshold": selection["validation_macro_f1"],
            "checkpoint_sha256": "generated_below",
            "best_epoch": selection["epoch"],
            "epochs_run": selection["epochs_run"],
            **result,
        }
        checkpoint_path = REPLICATION_CHECKPOINT_ROOT / f"locked_replication_seed_{seed}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "seed": seed,
            "learning_rate": LOCKED_REPLICATION_LR,
            "dropout": LOCKED_REPLICATION_DROPOUT,
            "selection": selection,
            "preprocessing": "TwiBot-20 official training split only",
            "hyperparameter_source": "locked from TwiBot-22 validation before external evaluation",
        }, checkpoint_path)
        row["checkpoint_sha256"] = sha256(checkpoint_path)
        rows.append(row)
        save_predictions(
            f"bertweet_mean_metadata_locked_replication_seed_{seed}.parquet",
            external_ids, test_indices, labels, test_scores, predictions,
            availability, selection["threshold"],
        )
        print(
            f"EXTERNAL locked replication, seed={seed}: macro-F1={result['macro_f1']:.6f}, "
            f"AUROC={result['auroc']:.6f}, AUPRC={result['auprc']:.6f}"
        )
        del model

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(OUTPUT_ROOT / "external_transfer_seed_results.csv", index=False)
    metric_names = [
        "accuracy", "balanced_accuracy", "precision_bot", "recall_bot",
        "specificity_human", "f1_bot", "macro_f1", "mcc", "auroc", "auprc",
    ]
    summary_rows = []
    for (model, protocol), group in seed_results.groupby(["model", "protocol"], sort=True):
        row = {"model": model, "protocol": protocol, "seeds": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["protocol", "macro_f1_mean"], ascending=[True, False])
    summary.to_csv(OUTPUT_ROOT / "external_transfer_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "external_test_users": len(test_indices),
        "protocols": {
            "strict_zero_shot_source_threshold": "TwiBot-22 preprocessing, checkpoint, and threshold; no TwiBot-20 labels used",
            "validation_threshold_adaptation": "TwiBot-22 preprocessing and checkpoint frozen; threshold selected on TwiBot-20 validation",
            "locked_within_twibot20_replication": "architecture and hyperparameters locked from TwiBot-22; preprocessing/training/threshold use official TwiBot-20 train/validation only",
        },
        "primary_external_protocol": "strict_zero_shot_source_threshold",
        "external_test_never_used_for_training_selection_or_thresholding": True,
        "sequence_length": 64,
        "known_external_token_saturation_rate": 0.670616,
        "seeds": SEEDS,
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "external_transfer_seed_results.csv": sha256(OUTPUT_ROOT / "external_transfer_seed_results.csv"),
            "external_transfer_summary.csv": sha256(OUTPUT_ROOT / "external_transfer_summary.csv"),
        },
    }
    (OUTPUT_ROOT / "external_transfer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL TRANSFER EVALUATION: PASS")
    print(summary[["model", "protocol", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auprc_mean"]].to_string(index=False))
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
