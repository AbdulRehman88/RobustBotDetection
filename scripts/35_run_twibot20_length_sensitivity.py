from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"
SOURCE_TOKENIZER_ROOT = PROJECT_ROOT / "outputs" / "07_bertweet_tokens" / "tokenizer"
SOURCE_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "09_bertweet_baselines" / "checkpoints"
DEBIASED_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "27_missingness_debiased_model" / "checkpoints"
LENGTH64_RESULT_ROOT = PROJECT_ROOT / "outputs" / "32_twibot20_external_transfer"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "34_twibot20_length_sensitivity"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions_128"

MODEL_NAME = "vinai/bertweet-base"
SEEDS = [42, 123, 2024, 3407, 7777]
MAX_LENGTH = 128
TOKEN_BATCH_SIZE = 4096
ENCODE_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 8192
HIDDEN_DIM = 256
BOTTLENECK_DIM = 64
VAL_ID, TEST_ID = 1, 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    pos_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    neg_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) != 0)
    macro = (pos_f1 + neg_f1) / 2
    all_negative = (2 * total_negative / (2 * total_negative + total_positive)) / 2
    best = int(np.argmax(macro))
    if all_negative > macro[best]:
        return float(sorted_scores[0] + 1e-6), float(all_negative)
    last = int(boundaries[best])
    threshold = (
        float(sorted_scores[-1] - 1e-6)
        if last == len(sorted_scores) - 1
        else float((sorted_scores[last] + sorted_scores[last + 1]) / 2)
    )
    return threshold, float(macro[best])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
    }


def prepare_features(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, np.maximum(norms, 1e-12), out=values)
    features = np.concatenate(
        [values, availability[indices, None].astype(np.float32), np.asarray(metadata[indices], dtype=np.float32)],
        axis=1,
    )
    return torch.from_numpy(features).to(device, non_blocking=True)


def predict(
    model: nn.Module,
    embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH_SIZE):
            selected = indices[start:start + EVAL_BATCH_SIZE]
            outputs.append(
                model(prepare_features(embeddings, metadata, availability, selected, device))
                .float().cpu().numpy()
            )
    return np.concatenate(outputs)


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    users = pd.read_parquet(
        FEATURE_ROOT / "users.parquet",
        columns=["row_id", "external_id", "semantic_text", "text_available", "label_id", "split_id"],
    ).sort_values("row_id").reset_index(drop=True)
    metadata = np.load(FEATURE_ROOT / "metadata_twibot22_scaled.npy", mmap_mode="r")
    labels = users["label_id"].to_numpy(dtype=np.int8)
    split_ids = users["split_id"].to_numpy(dtype=np.int8)
    availability = users["text_available"].to_numpy(dtype=bool)
    external_ids = users["external_id"].to_numpy(dtype=str)
    text = users["semantic_text"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_TOKENIZER_ROOT, use_fast=False, normalization=True, local_files_only=True
    )
    input_ids = np.empty((len(users), MAX_LENGTH), dtype=np.int32)
    attention_mask = np.empty((len(users), MAX_LENGTH), dtype=np.uint8)
    print("Tokenizing TwiBot-20 at 128 tokens...")
    for start in tqdm(range(0, len(users), TOKEN_BATCH_SIZE), unit="batch"):
        stop = min(start + TOKEN_BATCH_SIZE, len(users))
        encoded = tokenizer(
            text[start:stop], add_special_tokens=True, padding="max_length",
            truncation=True, max_length=MAX_LENGTH, return_attention_mask=True,
            return_tensors="np",
        )
        input_ids[start:stop] = encoded["input_ids"]
        attention_mask[start:stop] = encoded["attention_mask"]
    saturated = (attention_mask.sum(axis=1) == MAX_LENGTH) & availability

    print(f"Loading locked encoder: {MODEL_NAME}")
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device=device, dtype=torch.float16)
    encoder.eval()
    embeddings = np.zeros((len(users), int(encoder.config.hidden_size)), dtype=np.float16)
    available_indices = np.flatnonzero(availability)
    special_ids = {
        token_id for token_id in [
            tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
            tokenizer.bos_token_id, tokenizer.eos_token_id,
        ] if token_id is not None
    }
    print("Encoding TwiBot-20 at 128 tokens...")
    with torch.inference_mode():
        for start in tqdm(range(0, len(available_indices), ENCODE_BATCH_SIZE), unit="batch"):
            selected = available_indices[start:start + ENCODE_BATCH_SIZE]
            ids = torch.from_numpy(input_ids[selected].astype(np.int64, copy=False)).to(device)
            masks = torch.from_numpy(attention_mask[selected].astype(np.int64, copy=False)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = encoder(input_ids=ids, attention_mask=masks, return_dict=True).last_hidden_state
                semantic_mask = masks.bool()
                for token_id in special_ids:
                    semantic_mask &= ids.ne(token_id)
                counts = semantic_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (hidden * semantic_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / counts.to(hidden.dtype)
            embeddings[selected] = pooled.float().cpu().numpy().astype(np.float16, copy=False)
    del encoder

    if not np.isfinite(embeddings[available_indices]).all() or np.any(embeddings[~availability] != 0):
        raise RuntimeError("Invalid 128-token embeddings.")
    np.save(OUTPUT_ROOT / "mean_pooled_embeddings_128.npy", embeddings)
    np.save(OUTPUT_ROOT / "saturated_at_128.npy", saturated)

    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    rows = []
    families = [
        ("bertweet_mean_metadata", SOURCE_CHECKPOINT_ROOT, "bertweet_mean_metadata_seed_{seed}.pt"),
        ("debiased_text_metadata", DEBIASED_CHECKPOINT_ROOT, "debiased_seed_{seed}.pt"),
    ]
    for family, checkpoint_root, pattern in families:
        for seed in SEEDS:
            checkpoint_path = checkpoint_root / pattern.format(seed=seed)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = MLPClassifier(779, float(checkpoint["dropout"])).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            source_threshold = float(checkpoint["selection"]["threshold"])
            val_scores = predict(model, embeddings, metadata, availability, val_indices, device)
            adapted_threshold, val_macro_f1 = exact_macro_f1_threshold(labels[val_indices], val_scores)
            test_scores = predict(model, embeddings, metadata, availability, test_indices, device)
            for protocol, threshold in [
                ("strict_zero_shot_source_threshold", source_threshold),
                ("validation_threshold_adaptation", adapted_threshold),
            ]:
                result = metrics(labels[test_indices], test_scores, threshold)
                predictions = (test_scores >= threshold).astype(np.int8)
                rows.append({
                    "max_length": 128, "model": family, "protocol": protocol,
                    "seed": seed, "threshold": threshold,
                    "external_validation_macro_f1": val_macro_f1, **result,
                })
                pd.DataFrame({
                    "external_id": external_ids[test_indices], "label": labels[test_indices],
                    "score": test_scores.astype(np.float32), "prediction": predictions,
                    "threshold": threshold,
                }).to_parquet(
                    PREDICTION_ROOT / f"{family}_{protocol}_seed_{seed}.parquet", index=False
                )
                print(
                    f"128 {family}, {protocol}, seed={seed}: "
                    f"macro-F1={result['macro_f1']:.6f}, AUROC={result['auroc']:.6f}"
                )
            del model

    results128 = pd.DataFrame(rows)
    results128.to_csv(OUTPUT_ROOT / "length128_seed_results.csv", index=False)
    summary128 = (
        results128.groupby(["max_length", "model", "protocol"], as_index=False)
        .agg(
            seeds=("seed", "size"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
        )
    )
    results64 = pd.read_csv(LENGTH64_RESULT_ROOT / "external_transfer_seed_results.csv")
    results64 = results64[
        results64["model"].isin(["bertweet_mean_metadata", "debiased_text_metadata"])
        & results64["protocol"].isin([
            "strict_zero_shot_source_threshold", "validation_threshold_adaptation"
        ])
    ].copy()
    results64["max_length"] = 64
    combined = pd.concat([
        results64[["max_length", "model", "protocol", "seed", "macro_f1", "balanced_accuracy", "auroc", "auprc"]],
        results128[["max_length", "model", "protocol", "seed", "macro_f1", "balanced_accuracy", "auroc", "auprc"]],
    ], ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "length64_128_seed_results.csv", index=False)
    combined_summary = (
        combined.groupby(["max_length", "model", "protocol"], as_index=False)
        .agg(
            seeds=("seed", "size"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
        )
    )
    combined_summary.to_csv(OUTPUT_ROOT / "length64_128_summary.csv", index=False)

    token_summary = pd.DataFrame([
        {
            "max_length": 64,
            "available_text_users": int(availability.sum()),
            "saturation_count": 7301,
            "saturation_rate": 7301 / int(availability.sum()),
            "source": "locked Script 31 result",
        },
        {
            "max_length": 128,
            "available_text_users": int(availability.sum()),
            "saturation_count": int(saturated.sum()),
            "saturation_rate": float(saturated.sum() / availability.sum()),
            "source": "Script 35",
        },
    ])
    token_summary.to_csv(OUTPUT_ROOT / "token_saturation_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "primary_locked_length": 64,
        "sensitivity_length": 128,
        "no_hyperparameter_retuning": True,
        "models": ["bertweet_mean_metadata", "debiased_text_metadata"],
        "seeds": SEEDS,
        "runtime_minutes": (time.time() - started) / 60,
        "output_sha256": {
            "mean_pooled_embeddings_128.npy": sha256(OUTPUT_ROOT / "mean_pooled_embeddings_128.npy"),
            "length64_128_seed_results.csv": sha256(OUTPUT_ROOT / "length64_128_seed_results.csv"),
            "length64_128_summary.csv": sha256(OUTPUT_ROOT / "length64_128_summary.csv"),
            "token_saturation_summary.csv": sha256(OUTPUT_ROOT / "token_saturation_summary.csv"),
        },
    }
    (OUTPUT_ROOT / "external_length_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL LENGTH SENSITIVITY: PASS")
    print(token_summary.to_string(index=False))
    print(combined_summary.to_string(index=False))
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
