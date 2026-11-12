from __future__ import annotations

import copy
import hashlib
import json
import math
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
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIR = PROJECT_ROOT / "outputs" / "07_bertweet_tokens"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "14_finetuned_bertweet"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

MODEL_NAME = "vinai/bertweet-base"
SEEDS = [42, 123, 2024, 3407, 7777]

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 3
PATIENCE = 1
WARMUP_RATIO = 0.10
GRADIENT_CLIP_NORM = 1.0

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

    boundaries = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )

    tp = cumulative_tp[boundaries].astype(np.float64)
    fp = cumulative_fp[boundaries].astype(np.float64)
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

    all_negative_macro_f1 = (
        2 * total_negative
        / (2 * total_negative + total_positive)
    ) / 2

    best_index = int(np.argmax(macro_f1))
    if all_negative_macro_f1 > macro_f1[best_index]:
        return float(sorted_scores[0] + 1e-6), float(
            all_negative_macro_f1
        )

    boundary = int(boundaries[best_index])
    if boundary == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-6)
    else:
        threshold = float(
            (
                sorted_scores[boundary]
                + sorted_scores[boundary + 1]
            )
            / 2
        )

    return threshold, float(macro_f1[best_index])


def calculate_metrics(
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


class BERTweetClassifier(nn.Module):
    def __init__(
        self,
        special_token_ids: set[int],
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.special_token_ids = sorted(special_token_ids)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size, 1
        )

    def mean_pool(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_mask = attention_mask.bool()
        for token_id in self.special_token_ids:
            semantic_mask &= input_ids.ne(token_id)

        counts = semantic_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)

        return (
            hidden
            * semantic_mask.unsqueeze(-1).to(hidden.dtype)
        ).sum(dim=1) / counts.to(hidden.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        embeddings = self.mean_pool(
            outputs.last_hidden_state,
            input_ids,
            attention_mask,
        )
        logits = self.classifier(
            self.dropout(embeddings)
        ).squeeze(-1)

        if return_embeddings:
            return logits, embeddings
        return logits


def load_batch(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = np.asarray(
        input_ids[indices], dtype=np.int64
    )
    mask = np.asarray(
        attention_mask[indices], dtype=np.int64
    )
    return (
        torch.from_numpy(ids).to(device, non_blocking=True),
        torch.from_numpy(mask).to(
            device, non_blocking=True
        ),
    )


def predict(
    model: BERTweetClassifier,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []

    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH_SIZE):
            batch_indices = indices[
                start : start + EVAL_BATCH_SIZE
            ]
            ids, mask = load_batch(
                input_ids,
                attention_mask,
                batch_indices,
                device,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(ids, mask)
            parts.append(
                logits.float().cpu().numpy()
            )

    return np.concatenate(parts)


def extract_embeddings(
    model: BERTweetClassifier,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    available_indices: np.ndarray,
    n_users: int,
    output_path: Path,
    device: torch.device,
) -> None:
    hidden_size = int(
        model.encoder.config.hidden_size
    )
    embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_users, hidden_size),
    )
    embeddings[:] = 0

    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(available_indices), EVAL_BATCH_SIZE),
            desc=f"Extracting {output_path.stem}",
            unit="batch",
        ):
            batch_indices = available_indices[
                start : start + EVAL_BATCH_SIZE
            ]
            ids, mask = load_batch(
                input_ids,
                attention_mask,
                batch_indices,
                device,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                _, representations = model(
                    ids,
                    mask,
                    return_embeddings=True,
                )

            embeddings[batch_indices] = (
                representations.float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

    embeddings.flush()
    del embeddings


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    input_ids = np.load(
        TOKEN_DIR / "input_ids.npy", mmap_mode="r"
    )
    attention_mask = np.load(
        TOKEN_DIR / "attention_mask.npy", mmap_mode="r"
    )
    node_ids = np.load(
        TOKEN_DIR / "node_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        TOKEN_DIR / "labels.npy", mmap_mode="r"
    ).astype(np.int8)
    split_ids = np.load(
        TOKEN_DIR / "split_ids.npy", mmap_mode="r"
    )
    text_available = np.load(
        TOKEN_DIR / "text_available.npy", mmap_mode="r"
    ).astype(bool)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKEN_DIR / "tokenizer",
        use_fast=False,
        normalization=True,
        local_files_only=True,
    )
    special_token_ids = {
        token_id
        for token_id in [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
        if token_id is not None
    }

    train_indices = np.flatnonzero(
        (split_ids == TRAIN_ID) & text_available
    )
    val_indices = np.flatnonzero(
        (split_ids == VAL_ID) & text_available
    )
    test_available_indices = np.flatnonzero(
        (split_ids == TEST_ID) & text_available
    )
    test_all_indices = np.flatnonzero(
        split_ids == TEST_ID
    )
    available_indices = np.flatnonzero(text_available)

    train_positive = int(labels[train_indices].sum())
    train_negative = int(
        len(train_indices) - train_positive
    )

    steps_per_epoch = math.ceil(
        len(train_indices) / TRAIN_BATCH_SIZE
    )
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    device = torch.device("cuda")
    result_rows = []
    seed_records = []

    for seed in SEEDS:
        print(f"\nFine-tuning BERTweet, seed={seed}...")
        set_seed(seed)
        started = time.time()

        model = BERTweetClassifier(
            special_token_ids
        ).to(device)

        no_decay = ["bias", "LayerNorm.weight"]
        parameter_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not any(
                        term in name for term in no_decay
                    )
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if any(
                        term in name for term in no_decay
                    )
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=LEARNING_RATE,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda")

        positive_weight = torch.tensor(
            [train_negative / train_positive],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight
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

            progress = tqdm(
                range(0, len(permutation), TRAIN_BATCH_SIZE),
                desc=f"Seed {seed}, epoch {epoch}",
                unit="batch",
            )

            for start in progress:
                positions = permutation[
                    start : start + TRAIN_BATCH_SIZE
                ]
                batch_indices = train_indices[positions]

                ids, mask = load_batch(
                    input_ids,
                    attention_mask,
                    batch_indices,
                    device,
                )
                targets = torch.from_numpy(
                    labels[batch_indices].astype(np.float32)
                ).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(ids, mask)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += (
                    float(loss.item()) * len(batch_indices)
                )
                total_examples += len(batch_indices)

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            validation_scores = predict(
                model,
                input_ids,
                attention_mask,
                val_indices,
                device,
            )
            threshold, validation_macro_f1 = (
                exact_macro_f1_threshold(
                    labels[val_indices],
                    validation_scores,
                )
            )
            validation_auroc = roc_auc_score(
                labels[val_indices],
                validation_scores,
            )
            validation_auprc = average_precision_score(
                labels[val_indices],
                validation_scores,
            )

            record = {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "threshold": threshold,
                "validation_macro_f1": (
                    validation_macro_f1
                ),
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
            }
            history.append(record)

            print(
                f"Seed={seed}, epoch={epoch}: "
                f"loss={record['train_loss']:.6f}, "
                f"val macro-F1={validation_macro_f1:.6f}, "
                f"AUROC={validation_auroc:.6f}"
            )

            improved = (
                best_record is None
                or (
                    validation_macro_f1,
                    validation_auprc,
                    validation_auroc,
                )
                > (
                    best_record["validation_macro_f1"],
                    best_record["validation_auprc"],
                    best_record["validation_auroc"],
                )
            )

            if improved:
                best_record = record
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        if best_state is None or best_record is None:
            raise RuntimeError("No BERTweet checkpoint selected.")

        model.load_state_dict(best_state)
        model.eval()

        # Evaluate available-text test users after validation selection.
        test_available_scores = predict(
            model,
            input_ids,
            attention_mask,
            test_available_indices,
            device,
        )
        available_metrics, available_predictions = (
            calculate_metrics(
                labels[test_available_indices],
                test_available_scores,
                best_record["threshold"],
            )
        )

        # Overall text-only evaluation: missing-text users receive the
        # classifier bias, a label-independent learned fallback score.
        all_test_scores = np.full(
            len(test_all_indices),
            float(
                model.classifier.bias.detach()
                .float()
                .cpu()
                .item()
            ),
            dtype=np.float32,
        )
        test_position_lookup = np.full(
            len(labels), -1, dtype=np.int64
        )
        test_position_lookup[test_all_indices] = np.arange(
            len(test_all_indices)
        )
        available_positions = test_position_lookup[
            test_available_indices
        ]
        all_test_scores[available_positions] = (
            test_available_scores
        )

        overall_metrics, overall_predictions = (
            calculate_metrics(
                labels[test_all_indices],
                all_test_scores,
                best_record["threshold"],
            )
        )

        for scope, scope_metrics in [
            ("available_text_only", available_metrics),
            ("all_test_users", overall_metrics),
        ]:
            result_rows.append(
                {
                    "model": "finetuned_bertweet",
                    "seed": seed,
                    "scope": scope,
                    "best_epoch": best_record["epoch"],
                    "threshold": best_record["threshold"],
                    "validation_macro_f1": best_record[
                        "validation_macro_f1"
                    ],
                    **scope_metrics,
                }
            )

        pd.DataFrame(
            {
                "node_id": node_ids[test_all_indices],
                "label": labels[test_all_indices],
                "score": all_test_scores,
                "prediction": overall_predictions,
                "text_available": text_available[
                    test_all_indices
                ],
            }
        ).to_parquet(
            PREDICTION_DIR
            / f"seed_{seed}_test_predictions.parquet",
            index=False,
        )

        checkpoint_path = (
            CHECKPOINT_DIR / f"seed_{seed}.pt"
        )
        torch.save(
            {
                "state_dict": best_state,
                "seed": seed,
                "model_name": MODEL_NAME,
                "selection": best_record,
                "history": history,
                "special_token_ids": sorted(
                    special_token_ids
                ),
            },
            checkpoint_path,
        )

        embedding_path = (
            OUTPUT_DIR
            / f"finetuned_embeddings_seed_{seed}.npy"
        )
        extract_embeddings(
            model,
            input_ids,
            attention_mask,
            available_indices,
            len(labels),
            embedding_path,
            device,
        )

        seed_record = {
            "seed": seed,
            "best_epoch": best_record["epoch"],
            "validation_macro_f1": best_record[
                "validation_macro_f1"
            ],
            "available_test_macro_f1": (
                available_metrics["macro_f1"]
            ),
            "overall_test_macro_f1": (
                overall_metrics["macro_f1"]
            ),
            "overall_test_auroc": overall_metrics["auroc"],
            "overall_test_auprc": overall_metrics["auprc"],
            "runtime_seconds": time.time() - started,
            "checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
            "embedding_sha256": sha256_file(
                embedding_path
            ),
        }
        seed_records.append(seed_record)

        print(
            f"TEST seed={seed}: available macro-F1="
            f"{available_metrics['macro_f1']:.6f}; "
            f"overall macro-F1="
            f"{overall_metrics['macro_f1']:.6f}; "
            f"AUROC={overall_metrics['auroc']:.6f}"
        )

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / "finetuned_bertweet_seed_results.csv",
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

    for scope, group in results.groupby("scope"):
        row = {
            "scope": scope,
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

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "finetuned_bertweet_summary.csv",
        index=False,
    )

    np.save(
        OUTPUT_DIR / "node_ids.npy",
        np.asarray(node_ids),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "labels.npy",
        np.asarray(labels),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "split_ids.npy",
        np.asarray(split_ids),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "text_available.npy",
        text_available,
        allow_pickle=False,
    )

    manifest = {
        "status": "PASS",
        "model_name": MODEL_NAME,
        "training_scope": (
            "available-text users in official training split only"
        ),
        "model_selection_scope": (
            "available-text users in official validation split only"
        ),
        "test_policy": (
            "test labels accessed only after epoch and threshold "
            "selection"
        ),
        "seeds": SEEDS,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "warmup_ratio": WARMUP_RATIO,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "precision": "automatic mixed float16",
        },
        "pooling": (
            "mean of non-padding, non-special-token states"
        ),
        "missing_embedding": "exact all-zero vector",
        "records": seed_records,
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nFINE-TUNED BERTWEET: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()