from __future__ import annotations

import argparse
import copy
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
TEXT_DIR = PROJECT_ROOT / "outputs" / "08_frozen_bertweet"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"
GRAPH_ROOT = PROJECT_ROOT / "outputs" / "12_rgcn_embeddings"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "13_multimodal_fusion"

SEEDS = [42, 123, 2024, 3407, 7777]
MODALITIES = ["T", "G", "M"]
SCENARIOS = {
    "T": [1, 0, 0],
    "G": [0, 1, 0],
    "M": [0, 0, 1],
    "TG": [1, 1, 0],
    "TM": [1, 0, 1],
    "GM": [0, 1, 1],
    "TGM": [1, 1, 1],
}
LATENT_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3
MODALITY_DROP_RATE = 0.3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
MAX_EPOCHS = 30
PATIENCE = 5

TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusion",
        required=True,
        choices=["concat", "masked_mean", "static", "gated"],
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=["full_transductive", "split_contained"],
    )
    parser.add_argument(
        "--modality-dropout",
        required=True,
        choices=["on", "off"],
    )
    return parser.parse_args()


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


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(values),
        where=np.maximum(norms, 1e-12) != 0,
    )


class FusionModel(nn.Module):
    def __init__(self, fusion: str) -> None:
        super().__init__()
        self.fusion = fusion

        self.text_projector = nn.Sequential(
            nn.Linear(768, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.graph_projector = nn.Sequential(
            nn.Linear(64, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.metadata_projector = nn.Sequential(
            nn.Linear(10, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.null_representation = nn.Parameter(
            torch.zeros(LATENT_DIM)
        )

        if fusion == "static":
            self.static_logits = nn.Parameter(torch.zeros(3))
        elif fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(LATENT_DIM + 3, HIDDEN_DIM),
                nn.GELU(),
                nn.Linear(HIDDEN_DIM, 1),
            )

        classifier_input = (
            LATENT_DIM * 3 + 3
            if fusion == "concat"
            else LATENT_DIM + 3
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        text: torch.Tensor,
        graph: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.stack(
            [
                self.text_projector(text),
                self.graph_projector(graph),
                self.metadata_projector(metadata),
            ],
            dim=1,
        )
        float_mask = mask.to(projected.dtype)
        masked = projected * float_mask.unsqueeze(-1)

        if self.fusion == "concat":
            fused = torch.cat(
                [
                    masked.flatten(start_dim=1),
                    float_mask,
                ],
                dim=1,
            )
            weights = float_mask

        elif self.fusion == "masked_mean":
            denominator = float_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)
            representation = masked.sum(dim=1) / denominator
            all_missing = ~mask.any(dim=1)
            representation[all_missing] = (
                self.null_representation
            )
            fused = torch.cat(
                [representation, float_mask], dim=1
            )
            weights = float_mask / denominator

        else:
            if self.fusion == "static":
                logits = self.static_logits.unsqueeze(0).expand(
                    mask.shape[0], -1
                )
            elif self.fusion == "gated":
                context = float_mask.unsqueeze(1).expand(
                    -1, 3, -1
                )
                logits = self.gate(
                    torch.cat([projected, context], dim=-1)
                ).squeeze(-1)
            else:
                raise ValueError(self.fusion)

            logits = logits.masked_fill(~mask, -1e9)
            weights = torch.softmax(logits, dim=1)
            weights = weights * float_mask
            denominator = weights.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-12)
            weights = weights / denominator
            representation = (
                projected * weights.unsqueeze(-1)
            ).sum(dim=1)

            all_missing = ~mask.any(dim=1)
            representation[all_missing] = (
                self.null_representation
            )
            fused = torch.cat(
                [representation, float_mask], dim=1
            )

        return self.classifier(fused).squeeze(-1), weights


def prepare_batch(
    text_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    metadata_features: np.ndarray,
    natural_mask: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    text = l2_normalize(
        np.asarray(text_embeddings[indices], dtype=np.float32)
    )
    graph = l2_normalize(
        np.asarray(graph_embeddings[indices], dtype=np.float32)
    )
    metadata = np.asarray(
        metadata_features[indices], dtype=np.float32
    )
    mask = natural_mask[indices].astype(bool)

    return (
        torch.from_numpy(text).to(device, non_blocking=True),
        torch.from_numpy(graph).to(device, non_blocking=True),
        torch.from_numpy(metadata).to(device, non_blocking=True),
        torch.from_numpy(mask).to(device, non_blocking=True),
    )


def apply_training_dropout(
    mask: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return mask

    retained = (
        torch.rand(mask.shape, device=mask.device)
        >= MODALITY_DROP_RATE
    )
    dropped = mask & retained

    originally_available = mask.any(dim=1)
    all_dropped = originally_available & ~dropped.any(dim=1)

    if all_dropped.any():
        affected = torch.nonzero(
            all_dropped, as_tuple=False
        ).squeeze(1)
        for row in affected:
            available = torch.nonzero(
                mask[row], as_tuple=False
            ).squeeze(1)
            selected = available[
                torch.randint(
                    len(available),
                    (1,),
                    device=mask.device,
                )
            ]
            dropped[row, selected] = True

    return dropped


def predict_scenario(
    model: FusionModel,
    text_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    metadata_features: np.ndarray,
    natural_mask: np.ndarray,
    indices: np.ndarray,
    scenario_mask: np.ndarray | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_parts = []
    weight_parts = []

    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH_SIZE):
            batch_indices = indices[
                start : start + EVAL_BATCH_SIZE
            ]
            text, graph, metadata, mask = prepare_batch(
                text_embeddings,
                graph_embeddings,
                metadata_features,
                natural_mask,
                batch_indices,
                device,
            )

            if scenario_mask is not None:
                forced = torch.tensor(
                    scenario_mask,
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
                mask = mask & forced

            logits, weights = model(
                text, graph, metadata, mask
            )
            score_parts.append(
                logits.float().cpu().numpy()
            )
            weight_parts.append(
                weights.float().cpu().numpy()
            )

    return (
        np.concatenate(score_parts),
        np.concatenate(weight_parts),
    )


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    dropout_enabled = args.modality_dropout == "on"
    run_name = (
        f"{args.fusion}_{args.protocol}_"
        f"moddrop_{args.modality_dropout}"
    )
    output_dir = OUTPUT_ROOT / run_name
    prediction_dir = output_dir / "test_predictions"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    text_embeddings = np.load(
        TEXT_DIR / "mean_pooled_embeddings.npy",
        mmap_mode="r",
    )
    text_available = np.load(
        TEXT_DIR / "text_available.npy",
        mmap_mode="r",
    ).astype(bool)
    node_ids = np.load(
        TEXT_DIR / "node_ids.npy",
        mmap_mode="r",
    )
    labels = np.load(
        TEXT_DIR / "labels.npy",
        mmap_mode="r",
    ).astype(np.int8)
    split_ids = np.load(
        TEXT_DIR / "split_ids.npy",
        mmap_mode="r",
    )
    metadata_features = np.load(
        METADATA_DIR / "metadata_features.npy",
        mmap_mode="r",
    )

    graph_dir = GRAPH_ROOT / args.protocol
    graph_available = np.load(
        graph_dir / "graph_available.npy",
        mmap_mode="r",
    ).astype(bool)

    metadata_available = np.ones(
        len(labels), dtype=bool
    )
    natural_mask = np.column_stack(
        [
            text_available,
            graph_available,
            metadata_available,
        ]
    )

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    train_positive = int(labels[train_indices].sum())
    train_negative = int(
        len(train_indices) - train_positive
    )

    device = torch.device("cuda")
    result_rows = []
    selection_rows = []

    for seed in SEEDS:
        set_seed(seed)
        graph_embeddings = np.load(
            graph_dir / f"rgcn_embeddings_seed_{seed}.npy",
            mmap_mode="r",
        )

        model = FusionModel(args.fusion).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
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
        started = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            permutation = torch.randperm(
                len(train_indices),
                generator=generator,
            ).numpy()

            total_loss = 0.0
            total_examples = 0

            for start in range(
                0, len(permutation), BATCH_SIZE
            ):
                positions = permutation[
                    start : start + BATCH_SIZE
                ]
                batch_indices = train_indices[positions]

                text, graph, metadata, mask = prepare_batch(
                    text_embeddings,
                    graph_embeddings,
                    metadata_features,
                    natural_mask,
                    batch_indices,
                    device,
                )
                mask = apply_training_dropout(
                    mask, dropout_enabled
                )
                targets = torch.from_numpy(
                    labels[batch_indices].astype(np.float32)
                ).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(
                    text, graph, metadata, mask
                )
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += (
                    float(loss.item()) * len(batch_indices)
                )
                total_examples += len(batch_indices)

            natural_val_scores, _ = predict_scenario(
                model,
                text_embeddings,
                graph_embeddings,
                metadata_features,
                natural_mask,
                val_indices,
                None,
                device,
            )
            threshold, natural_macro_f1 = (
                exact_macro_f1_threshold(
                    labels[val_indices],
                    natural_val_scores,
                )
            )

            scenario_macro_f1 = {}
            for scenario_name, scenario in SCENARIOS.items():
                scenario_scores, _ = predict_scenario(
                    model,
                    text_embeddings,
                    graph_embeddings,
                    metadata_features,
                    natural_mask,
                    val_indices,
                    np.asarray(scenario, dtype=bool),
                    device,
                )
                scenario_metrics, _ = calculate_metrics(
                    labels[val_indices],
                    scenario_scores,
                    threshold,
                )
                scenario_macro_f1[scenario_name] = (
                    scenario_metrics["macro_f1"]
                )

            robust_validation_score = float(
                np.mean(list(scenario_macro_f1.values()))
            )
            natural_auroc = roc_auc_score(
                labels[val_indices],
                natural_val_scores,
            )
            natural_auprc = average_precision_score(
                labels[val_indices],
                natural_val_scores,
            )

            record = {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "threshold": threshold,
                "natural_validation_macro_f1": (
                    natural_macro_f1
                ),
                "natural_validation_auroc": natural_auroc,
                "natural_validation_auprc": natural_auprc,
                "robust_validation_macro_f1": (
                    robust_validation_score
                ),
                "scenario_macro_f1": scenario_macro_f1,
            }
            history.append(record)

            improved = (
                best_record is None
                or (
                    robust_validation_score,
                    natural_macro_f1,
                    natural_auprc,
                )
                > (
                    best_record[
                        "robust_validation_macro_f1"
                    ],
                    best_record[
                        "natural_validation_macro_f1"
                    ],
                    best_record[
                        "natural_validation_auprc"
                    ],
                )
            )

            if improved:
                best_record = record
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"{run_name}, seed={seed}, epoch={epoch}: "
                f"loss={record['train_loss']:.6f}, "
                f"natural={natural_macro_f1:.6f}, "
                f"robust={robust_validation_score:.6f}"
            )

            if epochs_without_improvement >= PATIENCE:
                break

        if best_state is None or best_record is None:
            raise RuntimeError("No fusion checkpoint selected.")

        model.load_state_dict(best_state)
        model.eval()

        selection_rows.append(
            {
                "model": run_name,
                "seed": seed,
                "best_epoch": best_record["epoch"],
                "epochs_run": len(history),
                "threshold": best_record["threshold"],
                "natural_validation_macro_f1": best_record[
                    "natural_validation_macro_f1"
                ],
                "robust_validation_macro_f1": best_record[
                    "robust_validation_macro_f1"
                ],
            }
        )

        evaluation_scenarios = {
            "natural": None,
            **{
                name: np.asarray(mask, dtype=bool)
                for name, mask in SCENARIOS.items()
            },
        }

        for scenario_name, scenario_mask in (
            evaluation_scenarios.items()
        ):
            test_scores, test_weights = predict_scenario(
                model,
                text_embeddings,
                graph_embeddings,
                metadata_features,
                natural_mask,
                test_indices,
                scenario_mask,
                device,
            )
            test_metrics, predictions = calculate_metrics(
                labels[test_indices],
                test_scores,
                best_record["threshold"],
            )

            row = {
                "model": run_name,
                "fusion": args.fusion,
                "protocol": args.protocol,
                "modality_dropout": dropout_enabled,
                "seed": seed,
                "scenario": scenario_name,
                "best_epoch": best_record["epoch"],
                "threshold": best_record["threshold"],
                "runtime_seconds": time.time() - started,
                "mean_text_weight": float(
                    test_weights[:, 0].mean()
                ),
                "mean_graph_weight": float(
                    test_weights[:, 1].mean()
                ),
                "mean_metadata_weight": float(
                    test_weights[:, 2].mean()
                ),
                **test_metrics,
            }
            result_rows.append(row)

            pd.DataFrame(
                {
                    "node_id": node_ids[test_indices],
                    "label": labels[test_indices],
                    "score": test_scores.astype(np.float32),
                    "prediction": predictions,
                    "natural_text_available": natural_mask[
                        test_indices, 0
                    ],
                    "natural_graph_available": natural_mask[
                        test_indices, 1
                    ],
                    "text_weight": test_weights[
                        :, 0
                    ].astype(np.float32),
                    "graph_weight": test_weights[
                        :, 1
                    ].astype(np.float32),
                    "metadata_weight": test_weights[
                        :, 2
                    ].astype(np.float32),
                }
            ).to_parquet(
                prediction_dir
                / f"seed_{seed}_{scenario_name}.parquet",
                index=False,
            )

            print(
                f"TEST seed={seed}, scenario={scenario_name}: "
                f"macro-F1={test_metrics['macro_f1']:.6f}, "
                f"AUROC={test_metrics['auroc']:.6f}"
            )

        torch.save(
            {
                "state_dict": best_state,
                "seed": seed,
                "fusion": args.fusion,
                "protocol": args.protocol,
                "modality_dropout": dropout_enabled,
                "selection": best_record,
                "history": history,
            },
            checkpoint_dir / f"seed_{seed}.pt",
        )

        del model, optimizer, graph_embeddings
        torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(
        output_dir / "seed_scenario_results.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "checkpoint_selection.csv",
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
            "model": run_name,
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
        row["text_weight_mean"] = float(
            group["mean_text_weight"].mean()
        )
        row["graph_weight_mean"] = float(
            group["mean_graph_weight"].mean()
        )
        row["metadata_weight_mean"] = float(
            group["mean_metadata_weight"].mean()
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "scenario"
    )
    summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "run_name": run_name,
        "fusion": args.fusion,
        "protocol": args.protocol,
        "modality_dropout": dropout_enabled,
        "modality_drop_rate": (
            MODALITY_DROP_RATE if dropout_enabled else 0.0
        ),
        "seeds": SEEDS,
        "modalities": {
            "T": "frozen mean-pooled BERTweet semantic text",
            "G": "validation-selected topology-only R-GCN",
            "M": "train-fitted standardized account metadata",
        },
        "selection": (
            "highest mean validation macro-F1 across all seven "
            "forced modality scenarios; single threshold selected "
            "on natural validation data"
        ),
        "test_policy": (
            "test evaluated only after checkpoint and threshold "
            "selection"
        ),
        "architecture": {
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
    }

    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nMULTIMODAL FUSION: PASS ({run_name})")
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
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()