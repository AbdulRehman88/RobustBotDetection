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
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "10_graph_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "11_graph_baselines"

SEEDS = [42, 123, 2024, 3407, 7777]
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 100
PATIENCE = 12
NUM_RELATIONS = 4

TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["mlp", "gcn", "graphsage", "rgcn"],
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=["full_transductive", "split_contained"],
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
        np.r_[
            sorted_scores[:-1] != sorted_scores[1:],
            True,
        ]
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
        (
            2 * total_negative
            / (2 * total_negative + total_positive)
        )
        / 2
    )

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


class StructuralMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, edge_type
        return self.network(x).squeeze(-1)


class GCNModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x, edge_index, edge_type):
        del edge_type
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        residual = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x + residual)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        return self.classifier(x).squeeze(-1)


class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, HIDDEN_DIM)
        self.conv2 = SAGEConv(HIDDEN_DIM, HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x, edge_index, edge_type):
        del edge_type
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        residual = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x + residual)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        return self.classifier(x).squeeze(-1)


class RGCNModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv1 = RGCNConv(
            input_dim,
            HIDDEN_DIM,
            num_relations=NUM_RELATIONS,
            num_bases=NUM_RELATIONS,
        )
        self.conv2 = RGCNConv(
            HIDDEN_DIM,
            HIDDEN_DIM,
            num_relations=NUM_RELATIONS,
            num_bases=NUM_RELATIONS,
        )
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        residual = x
        x = self.conv2(x, edge_index, edge_type)
        x = self.norm2(x)
        x = torch.relu(x + residual)
        x = nn.functional.dropout(
            x, p=DROPOUT, training=self.training
        )
        return self.classifier(x).squeeze(-1)


def build_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "mlp":
        return StructuralMLP(input_dim)
    if model_name == "gcn":
        return GCNModel(input_dim)
    if model_name == "graphsage":
        return GraphSAGEModel(input_dim)
    if model_name == "rgcn":
        return RGCNModel(input_dim)
    raise ValueError(model_name)


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    run_name = f"{args.model}_{args.protocol}"
    run_dir = OUTPUT_DIR / run_name
    prediction_dir = run_dir / "test_predictions"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = INPUT_DIR / args.protocol

    node_features_np = np.load(
        protocol_dir / "node_features.npy",
        mmap_mode="r",
    )
    edge_index_np = np.load(
        protocol_dir / "edge_index.npy",
        mmap_mode="r",
    )
    edge_type_np = np.load(
        protocol_dir / "edge_type.npy",
        mmap_mode="r",
    )
    graph_available = np.load(
        protocol_dir / "graph_available.npy",
        mmap_mode="r",
    ).astype(bool)

    node_ids = np.load(
        INPUT_DIR / "node_ids.npy", mmap_mode="r"
    )
    labels_np = np.load(
        INPUT_DIR / "labels.npy", mmap_mode="r"
    ).astype(np.int8)
    split_ids = np.load(
        INPUT_DIR / "split_ids.npy", mmap_mode="r"
    )

    train_indices_np = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices_np = np.flatnonzero(split_ids == VAL_ID)
    test_indices_np = np.flatnonzero(split_ids == TEST_ID)

    device = torch.device("cuda")
    print(f"Loading {args.protocol} graph onto GPU...")

    x = torch.from_numpy(
        np.asarray(node_features_np, dtype=np.float32)
    ).to(device)
    edge_index = torch.from_numpy(
        np.asarray(edge_index_np, dtype=np.int64)
    ).to(device)
    edge_type = torch.from_numpy(
        np.asarray(edge_type_np, dtype=np.int64)
    ).to(device)
    labels = torch.from_numpy(labels_np.astype(np.float32)).to(
        device
    )
    train_indices = torch.from_numpy(train_indices_np).to(device)
    val_indices = torch.from_numpy(val_indices_np).to(device)
    test_indices = torch.from_numpy(test_indices_np).to(device)

    train_positive = int(labels_np[train_indices_np].sum())
    train_negative = int(
        len(train_indices_np) - train_positive
    )
    positive_weight = torch.tensor(
        [train_negative / train_positive],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    result_rows = []

    for seed in SEEDS:
        set_seed(seed)
        model = build_model(
            args.model, node_features_np.shape[1]
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        best_state = None
        best_record = None
        epochs_without_improvement = 0
        history = []
        started = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index, edge_type)
            loss = criterion(
                logits[train_indices],
                labels[train_indices],
            )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                evaluation_logits = model(
                    x, edge_index, edge_type
                )
                val_scores = (
                    evaluation_logits[val_indices]
                    .float()
                    .cpu()
                    .numpy()
                )

            threshold, val_macro_f1 = (
                exact_macro_f1_threshold(
                    labels_np[val_indices_np],
                    val_scores,
                )
            )
            val_auroc = roc_auc_score(
                labels_np[val_indices_np], val_scores
            )
            val_auprc = average_precision_score(
                labels_np[val_indices_np], val_scores
            )

            record = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
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
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"{run_name}, seed={seed}, epoch={epoch}: "
                    f"loss={loss.item():.6f}, "
                    f"val macro-F1={val_macro_f1:.6f}"
                )

            del logits, evaluation_logits
            torch.cuda.empty_cache()

            if epochs_without_improvement >= PATIENCE:
                break

        if best_state is None or best_record is None:
            raise RuntimeError("No checkpoint selected.")

        model.load_state_dict(best_state)
        model.eval()

        # Test labels are accessed only after epoch and threshold have
        # been fixed using training and validation data.
        with torch.inference_mode():
            final_logits = model(x, edge_index, edge_type)
            test_scores = (
                final_logits[test_indices]
                .float()
                .cpu()
                .numpy()
            )

        test_metrics, predictions = calculate_metrics(
            labels_np[test_indices_np],
            test_scores,
            best_record["threshold"],
        )

        result = {
            "model": args.model,
            "protocol": args.protocol,
            "seed": seed,
            "best_epoch": best_record["epoch"],
            "epochs_run": len(history),
            "threshold": best_record["threshold"],
            "validation_macro_f1": best_record[
                "validation_macro_f1"
            ],
            "validation_auroc": best_record[
                "validation_auroc"
            ],
            "validation_auprc": best_record[
                "validation_auprc"
            ],
            "runtime_seconds": time.time() - started,
            **test_metrics,
        }
        result_rows.append(result)

        pd.DataFrame(
            {
                "node_id": node_ids[test_indices_np],
                "label": labels_np[test_indices_np],
                "score": test_scores.astype(np.float32),
                "prediction": predictions,
                "graph_available": graph_available[
                    test_indices_np
                ],
            }
        ).to_parquet(
            prediction_dir
            / f"{run_name}_seed_{seed}_test_predictions.parquet",
            index=False,
        )

        torch.save(
            {
                "state_dict": best_state,
                "model": args.model,
                "protocol": args.protocol,
                "seed": seed,
                "selection": best_record,
                "history": history,
            },
            checkpoint_dir / f"{run_name}_seed_{seed}.pt",
        )

        print(
            f"TEST {run_name}, seed={seed}: "
            f"macro-F1={test_metrics['macro_f1']:.6f}, "
            f"AUROC={test_metrics['auroc']:.6f}, "
            f"AUPRC={test_metrics['auprc']:.6f}"
        )

        del model, optimizer, final_logits
        torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(
        run_dir / "seed_results.csv", index=False
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
    summary = {
        "model": args.model,
        "protocol": args.protocol,
        "seeds": len(SEEDS),
    }
    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(
            results[metric].mean()
        )
        summary[f"{metric}_std"] = float(
            results[metric].std(ddof=1)
        )

    pd.DataFrame([summary]).to_csv(
        run_dir / "summary.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "model": args.model,
        "protocol": args.protocol,
        "seeds": SEEDS,
        "node_features": "topology only",
        "num_relations": NUM_RELATIONS,
        "training_labels": "official train nodes only",
        "model_selection_labels": (
            "official validation nodes only"
        ),
        "test_policy": (
            "test accessed after checkpoint and threshold selection"
        ),
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "optimizer": "AdamW",
            "loss": "class-weighted BCEWithLogitsLoss",
        },
        "claim_boundary": (
            "transductive node classification"
            if args.protocol == "full_transductive"
            else "split-contained inductive sensitivity"
        ),
    }
    with (run_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nGRAPH BASELINE: PASS ({run_name})")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()