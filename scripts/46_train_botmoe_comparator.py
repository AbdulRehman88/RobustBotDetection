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
import torch.nn as nn
import torch.nn.functional as functional
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
OUTPUT_DIR = OUTPUT_ROOT / "botmoe_common_evidence"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

TRAIN_ID = 0
VALIDATION_ID = 1
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
    if torch.cuda.is_available():
        # Sequence length is only three modalities. The math SDP
        # backend is inexpensive here and avoids the nondeterministic
        # memory-efficient attention backward kernel.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
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
        2 * total_negative / (2 * total_negative + total_positive)
    ) / 2
    best_index = int(np.argmax(macro_f1))
    if all_negative_macro_f1 > macro_f1[best_index]:
        return float(sorted_scores[0] + 1e-12), float(
            all_negative_macro_f1
        )

    boundary = int(boundaries[best_index])
    if boundary == len(sorted_scores) - 1:
        threshold = float(sorted_scores[-1] - 1e-12)
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


class ExpertNetwork(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class StraightThroughSparseMoE(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        number_of_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.number_of_experts = number_of_experts
        self.alignment = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dimension, number_of_experts)
        self.experts = nn.ModuleList(
            [
                ExpertNetwork(hidden_dimension, dropout)
                for _ in range(number_of_experts)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aligned = self.alignment(inputs)
        soft_gate = torch.softmax(self.gate(aligned), dim=1)
        selected = torch.argmax(soft_gate, dim=1)
        hard_gate = functional.one_hot(
            selected, num_classes=self.number_of_experts
        ).to(dtype=soft_gate.dtype)

        # Hard top-1 routing is used in the forward pass, while the
        # straight-through term preserves gradients for the gate.
        routing = hard_gate + soft_gate - soft_gate.detach()
        expert_outputs = torch.stack(
            [expert(aligned) for expert in self.experts],
            dim=1,
        )
        representation = torch.sum(
            routing.unsqueeze(-1) * expert_outputs,
            dim=1,
        )
        representation = representation * available.unsqueeze(1)

        if bool(available.any()):
            importance = soft_gate[available].mean(dim=0)
            balance_loss = (
                self.number_of_experts
                * torch.sum(importance.square())
                - 1.0
            ).clamp_min(0.0)
        else:
            balance_loss = aligned.new_zeros(())

        return representation, soft_gate, balance_loss


class BotMoECommonEvidence(nn.Module):
    def __init__(
        self,
        text_dimension: int,
        graph_dimension: int,
        metadata_dimension: int,
        hidden_dimension: int,
        text_experts: int,
        graph_experts: int,
        metadata_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_moe = StraightThroughSparseMoE(
            text_dimension,
            hidden_dimension,
            text_experts,
            dropout,
        )
        self.graph_moe = StraightThroughSparseMoE(
            graph_dimension,
            hidden_dimension,
            graph_experts,
            dropout,
        )
        self.metadata_moe = StraightThroughSparseMoE(
            metadata_dimension,
            hidden_dimension,
            metadata_experts,
            dropout,
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(3, hidden_dimension)
        )
        nn.init.normal_(
            self.modality_embedding, mean=0.0, std=0.02
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_dimension,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dimension)
        self.pooling_score = nn.Linear(hidden_dimension, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self,
        text: torch.Tensor,
        graph: torch.Tensor,
        metadata: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        text_representation, text_gate, text_balance = (
            self.text_moe(text, availability[:, 0])
        )
        graph_representation, graph_gate, graph_balance = (
            self.graph_moe(graph, availability[:, 1])
        )
        metadata_representation, metadata_gate, metadata_balance = (
            self.metadata_moe(metadata, availability[:, 2])
        )

        tokens = torch.stack(
            [
                text_representation,
                graph_representation,
                metadata_representation,
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)
        attention_output, _ = self.cross_modal_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~availability,
            need_weights=False,
        )
        tokens = self.attention_norm(tokens + attention_output)

        pooling_logits = self.pooling_score(tokens).squeeze(-1)
        pooling_logits = pooling_logits.masked_fill(
            ~availability,
            torch.finfo(pooling_logits.dtype).min,
        )
        pooling_weights = torch.softmax(pooling_logits, dim=1)
        pooled = torch.sum(
            pooling_weights.unsqueeze(-1) * tokens,
            dim=1,
        )
        logits = self.classifier(pooled).squeeze(1)
        balance_loss = (
            text_balance + graph_balance + metadata_balance
        )
        diagnostics = {
            "text_gate": text_gate,
            "graph_gate": graph_gate,
            "metadata_gate": metadata_gate,
            "modality_weights": pooling_weights,
        }
        return logits, balance_loss, diagnostics


def iterate_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    order = np.array(indices, copy=True)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def batch_tensors(
    batch: np.ndarray,
    text_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    text_tensor = torch.from_numpy(
        np.asarray(text_embeddings[batch], dtype=np.float32)
    ).to(device, non_blocking=True)
    graph_tensor = torch.from_numpy(
        np.asarray(graph_embeddings[batch], dtype=np.float32)
    ).to(device, non_blocking=True)
    metadata_tensor = torch.from_numpy(
        np.asarray(metadata[batch], dtype=np.float32)
    ).to(device, non_blocking=True)
    availability_tensor = torch.from_numpy(
        np.asarray(availability[batch], dtype=bool)
    ).to(device, non_blocking=True)
    label_tensor = torch.from_numpy(
        np.asarray(labels[batch], dtype=np.float32)
    ).to(device, non_blocking=True)
    return (
        text_tensor,
        graph_tensor,
        metadata_tensor,
        availability_tensor,
        label_tensor,
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    indices: np.ndarray,
    text_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    scores = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {
        "text_gate": [],
        "graph_gate": [],
        "metadata_gate": [],
        "modality_weights": [],
    }
    for batch in iterate_batches(indices, batch_size):
        tensors = batch_tensors(
            batch,
            text_embeddings,
            graph_embeddings,
            metadata,
            availability,
            labels,
            device,
        )
        logits, _, diagnostics = model(*tensors[:-1])
        scores.append(
            torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        )
        for key, values in diagnostics.items():
            diagnostic_parts[key].append(
                values.cpu().numpy().astype(np.float32)
            )
    return np.concatenate(scores), {
        key: np.concatenate(parts)
        for key, parts in diagnostic_parts.items()
    }


def available_gate_means(
    diagnostics: dict[str, np.ndarray],
    availability: np.ndarray,
) -> dict[str, float]:
    rows = {}
    specifications = [
        ("text_gate", "text", availability[:, 0]),
        ("graph_gate", "graph", availability[:, 1]),
        ("metadata_gate", "metadata", availability[:, 2]),
    ]
    for key, prefix, mask in specifications:
        values = diagnostics[key][mask]
        for expert_index in range(values.shape[1]):
            rows[
                f"{prefix}_expert_{expert_index + 1}_mean_gate"
            ] = float(values[:, expert_index].mean())
    for modality_index, modality in enumerate(
        ["text", "graph", "metadata"]
    ):
        rows[f"fusion_{modality}_mean_weight"] = float(
            diagnostics["modality_weights"][
                availability[:, modality_index], modality_index
            ].mean()
        )
    return rows


def run_training(
    seed: int,
    learning_rate: float,
    text_embeddings: np.ndarray,
    graph_embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    node_ids: np.ndarray,
    model_parameters: dict,
    training_parameters: dict,
    device: torch.device,
    save_artifacts: bool,
) -> dict:
    set_seed(seed)
    model = BotMoECommonEvidence(
        text_dimension=int(
            model_parameters["modalities"]["text"]["input_dimension"]
        ),
        graph_dimension=int(
            model_parameters["modalities"]["graph"]["input_dimension"]
        ),
        metadata_dimension=int(
            model_parameters["modalities"]["metadata"][
                "input_dimension"
            ]
        ),
        hidden_dimension=int(
            model_parameters["hidden_dimension"]
        ),
        text_experts=int(
            model_parameters["modalities"]["text"]["experts"]
        ),
        graph_experts=int(
            model_parameters["modalities"]["graph"]["experts"]
        ),
        metadata_experts=int(
            model_parameters["modalities"]["metadata"]["experts"]
        ),
        dropout=float(model_parameters["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_parameters["weight_decay"]),
    )
    positive = float(labels[train_indices].sum())
    negative = float(len(train_indices) - positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negative / positive,
            dtype=torch.float32,
            device=device,
        )
    )

    maximum_epochs = int(training_parameters["maximum_epochs"])
    patience = int(
        training_parameters["early_stopping_patience"]
    )
    batch_size = int(training_parameters["batch_size"])
    evaluation_batch_size = int(
        training_parameters["evaluation_batch_size"]
    )
    balance_weight = float(
        model_parameters["load_balance_weight"]
    )

    best_state = None
    best_epoch = -1
    best_validation_macro_f1 = -np.inf
    best_validation_auprc = -np.inf
    best_threshold = 0.5
    epochs_without_improvement = 0
    history_rows = []
    started = time.time()

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        total_loss = 0.0
        total_classification_loss = 0.0
        total_balance_loss = 0.0
        total_users = 0

        for batch in iterate_batches(
            train_indices, batch_size, rng=rng
        ):
            tensors = batch_tensors(
                batch,
                text_embeddings,
                graph_embeddings,
                metadata,
                availability,
                labels,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, balance_loss, _ = model(*tensors[:-1])
            classification_loss = criterion(
                logits, tensors[-1]
            )
            loss = (
                classification_loss
                + balance_weight * balance_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(batch)
            total_classification_loss += float(
                classification_loss.detach().cpu()
            ) * len(batch)
            total_balance_loss += float(
                balance_loss.detach().cpu()
            ) * len(batch)
            total_users += len(batch)

        validation_scores, _ = predict(
            model,
            validation_indices,
            text_embeddings,
            graph_embeddings,
            metadata,
            availability,
            labels,
            device,
            evaluation_batch_size,
        )
        threshold, validation_macro_f1 = exact_macro_f1_threshold(
            labels[validation_indices], validation_scores
        )
        validation_auprc = average_precision_score(
            labels[validation_indices], validation_scores
        )
        validation_auroc = roc_auc_score(
            labels[validation_indices], validation_scores
        )

        history_rows.append(
            {
                "seed": seed,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "training_loss": total_loss / total_users,
                "classification_loss": (
                    total_classification_loss / total_users
                ),
                "load_balance_loss": (
                    total_balance_loss / total_users
                ),
                "validation_macro_f1": validation_macro_f1,
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
                "threshold": threshold,
            }
        )
        print(
            f"seed={seed} lr={learning_rate:g} epoch={epoch:02d} "
            f"loss={total_loss / total_users:.6f} "
            f"balance={total_balance_loss / total_users:.6f} "
            f"val_macro_f1={validation_macro_f1:.6f} "
            f"val_auprc={validation_auprc:.6f}"
        )

        improved = (
            validation_macro_f1 > best_validation_macro_f1 + 1e-10
            or (
                abs(
                    validation_macro_f1
                    - best_validation_macro_f1
                )
                <= 1e-10
                and validation_auprc > best_validation_auprc
            )
        )
        if improved:
            best_validation_macro_f1 = validation_macro_f1
            best_validation_auprc = validation_auprc
            best_threshold = threshold
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("No valid BotMoE training state selected.")
    model.load_state_dict(best_state)

    validation_scores, validation_diagnostics = predict(
        model,
        validation_indices,
        text_embeddings,
        graph_embeddings,
        metadata,
        availability,
        labels,
        device,
        evaluation_batch_size,
    )
    best_threshold, best_validation_macro_f1 = (
        exact_macro_f1_threshold(
            labels[validation_indices], validation_scores
        )
    )
    result = {
        "model": "botmoe_style_common_evidence",
        "seed": seed,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation_macro_f1": best_validation_macro_f1,
        "validation_auroc": roc_auc_score(
            labels[validation_indices], validation_scores
        ),
        "validation_auprc": average_precision_score(
            labels[validation_indices], validation_scores
        ),
        "runtime_seconds": time.time() - started,
        **available_gate_means(
            validation_diagnostics,
            availability[validation_indices],
        ),
    }

    if save_artifacts:
        test_scores, test_diagnostics = predict(
            model,
            test_indices,
            text_embeddings,
            graph_embeddings,
            metadata,
            availability,
            labels,
            device,
            evaluation_batch_size,
        )
        test_metrics, test_predictions = calculate_metrics(
            labels[test_indices], test_scores, best_threshold
        )
        result.update(test_metrics)
        test_gate_means = available_gate_means(
            test_diagnostics, availability[test_indices]
        )
        result.update(
            {
                f"test_{key}": value
                for key, value in test_gate_means.items()
            }
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "seed": seed,
                "learning_rate": learning_rate,
                "threshold": best_threshold,
                "best_epoch": best_epoch,
                "model_designation": (
                    "botmoe_style_common_evidence"
                ),
            },
            MODEL_DIR / f"botmoe_seed_{seed}.pt",
        )
        pd.DataFrame(history_rows).to_csv(
            OUTPUT_DIR / f"training_history_seed_{seed}.csv",
            index=False,
        )

        prediction_data = {
            "node_id": node_ids[test_indices],
            "label": labels[test_indices],
            "score": test_scores,
            "prediction": test_predictions,
            "text_available": availability[test_indices, 0],
            "graph_available": availability[test_indices, 1],
            "metadata_available": availability[test_indices, 2],
            "fusion_text": test_diagnostics[
                "modality_weights"
            ][:, 0],
            "fusion_graph": test_diagnostics[
                "modality_weights"
            ][:, 1],
            "fusion_metadata": test_diagnostics[
                "modality_weights"
            ][:, 2],
        }
        for modality in ["text", "graph", "metadata"]:
            gates = test_diagnostics[f"{modality}_gate"]
            for expert_index in range(gates.shape[1]):
                prediction_data[
                    f"{modality}_expert_{expert_index + 1}_gate"
                ] = gates[:, expert_index]

        pd.DataFrame(prediction_data).to_parquet(
            PREDICTION_DIR
            / f"botmoe_seed_{seed}_test_predictions.parquet",
            index=False,
        )
    else:
        pd.DataFrame(history_rows).to_csv(
            OUTPUT_DIR
            / (
                f"tuning_seed_{seed}_lr_"
                f"{learning_rate:.0e}.csv"
            ),
            index=False,
        )
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

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
    seeds = [
        int(seed) for seed in config["comparison_policy"]["seeds"]
    ]
    model_parameters = config["botmoe_common_evidence"]
    training_parameters = config["training"]

    def path(key: str) -> Path:
        return PROJECT_ROOT / inputs[key]

    node_ids = np.load(path("node_ids"), mmap_mode="r")
    labels = np.load(path("labels"), mmap_mode="r").astype(
        np.int8, copy=False
    )
    split_ids = np.load(path("split_ids"), mmap_mode="r")
    text_embeddings = np.load(
        path("semantic_text_embeddings"), mmap_mode="r"
    )
    metadata = np.load(path("metadata_features"), mmap_mode="r")
    availability_archive = np.load(
        CONTRACT_DIR / "availability_masks.npz"
    )
    availability = np.column_stack(
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
        == availability.shape[0]
    ):
        raise RuntimeError("BotMoE comparator input row counts differ.")

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    validation_indices = np.flatnonzero(
        split_ids == VALIDATION_ID
    )
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    device_name = str(training_parameters["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    graph_pattern = inputs["graph_embedding_pattern"]
    tuning_seed = seeds[0]
    tuning_graph_path = PROJECT_ROOT / graph_pattern.format(
        seed=tuning_seed
    )
    tuning_graph = np.load(tuning_graph_path, mmap_mode="r")

    tuning_rows = []
    for learning_rate in training_parameters[
        "learning_rate_candidates"
    ]:
        print(
            f"\nTuning BotMoE-style comparator: "
            f"seed={tuning_seed}, lr={float(learning_rate):g}"
        )
        tuning_rows.append(
            run_training(
                tuning_seed,
                float(learning_rate),
                text_embeddings,
                tuning_graph,
                metadata,
                availability,
                labels,
                train_indices,
                validation_indices,
                test_indices,
                node_ids,
                model_parameters,
                training_parameters,
                device,
                save_artifacts=False,
            )
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_macro_f1", "validation_auprc"],
        ascending=False,
    )
    tuning.to_csv(
        OUTPUT_DIR / "validation_learning_rate_search.csv",
        index=False,
    )
    selected_learning_rate = float(
        tuning.iloc[0]["learning_rate"]
    )
    print(
        f"\nSelected learning rate: {selected_learning_rate:g}"
    )

    result_rows = []
    for seed in seeds:
        graph_path = PROJECT_ROOT / graph_pattern.format(seed=seed)
        graph_embeddings = np.load(graph_path, mmap_mode="r")
        if graph_embeddings.shape != (len(labels), 64):
            raise RuntimeError(
                f"Seed {seed} graph shape={graph_embeddings.shape}."
            )
        print(
            f"\nFinal BotMoE-style run: seed={seed}, "
            f"lr={selected_learning_rate:g}"
        )
        result = run_training(
            seed,
            selected_learning_rate,
            text_embeddings,
            graph_embeddings,
            metadata,
            availability,
            labels,
            train_indices,
            validation_indices,
            test_indices,
            node_ids,
            model_parameters,
            training_parameters,
            device,
            save_artifacts=True,
        )
        result_rows.append(result)
        print(
            f"TEST seed={seed}: "
            f"macro-F1={result['macro_f1']:.6f}, "
            f"AUROC={result['auroc']:.6f}, "
            f"AUPRC={result['auprc']:.6f}"
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / "botmoe_seed_results.csv", index=False
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
    summary = (
        results[metric_columns]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    summary["seeds"] = len(seeds)
    summary.to_csv(
        OUTPUT_DIR / "botmoe_summary.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "model_designation": "botmoe_style_common_evidence",
        "not_an_exact_reproduction": True,
        "source_repository": config["source_implementations"][
            "botmoe"
        ],
        "adaptation": model_parameters,
        "comparison_policy": config["comparison_policy"],
        "selected_learning_rate": selected_learning_rate,
        "seeds": seeds,
        "graph_embedding_policy": (
            "seed-specific locked R-GCN representation"
        ),
        "top_k_forward_routing": 1,
        "gate_gradient": "straight-through softmax estimator",
        "test_users_accessed_after_selection": True,
        "test_predictions_saved": True,
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_contract_sha256": sha256_file(contract_path),
    }
    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nBOTMOE-STYLE COMMON-EVIDENCE COMPARISON: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
