from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import sparse
from sklearn.linear_model import SGDClassifier
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
OUTPUT_DIR = OUTPUT_ROOT / "botbuster_common_evidence"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
IDENTITY_SCORE_DIR = OUTPUT_DIR / "identity_scores"

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
    torch.use_deterministic_algorithms(True, warn_only=True)
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


class BotBusterCommonEvidence(nn.Module):
    def __init__(
        self,
        text_dimension: int = 768,
        metadata_dimension: int = 10,
        hidden_dimension: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.identity_encoder = nn.Sequential(
            nn.Linear(1, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Dropout(dropout),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(hidden_dimension, 1)
        self.text_head = nn.Linear(hidden_dimension, 1)
        self.metadata_head = nn.Linear(hidden_dimension, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dimension * 3 + 3, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 3),
        )

    def forward(
        self,
        identity_score: torch.Tensor,
        text: torch.Tensor,
        metadata: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity_hidden = self.identity_encoder(
            identity_score.unsqueeze(1)
        )
        text_hidden = self.text_encoder(text)
        metadata_hidden = self.metadata_encoder(metadata)

        expert_logits = torch.cat(
            [
                self.identity_head(identity_hidden),
                self.text_head(text_hidden),
                self.metadata_head(metadata_hidden),
            ],
            dim=1,
        )
        gate_input = torch.cat(
            [
                identity_hidden,
                text_hidden,
                metadata_hidden,
                availability.float(),
            ],
            dim=1,
        )
        gate_logits = self.gate(gate_input)
        gate_logits = gate_logits.masked_fill(
            ~availability, torch.finfo(gate_logits.dtype).min
        )
        gate_weights = torch.softmax(gate_logits, dim=1)
        output = (gate_weights * expert_logits).sum(dim=1)
        return output, gate_weights


def train_identity_expert(
    identity: sparse.csr_matrix,
    labels: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    score_path = IDENTITY_SCORE_DIR / f"seed_{seed}_scores.npy"
    model_path = MODEL_DIR / f"identity_seed_{seed}.joblib"
    metadata_path = IDENTITY_SCORE_DIR / f"seed_{seed}_normalization.json"

    if score_path.is_file() and model_path.is_file() and metadata_path.is_file():
        scores = np.load(score_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if scores.shape == (len(labels),):
            return scores.astype(np.float32, copy=False), record

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight="balanced",
        max_iter=30,
        tol=1e-3,
        shuffle=True,
        random_state=seed,
        n_jobs=-1,
        average=True,
    )
    classifier.fit(identity[train_indices], labels[train_indices])
    raw_scores = classifier.decision_function(identity).astype(np.float64)
    train_scores = raw_scores[train_indices]
    mean = float(train_scores.mean())
    standard_deviation = float(train_scores.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise RuntimeError("Invalid identity-score standard deviation.")
    scores = np.clip(
        (raw_scores - mean) / standard_deviation,
        -8.0,
        8.0,
    ).astype(np.float32)

    np.save(score_path, scores, allow_pickle=False)
    joblib.dump(
        {
            "classifier": classifier,
            "fit_scope": "official training split only",
            "normalization_mean": mean,
            "normalization_standard_deviation": standard_deviation,
            "seed": seed,
        },
        model_path,
        compress=3,
    )
    record = {
        "seed": seed,
        "fit_scope": "official training split only",
        "alpha": 1e-5,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "clip": [-8.0, 8.0],
        "iterations": int(classifier.n_iter_),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return scores, record


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
    identity_scores: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    identity_tensor = torch.from_numpy(
        np.asarray(identity_scores[batch], dtype=np.float32)
    ).to(device, non_blocking=True)
    text_tensor = torch.from_numpy(
        np.asarray(text_embeddings[batch], dtype=np.float32)
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
        identity_tensor,
        text_tensor,
        metadata_tensor,
        availability_tensor,
        label_tensor,
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    indices: np.ndarray,
    identity_scores: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_parts = []
    gate_parts = []
    for batch in iterate_batches(indices, batch_size):
        tensors = batch_tensors(
            batch,
            identity_scores,
            text_embeddings,
            metadata,
            availability,
            labels,
            device,
        )
        logits, gates = model(*tensors[:-1])
        score_parts.append(
            torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        )
        gate_parts.append(gates.cpu().numpy().astype(np.float32))
    return np.concatenate(score_parts), np.concatenate(gate_parts)


def run_training(
    seed: int,
    learning_rate: float,
    text_embeddings: np.ndarray,
    metadata: np.ndarray,
    identity_scores: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    node_ids: np.ndarray,
    parameters: dict,
    device: torch.device,
    save_artifacts: bool,
) -> dict:
    set_seed(seed)
    model = BotBusterCommonEvidence(
        hidden_dimension=128,
        dropout=0.30,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(parameters["weight_decay"]),
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

    maximum_epochs = int(parameters["maximum_epochs"])
    patience = int(parameters["early_stopping_patience"])
    batch_size = int(parameters["batch_size"])
    evaluation_batch_size = int(parameters["evaluation_batch_size"])

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
        total_users = 0

        for batch in iterate_batches(
            train_indices, batch_size, rng=rng
        ):
            tensors = batch_tensors(
                batch,
                identity_scores,
                text_embeddings,
                metadata,
                availability,
                labels,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*tensors[:-1])
            loss = criterion(logits, tensors[-1])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_users += len(batch)

        validation_scores, _ = predict(
            model,
            validation_indices,
            identity_scores,
            text_embeddings,
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
                "validation_macro_f1": validation_macro_f1,
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
                "threshold": threshold,
            }
        )
        print(
            f"seed={seed} lr={learning_rate:g} epoch={epoch:02d} "
            f"loss={total_loss / total_users:.6f} "
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
        raise RuntimeError("No valid training state was selected.")

    model.load_state_dict(best_state)
    validation_scores, validation_gates = predict(
        model,
        validation_indices,
        identity_scores,
        text_embeddings,
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
        "model": "botbuster_style_common_evidence",
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
        "validation_gate_identity_mean": float(
            validation_gates[:, 0].mean()
        ),
        "validation_gate_text_mean": float(
            validation_gates[:, 1].mean()
        ),
        "validation_gate_metadata_mean": float(
            validation_gates[:, 2].mean()
        ),
    }

    if save_artifacts:
        test_scores, test_gates = predict(
            model,
            test_indices,
            identity_scores,
            text_embeddings,
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
        result.update(
            {
                "test_gate_identity_mean": float(
                    test_gates[:, 0].mean()
                ),
                "test_gate_text_mean": float(
                    test_gates[:, 1].mean()
                ),
                "test_gate_metadata_mean": float(
                    test_gates[:, 2].mean()
                ),
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
                    "botbuster_style_common_evidence"
                ),
            },
            MODEL_DIR / f"botbuster_seed_{seed}.pt",
        )
        pd.DataFrame(history_rows).to_csv(
            OUTPUT_DIR / f"training_history_seed_{seed}.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "node_id": node_ids[test_indices],
                "label": labels[test_indices],
                "score": test_scores,
                "prediction": test_predictions,
                "identity_available": availability[
                    test_indices, 0
                ],
                "text_available": availability[test_indices, 1],
                "metadata_available": availability[
                    test_indices, 2
                ],
                "gate_identity": test_gates[:, 0],
                "gate_text": test_gates[:, 1],
                "gate_metadata": test_gates[:, 2],
            }
        ).to_parquet(
            PREDICTION_DIR
            / f"botbuster_seed_{seed}_test_predictions.parquet",
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
    IDENTITY_SCORE_DIR.mkdir(parents=True, exist_ok=True)

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
    parameters = config["training"]

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
    identity = sparse.load_npz(path("identity_tfidf")).tocsr()
    availability_archive = np.load(
        CONTRACT_DIR / "availability_masks.npz"
    )
    availability = np.column_stack(
        [
            availability_archive["identity"],
            availability_archive["semantic_text"],
            availability_archive["metadata"],
        ]
    ).astype(bool)

    if not (
        len(node_ids)
        == len(labels)
        == len(split_ids)
        == text_embeddings.shape[0]
        == metadata.shape[0]
        == identity.shape[0]
        == availability.shape[0]
    ):
        raise RuntimeError("Comparator input row counts differ.")

    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    validation_indices = np.flatnonzero(
        split_ids == VALIDATION_ID
    )
    test_indices = np.flatnonzero(split_ids == TEST_ID)

    device_name = str(parameters["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    identity_scores_by_seed = {}
    identity_records = {}
    for seed in seeds:
        print(f"Preparing identity expert for seed {seed}...")
        scores, record = train_identity_expert(
            identity, labels, train_indices, seed
        )
        identity_scores_by_seed[seed] = scores
        identity_records[str(seed)] = record

    tuning_rows = []
    tuning_seed = seeds[0]
    for learning_rate in parameters["learning_rate_candidates"]:
        print(
            f"\nTuning BotBuster-style comparator: "
            f"seed={tuning_seed}, lr={float(learning_rate):g}"
        )
        tuning_rows.append(
            run_training(
                tuning_seed,
                float(learning_rate),
                text_embeddings,
                metadata,
                identity_scores_by_seed[tuning_seed],
                availability,
                labels,
                train_indices,
                validation_indices,
                test_indices,
                node_ids,
                parameters,
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
        print(
            f"\nFinal BotBuster-style run: seed={seed}, "
            f"lr={selected_learning_rate:g}"
        )
        result = run_training(
            seed,
            selected_learning_rate,
            text_embeddings,
            metadata,
            identity_scores_by_seed[seed],
            availability,
            labels,
            train_indices,
            validation_indices,
            test_indices,
            node_ids,
            parameters,
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
        OUTPUT_DIR / "botbuster_seed_results.csv",
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
    summary = (
        results[metric_columns]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    summary["seeds"] = len(seeds)
    summary.to_csv(
        OUTPUT_DIR / "botbuster_summary.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "model_designation": "botbuster_style_common_evidence",
        "not_an_exact_reproduction": True,
        "source_repository": config["source_implementations"][
            "botbuster"
        ],
        "adaptation": config["botbuster_common_evidence"],
        "comparison_policy": config["comparison_policy"],
        "selected_learning_rate": selected_learning_rate,
        "seeds": seeds,
        "identity_expert": identity_records,
        "test_users_accessed_after_selection": True,
        "test_predictions_saved": True,
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_contract_sha256": sha256_file(contract_path),
    }
    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nBOTBUSTER-STYLE COMMON-EVIDENCE COMPARISON: PASS")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
