from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import RGCNConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_INPUT_DIR = PROJECT_ROOT / "outputs" / "10_graph_inputs"
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "outputs" / "11_graph_baselines"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "12_rgcn_embeddings"

SEEDS = [42, 123, 2024, 3407, 7777]
HIDDEN_DIM = 64
NUM_RELATIONS = 4
DROPOUT = 0.3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        required=True,
        choices=["full_transductive", "split_contained"],
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )

        residual = x
        x = self.conv2(x, edge_index, edge_type)
        x = self.norm2(x)
        x = torch.relu(x + residual)
        x = nn.functional.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encode(
            x, edge_index, edge_type
        )
        return self.classifier(representation).squeeze(-1)


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    run_name = f"rgcn_{args.protocol}"
    protocol_input_dir = GRAPH_INPUT_DIR / args.protocol
    checkpoint_dir = (
        CHECKPOINT_ROOT / run_name / "checkpoints"
    )
    prediction_dir = (
        CHECKPOINT_ROOT / run_name / "test_predictions"
    )
    output_dir = OUTPUT_ROOT / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)

    node_features_np = np.load(
        protocol_input_dir / "node_features.npy",
        mmap_mode="r",
    )
    edge_index_np = np.load(
        protocol_input_dir / "edge_index.npy",
        mmap_mode="r",
    )
    edge_type_np = np.load(
        protocol_input_dir / "edge_type.npy",
        mmap_mode="r",
    )
    graph_available = np.load(
        protocol_input_dir / "graph_available.npy",
        mmap_mode="r",
    ).astype(bool)

    node_ids = np.load(
        GRAPH_INPUT_DIR / "node_ids.npy",
        mmap_mode="r",
    )
    labels = np.load(
        GRAPH_INPUT_DIR / "labels.npy",
        mmap_mode="r",
    )
    split_ids = np.load(
        GRAPH_INPUT_DIR / "split_ids.npy",
        mmap_mode="r",
    )
    test_indices = np.flatnonzero(split_ids == 2)

    device = torch.device("cuda")

    # Explicit copies remove the non-writable memmap warning and ensure
    # that PyTorch owns its input buffers.
    x = torch.from_numpy(
        np.array(node_features_np, dtype=np.float32, copy=True)
    ).to(device)
    edge_index = torch.from_numpy(
        np.array(edge_index_np, dtype=np.int64, copy=True)
    ).to(device)
    edge_type = torch.from_numpy(
        np.array(edge_type_np, dtype=np.int64, copy=True)
    ).to(device)

    seed_records = []

    for seed in SEEDS:
        checkpoint_path = (
            checkpoint_dir / f"{run_name}_seed_{seed}.pt"
        )
        prediction_path = (
            prediction_dir
            / f"{run_name}_seed_{seed}_test_predictions.parquet"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model = RGCNModel(
            input_dim=node_features_np.shape[1]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.inference_mode():
            representations = model.encode(
                x, edge_index, edge_type
            )
            logits = model.classifier(
                representations
            ).squeeze(-1)

        representation_np = (
            representations.float().cpu().numpy()
        )
        logits_np = logits.float().cpu().numpy()

        if not np.isfinite(representation_np).all():
            raise RuntimeError(
                f"Non-finite representations for seed {seed}."
            )
        if not np.isfinite(logits_np).all():
            raise RuntimeError(
                f"Non-finite logits for seed {seed}."
            )

        stored_predictions = pd.read_parquet(
            prediction_path,
            columns=["node_id", "score"],
        ).sort_values("node_id")

        expected_test_node_ids = np.asarray(
            node_ids[test_indices]
        )
        if not np.array_equal(
            stored_predictions["node_id"].to_numpy(),
            expected_test_node_ids,
        ):
            raise RuntimeError(
                f"Test-node ordering mismatch for seed {seed}."
            )

        stored_scores = stored_predictions[
            "score"
        ].to_numpy(dtype=np.float32)
        reproduced_scores = logits_np[test_indices]

        maximum_score_difference = float(
            np.max(
                np.abs(
                    stored_scores - reproduced_scores
                )
            )
        )
        if maximum_score_difference > 1e-5:
            raise RuntimeError(
                f"Checkpoint reproduction failed for seed {seed}: "
                f"maximum difference={maximum_score_difference}"
            )

        output_path = (
            output_dir
            / f"rgcn_embeddings_seed_{seed}.npy"
        )
        np.save(
            output_path,
            representation_np.astype(np.float16),
            allow_pickle=False,
        )

        logit_path = (
            output_dir / f"rgcn_logits_seed_{seed}.npy"
        )
        np.save(
            logit_path,
            logits_np.astype(np.float32),
            allow_pickle=False,
        )

        seed_record = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "selected_epoch": int(
                checkpoint["selection"]["epoch"]
            ),
            "validation_macro_f1": float(
                checkpoint["selection"][
                    "validation_macro_f1"
                ]
            ),
            "maximum_reproduction_difference": (
                maximum_score_difference
            ),
            "representation_shape": list(
                representation_np.shape
            ),
            "representation_mean": float(
                representation_np.mean()
            ),
            "representation_std": float(
                representation_np.std()
            ),
            "embedding_sha256": sha256_file(
                output_path
            ),
            "logit_sha256": sha256_file(logit_path),
        }
        seed_records.append(seed_record)

        print(
            f"Seed {seed}: extracted "
            f"{representation_np.shape[0]:,} × "
            f"{representation_np.shape[1]} embeddings; "
            f"max score difference="
            f"{maximum_score_difference:.3e}"
        )

        del model, representations, logits
        del representation_np, logits_np
        torch.cuda.empty_cache()

    np.save(
        output_dir / "node_ids.npy",
        np.asarray(node_ids),
        allow_pickle=False,
    )
    np.save(
        output_dir / "labels.npy",
        np.asarray(labels),
        allow_pickle=False,
    )
    np.save(
        output_dir / "split_ids.npy",
        np.asarray(split_ids),
        allow_pickle=False,
    )
    np.save(
        output_dir / "graph_available.npy",
        graph_available,
        allow_pickle=False,
    )

    manifest = {
        "status": "PASS",
        "protocol": args.protocol,
        "source_model": "validation-selected R-GCN",
        "representation": (
            "second R-GCN layer after residual addition, "
            "LayerNorm, and ReLU; dropout disabled"
        ),
        "dimensions": HIDDEN_DIM,
        "seeds": SEEDS,
        "records": seed_records,
        "integrity": {
            "row_order_matches_node_id": True,
            "checkpoint_logits_reproduced": True,
            "test_labels_not_used": True,
            "representations_finite": True,
            "graph_inputs_topology_only": True,
        },
        "claim_boundary": (
            "transductive"
            if args.protocol == "full_transductive"
            else "split-contained inductive sensitivity"
        ),
    }

    with (output_dir / "rgcn_embedding_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("R-GCN EMBEDDING EXTRACTION: PASS")
    print(f"Protocol: {args.protocol}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Dimensions: {HIDDEN_DIM}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()