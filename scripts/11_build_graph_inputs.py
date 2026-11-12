from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "10_graph_inputs"

USERS_PATH = CANONICAL_DIR / "users.parquet"
EDGES_PATH = CANONICAL_DIR / "user_user_edges.parquet"
RELATIONS_PATH = CANONICAL_DIR / "relation_vocabulary.csv"

RAW_RELATIONS = [0, 1]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def construct_bidirectional_edges(
    edges: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    source = edges["source"].to_numpy(dtype=np.int64, copy=False)
    target = edges["target"].to_numpy(dtype=np.int64, copy=False)
    relation = edges["relation_id"].to_numpy(
        dtype=np.int64, copy=False
    )

    # Forward raw relations retain IDs 0 and 1.
    # Explicit inverse message-passing relations use IDs 2 and 3.
    edge_index = np.empty((2, 2 * len(edges)), dtype=np.int64)
    edge_type = np.empty(2 * len(edges), dtype=np.int8)

    edge_index[0, : len(edges)] = source
    edge_index[1, : len(edges)] = target
    edge_type[: len(edges)] = relation.astype(np.int8)

    edge_index[0, len(edges) :] = target
    edge_index[1, len(edges) :] = source
    edge_type[len(edges) :] = (relation + 2).astype(np.int8)

    return edge_index, edge_type


def calculate_structural_features(
    edges: pd.DataFrame,
    n_nodes: int,
) -> tuple[np.ndarray, list[str]]:
    feature_columns = []
    feature_names = []

    total_out = np.zeros(n_nodes, dtype=np.int64)
    total_in = np.zeros(n_nodes, dtype=np.int64)

    for relation_id in RAW_RELATIONS:
        relation_edges = edges.loc[
            edges["relation_id"].eq(relation_id)
        ]
        source = relation_edges["source"].to_numpy(
            dtype=np.int64, copy=False
        )
        target = relation_edges["target"].to_numpy(
            dtype=np.int64, copy=False
        )

        out_degree = np.bincount(source, minlength=n_nodes)
        in_degree = np.bincount(target, minlength=n_nodes)

        total_out += out_degree
        total_in += in_degree

        feature_columns.extend(
            [
                np.log1p(out_degree),
                np.log1p(in_degree),
            ]
        )
        feature_names.extend(
            [
                f"log1p_relation_{relation_id}_out_degree",
                f"log1p_relation_{relation_id}_in_degree",
            ]
        )

    feature_columns.extend(
        [
            np.log1p(total_out),
            np.log1p(total_in),
        ]
    )
    feature_names.extend(
        [
            "log1p_total_out_degree",
            "log1p_total_in_degree",
        ]
    )

    matrix = np.column_stack(feature_columns).astype(np.float64)
    return matrix, feature_names


def scale_train_only(
    matrix: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(matrix[train_mask])
    scaled = scaler.transform(matrix).astype(np.float32)

    if not np.isfinite(scaled).all():
        raise RuntimeError(
            "Scaled structural features contain non-finite values."
        )
    return scaled, scaler


def save_protocol(
    protocol_name: str,
    edges: pd.DataFrame,
    users: pd.DataFrame,
    train_mask: np.ndarray,
    n_nodes: int,
) -> dict:
    protocol_dir = OUTPUT_DIR / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)

    structural, feature_names = calculate_structural_features(
        edges, n_nodes
    )
    scaled, scaler = scale_train_only(
        structural, train_mask
    )

    total_degree = np.zeros(n_nodes, dtype=np.int64)
    source = edges["source"].to_numpy(dtype=np.int64, copy=False)
    target = edges["target"].to_numpy(dtype=np.int64, copy=False)
    total_degree += np.bincount(source, minlength=n_nodes)
    total_degree += np.bincount(target, minlength=n_nodes)
    graph_available = total_degree > 0

    # Availability is appended after scaling and is never standardized.
    node_features = np.column_stack(
        [
            scaled,
            graph_available.astype(np.float32),
        ]
    ).astype(np.float32)
    final_feature_names = feature_names + ["graph_available"]

    edge_index, edge_type = construct_bidirectional_edges(edges)

    np.save(
        protocol_dir / "node_features.npy",
        node_features,
        allow_pickle=False,
    )
    np.save(
        protocol_dir / "edge_index.npy",
        edge_index,
        allow_pickle=False,
    )
    np.save(
        protocol_dir / "edge_type.npy",
        edge_type,
        allow_pickle=False,
    )
    np.save(
        protocol_dir / "graph_available.npy",
        graph_available,
        allow_pickle=False,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_names_before_scaling": feature_names,
            "final_feature_names": final_feature_names,
            "fit_scope": "official training nodes only",
        },
        protocol_dir / "structural_preprocessor.joblib",
    )

    availability_rows = []
    labels = users["label"].astype(str).to_numpy()
    splits = users["split"].astype(str).to_numpy()

    for split_name in ["train", "val", "test"]:
        for label_name in ["bot", "human"]:
            mask = (splits == split_name) & (labels == label_name)
            availability_rows.append(
                {
                    "protocol": protocol_name,
                    "split": split_name,
                    "label": label_name,
                    "users": int(mask.sum()),
                    "graph_available": int(
                        (mask & graph_available).sum()
                    ),
                    "graph_missing": int(
                        (mask & ~graph_available).sum()
                    ),
                    "availability_rate": float(
                        graph_available[mask].mean()
                    ),
                }
            )

    pd.DataFrame(availability_rows).to_csv(
        protocol_dir / "graph_availability.csv",
        index=False,
    )

    return {
        "protocol": protocol_name,
        "raw_edges": int(len(edges)),
        "message_passing_edges": int(edge_index.shape[1]),
        "relations": int(np.unique(edge_type).size),
        "nodes_with_edges": int(graph_available.sum()),
        "isolated_nodes": int((~graph_available).sum()),
        "node_feature_count": int(node_features.shape[1]),
        "node_feature_names": final_feature_names,
        "feature_source": "graph topology only",
        "scaler_fit_scope": "official training nodes only",
        "output_hashes": {
            "node_features.npy": sha256_file(
                protocol_dir / "node_features.npy"
            ),
            "edge_index.npy": sha256_file(
                protocol_dir / "edge_index.npy"
            ),
            "edge_type.npy": sha256_file(
                protocol_dir / "edge_type.npy"
            ),
            "graph_available.npy": sha256_file(
                protocol_dir / "graph_available.npy"
            ),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(
        USERS_PATH,
        columns=["node_id", "label", "split"],
    ).sort_values("node_id").reset_index(drop=True)

    edges = pd.read_parquet(
        EDGES_PATH,
        columns=[
            "source",
            "target",
            "relation_id",
            "source_split",
            "target_split",
        ],
    )

    relation_vocabulary = pd.read_csv(RELATIONS_PATH)

    n_nodes = len(users)
    node_ids = users["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        node_ids, np.arange(n_nodes, dtype=np.int64)
    ):
        raise RuntimeError("node_id is not contiguous and zero based.")

    observed_relations = set(
        edges["relation_id"].astype(int).unique()
    )
    if observed_relations != set(RAW_RELATIONS):
        raise RuntimeError(
            f"Unexpected raw relations: {sorted(observed_relations)}"
        )

    if edges.duplicated(
        ["source", "target", "relation_id"]
    ).any():
        raise RuntimeError("Duplicate canonical edges detected.")

    if edges["source"].eq(edges["target"]).any():
        raise RuntimeError("Self-loops remain in canonical edges.")

    train_mask = users["split"].eq("train").to_numpy()

    full_edges = edges.copy()
    contained_edges = edges.loc[
        edges["source_split"].eq(edges["target_split"])
    ].copy()

    full_summary = save_protocol(
        "full_transductive",
        full_edges,
        users,
        train_mask,
        n_nodes,
    )
    contained_summary = save_protocol(
        "split_contained",
        contained_edges,
        users,
        train_mask,
        n_nodes,
    )

    normalized_labels = users["label"].astype(str).str.lower()
    labels = normalized_labels.eq("bot").to_numpy(dtype=np.int8)
    split_ids = (
        users["split"]
        .astype(str)
        .map(SPLIT_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    np.save(
        OUTPUT_DIR / "node_ids.npy",
        node_ids,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "labels.npy",
        labels,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "split_ids.npy",
        split_ids,
        allow_pickle=False,
    )

    raw_relation_names = {
        int(row.relation_id): str(row.relation)
        for row in relation_vocabulary.itertuples(index=False)
    }
    message_relations = {
        "0": {
            "source": "raw",
            "raw_relation_id": 0,
            "raw_relation_name": raw_relation_names.get(0),
        },
        "1": {
            "source": "raw",
            "raw_relation_id": 1,
            "raw_relation_name": raw_relation_names.get(1),
        },
        "2": {
            "source": "explicit_inverse",
            "inverse_of_relation_id": 0,
        },
        "3": {
            "source": "explicit_inverse",
            "inverse_of_relation_id": 1,
        },
    }

    manifest = {
        "status": "PASS",
        "nodes": n_nodes,
        "raw_relation_vocabulary": raw_relation_names,
        "message_passing_relations": message_relations,
        "protocols": {
            "primary": full_summary,
            "sensitivity": contained_summary,
        },
        "scientific_constraints": {
            "graph_branch_uses_text": False,
            "graph_branch_uses_account_metadata": False,
            "graph_features": (
                "relation-specific and total directed degree only"
            ),
            "labels_used_in_feature_construction": False,
            "inverse_edges": (
                "explicitly typed and distinct from raw relations"
            ),
            "full_graph_claim": "transductive node classification",
            "split_contained_claim": (
                "inductive split-contained sensitivity"
            ),
        },
        "input_hashes": {
            "users.parquet": sha256_file(USERS_PATH),
            "user_user_edges.parquet": sha256_file(EDGES_PATH),
            "relation_vocabulary.csv": sha256_file(
                RELATIONS_PATH
            ),
        },
    }

    with (OUTPUT_DIR / "graph_input_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("GRAPH INPUT BUILD: PASS")
    print(f"Users: {n_nodes:,}")
    print(
        f"Full raw/message edges: "
        f"{full_summary['raw_edges']:,}/"
        f"{full_summary['message_passing_edges']:,}"
    )
    print(
        f"Split-contained raw/message edges: "
        f"{contained_summary['raw_edges']:,}/"
        f"{contained_summary['message_passing_edges']:,}"
    )
    print(
        f"Full-graph isolated users: "
        f"{full_summary['isolated_nodes']:,}"
    )
    print(
        f"Split-contained isolated users: "
        f"{contained_summary['isolated_nodes']:,}"
    )
    print("Graph features contain topology only: TRUE")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()