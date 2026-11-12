from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATH = PROJECT_ROOT.parent / "edge.csv"
USERS_PATH = (
    PROJECT_ROOT / "outputs" / "02_canonical_data" / "users.parquet"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "16_heterogeneous_graph_features"
)

CHUNK_SIZE = 1_000_000

RELATIONS = [
    "contain",
    "discuss",
    "followed",
    "followers",
    "following",
    "like",
    "membership",
    "mentioned",
    "own",
    "pinned",
    "post",
    "quoted",
    "replied_to",
    "retweeted",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_composite_counts(
    count_matrix: np.ndarray,
    node_indices: np.ndarray,
    relation_indices: np.ndarray,
    relation_count: int,
) -> int:
    valid = node_indices >= 0
    if not valid.any():
        return 0

    composite = (
        node_indices[valid].astype(np.int64)
        * np.int64(relation_count)
        + relation_indices[valid].astype(np.int64)
    )
    unique_keys, frequencies = np.unique(
        composite,
        return_counts=True,
    )
    flat_counts = count_matrix.reshape(-1)
    flat_counts[unique_keys] += frequencies.astype(
        flat_counts.dtype,
        copy=False,
    )
    return int(valid.sum())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EDGE_PATH.exists():
        raise FileNotFoundError(EDGE_PATH)

    users = pd.read_parquet(
        USERS_PATH,
        columns=["node_id", "id", "split", "label"],
    ).sort_values("node_id").reset_index(drop=True)

    n_users = len(users)
    node_ids = users["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        node_ids,
        np.arange(n_users, dtype=np.int64),
    ):
        raise RuntimeError("node_id is not contiguous and zero based.")

    user_index = pd.Index(
        users["id"].astype(str),
        dtype="object",
    )
    if not user_index.is_unique:
        raise RuntimeError("Canonical user IDs are not unique.")

    relation_to_id = {
        relation: index
        for index, relation in enumerate(RELATIONS)
    }
    relation_count = len(RELATIONS)

    outgoing_counts = np.zeros(
        (n_users, relation_count),
        dtype=np.uint64,
    )
    incoming_counts = np.zeros(
        (n_users, relation_count),
        dtype=np.uint64,
    )

    raw_relation_counts = np.zeros(
        relation_count,
        dtype=np.uint64,
    )
    source_user_incidents = np.zeros(
        relation_count,
        dtype=np.uint64,
    )
    target_user_incidents = np.zeros(
        relation_count,
        dtype=np.uint64,
    )

    total_edges = 0
    unknown_relations: set[str] = set()
    started = time.time()

    reader = pd.read_csv(
        EDGE_PATH,
        usecols=["source_id", "relation", "target_id"],
        dtype={
            "source_id": "string",
            "relation": "string",
            "target_id": "string",
        },
        chunksize=CHUNK_SIZE,
    )

    for chunk in tqdm(
        reader,
        desc="Streaming heterogeneous edges",
        unit="chunk",
    ):
        chunk_relations = chunk["relation"].astype(str)
        relation_indices = (
            chunk_relations.map(relation_to_id)
        )

        unknown_mask = relation_indices.isna()
        if unknown_mask.any():
            unknown_relations.update(
                chunk_relations.loc[unknown_mask].unique()
            )
            continue

        relation_array = relation_indices.to_numpy(
            dtype=np.int16
        )
        source_nodes = user_index.get_indexer(
            chunk["source_id"].astype(str)
        )
        target_nodes = user_index.get_indexer(
            chunk["target_id"].astype(str)
        )

        raw_relation_counts += np.bincount(
            relation_array,
            minlength=relation_count,
        ).astype(np.uint64)

        valid_source = source_nodes >= 0
        valid_target = target_nodes >= 0

        if valid_source.any():
            source_user_incidents += np.bincount(
                relation_array[valid_source],
                minlength=relation_count,
            ).astype(np.uint64)

        if valid_target.any():
            target_user_incidents += np.bincount(
                relation_array[valid_target],
                minlength=relation_count,
            ).astype(np.uint64)

        add_composite_counts(
            outgoing_counts,
            source_nodes,
            relation_array,
            relation_count,
        )
        add_composite_counts(
            incoming_counts,
            target_nodes,
            relation_array,
            relation_count,
        )

        total_edges += len(chunk)

    if unknown_relations:
        raise RuntimeError(
            f"Unknown relations detected: "
            f"{sorted(unknown_relations)}"
        )

    if total_edges != 170_185_937:
        raise RuntimeError(
            f"Expected 170,185,937 edges; found "
            f"{total_edges:,}."
        )

    raw_feature_names = []
    raw_feature_columns = []

    for relation_id, relation in enumerate(RELATIONS):
        raw_feature_columns.extend(
            [
                outgoing_counts[:, relation_id],
                incoming_counts[:, relation_id],
            ]
        )
        raw_feature_names.extend(
            [
                f"{relation}_out_count",
                f"{relation}_in_count",
            ]
        )

    raw_features = np.column_stack(
        raw_feature_columns
    ).astype(np.uint64)

    # Log transformation is deterministic; standardization parameters
    # are learned strictly from official training users.
    log_features = np.log1p(
        raw_features.astype(np.float64)
    )

    train_mask = users["split"].eq("train").to_numpy()
    scaler = StandardScaler()
    scaler.fit(log_features[train_mask])
    scaled_features = scaler.transform(
        log_features
    ).astype(np.float32)

    if not np.isfinite(scaled_features).all():
        raise RuntimeError(
            "Non-finite heterogeneous graph features detected."
        )

    total_incident_edges = raw_features.sum(axis=1)
    graph_available = total_incident_edges > 0

    final_features = np.column_stack(
        [
            scaled_features,
            graph_available.astype(np.float32),
        ]
    ).astype(np.float32)
    final_feature_names = raw_feature_names + [
        "heterogeneous_graph_available"
    ]

    np.save(
        OUTPUT_DIR / "raw_relation_counts.npy",
        raw_features,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "heterogeneous_graph_features.npy",
        final_features,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "graph_available.npy",
        graph_available,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "node_ids.npy",
        node_ids,
        allow_pickle=False,
    )

    normalized_labels = users["label"].astype(str).str.lower()
    labels = normalized_labels.eq("bot").to_numpy(dtype=np.int8)
    split_ids = (
        users["split"]
        .astype(str)
        .map({"train": 0, "val": 1, "test": 2})
        .to_numpy(dtype=np.int8)
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

    joblib.dump(
        {
            "scaler": scaler,
            "raw_feature_names": raw_feature_names,
            "final_feature_names": final_feature_names,
            "fit_scope": "official training users only",
            "transform": "log1p then StandardScaler",
        },
        OUTPUT_DIR / "preprocessor.joblib",
    )

    relation_summary = pd.DataFrame(
        {
            "relation": RELATIONS,
            "raw_edges": raw_relation_counts,
            "user_as_source_incidents": source_user_incidents,
            "user_as_target_incidents": target_user_incidents,
            "users_with_outgoing_relation": [
                int((outgoing_counts[:, index] > 0).sum())
                for index in range(relation_count)
            ],
            "users_with_incoming_relation": [
                int((incoming_counts[:, index] > 0).sum())
                for index in range(relation_count)
            ],
        }
    )
    relation_summary.to_csv(
        OUTPUT_DIR / "relation_incidence_summary.csv",
        index=False,
    )

    availability_rows = []
    split_values = users["split"].astype(str).to_numpy()
    label_values = users["label"].astype(str).to_numpy()

    for split_name in ["train", "val", "test"]:
        for label_name in ["bot", "human"]:
            mask = (
                (split_values == split_name)
                & (label_values == label_name)
            )
            availability_rows.append(
                {
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
                    "mean_incident_edges": float(
                        total_incident_edges[mask].mean()
                    ),
                    "median_incident_edges": float(
                        np.median(
                            total_incident_edges[mask]
                        )
                    ),
                }
            )

    pd.DataFrame(availability_rows).to_csv(
        OUTPUT_DIR / "availability_by_split_label.csv",
        index=False,
    )

    preprocessing_parameters = pd.DataFrame(
        {
            "feature": raw_feature_names,
            "training_mean": scaler.mean_,
            "training_scale": scaler.scale_,
        }
    )
    preprocessing_parameters.to_csv(
        OUTPUT_DIR / "preprocessing_parameters.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "users": n_users,
        "raw_edges": total_edges,
        "relations": RELATIONS,
        "raw_feature_count": len(raw_feature_names),
        "final_feature_count": len(final_feature_names),
        "feature_definition": (
            "Per-user incoming and outgoing edge counts for every "
            "heterogeneous relation; log1p transformed and "
            "training-standardized"
        ),
        "scientific_constraints": {
            "uses_labels_as_features": False,
            "uses_text": False,
            "uses_account_metadata": False,
            "uses_edge_topology_only": True,
            "preserves_relation_direction": True,
            "preprocessor_fit_scope": "official train only",
        },
        "graph_available_users": int(graph_available.sum()),
        "graph_missing_users": int((~graph_available).sum()),
        "runtime_seconds": time.time() - started,
        "input_hashes": {
            "edge.csv": sha256_file(EDGE_PATH),
            "users.parquet": sha256_file(USERS_PATH),
        },
        "output_hashes": {
            "raw_relation_counts.npy": sha256_file(
                OUTPUT_DIR / "raw_relation_counts.npy"
            ),
            "heterogeneous_graph_features.npy": sha256_file(
                OUTPUT_DIR
                / "heterogeneous_graph_features.npy"
            ),
        },
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("HETEROGENEOUS GRAPH FEATURE BUILD: PASS")
    print(f"Users: {n_users:,}")
    print(f"Edges processed: {total_edges:,}")
    print(f"Relations: {relation_count}")
    print(f"Raw directed features: {len(raw_feature_names)}")
    print(
        f"Users with heterogeneous graph activity: "
        f"{graph_available.sum():,} "
        f"({graph_available.mean():.4%})"
    )
    print("Feature source: topology only")
    print("Preprocessing fit: official training users only")
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()