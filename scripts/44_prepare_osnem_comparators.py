from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "osnem_sota_comparison.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "40_osnem_sota_comparison"
CONTRACT_DIR = OUTPUT_DIR / "input_contract"

SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
LABEL_NAMES = {0: "human", 1: "bot"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def clean_available(series: pd.Series) -> np.ndarray:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .ne("")
        .to_numpy(dtype=bool)
    )


def array_record(path: Path, array: np.ndarray) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }


def require_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        mismatch = np.flatnonzero(left != right)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise RuntimeError(f"{name} mismatch; first differing row={first}.")


def main() -> None:
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    inputs = config["canonical_inputs"]
    seeds = [int(seed) for seed in config["comparison_policy"]["seeds"]]

    node_path = resolve(inputs["node_ids"])
    label_path = resolve(inputs["labels"])
    split_path = resolve(inputs["split_ids"])
    text_path = resolve(inputs["semantic_text_embeddings"])
    text_available_path = resolve(inputs["semantic_text_available"])
    metadata_path = resolve(inputs["metadata_features"])
    graph_available_path = resolve(inputs["graph_available"])
    identity_path = resolve(inputs["identity_tfidf"])
    identity_available_path = resolve(inputs["identity_available"])
    users_path = resolve(inputs["users"])

    node_ids = np.load(node_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    split_ids = np.load(split_path, mmap_mode="r")
    text_embeddings = np.load(text_path, mmap_mode="r")
    text_available = np.load(text_available_path, mmap_mode="r")
    metadata = np.load(metadata_path, mmap_mode="r")
    graph_available = np.load(graph_available_path, mmap_mode="r")
    identity_available = np.load(identity_available_path, mmap_mode="r")
    identity = sparse.load_npz(identity_path)

    users = (
        pd.read_parquet(
            users_path,
            columns=[
                "node_id",
                "label",
                "split",
                "name_text",
                "username_text",
                "description_text",
                "location_text",
            ],
        )
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    n_users = len(node_ids)
    expected_ids = np.arange(n_users, dtype=np.int64)
    require_equal(
        "contiguous zero-based node identifiers",
        np.asarray(node_ids, dtype=np.int64),
        expected_ids,
    )

    if len(users) != n_users:
        raise RuntimeError(
            f"Canonical users={len(users):,}; array users={n_users:,}."
        )

    require_equal(
        "canonical node identifiers",
        users["node_id"].to_numpy(dtype=np.int64),
        expected_ids,
    )

    canonical_labels = (
        users["label"]
        .astype(str)
        .str.lower()
        .eq("bot")
        .to_numpy(dtype=np.int8)
    )
    canonical_splits = (
        users["split"]
        .astype(str)
        .map({"train": 0, "val": 1, "test": 2})
        .to_numpy(dtype=np.int8)
    )
    require_equal(
        "canonical labels",
        np.asarray(labels, dtype=np.int8),
        canonical_labels,
    )
    require_equal(
        "canonical split assignments",
        np.asarray(split_ids, dtype=np.int8),
        canonical_splits,
    )

    expected_shapes = {
        "semantic text embeddings": (n_users, 768),
        "metadata features": (n_users, 10),
        "semantic text availability": (n_users,),
        "graph availability": (n_users,),
        "identity availability": (n_users,),
    }
    observed_shapes = {
        "semantic text embeddings": text_embeddings.shape,
        "metadata features": metadata.shape,
        "semantic text availability": text_available.shape,
        "graph availability": graph_available.shape,
        "identity availability": identity_available.shape,
    }
    for name, expected in expected_shapes.items():
        if tuple(observed_shapes[name]) != expected:
            raise RuntimeError(
                f"{name}: expected {expected}, observed {observed_shapes[name]}."
            )

    if identity.shape[0] != n_users:
        raise RuntimeError(
            f"Identity TF-IDF rows={identity.shape[0]:,}; expected={n_users:,}."
        )

    name_available = clean_available(users["name_text"])
    username_available = clean_available(users["username_text"])
    description_available = clean_available(users["description_text"])
    location_available = clean_available(users["location_text"])
    expected_identity_available = name_available | username_available
    expected_text_available = description_available | location_available

    require_equal(
        "identity availability",
        np.asarray(identity_available, dtype=bool),
        expected_identity_available,
    )
    require_equal(
        "semantic text availability",
        np.asarray(text_available, dtype=bool),
        expected_text_available,
    )

    graph_records = {}
    graph_pattern = inputs["graph_embedding_pattern"]
    for seed in seeds:
        graph_path = resolve(graph_pattern.format(seed=seed))
        graph_embedding = np.load(graph_path, mmap_mode="r")
        if graph_embedding.shape != (n_users, 64):
            raise RuntimeError(
                f"Seed {seed} graph shape={graph_embedding.shape}; "
                f"expected={(n_users, 64)}."
            )
        graph_records[str(seed)] = array_record(
            graph_path, graph_embedding
        )

    metadata_available = np.ones(n_users, dtype=bool)
    availability = pd.DataFrame(
        {
            "split_id": np.asarray(split_ids, dtype=np.int8),
            "label": np.asarray(labels, dtype=np.int8),
            "name": name_available,
            "username": username_available,
            "identity": expected_identity_available,
            "semantic_text": expected_text_available,
            "graph": np.asarray(graph_available, dtype=bool),
            "metadata": metadata_available,
        }
    )

    summary_rows = []
    for split_id, split_name in SPLIT_NAMES.items():
        for label_id, label_name in LABEL_NAMES.items():
            mask = (
                availability["split_id"].eq(split_id)
                & availability["label"].eq(label_id)
            )
            row = {
                "split": split_name,
                "label": label_name,
                "users": int(mask.sum()),
            }
            for field in [
                "name",
                "username",
                "identity",
                "semantic_text",
                "graph",
                "metadata",
            ]:
                row[f"{field}_available"] = int(
                    availability.loc[mask, field].sum()
                )
                row[f"{field}_rate"] = float(
                    availability.loc[mask, field].mean()
                )
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        CONTRACT_DIR / "availability_by_split_and_label.csv",
        index=False,
    )

    np.savez_compressed(
        CONTRACT_DIR / "availability_masks.npz",
        name=name_available,
        username=username_available,
        identity=expected_identity_available,
        semantic_text=expected_text_available,
        graph=np.asarray(graph_available, dtype=bool),
        metadata=metadata_available,
    )

    records = {
        "node_ids": array_record(node_path, node_ids),
        "labels": array_record(label_path, labels),
        "split_ids": array_record(split_path, split_ids),
        "semantic_text_embeddings": array_record(
            text_path, text_embeddings
        ),
        "semantic_text_available": array_record(
            text_available_path, text_available
        ),
        "metadata_features": array_record(metadata_path, metadata),
        "graph_available": array_record(
            graph_available_path, graph_available
        ),
        "identity_available": array_record(
            identity_available_path, identity_available
        ),
        "identity_tfidf": {
            "path": str(identity_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "bytes": int(identity_path.stat().st_size),
            "shape": [int(value) for value in identity.shape],
            "dtype": str(identity.dtype),
            "nonzero_values": int(identity.nnz),
        },
        "seed_specific_graph_embeddings": graph_records,
    }

    manifest = {
        "status": "PASS",
        "purpose": (
            "Lock the common-evidence input contract for the "
            "reviewer-requested BotBuster and BotMoE comparisons."
        ),
        "users": int(n_users),
        "official_split_counts": {
            name: int((np.asarray(split_ids) == split_id).sum())
            for split_id, name in SPLIT_NAMES.items()
        },
        "positive_class": "bot",
        "source_commits": config["source_implementations"],
        "comparison_policy": config["comparison_policy"],
        "records": records,
        "config_sha256": sha256_file(CONFIG_PATH),
        "integrity": {
            "contiguous_zero_based_node_ids": True,
            "canonical_node_order_identical": True,
            "canonical_labels_identical": True,
            "official_splits_identical": True,
            "profile_field_availability_verified": True,
            "identity_availability_verified": True,
            "semantic_text_availability_verified": True,
            "seed_specific_graph_shapes_verified": True,
            "raw_profile_text_redistributed": False,
        },
    }
    with (CONTRACT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("OSNEM COMPARATOR INPUT CONTRACT: PASS")
    print(f"Users: {n_users:,}")
    print(
        "Official splits: "
        + ", ".join(
            f"{name}={(np.asarray(split_ids) == split_id).sum():,}"
            for split_id, name in SPLIT_NAMES.items()
        )
    )
    print(f"Identity TF-IDF: {identity.shape}; nnz={identity.nnz:,}")
    print(f"Text embeddings: {text_embeddings.shape}")
    print(f"Metadata features: {metadata.shape}")
    print(
        "Graph embeddings: "
        + ", ".join(
            f"{seed}:{graph_records[str(seed)]['shape']}"
            for seed in seeds
        )
    )
    print(f"Outputs: {CONTRACT_DIR}")


if __name__ == "__main__":
    main()
