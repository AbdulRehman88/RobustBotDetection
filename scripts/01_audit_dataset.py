from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_USER_FIELDS = {
    "id", "name", "username", "description", "location",
    "created_at", "public_metrics", "verified",
}
VALID_LABELS = {"human", "bot"}
VALID_SPLITS = {"train", "val", "test"}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def clean_text_available(record: dict[str, Any]) -> bool:
    for field in ("name", "username", "description", "location"):
        value = record.get(field)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"none", "nan", "null"}:
            return True
    return False


def metadata_available(record: dict[str, Any]) -> bool:
    created_at = record.get("created_at")
    metrics = record.get("public_metrics")
    return (
        isinstance(created_at, str)
        and bool(created_at.strip())
        and isinstance(metrics, dict)
        and any(
            key in metrics
            for key in ("tweet_count", "followers_count", "following_count", "listed_count")
        )
    )


def native(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)

    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    required = {
        "user_file": paths["user_file"],
        "label_file": paths["label_file"],
        "split_file": paths["split_file"],
        "edge_file": paths["edge_file"],
    }

    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    output_dir = paths["outputs_root"] / "00_dataset_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing immutable file hashes...")
    files = {}
    for name, path in required.items():
        print(f"  hashing {path.name} ({path.stat().st_size / 2**30:.3f} GiB)")
        files[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    print("Loading labels and official splits...")
    labels = pd.read_csv(paths["label_file"], dtype={"id": "string", "label": "string"})
    splits = pd.read_csv(paths["split_file"], dtype={"id": "string", "split": "string"})

    if list(labels.columns) != ["id", "label"]:
        raise ValueError(f"Unexpected label columns: {labels.columns.tolist()}")
    if list(splits.columns) != ["id", "split"]:
        raise ValueError(f"Unexpected split columns: {splits.columns.tolist()}")
    if labels["id"].isna().any() or splits["id"].isna().any():
        raise ValueError("Null IDs found in label.csv or split.csv")
    if labels["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in label.csv")
    if splits["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in split.csv")

    observed_labels = set(labels["label"].dropna().astype(str))
    observed_splits = set(splits["split"].dropna().astype(str))
    if observed_labels != VALID_LABELS:
        raise ValueError(f"Unexpected labels: {sorted(observed_labels)}")
    if observed_splits != VALID_SPLITS:
        raise ValueError(f"Unexpected splits: {sorted(observed_splits)}")

    index_df = labels.merge(splits, on="id", how="outer", validate="one_to_one", indicator=True)
    if not (index_df["_merge"] == "both").all():
        raise ValueError("label.csv and split.csv do not contain identical ID sets")
    index_df = index_df.drop(columns="_merge")

    print("Loading and auditing user.json...")
    with paths["user_file"].open("r", encoding="utf-8") as handle:
        users = json.load(handle)

    if not isinstance(users, list) or not users:
        raise ValueError("user.json must contain a non-empty list of user records")

    observed_user_fields = set().union(*(record.keys() for record in users[:1000]))
    missing_fields = sorted(REQUIRED_USER_FIELDS - observed_user_fields)
    if missing_fields:
        raise ValueError(f"Required user fields missing: {missing_fields}")

    user_ids = [str(record.get("id")) for record in users]
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("Duplicate IDs found in user.json")

    user_id_set = set(user_ids)
    indexed_id_set = set(index_df["id"].astype(str))
    if user_id_set != indexed_id_set:
        raise ValueError(
            "user.json, label.csv, and split.csv do not contain identical user ID sets"
        )

    text_available_ids = {
        str(record["id"]) for record in users if clean_text_available(record)
    }
    metadata_available_ids = {
        str(record["id"]) for record in users if metadata_available(record)
    }

    del users, user_ids

    print("Streaming and auditing the complete edge file...")
    edge_header = pd.read_csv(paths["edge_file"], nrows=0).columns.tolist()
    expected_edge_columns = ["source_id", "relation", "target_id"]
    if edge_header != expected_edge_columns:
        raise ValueError(
            f"Unexpected edge columns: {edge_header}; expected {expected_edge_columns}"
        )

    edge_rows = 0
    self_loops = 0
    valid_source_rows = 0
    valid_target_rows = 0
    both_valid_rows = 0
    relation_counts: Counter[str] = Counter()
    connected_ids: set[str] = set()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            paths["edge_file"],
            dtype={"source_id": "string", "relation": "string", "target_id": "string"},
            chunksize=args.edge_chunk_size,
        ),
        start=1,
    ):
        if chunk.isna().any().any():
            raise ValueError(f"Null edge values detected in chunk {chunk_number}")

        source = chunk["source_id"].astype(str)
        target = chunk["target_id"].astype(str)
        relation = chunk["relation"].astype(str).str.lower()

        source_valid = source.isin(user_id_set)
        target_valid = target.isin(user_id_set)
        both_valid = source_valid & target_valid

        rows = len(chunk)
        edge_rows += rows
        self_loops += int((source == target).sum())
        valid_source_rows += int(source_valid.sum())
        valid_target_rows += int(target_valid.sum())
        both_valid_rows += int(both_valid.sum())
        relation_counts.update(relation.value_counts().to_dict())

        if both_valid.any():
            connected_ids.update(source[both_valid].unique().tolist())
            connected_ids.update(target[both_valid].unique().tolist())

        print(
            f"  chunk={chunk_number:03d} "
            f"rows={edge_rows:,} both-endpoints-valid={both_valid_rows:,}",
            flush=True,
        )

    if edge_rows == 0 or both_valid_rows == 0:
        raise ValueError("Graph audit failed: no usable edges were retained")

    index_df["text_available"] = index_df["id"].isin(text_available_ids)
    index_df["graph_available"] = index_df["id"].isin(connected_ids)
    index_df["metadata_available"] = index_df["id"].isin(metadata_available_ids)

    def pattern(row: pd.Series) -> str:
        return "".join(
            [
                "T" if row["text_available"] else "-",
                "G" if row["graph_available"] else "-",
                "M" if row["metadata_available"] else "-",
            ]
        )

    index_df["availability_pattern"] = index_df.apply(pattern, axis=1)

    split_label_counts = (
        index_df.groupby(["split", "label"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    availability_counts = (
        index_df.groupby(
            ["split", "label", "availability_pattern"], observed=True
        )
        .size()
        .rename("count")
        .reset_index()
    )

    split_label_counts.to_csv(output_dir / "split_label_counts.csv", index=False)
    availability_counts.to_csv(output_dir / "availability_counts.csv", index=False)
    index_df[
        [
            "id", "label", "split",
            "text_available", "graph_available", "metadata_available",
            "availability_pattern",
        ]
    ].to_parquet(output_dir / "user_registry.parquet", index=False)

    audit = {
        "status": "PASS",
        "files": files,
        "schemas": {
            "user_fields": sorted(observed_user_fields),
            "label_columns": labels.columns.tolist(),
            "split_columns": splits.columns.tolist(),
            "edge_columns": edge_header,
        },
        "counts": {
            "users": len(user_id_set),
            "labels": len(labels),
            "splits": len(splits),
            "edges_total": edge_rows,
            "edges_valid_source": valid_source_rows,
            "edges_valid_target": valid_target_rows,
            "edges_both_endpoints_valid": both_valid_rows,
            "edge_self_loops": self_loops,
            "graph_connected_users": len(connected_ids),
            "text_available_users": len(text_available_ids),
            "metadata_available_users": len(metadata_available_ids),
        },
        "relations": {
            key: int(value) for key, value in sorted(relation_counts.items())
        },
        "split_label_counts": [
            {key: native(value) for key, value in row.items()}
            for row in split_label_counts.to_dict(orient="records")
        ],
        "availability_counts": [
            {key: native(value) for key, value in row.items()}
            for row in availability_counts.to_dict(orient="records")
        ],
    }

    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    manifest = {
        name: {
            "bytes": details["bytes"],
            "sha256": details["sha256"],
            "path": details["path"],
        }
        for name, details in files.items()
    }
    with (Path("manifests") / "input_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nDATASET AUDIT: PASS")
    print(f"Users: {len(user_id_set):,}")
    print(f"Edges: {edge_rows:,}")
    print(f"Usable edges: {both_valid_rows:,}")
    print(f"Connected users: {len(connected_ids):,}")
    print(f"Relations: {dict(sorted(relation_counts.items()))}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
