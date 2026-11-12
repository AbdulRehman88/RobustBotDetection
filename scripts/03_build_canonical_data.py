from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REFERENCE_DATE = pd.Timestamp("2022-03-14", tz="UTC")
TEXT_FIELDS = ("name", "username", "description", "location")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def valid_text(value) -> str:
    if not isinstance(value, str):
        return ""
    value = " ".join(value.split())
    return "" if value.lower() in {"none", "nan", "null"} else value


def metric(record, key: str) -> float:
    values = record.get("public_metrics")
    if not isinstance(values, dict):
        return 0.0
    value = values.get(key, 0)
    return float(value) if value is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8-sig") as handle:
        paths = {
            key: Path(value)
            for key, value in yaml.safe_load(handle)["paths"].items()
        }

    output_dir = paths["outputs_root"] / "02_canonical_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading canonical user records...")
    with paths["user_file"].open("r", encoding="utf-8") as handle:
        users = json.load(handle)

    labels = pd.read_csv(paths["label_file"], dtype="string")
    splits = pd.read_csv(paths["split_file"], dtype="string")
    audit_registry = pd.read_parquet(
        paths["outputs_root"] / "00_dataset_audit" / "user_registry.parquet"
    )

    records = []
    for record in users:
        text_parts = [valid_text(record.get(field)) for field in TEXT_FIELDS]
        created_at = pd.to_datetime(
            record.get("created_at"), errors="coerce", utc=True
        )
        age_days = (
            max(0.0, float((REFERENCE_DATE - created_at).days))
            if not pd.isna(created_at)
            else 0.0
        )

        tweets = metric(record, "tweet_count")
        followers = metric(record, "followers_count")
        following = metric(record, "following_count")
        listed = metric(record, "listed_count")

        records.append(
            {
                "id": str(record["id"]),
                "profile_text": " [SEP] ".join(
                    part for part in text_parts if part
                ),
                "name_text": text_parts[0],
                "username_text": text_parts[1],
                "description_text": text_parts[2],
                "location_text": text_parts[3],
                "account_age_days": age_days,
                "tweet_count": tweets,
                "followers_count": followers,
                "following_count": following,
                "listed_count": listed,
                "verified": int(bool(record.get("verified", False))),
                "ff_ratio": (followers + 1.0) / (following + 1.0),
                "tweets_per_day": tweets / (age_days + 1.0),
                "followers_per_day": followers / (age_days + 1.0),
                "following_per_day": following / (age_days + 1.0),
                "log_account_age": np.log1p(age_days),
                "log_tweet_count": np.log1p(tweets),
                "log_followers": np.log1p(followers),
                "log_following": np.log1p(following),
                "log_listed": np.log1p(listed),
            }
        )

    del users
    user_df = pd.DataFrame.from_records(records)
    del records

    user_df = (
        user_df
        .merge(labels, on="id", validate="one_to_one")
        .merge(splits, on="id", validate="one_to_one")
        .merge(
            audit_registry[
                [
                    "id", "text_available", "graph_available",
                    "metadata_available", "availability_pattern",
                ]
            ],
            on="id",
            validate="one_to_one",
        )
    )

    label_map = {"human": 0, "bot": 1}
    split_map = {"train": 0, "val": 1, "test": 2}
    user_df["label_id"] = user_df["label"].map(label_map).astype("int8")
    user_df["split_id"] = user_df["split"].map(split_map).astype("int8")
    user_df.insert(0, "node_id", np.arange(len(user_df), dtype=np.int32))

    if user_df["id"].duplicated().any():
        raise ValueError("Canonical registry contains duplicate user IDs")
    if user_df["profile_text"].str.len().eq(0).any():
        raise ValueError("Canonical profile text unexpectedly empty")

    users_path = output_dir / "users.parquet"
    user_df.to_parquet(users_path, index=False)
    print(f"Canonical users saved: {len(user_df):,}")

    id_to_node = dict(
        zip(user_df["id"].astype(str), user_df["node_id"].astype(int))
    )
    node_splits = user_df["split"].astype(str).to_numpy()

    print("Streaming the complete heterogeneous edge file...")
    retained_chunks = []
    scanned_rows = 0
    retained_rows = 0
    raw_relation_counts: Counter[str] = Counter()
    retained_relation_counts: Counter[str] = Counter()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            paths["edge_file"],
            dtype={
                "source_id": "string",
                "relation": "string",
                "target_id": "string",
            },
            chunksize=args.edge_chunk_size,
        ),
        start=1,
    ):
        scanned_rows += len(chunk)
        relations = chunk["relation"].astype(str).str.lower()
        raw_relation_counts.update(relations.value_counts().to_dict())

        source_node = chunk["source_id"].astype(str).map(id_to_node)
        target_node = chunk["target_id"].astype(str).map(id_to_node)
        keep = source_node.notna() & target_node.notna()

        if keep.any():
            retained = pd.DataFrame(
                {
                    "source": source_node[keep].astype(np.int32).to_numpy(),
                    "target": target_node[keep].astype(np.int32).to_numpy(),
                    "relation": relations[keep].to_numpy(),
                }
            )
            retained_relation_counts.update(
                retained["relation"].value_counts().to_dict()
            )
            retained_chunks.append(retained)
            retained_rows += len(retained)

        print(
            f"  chunk={chunk_number:03d} scanned={scanned_rows:,} "
            f"user-user={retained_rows:,}",
            flush=True,
        )

    if retained_rows == 0:
        raise ValueError("No user-to-user edges retained")

    edges = pd.concat(retained_chunks, ignore_index=True)
    del retained_chunks

    raw_user_edges = len(edges)
    self_loops = int((edges["source"] == edges["target"]).sum())
    edges = edges.loc[edges["source"] != edges["target"]].copy()
    pre_dedup = len(edges)
    edges = edges.drop_duplicates(
        subset=["source", "relation", "target"], keep="first"
    ).reset_index(drop=True)
    duplicate_edges_removed = pre_dedup - len(edges)

    relation_vocab = sorted(edges["relation"].unique().tolist())
    relation_to_id = {
        relation: index for index, relation in enumerate(relation_vocab)
    }
    edges["relation_id"] = (
        edges["relation"].map(relation_to_id).astype(np.int16)
    )

    source_values = edges["source"].to_numpy()
    target_values = edges["target"].to_numpy()
    edges["source_split"] = node_splits[source_values]
    edges["target_split"] = node_splits[target_values]

    edge_split_counts = (
        edges.groupby(
            ["source_split", "target_split", "relation"], observed=True
        )
        .size()
        .rename("count")
        .reset_index()
    )
    edge_split_counts.to_csv(
        output_dir / "edge_split_relation_counts.csv", index=False
    )

    edges_path = output_dir / "user_user_edges.parquet"
    edges[
        [
            "source", "target", "relation_id",
            "source_split", "target_split",
        ]
    ].to_parquet(edges_path, index=False)

    relation_table = pd.DataFrame(
        {
            "relation": relation_vocab,
            "relation_id": [
                relation_to_id[relation] for relation in relation_vocab
            ],
            "deduplicated_edges": [
                int((edges["relation"] == relation).sum())
                for relation in relation_vocab
            ],
        }
    )
    relation_table.to_csv(output_dir / "relation_vocabulary.csv", index=False)

    summary = {
        "status": "PASS",
        "users": len(user_df),
        "edge_rows_scanned": scanned_rows,
        "raw_user_user_edges": raw_user_edges,
        "self_loops_removed": self_loops,
        "duplicate_edges_removed": duplicate_edges_removed,
        "deduplicated_user_user_edges": len(edges),
        "relations": relation_table.to_dict(orient="records"),
        "raw_relation_counts_all_node_types": dict(
            sorted((key, int(value)) for key, value in raw_relation_counts.items())
        ),
        "retained_relation_counts_before_deduplication": dict(
            sorted(
                (key, int(value))
                for key, value in retained_relation_counts.items()
            )
        ),
        "artifacts": {
            "users": {
                "path": str(users_path.resolve()),
                "sha256": sha256_file(users_path),
            },
            "edges": {
                "path": str(edges_path.resolve()),
                "sha256": sha256_file(edges_path),
            },
            "relation_vocabulary": relation_to_id,
        },
    }

    with (output_dir / "canonical_data_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nCANONICAL DATA BUILD: PASS")
    print(f"Users: {len(user_df):,}")
    print(f"Raw user-user edges: {raw_user_edges:,}")
    print(f"Self-loops removed: {self_loops:,}")
    print(f"Duplicate edges removed: {duplicate_edges_removed:,}")
    print(f"Final edges: {len(edges):,}")
    print(relation_table.to_string(index=False))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
