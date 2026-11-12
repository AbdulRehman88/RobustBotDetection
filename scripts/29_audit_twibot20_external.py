from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "TwiBot-20-Format22" / "Twibot-20"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "28_twibot20_external_audit"
CHUNK_SIZE = 1_000_000
REQUIRED_FILES = ("node.json", "label.csv", "split.csv", "edge.csv")


def nonempty(value) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {"0": "human", "1": "bot", "human": "human", "bot": "bot"}
    if value not in aliases:
        raise ValueError(f"Unexpected label: {value!r}")
    return aliases[value]


def normalize_split(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "support": "support",
    }
    if value not in aliases:
        raise ValueError(f"Unexpected split: {value!r}")
    return aliases[value]


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_FILES if not (DATA_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required TwiBot-20 files: {missing}; root={DATA_ROOT}")

    print(f"Auditing official external dataset at: {DATA_ROOT}")
    print("Reading labels and official splits...")
    labels = pd.read_csv(DATA_ROOT / "label.csv", dtype=str)
    splits = pd.read_csv(DATA_ROOT / "split.csv", dtype=str)
    if list(labels.columns) != ["id", "label"]:
        raise ValueError(f"Unexpected label.csv columns: {labels.columns.tolist()}")
    if list(splits.columns) != ["id", "split"]:
        raise ValueError(f"Unexpected split.csv columns: {splits.columns.tolist()}")
    if labels["id"].duplicated().any():
        raise ValueError("Duplicate IDs in label.csv")
    if splits["id"].duplicated().any():
        raise ValueError("Duplicate IDs in split.csv")

    labels["label"] = labels["label"].map(normalize_label)
    splits["split"] = splits["split"].map(normalize_split)
    label_map = dict(zip(labels["id"], labels["label"]))
    split_map = dict(zip(splits["id"], splits["split"]))
    labeled_ids = set(label_map)
    split_ids = set(split_map)
    if not labeled_ids.issubset(split_ids):
        raise ValueError(
            f"Some labeled IDs are absent from split.csv: {len(labeled_ids-split_ids)}"
        )

    cohort = labels.merge(splits, on="id", validate="one_to_one")
    cohort_counts = (
        cohort.groupby(["split", "label"], observed=True).size().rename("count").reset_index()
    )
    cohort_counts.to_csv(OUTPUT_ROOT / "split_label_counts.csv", index=False)

    print("Streaming node.json (this is the longest audit stage)...")
    node_ids: set[str] = set()
    node_prefix_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    labeled_nodes_seen = 0
    duplicate_nodes = 0

    with (DATA_ROOT / "node.json").open("rb") as handle:
        for index, node in enumerate(ijson.items(handle, "item"), start=1):
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                raise ValueError(f"Node {index} has no usable id")
            if node_id in node_ids:
                duplicate_nodes += 1
            else:
                node_ids.add(node_id)
            prefix = node_id[0].lower() if node_id else "?"
            node_prefix_counts[prefix] += 1

            if node_id in labeled_ids:
                labeled_nodes_seen += 1
                description = node.get("description")
                location = node.get("location")
                name = node.get("name")
                username = node.get("username")
                metrics = node.get("public_metrics")
                field_counts["description_available"] += nonempty(description)
                field_counts["location_available"] += nonempty(location)
                field_counts["semantic_text_available"] += nonempty(description) or nonempty(location)
                field_counts["name_available"] += nonempty(name)
                field_counts["username_available"] += nonempty(username)
                field_counts["created_at_available"] += nonempty(node.get("created_at"))
                field_counts["verified_available"] += node.get("verified") is not None
                field_counts["public_metrics_available"] += isinstance(metrics, dict)
                if isinstance(metrics, dict):
                    for key in ("followers_count", "following_count", "tweet_count", "listed_count"):
                        field_counts[f"{key}_available"] += metrics.get(key) is not None
            if index % 50_000 == 0:
                print(f"  streamed {index:,} nodes")

    if duplicate_nodes:
        raise ValueError(f"Duplicate node IDs in node.json: {duplicate_nodes:,}")
    missing_labeled_nodes = labeled_ids - node_ids
    if missing_labeled_nodes:
        raise ValueError(f"Labeled IDs absent from node.json: {len(missing_labeled_nodes):,}")

    availability_rows = []
    for field, available in sorted(field_counts.items()):
        availability_rows.append(
            {
                "field": field,
                "available": int(available),
                "missing": int(labeled_nodes_seen - available),
                "availability_rate": float(available / labeled_nodes_seen),
            }
        )
    pd.DataFrame(availability_rows).to_csv(OUTPUT_ROOT / "modality_availability.csv", index=False)

    print("Streaming edge.csv...")
    relation_counts: Counter[str] = Counter()
    user_user_relation_counts: Counter[str] = Counter()
    edge_endpoint_classes: Counter[str] = Counter()
    graph_users: set[str] = set()
    same_split_user_edges = 0
    cross_split_user_edges = 0
    total_edges = 0
    self_loops = 0
    unknown_endpoint_edges = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_ROOT / "edge.csv",
            dtype=str,
            usecols=["source_id", "relation", "target_id"],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        if chunk.isna().any().any():
            raise ValueError(f"Missing edge values in chunk {chunk_number}")
        total_edges += len(chunk)
        relation_counts.update(chunk["relation"].str.strip().str.lower().value_counts().to_dict())
        self_loops += int((chunk["source_id"] == chunk["target_id"]).sum())

        sources = chunk["source_id"].astype(str)
        targets = chunk["target_id"].astype(str)
        source_prefix = sources.str[0].str.lower()
        target_prefix = targets.str[0].str.lower()
        classes = source_prefix + "->" + target_prefix
        edge_endpoint_classes.update(classes.value_counts().to_dict())

        known = sources.isin(node_ids) & targets.isin(node_ids)
        unknown_endpoint_edges += int((~known).sum())

        uu = source_prefix.eq("u") & target_prefix.eq("u")
        if uu.any():
            uu_chunk = chunk.loc[uu, ["source_id", "relation", "target_id"]].copy()
            uu_chunk["relation"] = uu_chunk["relation"].str.strip().str.lower()
            user_user_relation_counts.update(uu_chunk["relation"].value_counts().to_dict())
            graph_users.update(uu_chunk["source_id"])
            graph_users.update(uu_chunk["target_id"])

            source_splits = uu_chunk["source_id"].map(split_map)
            target_splits = uu_chunk["target_id"].map(split_map)
            both_supervised = source_splits.notna() & target_splits.notna()
            same_split_user_edges += int(
                (both_supervised & source_splits.eq(target_splits)).sum()
            )
            cross_split_user_edges += int(
                (both_supervised & source_splits.ne(target_splits)).sum()
            )
        print(f"  streamed {total_edges:,} edges")

    graph_labeled_users = labeled_ids & graph_users
    graph_availability = []
    for split_name in ("train", "val", "test", "support"):
        ids = {uid for uid, value in split_map.items() if value == split_name}
        if not ids:
            continue
        available = len(ids & graph_users)
        graph_availability.append(
            {
                "split": split_name,
                "users": len(ids),
                "graph_available": available,
                "graph_missing": len(ids) - available,
                "graph_availability_rate": available / len(ids),
            }
        )
    pd.DataFrame(graph_availability).to_csv(
        OUTPUT_ROOT / "graph_availability_by_split.csv", index=False
    )

    relation_table = pd.DataFrame(
        [
            {
                "relation": relation,
                "all_edges": count,
                "user_user_edges": user_user_relation_counts.get(relation, 0),
            }
            for relation, count in sorted(relation_counts.items())
        ]
    )
    relation_table.to_csv(OUTPUT_ROOT / "relation_counts.csv", index=False)
    pd.DataFrame(
        [{"endpoint_class": key, "count": value} for key, value in sorted(edge_endpoint_classes.items())]
    ).to_csv(OUTPUT_ROOT / "edge_endpoint_classes.csv", index=False)

    print("Computing SHA-256 provenance hashes...")
    provenance = []
    for name in REQUIRED_FILES:
        path = DATA_ROOT / name
        print(f"  hashing {name} ({path.stat().st_size / (1024**3):.2f} GiB)")
        provenance.append(
            {"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    pd.DataFrame(provenance).to_csv(OUTPUT_ROOT / "file_provenance_sha256.csv", index=False)

    summary = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "dataset_role": "locked_external_validation",
        "data_root": str(DATA_ROOT),
        "labeled_users": len(labeled_ids),
        "nodes": len(node_ids),
        "node_prefix_counts": dict(sorted(node_prefix_counts.items())),
        "labeled_nodes_seen": labeled_nodes_seen,
        "split_counts_all_nodes": splits["split"].value_counts().sort_index().to_dict(),
        "split_counts_labeled_nodes": cohort["split"].value_counts().sort_index().to_dict(),
        "label_counts": cohort["label"].value_counts().sort_index().to_dict(),
        "total_edges": total_edges,
        "relations": dict(sorted(relation_counts.items())),
        "user_user_relations": dict(sorted(user_user_relation_counts.items())),
        "labeled_graph_users": len(graph_labeled_users),
        "labeled_graph_availability_rate": len(graph_labeled_users) / len(labeled_ids),
        "same_split_supervised_user_edges": same_split_user_edges,
        "cross_split_supervised_user_edges": cross_split_user_edges,
        "self_loops": self_loops,
        "edges_with_unknown_endpoint": unknown_endpoint_edges,
        "unused_files": ["user_info.pt", "test.ipynb"],
        "runtime_minutes": (time.time() - started) / 60,
    }
    (OUTPUT_ROOT / "external_dataset_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL DATASET AUDIT: PASS")
    print(f"Labeled users: {len(labeled_ids):,}")
    print(f"Nodes: {len(node_ids):,}")
    print(f"Edges: {total_edges:,}")
    print(f"Relations: {dict(sorted(relation_counts.items()))}")
    print(
        f"Labeled graph availability: {len(graph_labeled_users):,}/{len(labeled_ids):,} "
        f"({len(graph_labeled_users)/len(labeled_ids):.4%})"
    )
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
