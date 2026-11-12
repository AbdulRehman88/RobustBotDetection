from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "02_canonical_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_graph_protocol_audit"

USERS_PATH = INPUT_DIR / "users.parquet"
EDGES_PATH = INPUT_DIR / "user_user_edges.parquet"

EXPECTED_SPLITS = {"train", "val", "test"}
EXPECTED_RELATIONS = {0, 1}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT_DIR / filename, index=False)


def encoded_edges(source: np.ndarray, target: np.ndarray, n_nodes: int) -> np.ndarray:
    return source.astype(np.int64) * np.int64(n_nodes) + target.astype(np.int64)


def reverse_overlap_count(
    source_a: np.ndarray,
    target_a: np.ndarray,
    source_b: np.ndarray,
    target_b: np.ndarray,
    n_nodes: int,
) -> int:
    keys_a = encoded_edges(source_a, target_a, n_nodes)
    reverse_b = encoded_edges(target_b, source_b, n_nodes)
    keys_a.sort()
    reverse_b.sort()
    return int(np.intersect1d(keys_a, reverse_b, assume_unique=True).size)


def degree_summary(
    users: pd.DataFrame,
    edges: pd.DataFrame,
    n_nodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = edges["source"].to_numpy(dtype=np.int64, copy=False)
    target = edges["target"].to_numpy(dtype=np.int64, copy=False)

    out_degree = np.bincount(source, minlength=n_nodes)
    in_degree = np.bincount(target, minlength=n_nodes)
    total_degree = out_degree + in_degree

    node_degrees = users[
        ["node_id", "id", "split", "label", "label_id"]
    ].copy()
    node_degrees["out_degree"] = out_degree
    node_degrees["in_degree"] = in_degree
    node_degrees["total_degree"] = total_degree
    node_degrees["isolated"] = total_degree == 0

    rows: list[dict] = []
    for (split, label), group in node_degrees.groupby(
        ["split", "label"], observed=True, sort=True
    ):
        for degree_name in ["out_degree", "in_degree", "total_degree"]:
            values = group[degree_name].to_numpy(dtype=np.int64)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "degree_type": degree_name,
                    "nodes": int(values.size),
                    "zero_degree_nodes": int((values == 0).sum()),
                    "zero_degree_rate": float((values == 0).mean()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(np.median(values)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p95": float(np.quantile(values, 0.95)),
                    "p99": float(np.quantile(values, 0.99)),
                    "maximum": int(values.max()),
                }
            )

    return node_degrees, pd.DataFrame(rows)


def relation_degree_summary(
    users: pd.DataFrame,
    edges: pd.DataFrame,
    n_nodes: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for relation_id, relation_edges in edges.groupby(
        "relation_id", observed=True, sort=True
    ):
        source = relation_edges["source"].to_numpy(dtype=np.int64, copy=False)
        target = relation_edges["target"].to_numpy(dtype=np.int64, copy=False)

        out_degree = np.bincount(source, minlength=n_nodes)
        in_degree = np.bincount(target, minlength=n_nodes)

        for split in sorted(EXPECTED_SPLITS):
            for label in sorted(users["label"].unique()):
                mask = (
                    users["split"].eq(split).to_numpy()
                    & users["label"].eq(label).to_numpy()
                )

                for degree_type, all_values in [
                    ("out_degree", out_degree),
                    ("in_degree", in_degree),
                ]:
                    values = all_values[mask]
                    rows.append(
                        {
                            "relation_id": int(relation_id),
                            "split": split,
                            "label": label,
                            "degree_type": degree_type,
                            "nodes": int(values.size),
                            "zero_degree_nodes": int((values == 0).sum()),
                            "zero_degree_rate": float((values == 0).mean()),
                            "mean": float(values.mean()),
                            "median": float(np.median(values)),
                            "p90": float(np.quantile(values, 0.90)),
                            "p99": float(np.quantile(values, 0.99)),
                            "maximum": int(values.max()),
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(
        USERS_PATH,
        columns=["node_id", "id", "split", "label", "label_id"],
    )
    edges = pd.read_parquet(EDGES_PATH)

    users = users.sort_values("node_id").reset_index(drop=True)
    edges = edges[
        ["source", "target", "relation_id", "source_split", "target_split"]
    ].copy()

    n_nodes = len(users)
    failures: list[str] = []

    if n_nodes != 1_000_000:
        failures.append(f"Expected 1,000,000 users; found {n_nodes:,}.")

    expected_node_ids = np.arange(n_nodes, dtype=np.int64)
    actual_node_ids = users["node_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(actual_node_ids, expected_node_ids):
        failures.append("node_id is not a contiguous zero-based index.")

    observed_splits = set(users["split"].astype(str).unique())
    if observed_splits != EXPECTED_SPLITS:
        failures.append(
            f"Unexpected user splits: {sorted(observed_splits)}."
        )

    observed_relations = set(
        edges["relation_id"].astype(int).unique().tolist()
    )
    if observed_relations != EXPECTED_RELATIONS:
        failures.append(
            f"Unexpected relation IDs: {sorted(observed_relations)}."
        )

    source = edges["source"].to_numpy(dtype=np.int64, copy=False)
    target = edges["target"].to_numpy(dtype=np.int64, copy=False)

    if source.min() < 0 or target.min() < 0:
        failures.append("Negative graph node index detected.")
    if source.max() >= n_nodes or target.max() >= n_nodes:
        failures.append("Graph node index exceeds the user table.")
    if np.any(source == target):
        failures.append("Self-loops remain in canonical graph.")

    split_lookup = users["split"].astype(str).to_numpy()
    derived_source_split = split_lookup[source]
    derived_target_split = split_lookup[target]

    if not np.array_equal(
        derived_source_split,
        edges["source_split"].astype(str).to_numpy(),
    ):
        failures.append("Stored source_split does not match source node.")
    if not np.array_equal(
        derived_target_split,
        edges["target_split"].astype(str).to_numpy(),
    ):
        failures.append("Stored target_split does not match target node.")

    edge_split_counts = (
        edges.groupby(
            ["relation_id", "source_split", "target_split"],
            observed=True,
        )
        .size()
        .rename("edge_count")
        .reset_index()
        .sort_values(
            ["relation_id", "source_split", "target_split"]
        )
    )
    edge_split_counts["edge_rate"] = (
        edge_split_counts["edge_count"] / len(edges)
    )
    save_csv(edge_split_counts, "edge_split_relation_counts.csv")

    same_split = edges["source_split"].eq(edges["target_split"])
    protocol_counts = pd.DataFrame(
        [
            {
                "protocol": "full_transductive_topology",
                "edges": int(len(edges)),
                "edge_rate": 1.0,
                "definition": (
                    "All canonical user-user edges; labels restricted to "
                    "training nodes during optimization."
                ),
            },
            {
                "protocol": "split_contained_sensitivity",
                "edges": int(same_split.sum()),
                "edge_rate": float(same_split.mean()),
                "definition": (
                    "Only edges whose endpoints belong to the same official "
                    "split; evaluated separately as an inductive sensitivity."
                ),
            },
            {
                "protocol": "cross_split_edges",
                "edges": int((~same_split).sum()),
                "edge_rate": float((~same_split).mean()),
                "definition": (
                    "Edges crossing official train/validation/test boundaries."
                ),
            },
        ]
    )
    save_csv(protocol_counts, "graph_protocol_counts.csv")

    node_degrees, degree_stats = degree_summary(users, edges, n_nodes)
    node_degrees.to_parquet(
        OUTPUT_DIR / "node_degree_features.parquet",
        index=False,
    )
    save_csv(degree_stats, "degree_summary_by_split_label.csv")

    relation_degree_stats = relation_degree_summary(
        users, edges, n_nodes
    )
    save_csv(
        relation_degree_stats,
        "relation_degree_summary_by_split_label.csv",
    )

    relation_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reciprocity_rows: list[dict] = []

    for relation_id in sorted(observed_relations):
        relation_edges = edges.loc[edges["relation_id"].eq(relation_id)]
        rel_source = relation_edges["source"].to_numpy(
            dtype=np.int64, copy=True
        )
        rel_target = relation_edges["target"].to_numpy(
            dtype=np.int64, copy=True
        )
        relation_arrays[relation_id] = (rel_source, rel_target)

        reciprocal = reverse_overlap_count(
            rel_source,
            rel_target,
            rel_source,
            rel_target,
            n_nodes,
        )
        reciprocity_rows.append(
            {
                "comparison": f"within_relation_{relation_id}",
                "relation_a": relation_id,
                "relation_b": relation_id,
                "edges_in_a": int(len(rel_source)),
                "reverse_matches": reciprocal,
                "reverse_match_rate": reciprocal / len(rel_source),
            }
        )

    source_0, target_0 = relation_arrays[0]
    source_1, target_1 = relation_arrays[1]

    cross_0_to_1 = reverse_overlap_count(
        source_0, target_0, source_1, target_1, n_nodes
    )
    cross_1_to_0 = reverse_overlap_count(
        source_1, target_1, source_0, target_0, n_nodes
    )

    reciprocity_rows.extend(
        [
            {
                "comparison": "relation_0_vs_reverse_relation_1",
                "relation_a": 0,
                "relation_b": 1,
                "edges_in_a": int(len(source_0)),
                "reverse_matches": cross_0_to_1,
                "reverse_match_rate": cross_0_to_1 / len(source_0),
            },
            {
                "comparison": "relation_1_vs_reverse_relation_0",
                "relation_a": 1,
                "relation_b": 0,
                "edges_in_a": int(len(source_1)),
                "reverse_matches": cross_1_to_0,
                "reverse_match_rate": cross_1_to_0 / len(source_1),
            },
        ]
    )
    save_csv(
        pd.DataFrame(reciprocity_rows),
        "relation_reciprocity_audit.csv",
    )

    isolated_counts = (
        node_degrees.groupby(
            ["split", "label"], observed=True, sort=True
        )
        .agg(
            nodes=("node_id", "size"),
            isolated_nodes=("isolated", "sum"),
        )
        .reset_index()
    )
    isolated_counts["isolated_rate"] = (
        isolated_counts["isolated_nodes"] / isolated_counts["nodes"]
    )
    save_csv(isolated_counts, "isolated_nodes_by_split_label.csv")

    protocol = {
        "status": "FAIL" if failures else "PASS",
        "primary_graph_protocol": {
            "name": "official_split_transductive_topology",
            "topology": (
                "Use the complete canonical user-user graph for message "
                "passing, including cross-split edges."
            ),
            "label_access": (
                "Only training-node labels may influence optimization. "
                "Validation labels are used only for model selection; test "
                "labels remain sealed until final evaluation."
            ),
            "feature_preprocessing": (
                "All learned preprocessing and feature models must be fit "
                "using training data only."
            ),
            "claim_boundary": (
                "This is transductive node classification, not deployment "
                "to an entirely unseen graph."
            ),
        },
        "required_sensitivity_protocol": {
            "name": "split_contained_inductive_sensitivity",
            "topology": (
                "Remove all cross-split edges and construct separate "
                "train, validation, and test subgraphs."
            ),
            "purpose": (
                "Quantify dependence on transductive cross-split topology "
                "and support a bounded unseen-subgraph generalization claim."
            ),
        },
        "prohibitions": [
            "Never use validation or test labels in graph features.",
            "Never select epochs or hyperparameters using test performance.",
            "Never call the primary full-graph protocol inductive.",
            "Never infer edge direction semantics beyond the raw documented "
            "relation labels without authoritative verification.",
        ],
        "failures": failures,
        "input_hashes": {
            "users.parquet": sha256_file(USERS_PATH),
            "user_user_edges.parquet": sha256_file(EDGES_PATH),
        },
        "counts": {
            "users": int(n_nodes),
            "edges": int(len(edges)),
            "same_split_edges": int(same_split.sum()),
            "cross_split_edges": int((~same_split).sum()),
            "isolated_users": int(node_degrees["isolated"].sum()),
        },
    }

    with (OUTPUT_DIR / "graph_protocol_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, indent=2)

    print(
        f"GRAPH PROTOCOL AUDIT: {protocol['status']}\n"
        f"Users: {n_nodes:,}\n"
        f"Edges: {len(edges):,}\n"
        f"Same-split edges: {same_split.sum():,}\n"
        f"Cross-split edges: {(~same_split).sum():,}\n"
        f"Isolated users: {node_degrees['isolated'].sum():,}\n"
        f"Relation-0 reverse matches in relation-1: "
        f"{cross_0_to_1:,}\n"
        f"Relation-1 reverse matches in relation-0: "
        f"{cross_1_to_0:,}\n"
        f"Outputs: {OUTPUT_DIR}"
    )

    if failures:
        for failure in failures:
            print(f"FAILURE: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()