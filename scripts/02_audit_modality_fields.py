from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


TEXT_FIELDS = ("name", "username", "description", "location")
METRIC_FIELDS = ("tweet_count", "followers_count", "following_count", "listed_count")


def present(value) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in {"none", "nan", "null"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.local.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)["paths"]

    user_path = Path(cfg["user_file"])
    label_path = Path(cfg["label_file"])
    split_path = Path(cfg["split_file"])
    registry_path = Path(cfg["outputs_root"]) / "00_dataset_audit" / "user_registry.parquet"
    output_dir = Path(cfg["outputs_root"]) / "01_modality_field_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    with user_path.open("r", encoding="utf-8") as handle:
        users = json.load(handle)

    rows = []
    for record in users:
        metrics = record.get("public_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}

        row = {"id": str(record["id"])}
        for field in TEXT_FIELDS:
            row[f"{field}_available"] = present(record.get(field))

        row["identity_text_available"] = (
            row["name_available"] or row["username_available"]
        )
        row["semantic_text_available"] = (
            row["description_available"] or row["location_available"]
        )
        row["any_profile_text_available"] = any(
            row[f"{field}_available"] for field in TEXT_FIELDS
        )
        row["complete_profile_text"] = all(
            row[f"{field}_available"] for field in TEXT_FIELDS
        )

        row["created_at_available"] = present(record.get("created_at"))
        row["verified_available"] = record.get("verified") is not None
        row["public_metrics_available"] = isinstance(
            record.get("public_metrics"), dict
        )

        for field in METRIC_FIELDS:
            row[f"{field}_available"] = field in metrics and metrics[field] is not None

        row["complete_metadata"] = (
            row["created_at_available"]
            and row["verified_available"]
            and row["public_metrics_available"]
            and all(row[f"{field}_available"] for field in METRIC_FIELDS)
        )
        rows.append(row)

    availability = pd.DataFrame(rows)
    labels = pd.read_csv(label_path, dtype="string")
    splits = pd.read_csv(split_path, dtype="string")
    graph_registry = pd.read_parquet(
        registry_path, columns=["id", "graph_available"]
    )

    audit = (
        availability
        .merge(labels, on="id", validate="one_to_one")
        .merge(splits, on="id", validate="one_to_one")
        .merge(graph_registry, on="id", validate="one_to_one")
    )

    boolean_columns = [
        column for column in audit.columns
        if column.endswith("_available")
        or column in {
            "identity_text_available",
            "semantic_text_available",
            "any_profile_text_available",
            "complete_profile_text",
            "complete_metadata",
            "graph_available",
        }
    ]

    summary_rows = []
    for (split, label), group in audit.groupby(["split", "label"], observed=True):
        for column in boolean_columns:
            count = int(group[column].sum())
            summary_rows.append(
                {
                    "split": split,
                    "label": label,
                    "field": column,
                    "available": count,
                    "missing": int(len(group) - count),
                    "availability_rate": count / len(group),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "field_availability_by_split_label.csv", index=False)
    audit.to_parquet(output_dir / "field_availability_registry.parquet", index=False)

    overall = []
    for column in boolean_columns:
        count = int(audit[column].sum())
        overall.append(
            {
                "field": column,
                "available": count,
                "missing": int(len(audit) - count),
                "availability_rate": count / len(audit),
            }
        )

    overall_df = pd.DataFrame(overall).sort_values("field")
    overall_df.to_csv(output_dir / "field_availability_overall.csv", index=False)

    result = {
        "status": "PASS",
        "users": len(audit),
        "overall": overall_df.to_dict(orient="records"),
    }
    with (output_dir / "modality_field_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("\nMODALITY FIELD AUDIT: PASS")
    print(
        overall_df.to_string(
            index=False,
            formatters={"availability_rate": lambda value: f"{value:.6f}"},
        )
    )
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
