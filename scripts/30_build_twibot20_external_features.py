from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import ijson
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "TwiBot-20-Format22" / "Twibot-20"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "28_twibot20_external_audit"
SOURCE_METADATA_ROOT = PROJECT_ROOT / "outputs" / "04_metadata_features"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"

REFERENCE_DATE = pd.Timestamp("2022-03-14", tz="UTC")
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
FEATURE_NAMES = [
    "log_account_age",
    "log_tweet_count",
    "log_followers",
    "log_following",
    "log_listed",
    "verified",
    "log1p_ff_ratio",
    "log1p_tweets_per_day",
    "log1p_followers_per_day",
    "log1p_following_per_day",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


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
        "dev": "val",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    if value not in aliases:
        raise ValueError(f"Unexpected supervised split: {value!r}")
    return aliases[value]


def numeric(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return max(result, 0.0)


def bool_value(value) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def build_record(node: dict, label: str, split: str) -> dict:
    metrics = node.get("public_metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    followers = numeric(metrics.get("followers_count"))
    following = numeric(metrics.get("following_count"))
    tweets = numeric(metrics.get("tweet_count"))
    listed = numeric(metrics.get("listed_count"))

    created_at = pd.to_datetime(node.get("created_at"), errors="coerce", utc=True)
    if pd.isna(created_at):
        age_days = 0.0
    else:
        age_days = max(0.0, float((REFERENCE_DATE - created_at).days))

    description = clean_text(node.get("description"))
    location = clean_text(node.get("location"))
    semantic_text = clean_text(f"{description} {location}")
    ff_ratio = (followers + 1.0) / (following + 1.0)
    tweets_per_day = tweets / (age_days + 1.0)
    followers_per_day = followers / (age_days + 1.0)
    following_per_day = following / (age_days + 1.0)

    return {
        "external_id": str(node["id"]).strip(),
        "label": label,
        "split": split,
        "name_text": clean_text(node.get("name")),
        "username_text": clean_text(node.get("username")),
        "description_text": description,
        "location_text": location,
        "semantic_text": semantic_text,
        "text_available": bool(semantic_text),
        "account_age_days": age_days,
        "tweet_count": tweets,
        "followers_count": followers,
        "following_count": following,
        "listed_count": listed,
        "verified": bool_value(node.get("verified", False)),
        "ff_ratio": ff_ratio,
        "tweets_per_day": tweets_per_day,
        "followers_per_day": followers_per_day,
        "following_per_day": following_per_day,
        "log_account_age": np.log1p(age_days),
        "log_tweet_count": np.log1p(tweets),
        "log_followers": np.log1p(followers),
        "log_following": np.log1p(following),
        "log_listed": np.log1p(listed),
        "log1p_ff_ratio": np.log1p(ff_ratio),
        "log1p_tweets_per_day": np.log1p(tweets_per_day),
        "log1p_followers_per_day": np.log1p(followers_per_day),
        "log1p_following_per_day": np.log1p(following_per_day),
    }


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    audit_manifest = AUDIT_ROOT / "external_dataset_audit.json"
    if not audit_manifest.is_file():
        raise FileNotFoundError("Run Script 29 external audit before Script 30.")
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("External dataset audit did not pass.")

    labels = pd.read_csv(DATA_ROOT / "label.csv", dtype=str)
    splits = pd.read_csv(DATA_ROOT / "split.csv", dtype=str)
    labels["label"] = labels["label"].map(normalize_label)

    split_raw = dict(zip(splits["id"].astype(str), splits["split"].astype(str)))
    supervised_ids = set(labels["id"].astype(str))
    missing_splits = supervised_ids - set(split_raw)
    if missing_splits:
        raise ValueError(f"Labeled users missing split assignments: {len(missing_splits)}")

    label_map = dict(zip(labels["id"].astype(str), labels["label"]))
    split_map = {uid: normalize_split(split_raw[uid]) for uid in supervised_ids}

    print(f"Extracting {len(supervised_ids):,} labeled profiles from node.json...")
    records = []
    found: set[str] = set()
    streamed = 0
    with (DATA_ROOT / "node.json").open("rb") as handle:
        for node in ijson.items(handle, "item"):
            streamed += 1
            node_id = str(node.get("id", "")).strip()
            if node_id in supervised_ids:
                if node_id in found:
                    raise ValueError(f"Duplicate labeled node in node.json: {node_id}")
                records.append(build_record(node, label_map[node_id], split_map[node_id]))
                found.add(node_id)
                if len(found) == len(supervised_ids):
                    break
            if streamed % 50_000 == 0:
                print(f"  streamed {streamed:,} nodes; recovered {len(found):,} labeled users")

    missing_nodes = supervised_ids - found
    if missing_nodes:
        raise ValueError(f"Labeled profiles absent from node.json: {len(missing_nodes)}")

    users = pd.DataFrame.from_records(records).sort_values("external_id").reset_index(drop=True)
    users.insert(0, "row_id", np.arange(len(users), dtype=np.int64))
    users["label_id"] = users["label"].eq("bot").astype(np.int8)
    users["split_id"] = users["split"].map(SPLIT_TO_ID).astype(np.int8)

    if len(users) != len(supervised_ids) or users["external_id"].duplicated().any():
        raise RuntimeError("External canonical user table failed cardinality/uniqueness checks.")
    if set(users["split"].unique()) != {"train", "val", "test"}:
        raise RuntimeError(f"Unexpected supervised splits: {sorted(users['split'].unique())}")

    raw_features = users[FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_features).all():
        raise RuntimeError("Non-finite raw metadata features detected.")

    print("Applying frozen TwiBot-22 metadata preprocessing (strict zero-shot)...")
    source_preprocessor_path = SOURCE_METADATA_ROOT / "metadata_preprocessor.joblib"
    source = joblib.load(source_preprocessor_path)
    if list(source["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("TwiBot-22 metadata feature order does not match Script 30.")
    source_lower = np.asarray(source["lower_bounds"], dtype=np.float64)
    source_upper = np.asarray(source["upper_bounds"], dtype=np.float64)
    source_clipped = np.clip(raw_features, source_lower, source_upper)
    source_scaled = source["scaler"].transform(source_clipped).astype(np.float32)

    print("Fitting separate preprocessing on TwiBot-20 training users only (replication)...")
    train_mask = users["split"].eq("train").to_numpy()
    train_raw = raw_features[train_mask]
    external_lower = np.quantile(train_raw, 0.001, axis=0)
    external_upper = np.quantile(train_raw, 0.999, axis=0)
    external_train_clipped = np.clip(train_raw, external_lower, external_upper)
    external_scaler = StandardScaler().fit(external_train_clipped)
    external_scaled = external_scaler.transform(
        np.clip(raw_features, external_lower, external_upper)
    ).astype(np.float32)

    if not np.isfinite(source_scaled).all() or not np.isfinite(external_scaled).all():
        raise RuntimeError("Non-finite standardized metadata features detected.")

    users.to_parquet(OUTPUT_ROOT / "users.parquet", index=False)
    np.save(OUTPUT_ROOT / "external_ids.npy", users["external_id"].to_numpy(dtype=str))
    np.save(OUTPUT_ROOT / "row_ids.npy", users["row_id"].to_numpy(dtype=np.int64))
    np.save(OUTPUT_ROOT / "labels.npy", users["label_id"].to_numpy(dtype=np.int8))
    np.save(OUTPUT_ROOT / "split_ids.npy", users["split_id"].to_numpy(dtype=np.int8))
    np.save(OUTPUT_ROOT / "text_available.npy", users["text_available"].to_numpy(dtype=bool))
    np.save(OUTPUT_ROOT / "metadata_twibot22_scaled.npy", source_scaled)
    np.save(OUTPUT_ROOT / "metadata_twibot20_train_scaled.npy", external_scaled)

    external_preprocessor = {
        "feature_names": FEATURE_NAMES,
        "lower_quantile": 0.001,
        "upper_quantile": 0.999,
        "lower_bounds": external_lower,
        "upper_bounds": external_upper,
        "scaler": external_scaler,
        "fit_scope": "TwiBot-20 official training split only",
    }
    joblib.dump(external_preprocessor, OUTPUT_ROOT / "twibot20_metadata_preprocessor.joblib")

    distribution_rows = []
    for feature_index, feature in enumerate(FEATURE_NAMES):
        distribution_rows.append(
            {
                "feature": feature,
                "twibot20_train_lower_bound": external_lower[feature_index],
                "twibot20_train_upper_bound": external_upper[feature_index],
                "twibot20_train_mean_after_clipping": external_scaler.mean_[feature_index],
                "twibot20_train_scale_after_clipping": external_scaler.scale_[feature_index],
                "twibot22_training_lower_bound": source_lower[feature_index],
                "twibot22_training_upper_bound": source_upper[feature_index],
                "twibot22_training_mean_after_clipping": source["scaler"].mean_[feature_index],
                "twibot22_training_scale_after_clipping": source["scaler"].scale_[feature_index],
            }
        )
    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT_ROOT / "metadata_preprocessing_comparison.csv", index=False
    )

    split_label_counts = (
        users.groupby(["split", "label"], observed=True).size().rename("count").reset_index()
    )
    split_label_counts.to_csv(OUTPUT_ROOT / "split_label_counts.csv", index=False)

    output_files = [
        "users.parquet",
        "external_ids.npy",
        "row_ids.npy",
        "labels.npy",
        "split_ids.npy",
        "text_available.npy",
        "metadata_twibot22_scaled.npy",
        "metadata_twibot20_train_scaled.npy",
        "twibot20_metadata_preprocessor.joblib",
        "metadata_preprocessing_comparison.csv",
        "split_label_counts.csv",
    ]
    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "role": "locked_external_validation",
        "users": len(users),
        "profiles_streamed_before_complete_recovery": streamed,
        "reference_date": str(REFERENCE_DATE),
        "features": FEATURE_NAMES,
        "semantic_text_definition": "clean(description + space + location)",
        "text_available": int(users["text_available"].sum()),
        "text_availability_rate": float(users["text_available"].mean()),
        "zero_shot_preprocessing": {
            "source": str(source_preprocessor_path),
            "source_sha256": sha256(source_preprocessor_path),
            "fit_scope": "TwiBot-22 training split only; no TwiBot-20 fitting",
        },
        "replication_preprocessing": {
            "fit_scope": "TwiBot-20 official training split only",
            "validation_and_test_excluded_from_fit": True,
        },
        "integrity": {
            "all_labeled_profiles_recovered": True,
            "unique_external_ids": True,
            "finite_features": True,
            "row_order": "lexicographic external_id",
        },
        "output_sha256": {name: sha256(OUTPUT_ROOT / name) for name in output_files},
        "runtime_minutes": (time.time() - started) / 60,
    }
    (OUTPUT_ROOT / "external_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL FEATURE BUILD: PASS")
    print(f"Users: {len(users):,}")
    print(f"Profiles streamed: {streamed:,}")
    print(
        f"Semantic text available: {users['text_available'].sum():,} "
        f"({users['text_available'].mean():.4%})"
    )
    print("Zero-shot preprocessing: frozen TwiBot-22 training fit")
    print("Replication preprocessing: TwiBot-20 training fit only")
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
