from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "outputs" / "02_canonical_data" / "users.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"

BASE_FEATURES = [
    "log_account_age",
    "log_tweet_count",
    "log_followers",
    "log_following",
    "log_listed",
    "verified",
]

RATE_FEATURES = [
    "ff_ratio",
    "tweets_per_day",
    "followers_per_day",
    "following_per_day",
]

FINAL_FEATURES = BASE_FEATURES + [
    f"log1p_{feature}" for feature in RATE_FEATURES
]

SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
LOWER_QUANTILE = 0.001
UPPER_QUANTILE = 0.999


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "node_id",
        "id",
        "label",
        "label_id",
        "split",
        *BASE_FEATURES,
        *RATE_FEATURES,
    ]
    users = pd.read_parquet(INPUT_PATH, columns=columns)
    users = users.sort_values("node_id").reset_index(drop=True)

    failures: list[str] = []

    expected_node_ids = np.arange(len(users), dtype=np.int64)
    node_ids = users["node_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(node_ids, expected_node_ids):
        failures.append("node_id is not contiguous and zero based.")

    if set(users["split"].astype(str).unique()) != set(SPLIT_TO_ID):
        failures.append("Official train/val/test splits are incomplete.")

    feature_frame = users[BASE_FEATURES].astype(np.float64).copy()

    for feature in RATE_FEATURES:
        values = users[feature].to_numpy(dtype=np.float64, copy=True)
        if np.any(values < 0):
            failures.append(f"Negative values detected in {feature}.")
        feature_frame[f"log1p_{feature}"] = np.log1p(
            np.maximum(values, 0.0)
        )

    feature_frame = feature_frame[FINAL_FEATURES]

    matrix = feature_frame.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        bad_columns = [
            FINAL_FEATURES[index]
            for index in np.where(~np.isfinite(matrix).all(axis=0))[0]
        ]
        failures.append(
            f"Non-finite metadata values detected: {bad_columns}."
        )

    train_mask = users["split"].eq("train").to_numpy()
    val_mask = users["split"].eq("val").to_numpy()
    test_mask = users["split"].eq("test").to_numpy()

    if failures:
        for failure in failures:
            print(f"FAILURE: {failure}")
        raise SystemExit(1)

    # All learned preprocessing parameters are estimated on training users only.
    train_matrix = matrix[train_mask]
    lower_bounds = np.quantile(
        train_matrix, LOWER_QUANTILE, axis=0
    )
    upper_bounds = np.quantile(
        train_matrix, UPPER_QUANTILE, axis=0
    )

    clipped_matrix = np.clip(matrix, lower_bounds, upper_bounds)

    scaler = StandardScaler()
    scaler.fit(clipped_matrix[train_mask])
    scaled_matrix = scaler.transform(clipped_matrix).astype(np.float32)

    if not np.isfinite(scaled_matrix).all():
        raise RuntimeError("Scaled metadata matrix contains non-finite values.")

    labels = users["label_id"].to_numpy(dtype=np.int64, copy=False)
    split_ids = (
        users["split"]
        .astype(str)
        .map(SPLIT_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    np.save(
        OUTPUT_DIR / "metadata_features.npy",
        scaled_matrix,
        allow_pickle=False,
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

    joblib.dump(
        {
            "feature_names": FINAL_FEATURES,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "scaler": scaler,
        },
        OUTPUT_DIR / "metadata_preprocessor.joblib",
    )

    split_summary = []
    for split_name, mask in [
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ]:
        split_values = scaled_matrix[mask]
        split_summary.append(
            {
                "split": split_name,
                "users": int(mask.sum()),
                "feature_count": int(scaled_matrix.shape[1]),
                "overall_mean": float(split_values.mean()),
                "overall_std": float(split_values.std()),
                "maximum_absolute_value": float(
                    np.abs(split_values).max()
                ),
            }
        )

    pd.DataFrame(split_summary).to_csv(
        OUTPUT_DIR / "metadata_feature_summary.csv",
        index=False,
    )

    preprocessing_parameters = pd.DataFrame(
        {
            "feature": FINAL_FEATURES,
            "training_lower_bound": lower_bounds,
            "training_upper_bound": upper_bounds,
            "training_mean_after_clipping": scaler.mean_,
            "training_scale_after_clipping": scaler.scale_,
        }
    )
    preprocessing_parameters.to_csv(
        OUTPUT_DIR / "metadata_preprocessing_parameters.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "users": int(len(users)),
        "features": FINAL_FEATURES,
        "feature_count": int(len(FINAL_FEATURES)),
        "split_encoding": SPLIT_TO_ID,
        "preprocessing": {
            "rate_transform": "log1p",
            "winsorization": {
                "lower_quantile": LOWER_QUANTILE,
                "upper_quantile": UPPER_QUANTILE,
                "fit_scope": "training_split_only",
            },
            "standardization": {
                "method": "StandardScaler",
                "fit_scope": "training_split_only",
            },
        },
        "integrity": {
            "ordered_by_node_id": True,
            "finite_output": True,
            "validation_and_test_excluded_from_fit": True,
        },
        "input_sha256": sha256_file(INPUT_PATH),
        "output_sha256": {
            "metadata_features.npy": sha256_file(
                OUTPUT_DIR / "metadata_features.npy"
            ),
            "node_ids.npy": sha256_file(OUTPUT_DIR / "node_ids.npy"),
            "labels.npy": sha256_file(OUTPUT_DIR / "labels.npy"),
            "split_ids.npy": sha256_file(OUTPUT_DIR / "split_ids.npy"),
        },
    }

    with (OUTPUT_DIR / "metadata_feature_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("METADATA FEATURE BUILD: PASS")
    print(f"Users: {len(users):,}")
    print(f"Features: {len(FINAL_FEATURES)}")
    print(f"Train: {train_mask.sum():,}")
    print(f"Validation: {val_mask.sum():,}")
    print(f"Test: {test_mask.sum():,}")
    print("Preprocessing fit: training split only")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()