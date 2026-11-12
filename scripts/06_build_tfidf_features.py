from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "outputs" / "02_canonical_data" / "users.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "05_tfidf_features"

SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}

WORD_MAX_FEATURES = 40_000
CHAR_MAX_FEATURES = 40_000
MIN_DF = 5
MAX_DF = 0.995


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "node_id",
            "label_id",
            "split",
            "description_text",
            "location_text",
        ],
    ).sort_values("node_id").reset_index(drop=True)

    node_ids = users["node_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(
        node_ids, np.arange(len(users), dtype=np.int64)
    ):
        raise RuntimeError("node_id is not contiguous and zero based.")

    description = clean_text(users["description_text"])
    location = clean_text(users["location_text"])

    # Primary text modality: semantic profile content only.
    # Names and usernames are deliberately excluded to avoid identity/handle
    # memorization and dataset-specific account-name shortcuts.
    semantic_text = (
        description.str.cat(location, sep=" ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    text_available = semantic_text.ne("").to_numpy(dtype=bool)

    split_ids = (
        users["split"]
        .astype(str)
        .map(SPLIT_TO_ID)
        .to_numpy(dtype=np.int8)
    )
    train_mask = split_ids == SPLIT_TO_ID["train"]
    train_available_mask = train_mask & text_available

    if train_available_mask.sum() == 0:
        raise RuntimeError("No available training text was found.")

    train_corpus = semantic_text.loc[train_available_mask]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    print(
        f"Fitting word TF-IDF on "
        f"{train_available_mask.sum():,} available training users..."
    )
    word_vectorizer.fit(train_corpus)

    print(
        f"Fitting character TF-IDF on "
        f"{train_available_mask.sum():,} available training users..."
    )
    char_vectorizer.fit(train_corpus)

    print("Transforming all users with training-fitted vectorizers...")
    word_matrix = word_vectorizer.transform(semantic_text)
    char_matrix = char_vectorizer.transform(semantic_text)

    feature_matrix = sparse.hstack(
        [word_matrix, char_matrix],
        format="csr",
        dtype=np.float32,
    )
    feature_matrix.sort_indices()

    unavailable_rows = np.flatnonzero(~text_available)
    if unavailable_rows.size:
        unavailable_nnz = int(
            feature_matrix[unavailable_rows].getnnz(axis=1).sum()
        )
        if unavailable_nnz != 0:
            raise RuntimeError(
                "Unavailable text rows contain nonzero TF-IDF features."
            )

    sparse.save_npz(
        OUTPUT_DIR / "semantic_tfidf_features.npz",
        feature_matrix,
        compressed=True,
    )
    np.save(
        OUTPUT_DIR / "text_available.npy",
        text_available,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "node_ids.npy",
        node_ids,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "labels.npy",
        users["label_id"].to_numpy(dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "split_ids.npy",
        split_ids,
        allow_pickle=False,
    )

    joblib.dump(
        {
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "feature_order": ["word_tfidf", "character_tfidf"],
            "primary_text_definition": (
                "description_text plus location_text"
            ),
            "excluded_identity_fields": ["name_text", "username_text"],
        },
        OUTPUT_DIR / "tfidf_vectorizers.joblib",
        compress=3,
    )

    availability_rows = []
    for split_name, split_id in SPLIT_TO_ID.items():
        split_mask = split_ids == split_id
        for label_id in sorted(users["label_id"].unique()):
            mask = (
                split_mask
                & users["label_id"].to_numpy(dtype=np.int64).astype(
                    np.int64
                ).__eq__(label_id)
            )
            availability_rows.append(
                {
                    "split": split_name,
                    "label_id": int(label_id),
                    "users": int(mask.sum()),
                    "text_available": int(
                        (mask & text_available).sum()
                    ),
                    "text_missing": int(
                        (mask & ~text_available).sum()
                    ),
                    "availability_rate": float(
                        text_available[mask].mean()
                    ),
                }
            )

    pd.DataFrame(availability_rows).to_csv(
        OUTPUT_DIR / "semantic_text_availability.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "users": int(len(users)),
        "primary_text_modality": {
            "included_fields": [
                "description_text",
                "location_text",
            ],
            "excluded_identity_fields": [
                "name_text",
                "username_text",
            ],
            "rationale": (
                "The primary text representation measures semantic profile "
                "content while limiting identity and username memorization."
            ),
        },
        "fit_scope": "available training users only",
        "word_features": int(len(word_vectorizer.vocabulary_)),
        "character_features": int(
            len(char_vectorizer.vocabulary_)
        ),
        "total_features": int(feature_matrix.shape[1]),
        "nonzero_values": int(feature_matrix.nnz),
        "text_available_users": int(text_available.sum()),
        "text_missing_users": int((~text_available).sum()),
        "parameters": {
            "word_ngram_range": [1, 2],
            "character_ngram_range": [3, 5],
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "sublinear_tf": True,
            "normalization": "l2",
            "dtype": "float32",
        },
        "integrity": {
            "validation_and_test_excluded_from_vocabulary_fit": True,
            "missing_text_encoded_as_all_zero_vector": True,
            "row_order_matches_node_id": True,
        },
        "input_sha256": sha256_file(INPUT_PATH),
        "outputs": {
            "matrix_shape": list(feature_matrix.shape),
            "matrix_sha256": sha256_file(
                OUTPUT_DIR / "semantic_tfidf_features.npz"
            ),
            "availability_sha256": sha256_file(
                OUTPUT_DIR / "text_available.npy"
            ),
        },
    }

    with (OUTPUT_DIR / "tfidf_feature_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("TF-IDF FEATURE BUILD: PASS")
    print(f"Users: {len(users):,}")
    print(
        f"Semantic text available: {text_available.sum():,} "
        f"({text_available.mean():.4%})"
    )
    print(f"Word features: {feature_matrix.shape[1] - len(char_vectorizer.vocabulary_):,}")
    print(f"Character features: {len(char_vectorizer.vocabulary_):,}")
    print(f"Total features: {feature_matrix.shape[1]:,}")
    print(f"Nonzero values: {feature_matrix.nnz:,}")
    print("Vocabulary fit: available training users only")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()