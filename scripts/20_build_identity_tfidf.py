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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "19_identity_tfidf"

WORD_MAX_FEATURES = 20_000
CHAR_MAX_FEATURES = 40_000
MIN_DF = 5
MAX_DF = 0.995
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


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
            "label",
            "split",
            "name_text",
            "username_text",
        ],
    ).sort_values("node_id").reset_index(drop=True)

    node_ids = users["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        node_ids, np.arange(len(users), dtype=np.int64)
    ):
        raise RuntimeError("Invalid canonical node order.")

    labels = (
        users["label"]
        .astype(str)
        .str.lower()
        .eq("bot")
        .to_numpy(dtype=np.int8)
    )
    split_ids = (
        users["split"]
        .astype(str)
        .map(SPLIT_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    name = clean_text(users["name_text"])
    username = clean_text(users["username_text"])

    # Explicit field markers prevent accidental ambiguity between a
    # display name and account handle.
    identity_text = (
        "name "
        + name
        + " username "
        + username
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    identity_available = (
        name.ne("") | username.ne("")
    ).to_numpy(dtype=bool)
    train_mask = split_ids == SPLIT_TO_ID["train"]
    fit_mask = train_mask & identity_available

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
        ngram_range=(2, 5),
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
        f"Fitting identity word TF-IDF on "
        f"{fit_mask.sum():,} training users..."
    )
    word_vectorizer.fit(identity_text.loc[fit_mask])

    print(
        f"Fitting identity character TF-IDF on "
        f"{fit_mask.sum():,} training users..."
    )
    char_vectorizer.fit(identity_text.loc[fit_mask])

    print("Transforming all identity profiles...")
    word_matrix = word_vectorizer.transform(identity_text)
    char_matrix = char_vectorizer.transform(identity_text)

    matrix = sparse.hstack(
        [word_matrix, char_matrix],
        format="csr",
        dtype=np.float32,
    )
    matrix.sort_indices()

    sparse.save_npz(
        OUTPUT_DIR / "identity_tfidf_features.npz",
        matrix,
        compressed=True,
    )
    np.save(
        OUTPUT_DIR / "identity_available.npy",
        identity_available,
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
            "word_vectorizer": word_vectorizer,
            "character_vectorizer": char_vectorizer,
            "fields": ["name_text", "username_text"],
            "fit_scope": "official training users only",
        },
        OUTPUT_DIR / "identity_vectorizers.joblib",
        compress=3,
    )

    availability_rows = []
    for split_name, split_id in SPLIT_TO_ID.items():
        split_mask = split_ids == split_id
        for label_id in [0, 1]:
            mask = split_mask & (labels == label_id)
            availability_rows.append(
                {
                    "split": split_name,
                    "label": (
                        "bot" if label_id == 1 else "human"
                    ),
                    "users": int(mask.sum()),
                    "identity_available": int(
                        (mask & identity_available).sum()
                    ),
                    "identity_missing": int(
                        (mask & ~identity_available).sum()
                    ),
                    "availability_rate": float(
                        identity_available[mask].mean()
                    ),
                }
            )

    pd.DataFrame(availability_rows).to_csv(
        OUTPUT_DIR / "identity_availability.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "users": int(len(users)),
        "fields": ["name_text", "username_text"],
        "purpose": (
            "Separately quantify predictive identity/handle cues "
            "without contaminating the primary semantic-text modality"
        ),
        "fit_scope": "official training users only",
        "word_features": int(
            len(word_vectorizer.vocabulary_)
        ),
        "character_features": int(
            len(char_vectorizer.vocabulary_)
        ),
        "total_features": int(matrix.shape[1]),
        "nonzero_values": int(matrix.nnz),
        "identity_available_users": int(
            identity_available.sum()
        ),
        "parameters": {
            "word_ngram_range": [1, 2],
            "character_ngram_range": [2, 5],
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "sublinear_tf": True,
            "normalization": "l2",
        },
        "integrity": {
            "validation_and_test_excluded_from_fit": True,
            "row_order_matches_node_id": True,
            "semantic_text_fields_excluded": True,
        },
        "input_sha256": sha256_file(INPUT_PATH),
        "output_sha256": sha256_file(
            OUTPUT_DIR / "identity_tfidf_features.npz"
        ),
    }

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("IDENTITY TF-IDF BUILD: PASS")
    print(f"Users: {len(users):,}")
    print(
        f"Identity available: {identity_available.sum():,} "
        f"({identity_available.mean():.4%})"
    )
    print(
        f"Features: {matrix.shape[1]:,}; "
        f"nonzero values: {matrix.nnz:,}"
    )
    print("Vocabulary fit: official training users only")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()