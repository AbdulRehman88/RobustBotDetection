from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "outputs" / "02_canonical_data" / "users.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "07_bertweet_tokens"

MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 64
TOKENIZATION_BATCH_SIZE = 4096
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
            "description_text",
            "location_text",
        ],
    ).sort_values("node_id").reset_index(drop=True)

    node_ids = users["node_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(
        node_ids, np.arange(len(users), dtype=np.int64)
    ):
        raise RuntimeError("node_id is not contiguous and zero based.")

    normalized_labels = users["label"].astype(str).str.lower()
    if set(normalized_labels.unique()) != {"bot", "human"}:
        raise RuntimeError(
            f"Unexpected labels: {sorted(normalized_labels.unique())}"
        )
    labels = normalized_labels.eq("bot").to_numpy(dtype=np.int8)

    split_ids = (
        users["split"]
        .astype(str)
        .map(SPLIT_TO_ID)
        .to_numpy(dtype=np.int8)
    )
    if np.isnan(split_ids.astype(np.float64)).any():
        raise RuntimeError("Unexpected split value detected.")

    description = clean_text(users["description_text"])
    location = clean_text(users["location_text"])

    # This exactly matches the primary semantic-text definition used by
    # the leakage-safe TF-IDF baseline.
    semantic_text = (
        description.str.cat(location, sep=" ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    text_available = semantic_text.ne("").to_numpy(dtype=bool)

    print(f"Loading locked tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        normalization=True,
    )

    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no padding token.")
    if tokenizer.cls_token_id is None:
        raise RuntimeError("Tokenizer has no start/classification token.")

    tokenizer_dir = OUTPUT_DIR / "tokenizer"
    if tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    input_ids_path = OUTPUT_DIR / "input_ids.npy"
    attention_mask_path = OUTPUT_DIR / "attention_mask.npy"

    input_ids = np.lib.format.open_memmap(
        input_ids_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(users), MAX_LENGTH),
    )
    attention_mask = np.lib.format.open_memmap(
        attention_mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(users), MAX_LENGTH),
    )

    started = time.time()

    for start in tqdm(
        range(0, len(users), TOKENIZATION_BATCH_SIZE),
        desc="Tokenizing BERTweet input",
        unit="batch",
    ):
        stop = min(start + TOKENIZATION_BATCH_SIZE, len(users))
        batch_text = semantic_text.iloc[start:stop].tolist()

        encoded = tokenizer(
            batch_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="np",
        )

        batch_ids = encoded["input_ids"]
        batch_mask = encoded["attention_mask"]

        if batch_ids.shape != (stop - start, MAX_LENGTH):
            raise RuntimeError(
                f"Unexpected token shape at rows {start}:{stop}: "
                f"{batch_ids.shape}"
            )
        if batch_mask.shape != (stop - start, MAX_LENGTH):
            raise RuntimeError(
                f"Unexpected mask shape at rows {start}:{stop}: "
                f"{batch_mask.shape}"
            )

        input_ids[start:stop] = batch_ids.astype(
            np.int32, copy=False
        )
        attention_mask[start:stop] = batch_mask.astype(
            np.uint8, copy=False
        )

    input_ids.flush()
    attention_mask.flush()
    del input_ids
    del attention_mask

    input_ids_check = np.load(input_ids_path, mmap_mode="r")
    attention_check = np.load(attention_mask_path, mmap_mode="r")

    if input_ids_check.shape != (len(users), MAX_LENGTH):
        raise RuntimeError("Saved input_ids shape is invalid.")
    if attention_check.shape != (len(users), MAX_LENGTH):
        raise RuntimeError("Saved attention-mask shape is invalid.")
    if not np.isin(attention_check, [0, 1]).all():
        raise RuntimeError("Attention mask contains values outside {0,1}.")

    token_lengths = np.asarray(
        attention_check.sum(axis=1),
        dtype=np.int16,
    )

    # Empty semantic text should contain only model special tokens.
    missing_lengths = token_lengths[~text_available]
    if missing_lengths.size and np.unique(missing_lengths).size != 1:
        raise RuntimeError(
            "Missing-text rows do not have a consistent token length."
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
    np.save(
        OUTPUT_DIR / "text_available.npy",
        text_available,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "token_lengths.npy",
        token_lengths,
        allow_pickle=False,
    )

    length_summary = []
    for split_name, split_id in SPLIT_TO_ID.items():
        split_mask = split_ids == split_id
        available_mask = split_mask & text_available
        available_lengths = token_lengths[available_mask]

        length_summary.append(
            {
                "split": split_name,
                "users": int(split_mask.sum()),
                "available_text_users": int(available_mask.sum()),
                "missing_text_users": int(
                    (split_mask & ~text_available).sum()
                ),
                "mean_token_length_available": float(
                    available_lengths.mean()
                ),
                "median_token_length_available": float(
                    np.median(available_lengths)
                ),
                "p95_token_length_available": float(
                    np.quantile(available_lengths, 0.95)
                ),
                "maximum_token_length_available": int(
                    available_lengths.max()
                ),
                "max_length_saturation_count": int(
                    (available_lengths == MAX_LENGTH).sum()
                ),
                "max_length_saturation_rate": float(
                    (available_lengths == MAX_LENGTH).mean()
                ),
            }
        )

    pd.DataFrame(length_summary).to_csv(
        OUTPUT_DIR / "token_length_summary.csv",
        index=False,
    )

    tokenizer_commit = getattr(
        tokenizer,
        "_commit_hash",
        None,
    )
    if tokenizer_commit is None:
        tokenizer_commit = tokenizer.init_kwargs.get(
            "_commit_hash"
        )

    manifest = {
        "status": "PASS",
        "model_name": MODEL_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_commit": tokenizer_commit,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "max_length": MAX_LENGTH,
        "padding": "max_length",
        "truncation": True,
        "normalization": True,
        "batch_size": TOKENIZATION_BATCH_SIZE,
        "users": int(len(users)),
        "semantic_text_available": int(text_available.sum()),
        "semantic_text_missing": int((~text_available).sum()),
        "primary_text_fields": [
            "description_text",
            "location_text",
        ],
        "excluded_identity_fields": [
            "name_text",
            "username_text",
        ],
        "label_encoding": {
            "human": 0,
            "bot": 1,
        },
        "split_encoding": SPLIT_TO_ID,
        "integrity": {
            "row_order_matches_node_id": True,
            "all_rows_fixed_length": True,
            "attention_mask_binary": True,
            "same_text_definition_as_tfidf": True,
        },
        "runtime_seconds": time.time() - started,
        "input_sha256": sha256_file(INPUT_PATH),
        "output_sha256": {
            "input_ids.npy": sha256_file(input_ids_path),
            "attention_mask.npy": sha256_file(
                attention_mask_path
            ),
            "node_ids.npy": sha256_file(
                OUTPUT_DIR / "node_ids.npy"
            ),
            "labels.npy": sha256_file(
                OUTPUT_DIR / "labels.npy"
            ),
            "split_ids.npy": sha256_file(
                OUTPUT_DIR / "split_ids.npy"
            ),
            "text_available.npy": sha256_file(
                OUTPUT_DIR / "text_available.npy"
            ),
            "token_lengths.npy": sha256_file(
                OUTPUT_DIR / "token_lengths.npy"
            ),
        },
    }

    with (OUTPUT_DIR / "bertweet_token_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("BERTWEET TOKENIZATION: PASS")
    print(f"Users: {len(users):,}")
    print(
        f"Semantic text available: {text_available.sum():,} "
        f"({text_available.mean():.4%})"
    )
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(
        f"Rows saturated at maximum length: "
        f"{(token_lengths[text_available] == MAX_LENGTH).sum():,}"
    )
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()