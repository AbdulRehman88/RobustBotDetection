from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "29_twibot20_external_features"
SOURCE_TOKEN_ROOT = PROJECT_ROOT / "outputs" / "07_bertweet_tokens"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "30_twibot20_external_tokens"

MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 64
BATCH_SIZE = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    users_path = INPUT_ROOT / "users.parquet"
    source_manifest_path = SOURCE_TOKEN_ROOT / "bertweet_token_manifest.json"
    tokenizer_root = SOURCE_TOKEN_ROOT / "tokenizer"
    if not users_path.is_file():
        raise FileNotFoundError("Run Script 30 before Script 31.")
    if not tokenizer_root.is_dir():
        raise FileNotFoundError("Locked TwiBot-22 tokenizer directory is missing.")

    users = pd.read_parquet(
        users_path,
        columns=["row_id", "external_id", "semantic_text", "text_available", "label_id", "split_id"],
    ).sort_values("row_id").reset_index(drop=True)
    expected_rows = np.arange(len(users), dtype=np.int64)
    if not np.array_equal(users["row_id"].to_numpy(dtype=np.int64), expected_rows):
        raise RuntimeError("External row IDs are not contiguous and zero based.")

    semantic_text = (
        users["semantic_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    text_available = users["text_available"].to_numpy(dtype=bool)
    if not np.array_equal(text_available, semantic_text.ne("").to_numpy(dtype=bool)):
        raise RuntimeError("Stored external text mask disagrees with semantic text.")

    print(f"Loading locked tokenizer from: {tokenizer_root}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        use_fast=False,
        normalization=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None or tokenizer.cls_token_id is None:
        raise RuntimeError("Locked tokenizer lacks required special tokens.")

    input_ids_path = OUTPUT_ROOT / "input_ids.npy"
    attention_mask_path = OUTPUT_ROOT / "attention_mask.npy"
    input_ids = np.lib.format.open_memmap(
        input_ids_path, mode="w+", dtype=np.int32, shape=(len(users), MAX_LENGTH)
    )
    attention_mask = np.lib.format.open_memmap(
        attention_mask_path, mode="w+", dtype=np.uint8, shape=(len(users), MAX_LENGTH)
    )

    saturated = np.zeros(len(users), dtype=bool)
    print("Tokenizing TwiBot-20 external profiles...")
    for start in tqdm(range(0, len(users), BATCH_SIZE), unit="batch"):
        stop = min(start + BATCH_SIZE, len(users))
        batch = semantic_text.iloc[start:stop].tolist()
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_tensors="np",
        )
        ids = encoded["input_ids"].astype(np.int32, copy=False)
        masks = encoded["attention_mask"].astype(np.uint8, copy=False)
        input_ids[start:stop] = ids
        attention_mask[start:stop] = masks
        saturated[start:stop] = masks.sum(axis=1) == MAX_LENGTH

    input_ids.flush()
    attention_mask.flush()
    del input_ids, attention_mask

    np.save(OUTPUT_ROOT / "row_ids.npy", users["row_id"].to_numpy(dtype=np.int64))
    np.save(OUTPUT_ROOT / "external_ids.npy", users["external_id"].to_numpy(dtype=str))
    np.save(OUTPUT_ROOT / "labels.npy", users["label_id"].to_numpy(dtype=np.int8))
    np.save(OUTPUT_ROOT / "split_ids.npy", users["split_id"].to_numpy(dtype=np.int8))
    np.save(OUTPUT_ROOT / "text_available.npy", text_available)
    np.save(OUTPUT_ROOT / "saturated_at_max_length.npy", saturated)

    missing_indices = np.flatnonzero(~text_available)
    stored_ids = np.load(input_ids_path, mmap_mode="r")
    stored_masks = np.load(attention_mask_path, mmap_mode="r")
    if stored_ids.shape != (len(users), MAX_LENGTH) or stored_masks.shape != stored_ids.shape:
        raise RuntimeError("External token arrays have invalid shapes.")
    if missing_indices.size and np.any(stored_masks[missing_indices].sum(axis=1) != 2):
        raise RuntimeError("Empty external profiles do not contain exactly the two special tokens.")

    source_manifest = None
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    output_files = [
        "input_ids.npy",
        "attention_mask.npy",
        "row_ids.npy",
        "external_ids.npy",
        "labels.npy",
        "split_ids.npy",
        "text_available.npy",
        "saturated_at_max_length.npy",
    ]
    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "role": "locked_external_validation",
        "model_name": MODEL_NAME,
        "tokenizer_source": str(tokenizer_root),
        "source_tokenization_manifest": source_manifest,
        "users": len(users),
        "semantic_text_available": int(text_available.sum()),
        "maximum_length": MAX_LENGTH,
        "saturated_available_profiles": int((saturated & text_available).sum()),
        "saturation_rate_among_available": float(
            (saturated & text_available).sum() / text_available.sum()
        ),
        "output_sha256": {name: sha256(OUTPUT_ROOT / name) for name in output_files},
        "runtime_minutes": (time.time() - started) / 60,
    }
    (OUTPUT_ROOT / "external_tokenization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 EXTERNAL TOKENIZATION: PASS")
    print(f"Users: {len(users):,}")
    print(f"Semantic text available: {text_available.sum():,} ({text_available.mean():.4%})")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(
        f"Available profiles saturated: {(saturated & text_available).sum():,} "
        f"({(saturated & text_available).sum()/text_available.sum():.4%})"
    )
    print(f"Runtime: {(time.time() - started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
