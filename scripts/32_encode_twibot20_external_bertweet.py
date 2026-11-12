from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_ROOT = PROJECT_ROOT / "outputs" / "30_twibot20_external_tokens"
SOURCE_TOKENIZER_ROOT = PROJECT_ROOT / "outputs" / "07_bertweet_tokens" / "tokenizer"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "31_twibot20_external_bertweet"

MODEL_NAME = "vinai/bertweet-base"
BATCH_SIZE = 256
SEED = 42
OUTPUT_DTYPE = np.float16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    total_started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for locked BERTweet encoding.")

    required = [
        "input_ids.npy", "attention_mask.npy", "row_ids.npy", "external_ids.npy",
        "labels.npy", "split_ids.npy", "text_available.npy",
    ]
    missing = [name for name in required if not (TOKEN_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run Script 31 first; missing: {missing}")

    input_ids = np.load(TOKEN_ROOT / "input_ids.npy", mmap_mode="r")
    attention_mask = np.load(TOKEN_ROOT / "attention_mask.npy", mmap_mode="r")
    row_ids = np.load(TOKEN_ROOT / "row_ids.npy", mmap_mode="r")
    external_ids = np.load(TOKEN_ROOT / "external_ids.npy", mmap_mode="r")
    labels = np.load(TOKEN_ROOT / "labels.npy", mmap_mode="r")
    split_ids = np.load(TOKEN_ROOT / "split_ids.npy", mmap_mode="r")
    text_available = np.load(TOKEN_ROOT / "text_available.npy", mmap_mode="r").astype(bool)

    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("External token IDs and attention masks differ in shape.")
    if input_ids.shape[0] != len(row_ids):
        raise RuntimeError("External token rows do not match row IDs.")
    if not np.array_equal(row_ids, np.arange(len(row_ids), dtype=np.int64)):
        raise RuntimeError("External row IDs are not contiguous and zero based.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    print(f"Loading locked encoder: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_TOKENIZER_ROOT, use_fast=False, normalization=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device=device, dtype=torch.float16)

    n_users, max_length = input_ids.shape
    hidden_size = int(model.config.hidden_size)
    mean_path = OUTPUT_ROOT / "mean_pooled_embeddings.npy"
    cls_path = OUTPUT_ROOT / "cls_embeddings.npy"
    mean_embeddings = np.lib.format.open_memmap(
        mean_path, mode="w+", dtype=OUTPUT_DTYPE, shape=(n_users, hidden_size)
    )
    cls_embeddings = np.lib.format.open_memmap(
        cls_path, mode="w+", dtype=OUTPUT_DTYPE, shape=(n_users, hidden_size)
    )
    mean_embeddings[:] = 0
    cls_embeddings[:] = 0

    available_indices = np.flatnonzero(text_available)
    missing_indices = np.flatnonzero(~text_available)
    special_token_ids = {
        token_id
        for token_id in [
            tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
            tokenizer.bos_token_id, tokenizer.eos_token_id,
        ]
        if token_id is not None
    }

    encoding_started = time.time()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(available_indices), BATCH_SIZE),
            desc="Encoding TwiBot-20 BERTweet profiles",
            unit="batch",
        ):
            indices = available_indices[start:start + BATCH_SIZE]
            ids = torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(
                device, non_blocking=True
            )
            masks = torch.from_numpy(np.asarray(attention_mask[indices], dtype=np.int64)).to(
                device, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = model(
                    input_ids=ids, attention_mask=masks, return_dict=True
                ).last_hidden_state
                semantic_mask = masks.bool()
                for token_id in special_token_ids:
                    semantic_mask &= ids.ne(token_id)
                counts = semantic_mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean_pooled = (
                    hidden * semantic_mask.unsqueeze(-1).to(hidden.dtype)
                ).sum(dim=1) / counts.to(hidden.dtype)
                cls_pooled = hidden[:, 0, :]

            mean_embeddings[indices] = mean_pooled.float().cpu().numpy().astype(
                OUTPUT_DTYPE, copy=False
            )
            cls_embeddings[indices] = cls_pooled.float().cpu().numpy().astype(
                OUTPUT_DTYPE, copy=False
            )

    mean_embeddings.flush()
    cls_embeddings.flush()
    del mean_embeddings, cls_embeddings

    mean_check = np.load(mean_path, mmap_mode="r")
    cls_check = np.load(cls_path, mmap_mode="r")
    if mean_check.shape != (n_users, hidden_size) or cls_check.shape != mean_check.shape:
        raise RuntimeError("External embedding shapes are invalid.")
    if not np.isfinite(np.asarray(mean_check[available_indices], dtype=np.float32)).all():
        raise RuntimeError("Non-finite external mean embeddings detected.")
    if not np.isfinite(np.asarray(cls_check[available_indices], dtype=np.float32)).all():
        raise RuntimeError("Non-finite external CLS embeddings detected.")
    if missing_indices.size and (
        np.any(mean_check[missing_indices] != 0) or np.any(cls_check[missing_indices] != 0)
    ):
        raise RuntimeError("Missing-text external profiles are not exactly zero encoded.")

    np.save(OUTPUT_ROOT / "row_ids.npy", np.asarray(row_ids, dtype=np.int64))
    np.save(OUTPUT_ROOT / "external_ids.npy", np.asarray(external_ids, dtype=str))
    np.save(OUTPUT_ROOT / "labels.npy", np.asarray(labels, dtype=np.int8))
    np.save(OUTPUT_ROOT / "split_ids.npy", np.asarray(split_ids, dtype=np.int8))
    np.save(OUTPUT_ROOT / "text_available.npy", text_available)

    output_files = [
        "mean_pooled_embeddings.npy", "cls_embeddings.npy", "row_ids.npy",
        "external_ids.npy", "labels.npy", "split_ids.npy", "text_available.npy",
    ]
    manifest = {
        "status": "PASS",
        "dataset": "TwiBot-20-Format22",
        "role": "locked_external_validation",
        "model_name": MODEL_NAME,
        "encoder_mode": "frozen",
        "pooling": ["semantic_mean_excluding_special_tokens", "CLS"],
        "users": n_users,
        "available_profiles_encoded": int(text_available.sum()),
        "missing_profiles_zero_encoded": int((~text_available).sum()),
        "embedding_dimensions": hidden_size,
        "maximum_sequence_length": max_length,
        "dtype": str(np.dtype(OUTPUT_DTYPE)),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "runtime_minutes": (time.time() - encoding_started) / 60,
        "output_sha256": {name: sha256(OUTPUT_ROOT / name) for name in output_files},
    }
    (OUTPUT_ROOT / "external_bertweet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTWIBOT-20 FROZEN BERTWEET ENCODING: PASS")
    print(f"Users: {n_users:,}")
    print(f"Encoded available profiles: {text_available.sum():,}")
    print(f"Zero-coded missing profiles: {(~text_available).sum():,}")
    print(f"Embedding dimensions: {hidden_size}")
    print(f"Runtime: {(time.time() - total_started) / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
