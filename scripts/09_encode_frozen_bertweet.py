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
TOKEN_DIR = PROJECT_ROOT / "outputs" / "07_bertweet_tokens"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "08_frozen_bertweet"

MODEL_NAME = "vinai/bertweet-base"
BATCH_SIZE = 256
SEED = 42
OUTPUT_DTYPE = np.float16
CHECK_CHUNK_SIZE = 50_000


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_finite_in_chunks(
    matrix: np.ndarray,
    row_indices: np.ndarray,
) -> bool:
    for start in range(0, len(row_indices), CHECK_CHUNK_SIZE):
        selected = row_indices[start : start + CHECK_CHUNK_SIZE]
        if not np.isfinite(matrix[selected]).all():
            return False
    return True


def check_zero_in_chunks(
    matrix: np.ndarray,
    row_indices: np.ndarray,
) -> bool:
    for start in range(0, len(row_indices), CHECK_CHUNK_SIZE):
        selected = row_indices[start : start + CHECK_CHUNK_SIZE]
        if np.any(matrix[selected] != 0):
            return False
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for BERTweet encoding.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    input_ids = np.load(
        TOKEN_DIR / "input_ids.npy",
        mmap_mode="r",
    )
    attention_mask = np.load(
        TOKEN_DIR / "attention_mask.npy",
        mmap_mode="r",
    )
    node_ids = np.load(
        TOKEN_DIR / "node_ids.npy",
        mmap_mode="r",
    )
    labels = np.load(
        TOKEN_DIR / "labels.npy",
        mmap_mode="r",
    )
    split_ids = np.load(
        TOKEN_DIR / "split_ids.npy",
        mmap_mode="r",
    )
    text_available = np.load(
        TOKEN_DIR / "text_available.npy",
        mmap_mode="r",
    ).astype(bool)

    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("Token IDs and attention masks differ in shape.")
    if input_ids.shape[0] != len(node_ids):
        raise RuntimeError("Token rows do not match node IDs.")
    if not np.array_equal(
        node_ids,
        np.arange(len(node_ids), dtype=np.int64),
    ):
        raise RuntimeError("Node IDs are not contiguous and zero based.")

    print(f"Loading locked encoder: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKEN_DIR / "tokenizer",
        use_fast=False,
        normalization=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device=device, dtype=torch.float16)

    hidden_size = int(model.config.hidden_size)
    n_users, max_length = input_ids.shape

    mean_path = OUTPUT_DIR / "mean_pooled_embeddings.npy"
    cls_path = OUTPUT_DIR / "cls_embeddings.npy"

    mean_embeddings = np.lib.format.open_memmap(
        mean_path,
        mode="w+",
        dtype=OUTPUT_DTYPE,
        shape=(n_users, hidden_size),
    )
    cls_embeddings = np.lib.format.open_memmap(
        cls_path,
        mode="w+",
        dtype=OUTPUT_DTYPE,
        shape=(n_users, hidden_size),
    )
    mean_embeddings[:] = 0
    cls_embeddings[:] = 0

    available_indices = np.flatnonzero(text_available)
    missing_indices = np.flatnonzero(~text_available)

    special_token_ids = {
        token_id
        for token_id in [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
        if token_id is not None
    }

    started = time.time()

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(available_indices), BATCH_SIZE),
            desc="Encoding BERTweet profiles",
            unit="batch",
        ):
            batch_indices = available_indices[
                start : start + BATCH_SIZE
            ]

            batch_ids_np = np.asarray(
                input_ids[batch_indices],
                dtype=np.int64,
            )
            batch_attention_np = np.asarray(
                attention_mask[batch_indices],
                dtype=np.int64,
            )

            batch_ids = torch.from_numpy(batch_ids_np).to(
                device,
                non_blocking=True,
            )
            batch_attention = torch.from_numpy(
                batch_attention_np
            ).to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(
                    input_ids=batch_ids,
                    attention_mask=batch_attention,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state

                semantic_mask = batch_attention.bool()
                for token_id in special_token_ids:
                    semantic_mask &= batch_ids.ne(token_id)

                # Defensive fallback for any row containing only special
                # tokens, although unavailable rows are not encoded.
                semantic_counts = semantic_mask.sum(
                    dim=1,
                    keepdim=True,
                ).clamp(min=1)

                mean_pooled = (
                    hidden
                    * semantic_mask.unsqueeze(-1).to(hidden.dtype)
                ).sum(dim=1) / semantic_counts.to(hidden.dtype)

                cls_pooled = hidden[:, 0, :]

            mean_embeddings[batch_indices] = (
                mean_pooled.float().cpu().numpy().astype(
                    OUTPUT_DTYPE,
                    copy=False,
                )
            )
            cls_embeddings[batch_indices] = (
                cls_pooled.float().cpu().numpy().astype(
                    OUTPUT_DTYPE,
                    copy=False,
                )
            )

    mean_embeddings.flush()
    cls_embeddings.flush()
    del mean_embeddings
    del cls_embeddings

    mean_check = np.load(mean_path, mmap_mode="r")
    cls_check = np.load(cls_path, mmap_mode="r")

    if mean_check.shape != (n_users, hidden_size):
        raise RuntimeError("Mean-pooled embedding shape is invalid.")
    if cls_check.shape != (n_users, hidden_size):
        raise RuntimeError("CLS embedding shape is invalid.")

    if not check_finite_in_chunks(mean_check, available_indices):
        raise RuntimeError(
            "Non-finite mean-pooled embeddings detected."
        )
    if not check_finite_in_chunks(cls_check, available_indices):
        raise RuntimeError("Non-finite CLS embeddings detected.")

    if not check_zero_in_chunks(mean_check, missing_indices):
        raise RuntimeError(
            "Missing-text mean embeddings are not exactly zero."
        )
    if not check_zero_in_chunks(cls_check, missing_indices):
        raise RuntimeError(
            "Missing-text CLS embeddings are not exactly zero."
        )

    sample_indices = available_indices[
        np.linspace(
            0,
            len(available_indices) - 1,
            min(10_000, len(available_indices)),
            dtype=np.int64,
        )
    ]
    mean_sample = np.asarray(
        mean_check[sample_indices],
        dtype=np.float32,
    )
    cls_sample = np.asarray(
        cls_check[sample_indices],
        dtype=np.float32,
    )

    model_commit = getattr(model.config, "_commit_hash", None)
    tokenizer_commit = getattr(
        tokenizer,
        "_commit_hash",
        None,
    )
    if tokenizer_commit is None:
        tokenizer_commit = tokenizer.init_kwargs.get(
            "_commit_hash"
        )

    runtime_seconds = time.time() - started

    manifest = {
        "status": "PASS",
        "encoder": {
            "model_name": MODEL_NAME,
            "model_class": model.__class__.__name__,
            "model_commit": model_commit,
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_commit": tokenizer_commit,
            "hidden_size": hidden_size,
            "fine_tuned": False,
            "encoder_mode": "frozen inference",
        },
        "pooling": {
            "mean": (
                "Mean of non-padding, non-special-token hidden states"
            ),
            "cls": "Final hidden state at position zero",
        },
        "data": {
            "users": int(n_users),
            "available_text_users": int(
                len(available_indices)
            ),
            "missing_text_users": int(len(missing_indices)),
            "max_length": int(max_length),
            "missing_text_encoding": (
                "exact all-zero embedding"
            ),
        },
        "execution": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "model_dtype": "float16",
            "output_dtype": "float16",
            "device": str(device),
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "tf32_enabled": False,
            "runtime_seconds": runtime_seconds,
        },
        "sample_statistics": {
            "mean_pool_global_mean": float(mean_sample.mean()),
            "mean_pool_global_std": float(mean_sample.std()),
            "mean_pool_l2_norm_mean": float(
                np.linalg.norm(mean_sample, axis=1).mean()
            ),
            "cls_global_mean": float(cls_sample.mean()),
            "cls_global_std": float(cls_sample.std()),
            "cls_l2_norm_mean": float(
                np.linalg.norm(cls_sample, axis=1).mean()
            ),
        },
        "integrity": {
            "row_order_matches_node_id": True,
            "available_embeddings_finite": True,
            "missing_embeddings_exactly_zero": True,
            "labels_not_used": True,
            "validation_labels_not_used": True,
            "test_labels_not_used": True,
        },
        "input_hashes": {
            "input_ids.npy": sha256_file(
                TOKEN_DIR / "input_ids.npy"
            ),
            "attention_mask.npy": sha256_file(
                TOKEN_DIR / "attention_mask.npy"
            ),
            "text_available.npy": sha256_file(
                TOKEN_DIR / "text_available.npy"
            ),
        },
        "output_hashes": {
            "mean_pooled_embeddings.npy": sha256_file(mean_path),
            "cls_embeddings.npy": sha256_file(cls_path),
        },
    }

    with (OUTPUT_DIR / "frozen_bertweet_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    np.save(
        OUTPUT_DIR / "node_ids.npy",
        np.asarray(node_ids),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "labels.npy",
        np.asarray(labels),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "split_ids.npy",
        np.asarray(split_ids),
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "text_available.npy",
        text_available,
        allow_pickle=False,
    )

    print("FROZEN BERTWEET ENCODING: PASS")
    print(f"Users: {n_users:,}")
    print(f"Encoded available profiles: {len(available_indices):,}")
    print(f"Zero-coded missing profiles: {len(missing_indices):,}")
    print(f"Embedding dimensions: {hidden_size}")
    print("Pooling outputs: semantic mean and CLS")
    print(f"Runtime: {runtime_seconds / 60:.2f} minutes")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()