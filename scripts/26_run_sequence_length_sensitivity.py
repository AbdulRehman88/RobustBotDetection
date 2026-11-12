from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "25_sequence_length_sensitivity"
TOKEN_DIR_128 = OUTPUT_DIR / "tokens_128"
EMBEDDING_DIR_128 = OUTPUT_DIR / "frozen_bertweet_128"
BASELINE_DIR_128 = OUTPUT_DIR / "bertweet_mean_metadata_128"
PREDICTION_DIR_128 = BASELINE_DIR_128 / "test_predictions"
CHECKPOINT_DIR_128 = BASELINE_DIR_128 / "checkpoints"

ORIGINAL_TOKEN_DIR = PROJECT_ROOT / "outputs" / "07_bertweet_tokens"
ORIGINAL_BASELINE_DIR = PROJECT_ROOT / "outputs" / "09_bertweet_baselines"
METADATA_DIR = PROJECT_ROOT / "outputs" / "04_metadata_features"

SEEDS = [42, 123, 2024, 3407, 7777]
MAX_LENGTH = 128
ENCODING_BATCH_SIZE = 128
TRAIN_ID = 0
VAL_ID = 1
TEST_ID = 2


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import {path}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def manifest_has(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        current = payload
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        if current != value:
            return False
    return True


def tokenize_128() -> None:
    required = [
        TOKEN_DIR_128 / "input_ids.npy",
        TOKEN_DIR_128 / "attention_mask.npy",
        TOKEN_DIR_128 / "node_ids.npy",
        TOKEN_DIR_128 / "labels.npy",
        TOKEN_DIR_128 / "split_ids.npy",
        TOKEN_DIR_128 / "text_available.npy",
        TOKEN_DIR_128 / "token_lengths.npy",
    ]
    manifest_path = TOKEN_DIR_128 / "bertweet_token_manifest.json"
    if all(path.exists() for path in required) and manifest_has(
        manifest_path, {"status": "PASS", "max_length": MAX_LENGTH}
    ):
        print("128-token artifacts already complete; tokenization skipped.")
        return

    module = load_module(
        "bertweet_tokenizer_128",
        SCRIPTS_DIR / "08_tokenize_bertweet.py",
    )
    module.OUTPUT_DIR = TOKEN_DIR_128
    module.MAX_LENGTH = MAX_LENGTH
    print("\nSTAGE 1/3: Tokenizing at maximum length 128...")
    module.main()


def encode_128() -> None:
    required = [
        EMBEDDING_DIR_128 / "mean_pooled_embeddings.npy",
        EMBEDDING_DIR_128 / "node_ids.npy",
        EMBEDDING_DIR_128 / "labels.npy",
        EMBEDDING_DIR_128 / "split_ids.npy",
        EMBEDDING_DIR_128 / "text_available.npy",
    ]
    manifest_path = EMBEDDING_DIR_128 / "frozen_bertweet_manifest.json"
    if all(path.exists() for path in required) and manifest_has(
        manifest_path,
        {"status": "PASS", "data.max_length": MAX_LENGTH},
    ):
        print("128-token embeddings already complete; encoding skipped.")
        return

    module = load_module(
        "bertweet_encoder_128",
        SCRIPTS_DIR / "09_encode_frozen_bertweet.py",
    )
    module.TOKEN_DIR = TOKEN_DIR_128
    module.OUTPUT_DIR = EMBEDDING_DIR_128
    module.BATCH_SIZE = ENCODING_BATCH_SIZE
    print("\nSTAGE 2/3: Encoding frozen BERTweet at length 128...")
    module.main()


def locked_hyperparameters() -> tuple[float, float]:
    results = pd.read_csv(
        ORIGINAL_BASELINE_DIR / "bertweet_baseline_seed_results.csv"
    )
    selected = results.loc[
        results["model"].eq("bertweet_mean_metadata")
    ]
    if selected["seed"].nunique() != len(SEEDS):
        raise RuntimeError("Incomplete 64-token baseline seed results.")
    learning_rates = selected["learning_rate"].unique()
    dropouts = selected["dropout"].unique()
    if len(learning_rates) != 1 or len(dropouts) != 1:
        raise RuntimeError(
            "The original final runs do not share one locked configuration."
        )
    return float(learning_rates[0]), float(dropouts[0])


def train_128() -> pd.DataFrame:
    result_path = BASELINE_DIR_128 / "seed_results.csv"
    if result_path.exists():
        existing = pd.read_csv(result_path)
        if set(existing["seed"].astype(int)) == set(SEEDS):
            prediction_files = [
                PREDICTION_DIR_128
                / f"bertweet_mean_metadata_len128_seed_{seed}_test_predictions.parquet"
                for seed in SEEDS
            ]
            if all(path.exists() for path in prediction_files):
                print("128-token five-seed training already complete; skipped.")
                return existing

    module = load_module(
        "bertweet_baseline_128",
        SCRIPTS_DIR / "10_run_bertweet_baselines.py",
    )
    BASELINE_DIR_128.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR_128.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR_128.mkdir(parents=True, exist_ok=True)
    module.PREDICTION_DIR = PREDICTION_DIR_128
    module.CHECKPOINT_DIR = CHECKPOINT_DIR_128

    device = torch.device("cuda")
    node_ids = np.load(
        EMBEDDING_DIR_128 / "node_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        EMBEDDING_DIR_128 / "labels.npy", mmap_mode="r"
    ).astype(np.int8)
    split_ids = np.load(
        EMBEDDING_DIR_128 / "split_ids.npy", mmap_mode="r"
    )
    availability = np.load(
        EMBEDDING_DIR_128 / "text_available.npy", mmap_mode="r"
    ).astype(bool)
    embeddings = np.load(
        EMBEDDING_DIR_128 / "mean_pooled_embeddings.npy", mmap_mode="r"
    )
    metadata = np.load(
        METADATA_DIR / "metadata_features.npy", mmap_mode="r"
    )
    train_indices = np.flatnonzero(split_ids == TRAIN_ID)
    val_indices = np.flatnonzero(split_ids == VAL_ID)
    test_indices = np.flatnonzero(split_ids == TEST_ID)
    learning_rate, dropout = locked_hyperparameters()

    print(
        "\nSTAGE 3/3: Training the locked mean+metadata head "
        f"(lr={learning_rate:g}, dropout={dropout:.1f})..."
    )
    rows = []
    for seed in SEEDS:
        rows.append(
            module.run_configuration(
                "bertweet_mean_metadata_len128",
                embeddings,
                metadata,
                availability,
                labels,
                train_indices,
                val_indices,
                test_indices,
                node_ids,
                True,
                learning_rate,
                dropout,
                seed,
                device,
                save_test=True,
            )
        )
    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    return results


def paired_length_inference(results_128: pd.DataFrame) -> pd.DataFrame:
    inference = load_module(
        "statistical_inference_length",
        SCRIPTS_DIR / "25_run_statistical_inference.py",
    )
    rng = np.random.default_rng(261_128)
    seed_data = []
    observed = {"macro_f1": [], "balanced_accuracy": []}

    for seed in SEEDS:
        frame_128 = inference.load_predictions(
            PREDICTION_DIR_128
            / f"bertweet_mean_metadata_len128_seed_{seed}_test_predictions.parquet"
        )
        frame_64 = inference.load_predictions(
            ORIGINAL_BASELINE_DIR
            / "test_predictions"
            / f"bertweet_mean_metadata_seed_{seed}_test_predictions.parquet"
        )
        if not np.array_equal(
            frame_128["node_id"].to_numpy(dtype=np.int64),
            frame_64["node_id"].to_numpy(dtype=np.int64),
        ):
            raise RuntimeError(f"64/128 node mismatch for seed {seed}.")
        labels = frame_128["label"].to_numpy(dtype=np.int8)
        if not np.array_equal(
            labels, frame_64["label"].to_numpy(dtype=np.int8)
        ):
            raise RuntimeError(f"64/128 label mismatch for seed {seed}.")
        prediction_128 = frame_128["prediction"].to_numpy(dtype=np.int8)
        prediction_64 = frame_64["prediction"].to_numpy(dtype=np.int8)
        human_p, bot_p, human_n, bot_n = inference.joint_probabilities(
            labels, prediction_128, prediction_64
        )
        seed_data.append(
            {
                "human_probabilities": human_p,
                "bot_probabilities": bot_p,
                "human_total": human_n,
                "bot_total": bot_n,
            }
        )
        for metric in observed:
            observed[metric].append(
                inference.metric_value(labels, prediction_128, metric)
                - inference.metric_value(labels, prediction_64, metric)
            )

    rows = []
    for metric in ["macro_f1", "balanced_accuracy"]:
        bootstrap = inference.paired_hierarchical_bootstrap(
            seed_data, metric, rng
        )
        differences = np.asarray(observed[metric], dtype=np.float64)
        rows.append(
            {
                "comparison": "length_128_minus_length_64",
                "metric": metric,
                "seeds": len(SEEDS),
                "test_users": int(
                    seed_data[0]["human_total"] + seed_data[0]["bot_total"]
                ),
                "mean_difference": float(differences.mean()),
                "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
                "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
                "two_sided_bootstrap_p": inference.two_sided_bootstrap_p(
                    bootstrap
                ),
                "probability_128_superior": float(np.mean(bootstrap > 0)),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    results_128: pd.DataFrame,
    inference: pd.DataFrame,
) -> pd.DataFrame:
    results_64 = pd.read_csv(
        ORIGINAL_BASELINE_DIR / "bertweet_baseline_seed_results.csv"
    )
    results_64 = results_64.loc[
        results_64["model"].eq("bertweet_mean_metadata")
    ]
    metrics = ["macro_f1", "balanced_accuracy", "auroc", "auprc"]
    rows = []
    for length, frame in [(64, results_64), (128, results_128)]:
        row = {"max_length": length, "seeds": int(frame["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenize_128()
    encode_128()
    results_128 = train_128()
    inference = paired_length_inference(results_128)
    inference.to_csv(OUTPUT_DIR / "paired_64_vs_128_inference.csv", index=False)
    summary = build_summary(results_128, inference)
    summary.to_csv(OUTPUT_DIR / "sequence_length_summary.csv", index=False)

    length_64 = pd.read_csv(ORIGINAL_TOKEN_DIR / "token_length_summary.csv")
    length_128 = pd.read_csv(TOKEN_DIR_128 / "token_length_summary.csv")
    saturation = pd.concat(
        [
            length_64.assign(max_length=64),
            length_128.assign(max_length=128),
        ],
        ignore_index=True,
    )
    saturation.to_csv(OUTPUT_DIR / "token_saturation_comparison.csv", index=False)

    learning_rate, dropout = locked_hyperparameters()
    manifest = {
        "status": "PASS",
        "comparison": "maximum sequence length 64 versus 128",
        "primary_model": "frozen BERTweet semantic mean pooling plus metadata",
        "controlled_components": [
            "semantic text fields",
            "tokenizer and normalization",
            "frozen encoder",
            "semantic mean pooling",
            "metadata features",
            "classifier architecture",
            "official splits",
            "five random seeds",
        ],
        "locked_from_64_validation": {
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
        "test_policy": (
            "128-token configuration and inference protocol fixed before test evaluation; "
            "no 128-token hyperparameter search"
        ),
        "paired_inference": (
            "10,000-replicate two-level class-stratified paired bootstrap"
        ),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nSEQUENCE-LENGTH SENSITIVITY: PASS")
    print("\nPerformance:")
    print(summary.to_string(index=False))
    print("\nPaired 128-minus-64 inference:")
    print(inference.to_string(index=False))
    print("\nToken saturation:")
    print(
        saturation[
            [
                "max_length",
                "split",
                "available_text_users",
                "max_length_saturation_count",
                "max_length_saturation_rate",
            ]
        ].to_string(index=False)
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
