from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "40_osnem_sota_comparison"
EVIDENCE_DIR = PROJECT_ROOT / "evidence" / "osnem_sota_comparison"
CONFIG_PATH = PROJECT_ROOT / "configs" / "osnem_sota_comparison.yaml"

CONTROLLED_DIR = OUTPUT_ROOT / "controlled_comparison"
FORCED_DIR = OUTPUT_ROOT / "forced_availability_comparison"
MANIFEST_PATHS = {
    "input_contract": OUTPUT_ROOT / "input_contract" / "manifest.json",
    "botbuster": OUTPUT_ROOT / "botbuster_common_evidence" / "manifest.json",
    "botmoe": OUTPUT_ROOT / "botmoe_common_evidence" / "manifest.json",
    "controlled_inference": CONTROLLED_DIR / "manifest.json",
    "botmoe_forced": OUTPUT_ROOT / "botmoe_forced_availability" / "manifest.json",
    "forced_inference": FORCED_DIR / "manifest.json",
}

PRIMARY = "proposed_reliability_route_natural_tgm"
ROUTER = "validation_selected_reliability_router"
BOTBUSTER = "botbuster_style_common_evidence"
BOTMOE = "botmoe_style_common_evidence"
SCENARIOS = ["T", "G", "M", "TG", "TM", "GM", "TGM"]
ALIGNED_SCENARIOS = ["T", "M", "TM"]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(
    path: Path, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    with require(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "PASS":
        raise RuntimeError(
            f"Manifest is not PASS: {path}"
        )
    return manifest


def summarize_scope(
    seed_results: pd.DataFrame,
    scenarios: list[str],
    scope: str,
) -> pd.DataFrame:
    selected = seed_results.loc[
        seed_results["scenario"].isin(scenarios)
    ].copy()
    seed_means = (
        selected.groupby(["model", "seed"], as_index=False)[
            ["macro_f1", "balanced_accuracy"]
        ]
        .mean()
    )
    rows = []
    for model, frame in seed_means.groupby(
        "model", sort=False
    ):
        rows.append(
            {
                "scope": scope,
                "model": model,
                "scenarios": ",".join(scenarios),
                "scenario_count": len(scenarios),
                "seeds": int(frame["seed"].nunique()),
                "macro_f1_mean": float(
                    frame["macro_f1"].mean()
                ),
                "macro_f1_std": float(
                    frame["macro_f1"].std(ddof=1)
                ),
                "balanced_accuracy_mean": float(
                    frame["balanced_accuracy"].mean()
                ),
                "balanced_accuracy_std": float(
                    frame["balanced_accuracy"].std(ddof=1)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {
        name: read_manifest(path)
        for name, path in MANIFEST_PATHS.items()
    }

    with require(CONFIG_PATH).open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    source_implementations = config[
        "source_implementations"
    ]

    natural_summary = pd.read_csv(
        require(
            CONTROLLED_DIR
            / "controlled_comparison_summary.csv"
        )
    )
    natural_inference = pd.read_csv(
        require(
            CONTROLLED_DIR
            / "controlled_paired_inference.csv"
        )
    )
    forced_seed = pd.read_csv(
        require(
            FORCED_DIR
            / "forced_comparison_seed_results.csv"
        )
    )
    forced_summary = pd.read_csv(
        require(
            FORCED_DIR
            / "forced_comparison_summary.csv"
        )
    )
    scenario_inference = pd.read_csv(
        require(
            FORCED_DIR
            / "forced_scenario_paired_inference.csv"
        )
    )
    robust_inference = pd.read_csv(
        require(
            FORCED_DIR
            / "robust_mean_paired_inference.csv"
        )
    )
    graph_audit = pd.read_csv(
        require(FORCED_DIR / "graph_availability_audit.csv")
    )

    expected_natural_models = {
        PRIMARY, BOTBUSTER, BOTMOE
    }
    if set(natural_summary["model"]) != expected_natural_models:
        raise RuntimeError(
            "Natural-comparison model registry differs."
        )
    if set(forced_seed["model"]) != {ROUTER, BOTMOE}:
        raise RuntimeError(
            "Forced-comparison model registry differs."
        )
    if set(forced_seed["scenario"]) != set(SCENARIOS):
        raise RuntimeError(
            "Forced-comparison scenario registry differs."
        )
    if not (
        forced_seed.groupby(["model", "scenario"])[
            "seed"
        ].nunique()
        == 5
    ).all():
        raise RuntimeError(
            "Forced comparison does not contain five seeds."
        )

    graph_enabled = graph_audit.loc[
        graph_audit["scenario"].isin(
            ["G", "TG", "GM", "TGM"]
        )
    ]
    if (
        graph_enabled["graph_mask_mismatches"] <= 0
    ).any():
        raise RuntimeError(
            "Expected native graph-mask divergence was not audited."
        )
    non_graph = graph_audit.loc[
        graph_audit["scenario"].isin(
            ALIGNED_SCENARIOS
        )
    ]
    if (
        non_graph["graph_mask_mismatches"] != 0
    ).any():
        raise RuntimeError(
            "Non-graph aligned sensitivity has mask divergence."
        )

    natural_table = natural_summary[
        [
            "model",
            "seeds",
            "test_users",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "f1_bot_mean",
            "f1_bot_std",
            "macro_f1_mean",
            "macro_f1_std",
            "auroc_mean",
            "auroc_std",
            "auprc_mean",
            "auprc_std",
        ]
    ].copy()
    natural_table.to_csv(
        EVIDENCE_DIR / "natural_comparison.csv",
        index=False,
    )

    scenario_table = forced_summary[
        [
            "model",
            "scenario",
            "seeds",
            "test_users",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
        ]
    ].copy()
    scenario_table["scenario"] = pd.Categorical(
        scenario_table["scenario"],
        categories=SCENARIOS,
        ordered=True,
    )
    scenario_table = scenario_table.sort_values(
        ["scenario", "model"]
    )
    scenario_table.to_csv(
        EVIDENCE_DIR / "forced_scenario_comparison.csv",
        index=False,
    )

    robust_summary = pd.concat(
        [
            summarize_scope(
                forced_seed,
                SCENARIOS,
                "native_evidence_mean_across_7_scenarios",
            ),
            summarize_scope(
                forced_seed,
                ALIGNED_SCENARIOS,
                "aligned_non_graph_mean_across_T_M_TM",
            ),
        ],
        ignore_index=True,
    )
    robust_summary.to_csv(
        EVIDENCE_DIR / "robust_mean_comparison.csv",
        index=False,
    )

    natural_inference_output = natural_inference.copy()
    natural_inference_output.insert(
        0, "evaluation_scope", "natural_distribution"
    )
    natural_inference_output.to_csv(
        EVIDENCE_DIR / "natural_paired_inference.csv",
        index=False,
    )
    scenario_inference.to_csv(
        EVIDENCE_DIR / "forced_scenario_paired_inference.csv",
        index=False,
    )
    robust_inference.to_csv(
        EVIDENCE_DIR / "robust_mean_paired_inference.csv",
        index=False,
    )

    graph_audit.to_csv(
        EVIDENCE_DIR / "graph_availability_audit.csv",
        index=False,
    )

    key_results = {
        "status": "PASS",
        "reviewer_request": (
            "Quantitative comparison with BotBuster and BotMoE"
        ),
        "natural_distribution": natural_table.to_dict(
            orient="records"
        ),
        "robust_summary": robust_summary.to_dict(
            orient="records"
        ),
        "natural_paired_inference": (
            natural_inference_output.loc[
                natural_inference_output["metric"].eq(
                    "macro_f1"
                )
            ].to_dict(orient="records")
        ),
        "forced_robust_paired_inference": (
            robust_inference.loc[
                robust_inference["metric"].eq("macro_f1")
            ].to_dict(orient="records")
        ),
        "graph_availability_interpretation": (
            "The seven-condition comparison preserves each system's "
            "native graph representation. The T/M/TM sensitivity "
            "excludes graph evidence and therefore has identical "
            "availability across systems."
        ),
        "source_implementations": source_implementations,
        "naming_constraint": (
            "Comparator models are common-evidence, audited "
            "reimplementations and are not described as exact "
            "reproductions of the original systems."
        ),
    }
    with (
        EVIDENCE_DIR / "key_results.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(key_results, handle, indent=2)

    evidence_files = sorted(
        path
        for path in EVIDENCE_DIR.iterdir()
        if path.is_file()
        and path.name != "manifest.json"
    )
    evidence_hashes = {
        path.name: {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in evidence_files
    }
    manifest = {
        "status": "PASS",
        "manuscript_reference": "OSNEM-D-26-00779",
        "evidence_scope": (
            "Reviewer-requested BotBuster and BotMoE comparisons"
        ),
        "contains_user_level_data": False,
        "contains_predictions": False,
        "contains_model_checkpoints": False,
        "aggregate_evidence_only": True,
        "verified_source_manifests": {
            name: str(path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            )
            for name, path in MANIFEST_PATHS.items()
        },
        "source_manifest_hashes": {
            name: sha256_file(path)
            for name, path in MANIFEST_PATHS.items()
        },
        "source_implementation_commits": {
            name: details["commit"]
            for name, details in source_implementations.items()
        },
        "evidence_files": evidence_hashes,
        "finalization_script_sha256": sha256_file(
            Path(__file__)
        ),
    }
    with (
        EVIDENCE_DIR / "manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("OSNEM COMPARATOR EVIDENCE FINALIZATION: PASS")
    print("\nNatural comparison:")
    print(
        natural_table[
            [
                "model",
                "macro_f1_mean",
                "macro_f1_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
            ]
        ].to_string(index=False)
    )
    print("\nRobust comparison:")
    print(robust_summary.to_string(index=False))
    print("\nMacro-F1 paired inference:")
    print(
        pd.concat(
            [
                natural_inference_output.loc[
                    natural_inference_output["metric"].eq(
                        "macro_f1"
                    ),
                    [
                        "evaluation_scope",
                        "comparator",
                        "mean_observed_difference",
                        "bootstrap_ci_lower_95",
                        "bootstrap_ci_upper_95",
                        "bootstrap_p_holm",
                        "interpretation",
                    ],
                ].rename(
                    columns={
                        "evaluation_scope": "scope"
                    }
                ),
                robust_inference.loc[
                    robust_inference["metric"].eq(
                        "macro_f1"
                    ),
                    [
                        "scope",
                        "comparator",
                        "mean_observed_difference",
                        "bootstrap_ci_lower_95",
                        "bootstrap_ci_upper_95",
                        "bootstrap_p_holm",
                        "interpretation",
                    ],
                ],
            ],
            ignore_index=True,
        ).to_string(index=False)
    )
    print(f"\nEvidence: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
