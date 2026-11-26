from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "40_create_manuscript_figures.py"
)
FIGURE_DIR = (
    PROJECT_ROOT / "outputs" / "39_manuscript_figures"
)
FIGURE_STEM = "fig03_forced_modality_router_vs_fusion"


def load_figure_module():
    specification = importlib.util.spec_from_file_location(
        "manuscript_figure_generator", FIGURE_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"Unable to import {FIGURE_SCRIPT}."
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    module = load_figure_module()
    module.configure_style()
    module.figure_3_forced_modalities()

    generated = []
    for extension in ["pdf", "svg", "png"]:
        path = FIGURE_DIR / f"{FIGURE_STEM}.{extension}"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing regenerated figure artifact: {path}"
            )
        generated.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "bytes": int(path.stat().st_size),
            }
        )

    manifest = {
        "status": "PASS",
        "figure": FIGURE_STEM,
        "comparison_series": [
            "Reliability Router",
            "Gated Fusion + Modality Dropout",
            "BotMoE-style (Common Evidence)",
        ],
        "data_policy": (
            "Locked aggregate test results only; no model "
            "retraining or prediction recomputation."
        ),
        "generated": generated,
    }
    with (
        FIGURE_DIR / "fig03_osnem_revision_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("OSNEM COMPARATOR FIGURE 3: PASS")
    for record in generated:
        print(f"{record['path']} | {record['bytes']} bytes")


if __name__ == "__main__":
    main()
