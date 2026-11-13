"""Regenerate manuscript Figure 7 from overlap-excluded TwiBot-20 results.

Run from the RobustBotDetection project root:

    python ./scripts/43_regenerate_twibot20_disjoint_figure.py

The script reads only the locked outputs produced by Script 42. It does not
train models, select thresholds, or recompute predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
INPUT_DIR = OUTPUTS / "39_twibot20_overlap_sensitivity"
FIGURE_DIR = OUTPUTS / "39_manuscript_figures"

COLORS = [
    "#165B5F", "#1F7E84", "#27A0A8", "#30C5CF", "#55D0D8",
    "#7BDAE0", "#A0E4E9", "#C5EFF2", "#EAF9FA",
]
TICK_SIZE = 10
LABEL_SIZE = 12
TITLE_SIZE = 14
AXIS_LINEWIDTH = 1.0
GRID_COLOR = "#D6E8E9"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required overlap-excluded result is missing: {path}")
    return path


def configure_style() -> None:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError("Times New Roman is not available to Matplotlib.") from exc

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": TICK_SIZE,
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": AXIS_LINEWIDTH,
            "ytick.major.width": AXIS_LINEWIDTH,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.fontsize": LABEL_SIZE,
            "legend.frameon": False,
            "figure.titlesize": TITLE_SIZE,
            "figure.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, *, y_minor: bool = True, x_minor: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
        spine.set_color("black")
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)
    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if x_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def bold_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def main() -> None:
    configure_style()
    summary = pd.read_csv(require(INPUT_DIR / "disjoint_summary.csv"))
    inference = pd.read_csv(require(INPUT_DIR / "disjoint_paired_inference.csv"))

    protocols = [
        "strict_zero_shot_source_threshold",
        "validation_threshold_adaptation",
        "locked_within_twibot20_replication",
    ]
    protocol_names = ["Strict Zero-Shot", "Threshold-Adapted", "Locked Replication"]
    series = [
        ("bertweet_mean_metadata", "Text + Metadata", COLORS[0], "o"),
        ("debiased_text_metadata", "Debiased Text + Metadata", COLORS[2], "s"),
        ("xgboost_metadata", "Metadata", COLORS[5], "^"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
    )

    for model, label, color, marker in series:
        xs, ys, errors = [], [], []
        for index, protocol in enumerate(protocols):
            row = summary.loc[
                summary["model"].eq(model)
                & summary["protocol"].eq(protocol)
                & summary["max_length"].eq(64)
            ]
            if not row.empty:
                xs.append(index)
                ys.append(float(row.iloc[0]["macro_f1_mean"]))
                errors.append(float(row.iloc[0]["macro_f1_std"]))
        axes[0].errorbar(
            xs,
            ys,
            yerr=errors,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            capsize=3,
            label=label,
        )

    axes[0].set_xticks(np.arange(3), protocol_names, rotation=18, ha="right")
    axes[0].set_ylabel("TwiBot-20 Macro-F1")
    axes[0].set_ylim(0.40, 0.78)
    axes[0].set_title("External Transfer Protocols", pad=8)
    style_axis(axes[0])
    bold_legend(axes[0], loc="lower right", fontsize=10)
    panel_label(axes[0], "(a)")

    chosen = [
        "debiased_vs_original_strict_disjoint",
        "debiased_vs_metadata_strict_disjoint",
        "original_vs_metadata_strict_disjoint",
        "multimodal_vs_metadata_replication_disjoint",
        "multimodal_vs_metadata_graph_replication_disjoint",
        "metadata_graph_vs_metadata_replication_disjoint",
    ]
    label_map = {
        "debiased_vs_original_strict_disjoint": "Debiased vs Original (Zero-Shot)",
        "debiased_vs_metadata_strict_disjoint": "Debiased vs Metadata (Zero-Shot)",
        "original_vs_metadata_strict_disjoint": "Original vs Metadata (Zero-Shot)",
        "multimodal_vs_metadata_replication_disjoint": "Multimodal vs Metadata (Replication)",
        "multimodal_vs_metadata_graph_replication_disjoint": "Multimodal vs Metadata + Graph",
        "metadata_graph_vs_metadata_replication_disjoint": "Metadata + Graph vs Metadata",
    }
    effects = inference.set_index("comparison").reindex(chosen).reset_index()
    if effects[["mean_observed_difference", "ci_lower_95", "ci_upper_95"]].isna().any().any():
        raise RuntimeError("One or more required paired comparisons are missing.")
    effects["display"] = effects["comparison"].map(label_map)
    y = np.arange(len(effects))[::-1]
    means = effects["mean_observed_difference"].to_numpy(float)
    lower = effects["ci_lower_95"].to_numpy(float)
    upper = effects["ci_upper_95"].to_numpy(float)

    axes[1].axvline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color=COLORS[0],
        ecolor=COLORS[2],
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    axes[1].set_yticks(y, effects["display"])
    axes[1].set_xlabel("Paired Macro-F1 Difference (95% CI)")
    axes[1].set_title("External Paired Effect Sizes", pad=8)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.05))
    axes[1].set_xlim(-0.02, 0.30)
    style_axis(axes[1], y_minor=False, x_minor=True)
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axes[1].grid(axis="y", visible=False)
    panel_label(axes[1], "(b)")

    fig.suptitle(
        "Identifier-Disjoint TwiBot-20 Validation and Statistical Inference",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = "fig07_twibot20_external_validation"
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "PASS",
        "figure": stem,
        "cohort": "885 TwiBot-20 test accounts absent from the labeled TwiBot-22 release",
        "source": "outputs/39_twibot20_overlap_sensitivity",
        "policy": "locked predictions and thresholds; no retraining or reselection",
        "formats": ["PDF", "SVG", "PNG (600 dpi)"],
        "font_family": "Times New Roman",
        "font_sizes_pt": {
            "tick_and_values": TICK_SIZE,
            "axes_and_legends": LABEL_SIZE,
            "titles": TITLE_SIZE,
        },
        "palette_dark_to_light": COLORS,
    }
    (FIGURE_DIR / "fig07_disjoint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("TWIBOT-20 DISJOINT FIGURE 7: PASS")
    print(f"PDF: {FIGURE_DIR / (stem + '.pdf')}")
    print(f"PNG: {FIGURE_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
