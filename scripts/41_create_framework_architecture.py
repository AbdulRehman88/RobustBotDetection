"""Create the code-grounded framework figure for the bot-detection manuscript.

Place this file in ``RobustBotDetection/scripts`` and run from the project root:

    python ./scripts/41_create_framework_architecture.py

The diagram mirrors the implemented pipeline: leakage-safe modality builders,
the four locked experts, validation-only routing, no-modality fallback, and the
separate calibration, robustness, bias, and external-validation audits.  It
does not read results or execute any model.  PDF, SVG, and 600-dpi PNG outputs
are written to ``outputs/39_manuscript_figures``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "39_manuscript_figures"

# User-locked palette, darkest to lightest.
COLORS = [
    "#165B5F", "#1F7E84", "#27A0A8", "#30C5CF", "#55D0D8",
    "#7BDAE0", "#A0E4E9", "#C5EFF2", "#EAF9FA",
]

EDGE = "#123D40"
BLACK = "#111111"
WHITE = "#FFFFFF"
FONT_FAMILY = os.environ.get("BOT_FIGURE_FONT", "Times New Roman")

# Publication typography: detail, stage/box title, overall title.
DETAIL_SIZE = 8.5
BOX_TITLE_SIZE = 10.5
STAGE_SIZE = 11.5
TITLE_SIZE = 14.5
LINE_WIDTH = 1.0


def configure_style() -> None:
    try:
        fm.findfont(FONT_FAMILY, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            f"{FONT_FAMILY!r} is not installed or visible to Matplotlib. "
            "Install/enable Times New Roman before creating the final figure."
        ) from exc

    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.weight": "bold",
            "font.size": DETAIL_SIZE,
            "axes.titleweight": "bold",
            "figure.titlesize": TITLE_SIZE,
            "figure.titleweight": "bold",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...] = (),
    *,
    facecolor: str = COLORS[8],
    edgecolor: str = EDGE,
    title_color: str = BLACK,
    text_color: str = BLACK,
    linewidth: float = LINE_WIDTH,
    linestyle: str = "solid",
    radius: float = 0.012,
    title_size: float = BOX_TITLE_SIZE,
    detail_size: float = DETAIL_SIZE,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)

    if lines:
        title_y = y + height * 0.73
        detail_y = y + height * 0.38
    else:
        title_y = y + height * 0.50
        detail_y = y + height * 0.25

    ax.text(
        x + width / 2, title_y, title,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=title_size, fontweight="bold", color=title_color,
        linespacing=0.95, zorder=zorder + 1,
    )
    if lines:
        ax.text(
            x + width / 2, detail_y, "\n".join(lines),
            transform=ax.transAxes, ha="center", va="center",
            fontsize=detail_size, fontweight="bold", color=text_color,
            linespacing=1.12, zorder=zorder + 1,
        )
    return patch


def stage_header(
    ax: plt.Axes, x: float, width: float, number: str, label: str
) -> None:
    ax.text(
        x + width / 2, 0.915, f"{number}  {label}",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=STAGE_SIZE, fontweight="bold", color=COLORS[0],
    )
    ax.plot(
        [x, x + width], [0.897, 0.897], transform=ax.transAxes,
        color=COLORS[2], linewidth=2.0, solid_capstyle="round", zorder=2,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = EDGE,
    linestyle: str = "solid",
    linewidth: float = 1.35,
    mutation_scale: float = 12,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def line(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = EDGE,
    linestyle: str = "solid",
    linewidth: float = 1.2,
    zorder: int = 1,
) -> None:
    ax.plot(
        [start[0], end[0]], [start[1], end[1]],
        transform=ax.transAxes, color=color, linestyle=linestyle,
        linewidth=linewidth, solid_capstyle="round", zorder=zorder,
    )


def audit_box(
    ax: plt.Axes,
    x: float,
    width: float,
    title: str,
    text: str,
    color: str,
) -> None:
    rounded_box(
        ax, x, 0.055, width, 0.115, title, (text,),
        facecolor=color, edgecolor=EDGE,
        title_size=9.8, detail_size=7.9, radius=0.010,
    )


def create_framework() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.0, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Code-Grounded Reliability-Routed Multimodal Bot-Detection Framework",
        y=0.982,
    )

    # Stage headers.
    stage_header(ax, 0.025, 0.135, "1", "DATA")
    stage_header(ax, 0.185, 0.255, "2", "MODALITY BUILDERS")
    stage_header(ax, 0.475, 0.235, "3", "LOCKED EXPERTS")
    stage_header(ax, 0.755, 0.155, "4", "ROUTING")
    stage_header(ax, 0.935, 0.055, "5", "OUTPUT")

    # Stage 1: datasets and strict role separation.
    rounded_box(
        ax, 0.030, 0.655, 0.125, 0.165,
        "TwiBot-22",
        ("Primary: 1,000,000 users", "Official train / val / test"),
        facecolor=COLORS[7],
    )
    rounded_box(
        ax, 0.030, 0.430, 0.125, 0.145,
        "TwiBot-20",
        ("External: 11,826 labels", "Zero-shot / adaptation / replication"),
        facecolor=WHITE, linestyle="dashed",
    )
    rounded_box(
        ax, 0.030, 0.270, 0.125, 0.090,
        "Split Firewall",
        ("Train fit | Val select | Test once",),
        facecolor=COLORS[8], title_size=9.8, detail_size=7.7,
    )

    # Stage 2: exact implemented modality construction.
    rounded_box(
        ax, 0.195, 0.700, 0.235, 0.120,
        "Text Representation  T",
        ("name + username + description + location",
         "BERTweet, L=64, frozen mean pool  ->  z_T in R^768"),
        facecolor=COLORS[6],
    )
    rounded_box(
        ax, 0.195, 0.555, 0.235, 0.120,
        "Metadata Representation  M",
        ("10 count, rate, age, ratio, and verified features",
         "train-fit winsorization + standardization  ->  z_M in R^10"),
        facecolor=COLORS[7],
    )
    rounded_box(
        ax, 0.195, 0.410, 0.235, 0.120,
        "Graph Representation  G",
        ("14 heterogeneous relations; topology only",
         "in/out activity features  ->  z_G in R^28"),
        facecolor=COLORS[8],
    )
    rounded_box(
        ax, 0.215, 0.275, 0.195, 0.080,
        "Availability Mask",
        ("a = (T, G, M) in {0,1}^3",),
        facecolor=WHITE, title_size=9.8,
    )

    # Dataset-to-builder buses. Both datasets enter every modality builder;
    # solid = primary and dashed = external transfer/evaluation.
    line(ax, (0.155, 0.738), (0.173, 0.738), linewidth=1.4)
    line(ax, (0.173, 0.315), (0.173, 0.760), linewidth=1.4)
    for y in (0.760, 0.615, 0.470):
        arrow(ax, (0.173, y), (0.195, y), linewidth=1.4)
    arrow(ax, (0.173, 0.315), (0.215, 0.315), linewidth=1.4)

    line(
        ax, (0.155, 0.502), (0.184, 0.502),
        linestyle="dashed", color=COLORS[2],
    )
    line(
        ax, (0.184, 0.315), (0.184, 0.760),
        linestyle="dashed", color=COLORS[2],
    )
    for y in (0.760, 0.615, 0.470):
        arrow(
            ax, (0.184, y), (0.195, y), linestyle="dashed",
            color=COLORS[2], linewidth=1.0, mutation_scale=9,
        )
    arrow(
        ax, (0.184, 0.315), (0.215, 0.315), linestyle="dashed",
        color=COLORS[2], linewidth=1.0, mutation_scale=9,
    )

    # Stage 3: implemented experts and exact neural input sizes.
    rounded_box(
        ax, 0.485, 0.705, 0.215, 0.105,
        "E_T : Text MLP",
        ("[z_T, a_T] : 769 -> 256 -> 64 -> 1",),
        facecolor=COLORS[5],
    )
    rounded_box(
        ax, 0.485, 0.575, 0.215, 0.105,
        "E_M : Metadata XGBoost",
        ("z_M : 10 -> score + validation threshold",),
        facecolor=COLORS[6],
    )
    rounded_box(
        ax, 0.485, 0.445, 0.215, 0.105,
        "E_G : Graph XGBoost",
        ("z_G : 28 -> score + validation threshold",),
        facecolor=COLORS[7],
    )
    rounded_box(
        ax, 0.485, 0.315, 0.215, 0.105,
        "E_TM : Text + Metadata MLP",
        ("[z_T, z_M, a_T] : 779 -> 256 -> 64 -> 1",),
        facecolor=COLORS[4],
    )

    # Builder-to-expert connections.
    arrow(ax, (0.430, 0.760), (0.485, 0.758))
    arrow(ax, (0.430, 0.615), (0.485, 0.628))
    arrow(ax, (0.430, 0.470), (0.485, 0.498))
    arrow(
        ax, (0.430, 0.735), (0.485, 0.385),
        color=COLORS[1], connectionstyle="arc3,rad=0.20",
    )
    arrow(
        ax, (0.430, 0.590), (0.485, 0.365),
        color=COLORS[1], connectionstyle="arc3,rad=0.12",
    )

    # Stage 4: validation-selected compatible-expert router.
    rounded_box(
        ax, 0.755, 0.390, 0.155, 0.420,
        "Availability-Pattern Router",
        (
            "Select compatible expert with",
            "highest validation Macro-F1",
            "",
            "T, TG     ->  E_T",
            "M, GM    ->  E_M",
            "G         ->  E_G",
            "TM, TGM  ->  E_TM",
            "none      ->  fallback",
        ),
        facecolor=COLORS[0], edgecolor=COLORS[0],
        title_color=WHITE, text_color=WHITE,
        title_size=10.6, detail_size=8.4,
    )
    # Expert-score bus into the router.
    for y in (0.758, 0.628, 0.498, 0.368):
        line(ax, (0.700, y), (0.730, y), color=COLORS[0], linewidth=1.2)
    line(ax, (0.730, 0.368), (0.730, 0.758), color=COLORS[0], linewidth=1.2)
    arrow(ax, (0.730, 0.600), (0.755, 0.600), color=COLORS[0], linewidth=1.5)
    arrow(
        ax, (0.410, 0.315), (0.755, 0.425),
        color=COLORS[2], connectionstyle="arc3,rad=-0.12", linewidth=1.4,
    )

    # Stage 5: selected score and validation-locked decision.
    rounded_box(
        ax, 0.935, 0.515, 0.055, 0.175,
        "p(bot)",
        ("locked", "threshold", "Bot / Human"),
        facecolor=COLORS[3], edgecolor=EDGE,
        title_color=BLACK, text_color=BLACK,
        title_size=10.5, detail_size=8.0,
    )
    arrow(ax, (0.910, 0.600), (0.935, 0.600), color=COLORS[0], linewidth=1.6)

    # Audit layer: explicitly separate from fitting the primary locked test result.
    ax.text(
        0.5, 0.215,
        "SEPARATE VALIDATION, ROBUSTNESS, AND INFERENCE LAYER",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=STAGE_SIZE, fontweight="bold", color=COLORS[0],
    )
    ax.plot(
        [0.025, 0.990], [0.198, 0.198], transform=ax.transAxes,
        color=COLORS[2], linewidth=2.0, solid_capstyle="round",
    )

    audit_box(
        ax, 0.030, 0.220, "Robustness",
        "forced patterns | MCAR + label-dependent drops | feature noise",
        COLORS[8],
    )
    audit_box(
        ax, 0.270, 0.220, "Missingness Bias",
        "Cramer's V | pattern-label standardization | weighted debiasing",
        COLORS[7],
    )
    audit_box(
        ax, 0.510, 0.220, "Probability Reliability",
        "validation-only Platt scaling | ECE | Brier score",
        COLORS[6],
    )
    audit_box(
        ax, 0.750, 0.240, "Independent TwiBot-20 Validation",
        "strict zero-shot | threshold adaptation | locked replication | graph controls",
        COLORS[5],
    )

    # Dashed audit connectors indicate evaluation paths, not model inputs.
    arrow(
        ax, (0.962, 0.515), (0.870, 0.174),
        color=COLORS[2], linestyle="dashed", linewidth=1.0,
        connectionstyle="arc3,rad=0.18", mutation_scale=10, zorder=1,
    )
    arrow(
        ax, (0.830, 0.390), (0.620, 0.174),
        color=COLORS[2], linestyle="dashed", linewidth=1.0,
        connectionstyle="arc3,rad=0.15", mutation_scale=10, zorder=1,
    )

    # Compact visual key.
    ax.text(
        0.030, 0.018,
        "Solid path: primary fitting/inference     Dashed path: external or audit-only evaluation",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=7.6, fontweight="bold", color=BLACK,
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.94)
    return fig


def save_figure(fig: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = "fig08_code_grounded_framework"
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        FIGURE_DIR / f"{stem}.png", dpi=600,
        bbox_inches="tight", pad_inches=0.05,
    )
    plt.close(fig)


def main() -> None:
    configure_style()
    figure = create_framework()
    save_figure(figure)
    print("FRAMEWORK FIGURE: PASS")
    print(f"Outputs: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
