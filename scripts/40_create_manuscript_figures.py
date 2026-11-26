"""Create the seven locked result figures for the bot-detection manuscript.

Place this file in ``RobustBotDetection/scripts`` and run from the project root:

    python ./scripts/40_create_manuscript_figures.py

The script reads only completed experimental CSV artifacts. It does not retrain,
retune, recompute predictions, or invent missing values. Each figure is exported
as PDF, SVG, and 600-dpi PNG to ``outputs/39_manuscript_figures``.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUTS / "39_manuscript_figures"

# User-locked palette: darkest to lightest.
COLORS = [
    "#165B5F", "#1F7E84", "#27A0A8", "#30C5CF", "#55D0D8",
    "#7BDAE0", "#A0E4E9", "#C5EFF2", "#EAF9FA",
]

# Full-width publication typography (points): smallest, +2, +2.
TICK_SIZE = 10
LABEL_SIZE = 12
TITLE_SIZE = 14
AXIS_LINEWIDTH = 1.0
BAR_EDGE = "#123D40"
GRID_COLOR = "#D6E8E9"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required locked result is missing: {path}")
    return path


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(require(OUTPUTS / relative))


def configure_style() -> None:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is not installed or visible to Matplotlib. "
            "Install/enable it before creating the final figures."
        ) from exc

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


def bold_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075, 1.025, label, transform=ax.transAxes, fontsize=TITLE_SIZE,
        fontweight="bold", ha="left", va="bottom"
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_bar_values(ax: plt.Axes, bars, decimals: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=TICK_SIZE, fontweight="bold", rotation=0,
        )


def figure_1_availability_bias() -> None:
    dist = read_csv("26_missingness_bias_audit/availability_pattern_distribution.csv")
    assoc = read_csv("26_missingness_bias_audit/pattern_label_association.csv")
    split_order = ["train", "val", "test"]
    pattern_order = ["M", "GM", "TM", "TGM"]
    titles = {"train": "Training", "val": "Validation", "test": "Test"}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    x = np.arange(len(pattern_order))
    width = 0.36
    for idx, (ax, split) in enumerate(zip(axes, split_order)):
        frame = dist.loc[dist["split"].eq(split)].set_index("availability_pattern")
        frame = frame.reindex(pattern_order).fillna(0)
        human_total = frame["humans"].sum()
        bot_total = frame["bots"].sum()
        human_pct = 100 * frame["humans"].to_numpy(float) / human_total
        bot_pct = 100 * frame["bots"].to_numpy(float) / bot_total
        ax.bar(x - width / 2, human_pct, width, color=COLORS[5], edgecolor=BAR_EDGE,
               linewidth=0.8, label="Human")
        ax.bar(x + width / 2, bot_pct, width, color=COLORS[0], edgecolor=BAR_EDGE,
               linewidth=0.8, label="Bot")
        v = float(assoc.loc[assoc["split"].eq(split), "cramers_v"].iloc[0])
        ax.set_title(f"{titles[split]} (Cramér's V = {v:.3f})", pad=8)
        ax.set_xticks(x, pattern_order)
        ax.set_xlabel("Available Modality Pattern")
        ax.set_ylim(0, 80)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        style_axis(ax)
        panel_label(ax, f"({chr(97 + idx)})")
    axes[0].set_ylabel("Within-Class Users (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
        ncol=2, frameon=False, fontsize=LABEL_SIZE,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.suptitle("Modality Availability Is Associated with Bot Labels", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.835), w_pad=1.0)
    save_figure(fig, "fig01_availability_pattern_bias")


def load_internal_models() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    classical = read_csv("06_classical_baselines/classical_baseline_summary.csv")
    classical_map = {
        "tfidf_metadata_concat": "TF–IDF + Metadata",
        "tfidf_only": "TF–IDF",
        "mask_only": "Availability Mask",
    }
    for key, label in classical_map.items():
        row = classical.loc[classical["model"].eq(key)].iloc[0]
        rows.append({"model": label, "mean": row["macro_f1_mean"], "std": row["macro_f1_std"]})

    bert = read_csv("09_bertweet_baselines/bertweet_baseline_summary.csv")
    bert_map = {
        "bertweet_mean_metadata": "BERTweet + Metadata",
        "bertweet_mean": "BERTweet",
    }
    for key, label in bert_map.items():
        row = bert.loc[bert["model"].eq(key)].iloc[0]
        rows.append({"model": label, "mean": row["macro_f1_mean"], "std": row["macro_f1_std"]})

    metadata = read_csv("15_xgboost_metadata/xgboost_metadata_summary.csv").iloc[0]
    rows.append({"model": "Metadata (XGBoost)", "mean": metadata["macro_f1_mean"],
                 "std": metadata["macro_f1_std"]})

    graph = read_csv(
        "17_heterogeneous_graph_baseline/xgboost_heterogeneous_graph_summary.csv"
    ).iloc[0]
    rows.append({"model": "Graph Topology", "mean": graph["macro_f1_mean"],
                 "std": graph["macro_f1_std"]})
    return pd.DataFrame(rows).sort_values("mean", ascending=True).reset_index(drop=True)


def figure_2_internal_models() -> None:
    frame = load_internal_models()
    colors = [COLORS[1] if m == "BERTweet + Metadata" else COLORS[4]
              for m in frame["model"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    y = np.arange(len(frame))
    bars = ax.barh(
        y, frame["mean"], xerr=frame["std"], color=colors, edgecolor=BAR_EDGE,
        linewidth=0.8, capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
    )
    ax.set_yticks(y, frame["model"])
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Internal Performance across Modality Experts and Baselines", pad=10)
    ax.set_xlim(0.38, 0.72)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    for bar, value in zip(bars, frame["mean"]):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=TICK_SIZE, fontweight="bold")
    style_axis(ax, y_minor=False, x_minor=True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "fig02_internal_model_comparison")


def figure_3_forced_modalities() -> None:
    router = read_csv("21_reliability_router/router_scenario_summary.csv")
    gated = read_csv(
        "13_multimodal_fusion/gated_full_transductive_moddrop_on/scenario_summary.csv"
    )
    botmoe = read_csv(
        "40_osnem_sota_comparison/botmoe_forced_availability/"
        "forced_scenario_summary.csv"
    )
    scenarios = ["T", "G", "M", "TG", "TM", "GM", "TGM"]
    router = router.set_index("scenario").reindex(scenarios)
    gated = gated.set_index("scenario").reindex(scenarios)
    botmoe = botmoe.set_index("scenario").reindex(scenarios)
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width, router["macro_f1_mean"], width,
           yerr=router["macro_f1_std"], color=COLORS[0], edgecolor=BAR_EDGE,
           linewidth=0.8, capsize=3, label="Reliability Router")
    ax.bar(x, gated["macro_f1_mean"], width,
           yerr=gated["macro_f1_std"], color=COLORS[5], edgecolor=BAR_EDGE,
           linewidth=0.8, capsize=3, label="Gated Fusion + Modality Dropout")
    ax.bar(x + width, botmoe["macro_f1_mean"], width,
           yerr=botmoe["macro_f1_std"], color=COLORS[2], edgecolor=BAR_EDGE,
           linewidth=0.8, capsize=3, label="BotMoE-style (Common Evidence)")
    ax.set_xticks(x, scenarios)
    ax.set_xlabel("Available Modalities (T: Text, G: Graph, M: Metadata)")
    ax.set_ylabel("Test Macro-F1")
    ax.set_ylim(0.35, 0.74)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_title("Reliability Routing under Forced Modality Availability", pad=10)
    style_axis(ax)
    bold_legend(ax, loc="upper center", ncol=3)
    fig.tight_layout()
    save_figure(fig, "fig03_forced_modality_router_vs_fusion")


def figure_4_missingness_stress() -> None:
    frame = read_csv("23_robustness_stress_tests/missingness_stress_summary.csv")
    order = ["mcar", "bot_heavier", "human_heavier"]
    labels = {
        "mcar": "MCAR",
        "bot_heavier": "Bot-Heavier",
        "human_heavier": "Human-Heavier",
    }
    styles = {
        "mcar": (COLORS[0], "o"),
        "bot_heavier": (COLORS[2], "s"),
        "human_heavier": (COLORS[5], "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for mechanism in order:
        sub = frame.loc[frame["mechanism"].eq(mechanism)].sort_values("nominal_drop_rate")
        x = 100 * sub["nominal_drop_rate"].to_numpy(float)
        color, marker = styles[mechanism]
        axes[0].plot(x, sub["macro_f1_mean"], color=color, marker=marker,
                     linewidth=2.0, markersize=6, label=labels[mechanism])
        axes[0].fill_between(
            x,
            sub["macro_f1_mean"] - sub["macro_f1_std"],
            sub["macro_f1_mean"] + sub["macro_f1_std"],
            color=color, alpha=0.16, linewidth=0,
        )
        axes[1].plot(x, 100 * sub["fallback_rate_mean"], color=color, marker=marker,
                     linewidth=2.0, markersize=6, label=labels[mechanism])

    axes[0].set_title("Detection Performance", pad=8)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.55, 0.72)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.04))
    axes[1].set_title("No-Modality Fallback", pad=8)
    axes[1].set_ylabel("Fallback Rate (%)")
    axes[1].set_ylim(0, 40)
    axes[1].yaxis.set_major_locator(MultipleLocator(10))
    for idx, ax in enumerate(axes):
        ax.set_xlabel("Nominal Modality Drop Rate (%)")
        ax.set_xlim(-2, 72)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        style_axis(ax, x_minor=True)
        panel_label(ax, f"({chr(97 + idx)})")
    bold_legend(axes[0], loc="lower left")
    fig.suptitle("Robustness to Random and Label-Dependent Missingness", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    save_figure(fig, "fig04_missingness_stress")


def figure_5_calibration() -> None:
    frame = read_csv("22_expert_calibration/test_calibration_summary.csv")
    experts = ["text", "metadata", "graph", "text_metadata"]
    names = ["Text", "Metadata", "Graph", "Text + Metadata"]
    uncal = frame.loc[frame["calibration"].eq("uncalibrated")].set_index("expert")
    cal = frame.loc[frame["calibration"].eq("platt_calibrated")].set_index("expert")
    x = np.arange(len(experts))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    metrics = [("ece_15_mean", "Expected Calibration Error"),
               ("brier_mean", "Brier Score")]
    for idx, (ax, (column, title)) in enumerate(zip(axes, metrics)):
        before = uncal.reindex(experts)[column].to_numpy(float)
        after = cal.reindex(experts)[column].to_numpy(float)
        ax.bar(x - width / 2, before, width, color=COLORS[5], edgecolor=BAR_EDGE,
               linewidth=0.8, label="Uncalibrated")
        ax.bar(x + width / 2, after, width, color=COLORS[0], edgecolor=BAR_EDGE,
               linewidth=0.8, label="Platt-Calibrated")
        ax.set_xticks(x, names, rotation=18, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title, pad=8)
        ax.set_ylim(0, max(before.max(), after.max()) * 1.25)
        style_axis(ax)
        panel_label(ax, f"({chr(97 + idx)})")
    bold_legend(axes[0], loc="upper left")
    fig.suptitle("Validation-Only Calibration Improves Probability Reliability", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    save_figure(fig, "fig05_expert_calibration")


def figure_6_bias_and_debiasing() -> None:
    bias = read_csv("26_missingness_bias_audit/pattern_label_standardized_summary.csv")
    debiased = read_csv("27_missingness_debiased_model/summary.csv")
    model_order = [
        "bertweet_mean_metadata", "tfidf_metadata_concat", "xgboost_metadata",
        "bertweet_mean", "xgboost_heterogeneous_graph", "mask_only",
    ]
    display = ["BERTweet + M", "TF–IDF + M", "Metadata", "BERTweet", "Graph", "Mask"]
    left = bias.set_index("model").reindex(model_order)
    x = np.arange(len(model_order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.65, 1]})
    axes[0].bar(x - width / 2, left["natural_macro_f1_mean"], width,
                color=COLORS[5], edgecolor=BAR_EDGE, linewidth=0.8, label="Natural")
    axes[0].bar(x + width / 2, left["standardized_macro_f1_mean"], width,
                yerr=left["standardized_macro_f1_std"], color=COLORS[0],
                edgecolor=BAR_EDGE, linewidth=0.8, capsize=3,
                label="Pattern–Label Standardized")
    axes[0].set_xticks(x, display, rotation=22, ha="right")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.45, 0.73)
    axes[0].set_title("Natural versus Standardized Evaluation", pad=8)
    style_axis(axes[0])
    bold_legend(axes[0], loc="upper right")
    panel_label(axes[0], "(a)")

    d = debiased.set_index("model")
    original_key = "original_validation_standardized_threshold"
    debiased_key = "debiased_pattern_label_weighted"
    labels = ["Original\nNatural", "Original\nStandardized", "Debiased\nNatural",
              "Debiased\nStandardized"]
    values = [
        d.loc[original_key, "natural_macro_f1_mean"],
        d.loc[original_key, "standardized_macro_f1_mean"],
        d.loc[debiased_key, "natural_macro_f1_mean"],
        d.loc[debiased_key, "standardized_macro_f1_mean"],
    ]
    errors = [
        d.loc[original_key, "natural_macro_f1_std"],
        d.loc[original_key, "standardized_macro_f1_std"],
        d.loc[debiased_key, "natural_macro_f1_std"],
        d.loc[debiased_key, "standardized_macro_f1_std"],
    ]
    bars = axes[1].bar(np.arange(4), values, yerr=errors,
                       color=[COLORS[5], COLORS[2], COLORS[4], COLORS[0]],
                       edgecolor=BAR_EDGE, linewidth=0.8, capsize=3)
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.56, 0.72)
    axes[1].set_title("Pattern–Label Debiasing", pad=8)
    add_bar_values(axes[1], bars, decimals=3)
    style_axis(axes[1])
    panel_label(axes[1], "(b)")
    fig.suptitle("Informative Missingness Changes Apparent Model Performance", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.3)
    save_figure(fig, "fig06_missingness_bias_and_debiasing")


def figure_7_external_validation() -> None:
    transfer = read_csv("32_twibot20_external_transfer/external_transfer_summary.csv")
    metadata = read_csv("33_twibot20_external_metadata/external_metadata_summary.csv")
    graph = read_csv("36_twibot20_external_graph_controls/external_graph_control_summary.csv")
    inference = read_csv("35_twibot20_external_inference/external_paired_inference.csv")
    graph_inf = read_csv("37_twibot20_graph_inference/external_graph_paired_inference.csv")

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
    combined = pd.concat([transfer, metadata], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.35]})

    for model, label, color, marker in series:
        xs, ys, es = [], [], []
        for i, protocol in enumerate(protocols):
            row = combined.loc[(combined["model"].eq(model)) & (combined["protocol"].eq(protocol))]
            if not row.empty:
                xs.append(i); ys.append(float(row.iloc[0]["macro_f1_mean"]))
                es.append(float(row.iloc[0]["macro_f1_std"]))
        axes[0].errorbar(xs, ys, yerr=es, color=color, marker=marker, linewidth=2,
                         markersize=7, capsize=3, label=label)
    axes[0].set_xticks(np.arange(3), protocol_names, rotation=18, ha="right")
    axes[0].set_ylabel("TwiBot-20 Macro-F1")
    axes[0].set_ylim(0.45, 0.86)
    axes[0].set_title("External Transfer Protocols", pad=8)
    style_axis(axes[0])
    bold_legend(axes[0], loc="lower right", fontsize=10)
    panel_label(axes[0], "(a)")

    chosen_external = [
        "debiased_vs_original_strict_zero_shot",
        "debiased_vs_metadata_strict_zero_shot",
        "original_vs_metadata_strict_zero_shot",
        "multimodal_vs_metadata_locked_replication",
    ]
    chosen_graph = ["multimodal_vs_metadata_graph", "metadata_graph_vs_metadata"]
    eff = pd.concat([
        inference.loc[inference["comparison"].isin(chosen_external)],
        graph_inf.loc[graph_inf["comparison"].isin(chosen_graph)],
    ], ignore_index=True)
    label_map = {
        "debiased_vs_original_strict_zero_shot": "Debiased vs Original (Zero-Shot)",
        "debiased_vs_metadata_strict_zero_shot": "Debiased vs Metadata (Zero-Shot)",
        "original_vs_metadata_strict_zero_shot": "Original vs Metadata (Zero-Shot)",
        "multimodal_vs_metadata_locked_replication": "Multimodal vs Metadata (Replication)",
        "multimodal_vs_metadata_graph": "Multimodal vs Metadata + Graph",
        "metadata_graph_vs_metadata": "Metadata + Graph vs Metadata",
    }
    eff["display"] = eff["comparison"].map(label_map)
    eff = eff.set_index("comparison").reindex(chosen_external + chosen_graph).reset_index()
    y = np.arange(len(eff))[::-1]
    means = eff["mean_observed_difference"].to_numpy(float)
    lower = eff["bootstrap_ci_lower_95"].to_numpy(float)
    upper = eff["bootstrap_ci_upper_95"].to_numpy(float)
    axes[1].axvline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].errorbar(
        means, y, xerr=np.vstack([means - lower, upper - means]), fmt="o",
        color=COLORS[0], ecolor=COLORS[2], elinewidth=2, capsize=4, markersize=7,
    )
    axes[1].set_yticks(y, eff["display"])
    axes[1].set_xlabel("Paired Macro-F1 Difference (95% CI)")
    axes[1].set_title("External Paired Effect Sizes", pad=8)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.05))
    style_axis(axes[1], y_minor=False, x_minor=True)
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axes[1].grid(axis="y", visible=False)
    panel_label(axes[1], "(b)")
    fig.suptitle("TwiBot-20 Validation and Statistical Inference", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    save_figure(fig, "fig07_twibot20_external_validation")


def main() -> None:
    configure_style()
    figure_1_availability_bias()
    figure_2_internal_models()
    figure_3_forced_modalities()
    figure_4_missingness_stress()
    figure_5_calibration()
    figure_6_bias_and_debiasing()
    figure_7_external_validation()

    generated = sorted(p.name for p in FIGURE_DIR.glob("fig*.*"))
    manifest = {
        "status": "PASS",
        "figures": 7,
        "formats": ["PDF", "SVG", "PNG (600 dpi)"],
        "font_family": "Times New Roman",
        "font_sizes_pt": {"tick_and_values": TICK_SIZE, "axes_and_legends": LABEL_SIZE,
                          "titles": TITLE_SIZE},
        "axis_linewidth_pt": AXIS_LINEWIDTH,
        "palette_dark_to_light": COLORS,
        "generated_files": generated,
        "data_policy": "locked experimental CSV artifacts only; no retraining or reconstruction",
    }
    (FIGURE_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("MANUSCRIPT RESULT FIGURES: PASS")
    print(f"Figures: 7; files: {len(generated)}")
    print(f"Outputs: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
