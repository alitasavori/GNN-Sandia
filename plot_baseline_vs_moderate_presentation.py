#!/usr/bin/env python3
"""Presentation-quality epoch curves: Baseline vs Moderate DA-GPS."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "plots" / "presentation"

# Baseline: no add-ons (da_gps_chunked_l2_h64_mvagg_gine_metaaux_regce_20260630_175219)
# Source: agent transcript / _tmp_transcript_metrics.txt (epochs 1–100, eval every 10)
BASELINE = {
    "label": "Baseline (no add-ons)",
    "run_id": "20260630_175219",
    "epochs": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "val_tot": [0.2459, 0.1474, 0.1424, 0.1272, 0.1183, 0.1135, 0.1246, 0.1167, 0.1092, 0.1047, 0.1052],
    "val_reg": [1.3872, 1.0909, 1.0634, 0.9970, 0.9403, 0.9201, 0.9668, 0.9239, 0.8882, 0.8624, 0.8654],
    "val_volt": [0.0646, 0.0285, 0.0276, 0.0195, 0.0164, 0.0139, 0.0198, 0.0168, 0.0127, 0.0114, 0.0117],
    "val_r2_mean": [0.8366, 0.8868, 0.8923, 0.9149, 0.9180, 0.9251, 0.9079, 0.9180, 0.9283, 0.9304, 0.9306],
    "val_r2_min": [-4.2531, -0.1358, -0.0399, 0.0911, 0.0332, 0.1105, 0.1091, -0.0039, 0.1717, 0.1427, 0.1694],
    # val_tap_acc was not logged in the baseline run
}

# Moderate full: addons_moderate_full_20260708_050401 (α=1.75, β=7, V=6)
# Source: da_gps_ordinal225_colab.ipynb user logs (_tmp_cell_3.txt)
MODERATE = {
    "label": "Moderate add-ons (β=7, α=1.75, V=6)",
    "run_id": "20260708_050401",
    "epochs": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    "val_tot": [
        0.2469, 0.1498, 0.1425, 0.1255, 0.1314, 0.1120, 0.1151, 0.1114, 0.1093, 0.1083, 0.1085,
        0.1098, 0.1095, 0.1110, 0.1068, 0.1067, 0.1038, 0.1040, 0.1045, 0.1036, 0.1048,
    ],
    "val_reg": [
        1.4004, 1.0993, 1.0699, 0.9862, 0.9611, 0.9099, 0.9332, 0.8882, 0.8957, 0.8819, 0.8894,
        0.8952, 0.8925, 0.9117, 0.8756, 0.8837, 0.8595, 0.8622, 0.8603, 0.8584, 0.8709,
    ],
    "val_volt": [
        0.0642, 0.0303, 0.0272, 0.0190, 0.0274, 0.0133, 0.0140, 0.0149, 0.0124, 0.0131, 0.0123,
        0.0117, 0.0127, 0.0128, 0.0120, 0.0108, 0.0109, 0.0108, 0.0114, 0.0110, 0.0108,
    ],
    "val_r2_mean": [
        0.8317, 0.8831, 0.8910, 0.9138, 0.8931, 0.9256, 0.9258, 0.9232, 0.9281, 0.9279, 0.9290,
        0.9303, 0.9287, 0.9294, 0.9318, 0.9320, 0.9321, 0.9332, 0.9322, 0.9333, 0.9321,
    ],
    "val_tap_acc": [
        0.4395, 0.5408, 0.5497, 0.5777, 0.5895, 0.6075, 0.5980, 0.6184, 0.6130, 0.6188, 0.6177,
        0.6117, 0.6138, 0.6062, 0.6229, 0.6193, 0.6282, 0.6273, 0.6281, 0.6293, 0.6233,
    ],
    # |val_reg(with territory) - val_reg(no territory)| from counterfactual logs
    "territory_reg_delta_abs": [
        0.1381, 0.2643, 0.3609, 1.8294, 2.4514, 2.7687, 3.3057, 3.1495, 3.0931, 2.4318, 2.4898,
        2.4934, 2.3416, 2.4222, 2.2579, 2.4407, 2.3231, 2.3890, 2.2968, 2.2218, 2.3578,
    ],
}

BASELINE_COLOR = "#2563eb"
MODERATE_COLOR = "#dc2626"
TERRITORY_COLOR = "#7c3aed"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)


def _plot_series(ax, epochs, values, *, label, color, marker, linestyle="-") -> None:
    ax.plot(
        epochs,
        values,
        label=label,
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
        marker=marker,
        markersize=7,
        markevery=1,
    )


def _style_panel(ax, title: str, ylabel: str, *, higher_better: bool | None = None) -> None:
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlim(left=0)
    if higher_better is True:
        ax.text(
            0.02,
            0.04,
            "↑ higher is better",
            transform=ax.transAxes,
            fontsize=10,
            color="dimgray",
            ha="left",
        )
    elif higher_better is False:
        ax.text(
            0.02,
            0.96,
            "↓ lower is better",
            transform=ax.transAxes,
            fontsize=10,
            color="dimgray",
            ha="left",
            va="top",
        )


def plot_figure1_core_metrics(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "DA-GPS Validation Metrics: Baseline vs Moderate Add-ons",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    panels = [
        (axes[0, 0], "val_tot", "Total validation loss", False),
        (axes[0, 1], "val_reg", "Regulator CE (plain)", False),
        (axes[1, 0], "val_volt", "Voltage MSE", False),
        (axes[1, 1], "val_r2_mean", "Mean per-node R²", True),
    ]

    for ax, key, title, higher_better in panels:
        _plot_series(
            ax,
            BASELINE["epochs"],
            BASELINE[key],
            label=BASELINE["label"],
            color=BASELINE_COLOR,
            marker="o",
        )
        _plot_series(
            ax,
            MODERATE["epochs"],
            MODERATE[key],
            label=MODERATE["label"],
            color=MODERATE_COLOR,
            marker="s",
        )
        _style_panel(ax, title, key, higher_better=higher_better)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure2_tap_accuracy(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    _plot_series(
        ax,
        MODERATE["epochs"],
        MODERATE["val_tap_acc"],
        label=MODERATE["label"],
        color=MODERATE_COLOR,
        marker="s",
    )
    _style_panel(ax, "Regulator Tap Accuracy (val_tap_acc)", "Accuracy", higher_better=True)
    ax.set_ylim(0.35, 0.68)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.96,
        "Baseline did not log val_tap_acc\n(counterfactual logging added with add-ons)",
        transform=ax.transAxes,
        fontsize=11,
        color="dimgray",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    fig.suptitle(
        "Tap Prediction Accuracy — Moderate Run Only",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure3_territory_mechanism(out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)

    ax1.bar(
        MODERATE["epochs"],
        MODERATE["territory_reg_delta_abs"],
        width=6,
        color=TERRITORY_COLOR,
        alpha=0.55,
        label="|Δ val_reg| with vs without territory bias",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("|Δ val_reg| (territory counterfactual)", color=TERRITORY_COLOR)
    ax1.tick_params(axis="y", labelcolor=TERRITORY_COLOR)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax1.set_xlim(0, 205)

    ax2 = ax1.twinx()
    _plot_series(
        ax2,
        MODERATE["epochs"],
        MODERATE["val_reg"],
        label="val_reg (with territory active)",
        color=MODERATE_COLOR,
        marker="s",
        linestyle="-",
    )
    ax2.set_ylabel("val_reg (with territory)", color=MODERATE_COLOR)
    ax2.tick_params(axis="y", labelcolor=MODERATE_COLOR)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(
        "Territory Bias Is the Active Mechanism\n"
        "Removing territory bias sharply worsens val_reg at every eval epoch",
        fontweight="bold",
        pad=12,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "fig1": OUT_DIR / "fig1_core_metrics_2x2.png",
        "fig2": OUT_DIR / "fig2_val_tap_acc.png",
        "fig3": OUT_DIR / "fig3_territory_counterfactual.png",
    }

    plot_figure1_core_metrics(paths["fig1"])
    plot_figure2_tap_accuracy(paths["fig2"])
    plot_figure3_territory_mechanism(paths["fig3"])

    for name, path in paths.items():
        print(f"Saved {name}: {path}")

    best_baseline = min(zip(BASELINE["epochs"], BASELINE["val_tot"]), key=lambda x: x[1])
    best_moderate = min(zip(MODERATE["epochs"], MODERATE["val_tot"]), key=lambda x: x[1])
    print(f"Best baseline val_tot: ep{best_baseline[0]} = {best_baseline[1]:.4f}")
    print(f"Best moderate val_tot: ep{best_moderate[0]} = {best_moderate[1]:.4f}")
    print(f"Moderate val_tap_acc at best val_tot epoch: {MODERATE['val_tap_acc'][MODERATE['epochs'].index(best_moderate[0])]:.4f}")


if __name__ == "__main__":
    main()
