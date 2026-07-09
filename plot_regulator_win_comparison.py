#!/usr/bin/env python3
"""Focused presentation plots: Moderate wins on Regulator CE & tap accuracy."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_baseline_vs_moderate_presentation import (
    BASELINE,
    BASELINE_COLOR,
    MODERATE,
    MODERATE_COLOR,
    OUT_DIR,
)

# Reuse presentation styling; bump fonts slightly for slide legibility.
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

BASELINE_BEST_REG_EP = 90
BASELINE_BEST_REG = 0.8624
MODERATE_BEST_REG_EP = 190
MODERATE_BEST_REG = 0.8584
MODERATE_BEST_TAP_EP = 190
MODERATE_BEST_TAP = 0.6293


def _plot_series(ax, epochs, values, *, label, color, marker, linestyle="-") -> None:
    ax.plot(
        epochs,
        values,
        label=label,
        color=color,
        linewidth=2.8,
        linestyle=linestyle,
        marker=marker,
        markersize=7,
        markevery=1,
    )


def _annotate_point(
    ax,
    epoch: int,
    value: float,
    text: str,
    *,
    xytext: tuple[float, float],
    color: str,
    ha: str = "left",
) -> None:
    ax.scatter([epoch], [value], s=90, color=color, zorder=5, edgecolors="white", linewidths=1.2)
    ax.annotate(
        text,
        xy=(epoch, value),
        xytext=xytext,
        textcoords="offset points",
        fontsize=11,
        color=color,
        ha=ha,
        va="center",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, shrinkA=4, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, alpha=0.92),
    )


def _style_reg_axis(ax) -> None:
    ax.set_title("Regulator CE (val_reg) — Moderate Wins", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_reg (plain CE)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlim(0, 205)
    ax.text(
        0.02,
        0.96,
        "↓ lower is better",
        transform=ax.transAxes,
        fontsize=11,
        color="dimgray",
        ha="left",
        va="top",
    )


def _style_tap_axis(ax) -> None:
    ax.set_title("Regulator Tap Accuracy (val_tap_acc)", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlim(0, 205)
    ax.set_ylim(0.40, 0.66)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.text(
        0.02,
        0.04,
        "↑ higher is better",
        transform=ax.transAxes,
        fontsize=11,
        color="dimgray",
        ha="left",
    )


def plot_regulator_ce(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

    _plot_series(
        ax,
        BASELINE["epochs"],
        BASELINE["val_reg"],
        label=BASELINE["label"],
        color=BASELINE_COLOR,
        marker="o",
    )
    _plot_series(
        ax,
        MODERATE["epochs"],
        MODERATE["val_reg"],
        label=MODERATE["label"],
        color=MODERATE_COLOR,
        marker="s",
    )
    _style_reg_axis(ax)
    ax.legend(loc="upper right", frameon=True)

    _annotate_point(
        ax,
        BASELINE_BEST_REG_EP,
        BASELINE_BEST_REG,
        f"Baseline best\nep{BASELINE_BEST_REG_EP}: {BASELINE_BEST_REG:.4f}",
        xytext=(-95, 18),
        color=BASELINE_COLOR,
        ha="right",
    )
    _annotate_point(
        ax,
        MODERATE_BEST_REG_EP,
        MODERATE_BEST_REG,
        f"Moderate best\nep{MODERATE_BEST_REG_EP}: {MODERATE_BEST_REG:.4f}",
        xytext=(12, -28),
        color=MODERATE_COLOR,
    )

    delta_pct = (BASELINE_BEST_REG - MODERATE_BEST_REG) / BASELINE_BEST_REG * 100
    fig.suptitle(
        "Moderate Add-ons Lower Regulator CE at Best Checkpoints",
        fontsize=19,
        fontweight="bold",
        y=1.03,
    )
    ax.text(
        0.98,
        0.05,
        f"Δ best val_reg = {BASELINE_BEST_REG - MODERATE_BEST_REG:.4f} ({delta_pct:.1f}% lower)",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tap_accuracy_moderate_only(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

    _plot_series(
        ax,
        MODERATE["epochs"],
        MODERATE["val_tap_acc"],
        label=MODERATE["label"],
        color=MODERATE_COLOR,
        marker="s",
    )
    _style_tap_axis(ax)
    ax.legend(loc="lower right", frameon=True)

    _annotate_point(
        ax,
        MODERATE_BEST_TAP_EP,
        MODERATE_BEST_TAP,
        f"Best tap acc\nep{MODERATE_BEST_TAP_EP}: {MODERATE_BEST_TAP:.1%}",
        xytext=(12, 16),
        color=MODERATE_COLOR,
    )
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
        "Tap Prediction Accuracy — Moderate Run (~63% at Convergence)",
        fontsize=19,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_1x2(out_path: Path) -> None:
    fig, (ax_reg, ax_tap) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    _plot_series(
        ax_reg,
        BASELINE["epochs"],
        BASELINE["val_reg"],
        label=BASELINE["label"],
        color=BASELINE_COLOR,
        marker="o",
    )
    _plot_series(
        ax_reg,
        MODERATE["epochs"],
        MODERATE["val_reg"],
        label=MODERATE["label"],
        color=MODERATE_COLOR,
        marker="s",
    )
    _style_reg_axis(ax_reg)
    ax_reg.legend(loc="upper right", frameon=True, fontsize=11)
    _annotate_point(
        ax_reg,
        BASELINE_BEST_REG_EP,
        BASELINE_BEST_REG,
        f"Baseline ep{BASELINE_BEST_REG_EP}\n{BASELINE_BEST_REG:.4f}",
        xytext=(-70, 16),
        color=BASELINE_COLOR,
        ha="right",
    )
    _annotate_point(
        ax_reg,
        MODERATE_BEST_REG_EP,
        MODERATE_BEST_REG,
        f"Moderate ep{MODERATE_BEST_REG_EP}\n{MODERATE_BEST_REG:.4f}",
        xytext=(10, -24),
        color=MODERATE_COLOR,
    )

    _plot_series(
        ax_tap,
        MODERATE["epochs"],
        MODERATE["val_tap_acc"],
        label=MODERATE["label"],
        color=MODERATE_COLOR,
        marker="s",
    )
    _style_tap_axis(ax_tap)
    ax_tap.legend(loc="lower right", frameon=True, fontsize=11)
    _annotate_point(
        ax_tap,
        MODERATE_BEST_TAP_EP,
        MODERATE_BEST_TAP,
        f"ep{MODERATE_BEST_TAP_EP}: {MODERATE_BEST_TAP:.1%}",
        xytext=(10, 14),
        color=MODERATE_COLOR,
    )
    ax_tap.text(
        0.02,
        0.96,
        "Baseline: not logged",
        transform=ax_tap.transAxes,
        fontsize=10,
        color="dimgray",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    fig.suptitle(
        "Moderate Wins on Regulator CE & Tap Accuracy",
        fontsize=20,
        fontweight="bold",
        y=1.04,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "reg_ce": OUT_DIR / "fig_regulator_ce_baseline_vs_moderate.png",
        "tap_acc": OUT_DIR / "fig_tap_acc_moderate.png",
        "combined": OUT_DIR / "fig_regulator_win_1x2.png",
    }

    plot_regulator_ce(paths["reg_ce"])
    plot_tap_accuracy_moderate_only(paths["tap_acc"])
    plot_combined_1x2(paths["combined"])

    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
