#!/usr/bin/env python3
"""Presentation PNG plots: with-DER fine-tune metrics over epochs (3 eval pools).

Run: da_gps_finetune_withder_l2_h96_regce_20260710_011052
Source: Colab fine-tune log (canvas / transcript extract).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "plots" / "presentation"

RUN_ID = "da_gps_finetune_withder_l2_h96_regce_20260710_011052"

# Epoch 0 = eval_before_train (init); then 1, 10, ..., 60 (eval_every=10)
EPOCHS = [0, 1, 10, 20, 30, 40, 50, 60]

# Test |V| MAE (pu)
V_MAE = {
    "train_pool_eval": [0.004957, 0.005021, 0.005268, 0.005179, 0.004521, 0.004601, 0.004761, 0.004365],
    "nobess_40": [0.003896, 0.004008, 0.004063, 0.004247, 0.003965, 0.004085, 0.004133, 0.003835],
    "withder_4": [0.017810, 0.017289, 0.019870, 0.016473, 0.011257, 0.010853, 0.012381, 0.010781],
}

# Test angle MAE (degrees)
ANGLE_MAE = {
    "train_pool_eval": [0.421952, 0.419651, 0.452630, 0.420071, 0.259660, 0.269221, 0.308815, 0.226651],
    "nobess_40": [0.196201, 0.205977, 0.199328, 0.246745, 0.196844, 0.206704, 0.236690, 0.172160],
    "withder_4": [3.157686, 3.009040, 3.522249, 2.520502, 1.020881, 1.026826, 1.182856, 0.886990],
}

# Test tot (composite loss)
TOT = {
    "train_pool_eval": [0.1823, 0.1698, 0.1657, 0.1472, 0.1166, 0.1167, 0.1176, 0.1148],
    "nobess_40": [0.0998, 0.1019, 0.1006, 0.1044, 0.0996, 0.1000, 0.1005, 0.0993],
    "withder_4": [1.1824, 0.9927, 0.9536, 0.6657, 0.3233, 0.3190, 0.3249, 0.3026],
}

POOL_STYLE = {
    "train_pool_eval": {
        "label": "train_pool_eval (44 chunks)",
        "color": "#2563eb",
        "marker": "o",
    },
    "nobess_40": {
        "label": "nobess_40 (no-DER)",
        "color": "#16a34a",
        "marker": "s",
    },
    "withder_4": {
        "label": "withder_4 (with-DER)",
        "color": "#dc2626",
        "marker": "D",
    },
}

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.8,
        "lines.markersize": 9,
    }
)


def _plot_three_pools(ax, series: dict[str, list[float]]) -> None:
    for key, style in POOL_STYLE.items():
        ax.plot(
            EPOCHS,
            series[key],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markevery=1,
        )


def _style_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch (0 = init)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.FixedLocator(EPOCHS))
    ax.set_xlim(-2, 62)
    ax.legend(loc="best", framealpha=0.95)
    ax.text(
        0.02,
        0.96,
        "↓ lower is better",
        transform=ax.transAxes,
        fontsize=12,
        color="dimgray",
        ha="left",
        va="top",
    )


def _savefig(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def fig_v_mae() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _plot_three_pools(ax, V_MAE)
    _style_ax(ax, "DA-GPS with-DER fine-tune: Test |V| MAE", "Test |V| MAE (pu)")
    ax.annotate(
        "best @ epoch 60",
        xy=(60, V_MAE["withder_4"][-1]),
        xytext=(42, 0.0185),
        fontsize=12,
        color=POOL_STYLE["withder_4"]["color"],
        arrowprops=dict(arrowstyle="->", color=POOL_STYLE["withder_4"]["color"], lw=1.5),
    )
    fig.suptitle(
        f"Run {RUN_ID}",
        fontsize=12,
        color="dimgray",
        y=0.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _savefig(fig, "withder_finetune_test_v_mae_three_pools.png")


def fig_angle_mae() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _plot_three_pools(ax, ANGLE_MAE)
    _style_ax(ax, "DA-GPS with-DER fine-tune: Test angle MAE", "Test angle MAE (degrees)")
    fig.suptitle(
        f"Run {RUN_ID}",
        fontsize=12,
        color="dimgray",
        y=0.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _savefig(fig, "withder_finetune_test_angle_mae_three_pools.png")


def fig_tot() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _plot_three_pools(ax, TOT)
    _style_ax(ax, "DA-GPS with-DER fine-tune: Test tot", "Test tot (composite loss)")
    fig.suptitle(
        f"Run {RUN_ID}",
        fontsize=12,
        color="dimgray",
        y=0.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _savefig(fig, "withder_finetune_test_tot_three_pools.png")


def main() -> None:
    paths = [fig_v_mae(), fig_angle_mae(), fig_tot()]
    print("\nGenerated:")
    for p in paths:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
