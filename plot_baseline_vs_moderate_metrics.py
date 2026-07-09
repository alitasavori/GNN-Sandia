#!/usr/bin/env python3
"""Plot baseline vs moderate DA-GPS training metrics along epochs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "plots"

# Baseline: no add-ons (da_gps_chunked_l2_h64_mvagg_gine_metaaux_regce_20260630_175219)
# val_r2_mean sourced from training transcript (_tmp_transcript_metrics.txt).
BASELINE = {
    "epochs": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "val_volt": [0.0646, 0.0285, 0.0276, 0.0195, 0.0164, 0.0139, 0.0198, 0.0168, 0.0127, 0.0114, 0.0117],
    "val_r2_mean": [0.8366, 0.8868, 0.8923, 0.9149, 0.9180, 0.9251, 0.9079, 0.9180, 0.9283, 0.9304, 0.9306],
    "val_r2_min": [-4.2531, -0.1358, -0.0399, 0.0911, 0.0332, 0.1105, 0.1091, -0.0039, 0.1717, 0.1427, 0.1694],
}

# Moderate full: addons_moderate_full_20260708_050401
MODERATE = {
    "epochs": list(range(1, 201, 10)) + [200],
    "val_volt": [
        0.0642, 0.0303, 0.0272, 0.0190, 0.0274, 0.0133, 0.0140, 0.0149, 0.0124, 0.0131,
        0.0123, 0.0117, 0.0127, 0.0128, 0.0120, 0.0108, 0.0109, 0.0108, 0.0114, 0.0110, 0.0108,
    ],
    "val_r2_mean": [
        0.8317, 0.8831, 0.8910, 0.9138, 0.8931, 0.9256, 0.9258, 0.9232, 0.9281, 0.9279,
        0.9290, 0.9303, 0.9287, 0.9294, 0.9318, 0.9320, 0.9321, 0.9332, 0.9322, 0.9333, 0.9321,
    ],
    "val_r2_min": [
        -1.7269, -0.8489, -0.0360, -0.0506, -0.4671, -0.1239, -0.1209, -0.1769, -0.0096, -0.0633,
        0.0414, 0.1007, 0.0561, 0.0994, 0.0903, 0.1124, 0.0360, 0.1056, 0.1397, 0.1505, 0.1216,
    ],
}

BASELINE_HAS_R2_MEAN = bool(BASELINE.get("val_r2_mean"))


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best", fontsize=10)


def plot_val_volt(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        BASELINE["epochs"],
        BASELINE["val_volt"],
        marker="o",
        linewidth=2,
        label="Baseline (no add-ons)",
    )
    ax.plot(
        MODERATE["epochs"],
        MODERATE["val_volt"],
        marker="s",
        linewidth=2,
        label="Moderate full add-ons",
    )
    _style_axes(
        ax,
        "Validation Voltage Loss: Baseline vs Moderate DA-GPS",
        "val_volt (lower is better)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_r2_mean(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    if BASELINE_HAS_R2_MEAN:
        ax.plot(
            BASELINE["epochs"],
            BASELINE["val_r2_mean"],
            marker="o",
            linewidth=2,
            label="Baseline (no add-ons)",
        )
        ax.plot(
            MODERATE["epochs"],
            MODERATE["val_r2_mean"],
            marker="s",
            linewidth=2,
            label="Moderate full add-ons",
        )
        title = "Validation R² Mean: Baseline vs Moderate DA-GPS"
        subtitle = None
    else:
        ax.plot(
            MODERATE["epochs"],
            MODERATE["val_r2_mean"],
            marker="s",
            linewidth=2,
            label="Moderate full add-ons",
        )
        title = "Validation R² Mean: Moderate DA-GPS"
        subtitle = "Baseline val_r2_mean not available in transcript"

    _style_axes(ax, title, "val_r2_mean (higher is better)")
    if subtitle:
        fig.suptitle(subtitle, fontsize=10, y=0.98, color="dimgray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    volt_path = OUT_DIR / "baseline_vs_moderate_val_volt.png"
    r2_path = OUT_DIR / "baseline_vs_moderate_val_r2_mean.png"

    plot_val_volt(volt_path)
    plot_val_r2_mean(r2_path)

    print(f"Saved: {volt_path}")
    print(f"Saved: {r2_path}")
    print(f"baseline_has_r2_mean={BASELINE_HAS_R2_MEAN}")


if __name__ == "__main__":
    main()
