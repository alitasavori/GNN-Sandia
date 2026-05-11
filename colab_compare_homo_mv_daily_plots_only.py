"""
Colab / local launcher: compare_homo_mv_daily_global_localres.py with --plots-only.

Writes only PNGs for each --plot-node under OUT_DIR/monitoring_plots/
(no daily_metrics_global_localres.json when --plots-only is set).

IMPORTANT — checkpoint compatibility
--------------------------------------
This daily compare script builds **per-timestep node features as [P_kw, Q_kvar] only**
from OpenDSS and normalizes with x_mean.pt / x_std.pt (length 2).

Checkpoints from ``train_gnn_only_compare_complex_voltage.py`` **chunk/mvagg** runs
(5 power columns + positional encodings → 13-dim input) are **not** compatible with this
script as-is: you need either (a) a GNN-only checkpoint trained with **2** PQ features
on the **same** heterogeneous MV graph used here, or (b) extend the compare script to
build the same feature vector your training used.

Edit the constants below, then run:
  python -u colab_compare_homo_mv_daily_plots_only.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- edit for your Colab / machine ---
REPO = "/content/GNN-Sandia"
CHECKPOINT = "/content/drive/MyDrive/datasets_gnn2/runs/gnn_only_chunked_mvagg_20260504_180402/gine_gnn_only_best.pt"
DATASET_DIR = "/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"
OUT_DIR = "/content/drive/MyDrive/datasets_gnn2/runs/gnn2_daily_compare_gnn_only_node_plots_only"
DAILY_PROFILE = "5minDayShape4.csv"

# One PNG per node (repeat --plot-node for each)
PLOT_NODES = [
    "l3010568.1",
    "l2823592.1",
]

# Optional: absolute path to 8500-node/*.csv if not under repo
# DAILY_PROFILE = "/content/drive/.../5minDayShape4.csv"


def main() -> None:
    repo = Path(REPO)
    if not repo.is_dir():
        raise FileNotFoundError(f"REPO not found: {repo}")

    os.chdir(repo)
    os.environ["PYTHONUNBUFFERED"] = "1"

    script = repo / "compare_homo_mv_daily_global_localres.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(script),
        "--checkpoint",
        str(Path(CHECKPOINT).resolve()),
        "--dataset-dir",
        str(Path(DATASET_DIR).resolve()),
        "--out-dir",
        str(Path(OUT_DIR).resolve()),
        "--daily-profile",
        DAILY_PROFILE,
        "--plots-only",
    ]
    for nk in PLOT_NODES:
        cmd.extend(["--plot-node", nk])

    print(" ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)
    print("Figures:", Path(OUT_DIR).resolve() / "monitoring_plots", flush=True)


if __name__ == "__main__":
    main()
