"""
Colab / local launcher: compare_homo_mv_daily_global_localres.py with --plots-only.

Writes only PNGs for each --plot-node under OUT_DIR/monitoring_plots/
(no daily_metrics_global_localres.json when --plots-only is set).

For GNN-only checkpoints trained with extra inputs (e.g. 5 power + PE), pass the same
--gnn-node-pe-csv and optional --gnn-static-mvagg-csv as used in training so in_dim matches.

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

# GNN-only mvagg+PE (same paths as your Colab training)
GNN_NODE_PE_CSV = "/content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_2000_40/run_001_scen_0000_0049_seed_20360133/gnn_node_index_master.csv"
GNN_STATIC_MVAGG_CSV = "/content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_2000_40/run_001_scen_0000_0049_seed_20360133/gnn_node_features_and_targets_mvagg.csv"

# One PNG per node (repeat --plot-node for each)
PLOT_NODES = [
    "l3010568.1",
    "l2823592.1",
]


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
        "--gnn-node-pe-csv",
        str(Path(GNN_NODE_PE_CSV).resolve()),
        "--gnn-node-pe-cols",
        "auto",
        "--gnn-static-mvagg-csv",
        str(Path(GNN_STATIC_MVAGG_CSV).resolve()),
    ]
    for nk in PLOT_NODES:
        cmd.extend(["--plot-node", nk])

    print(" ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)
    print("Figures:", Path(OUT_DIR).resolve() / "monitoring_plots", flush=True)


if __name__ == "__main__":
    main()
