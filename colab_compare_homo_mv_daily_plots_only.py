"""
Colab / local launcher: compare_homo_mv_daily_global_localres.py with --plots-only.

Uses the **same graph** as training (run_* chunk): 3817 nodes + shared edges.
The smaller ``loadtype_8500_dailyagg`` hetero folder (~1177 nodes) will **not** match
your ``gnn_only_chunked_mvagg_*`` checkpoint — pass ``--nodes-csv`` / ``--edge-csv``.

Edit RUN_CHUNK and paths below, then:
  python -u colab_compare_homo_mv_daily_plots_only.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "/content/GNN-Sandia"

# Training chunk (same node count / edges as checkpoint)
RUN_CHUNK = "/content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_2000_40/run_001_scen_0000_0049_seed_20360133"

CHECKPOINT = "/content/drive/MyDrive/datasets_gnn2/runs/gnn_only_chunked_mvagg_20260504_180402/gine_gnn_only_best.pt"
# dataset-dir must exist; only used as fallback root — we override nodes/edges to RUN_CHUNK.
DATASET_DIR = RUN_CHUNK
OUT_DIR = "/content/drive/MyDrive/datasets_gnn2/runs/gnn2_daily_compare_gnn_only_node_plots_only"
DAILY_PROFILE = "5minDayShape4.csv"

NODES_CSV = str(Path(RUN_CHUNK) / "gnn_node_features_and_targets_mvagg.csv")
EDGE_CSV = str(Path(RUN_CHUNK) / "gnn_edges_phase_static.csv")
NODE_PE_CSV = str(Path(RUN_CHUNK) / "gnn_node_index_master.csv")
STATIC_MVAGG_CSV = str(Path(RUN_CHUNK) / "gnn_node_features_and_targets_mvagg.csv")

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
        "--nodes-csv",
        str(Path(NODES_CSV).resolve()),
        "--edge-csv",
        str(Path(EDGE_CSV).resolve()),
        "--out-dir",
        str(Path(OUT_DIR).resolve()),
        "--daily-profile",
        DAILY_PROFILE,
        "--plots-only",
        "--gnn-node-pe-csv",
        str(Path(NODE_PE_CSV).resolve()),
        "--gnn-node-pe-cols",
        "auto",
        "--gnn-static-mvagg-csv",
        str(Path(STATIC_MVAGG_CSV).resolve()),
    ]
    for nk in PLOT_NODES:
        cmd.extend(["--plot-node", nk])

    print(" ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)
    print("Figures:", Path(OUT_DIR).resolve() / "monitoring_plots", flush=True)


if __name__ == "__main__":
    main()
