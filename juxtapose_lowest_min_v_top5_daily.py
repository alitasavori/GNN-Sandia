"""
Top **5** hetero loads with the **lowest daily minimum** OpenDSS voltage |V| (most stressed).

One OpenDSS day, SAGE + GINE each step; rank loads by ``min_t V_dss(t)`` ascending (smallest min
first). Plots show **OpenDSS + SAGE + GINE** for those five nodes.

Outputs (under ``OUT_DIR``):
  - ``daily_juxtapose_lowest_min_v_dss_<SAGE>_vs_<GINE>.csv``
  - ``daily_juxtapose_lowest_min_v_dss_<SAGE>_vs_<GINE>_<node>.png``

Usage (notebook)::

    %cd C:\\Users\\alita\\OneDrive\\Desktop\\GNN2
    %run juxtapose_lowest_min_v_top5_daily.py

Adjust ``N_TOP`` for a different number of plots.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_cmp = "compare_hetero_mv_daily"
if _cmp in sys.modules:
    importlib.reload(sys.modules[_cmp])
from compare_hetero_mv_daily import run_compare_juxtapose

REPO = Path(__file__).resolve().parent

N_TOP = 5

DATASET_DIR = REPO / "datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"
NODE_INDEX = REPO / "datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv"
CKPT_SAGE = REPO / "gnn2_architecture_search/hetero_mv_8500/NOT EDGE AWARE/hetero_sage_4x64_ln_drop_best.pt"
CKPT_GINE = REPO / "gnn2_architecture_search/hetero_mv_8500/EDGE AWARE/hetero_gine_3x80_best.pt"
OUT_DIR = REPO / "gnn2_daily_compare_8500_output_lowest_min_v_top5"


def main() -> None:
    run_compare_juxtapose(
        checkpoint_a=CKPT_SAGE,
        checkpoint_b=CKPT_GINE,
        dataset_dir=DATASET_DIR,
        node_index=NODE_INDEX,
        out_dir=OUT_DIR,
        npts=288,
        step_min=5,
        ymin=0.85,
        ymax=1.10,
        mv_sx_mapping=None,
        top_disagree=int(N_TOP),
        disagree_scope="load",
        also_plot_nodes=[],
        juxtapose_mode="lowest_min_v_dss",
    )


if __name__ == "__main__":
    main()
