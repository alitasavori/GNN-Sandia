"""
SAGE + GINE vs OpenDSS: rank loads where **both** models match DSS poorly.

Same daily trajectory as ``juxtapose_sage_gine_daily.py``, but nodes are sorted by
``min(MAE_A, MAE_B)`` vs OpenDSS (high ⇒ neither model is accurate on average).

Outputs (under ``OUT_DIR``):
  - ``daily_juxtapose_both_fail_vs_dss_<A>_vs_<B>.csv``
  - ``daily_juxtapose_both_fail_vs_dss_<A>_vs_<B>_<node>.png``

Usage (notebook):
    %cd C:\\Users\\alita\\OneDrive\\Desktop\\GNN2
    %run juxtapose_both_fail_vs_dss_daily.py

Or CLI from repo root::
    python compare_hetero_mv_daily.py --checkpoint <A.pt> --vs-checkpoint <B.pt> --out-dir <dir> --juxtapose-mode both-fail-dss
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Jupyter caches imports; reload so `juxtapose_mode` and other updates are always picked up.
_cmp = "compare_hetero_mv_daily"
if _cmp in sys.modules:
    importlib.reload(sys.modules[_cmp])
from compare_hetero_mv_daily import run_compare_juxtapose

REPO = Path(__file__).resolve().parent

DATASET_DIR = REPO / "datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"
NODE_INDEX = REPO / "datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv"
CKPT_SAGE = REPO / "gnn2_architecture_search/hetero_mv_8500/NOT EDGE AWARE/hetero_sage_4x64_ln_drop_best.pt"
CKPT_GINE = REPO / "gnn2_architecture_search/hetero_mv_8500/EDGE AWARE/hetero_gine_3x80_best.pt"
OUT_DIR = REPO / "gnn2_daily_compare_8500_output_both_fail_vs_dss"


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
        top_disagree=10,
        disagree_scope="load",
        also_plot_nodes=[],
        juxtapose_mode="both_fail_dss",
    )


if __name__ == "__main__":
    main()
