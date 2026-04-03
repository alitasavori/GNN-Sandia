"""
OpenDSS snapshot-mode helpers for daily compare scripts.

Kept separate from ``run_daily_aggregate_dataset_8500`` so notebook sessions that cache an
old ``rd8500`` module still get correct behavior after ``git pull`` (no missing attributes).
"""

from __future__ import annotations

import os

import opendssdirect as dss

# ``run_daily_aggregate_dataset_8500._compile_8500_daily_setup`` sets ``maxcontroliter=20000``.
# A previous compare default of 10 matched an old MLP script but causes **(#485) Max Control
# Iterations Exceeded** on IEEE 8500 with regulators/caps. Default **20000** matches compile.
# For timing experiments with a tighter cap, set env ``GNN_COMPARE_MAXCONTROLITER`` (e.g. 200).
_DEFAULT_MAX_CONTROLITER = 20000


def _max_control_iter() -> int:
    return int(os.environ.get("GNN_COMPARE_MAXCONTROLITER", str(_DEFAULT_MAX_CONTROLITER)))


def force_snapshot_mode_for_compare_timing() -> None:
    """
    After compiling the daily 8500 circuit, switch to snapshot solves for fair timing vs
    warm-started daily marching.

    - ``maxiterations=20`` matches ``8500-node/Run_8500Node_Daily_5min.dss`` and
      ``compare_daily_8500_mlp_gnn.py`` (ResidualGCN path).
    - ``maxcontroliter`` defaults to **20000** (same order of magnitude as compile). Override with
      env ``GNN_COMPARE_MAXCONTROLITER`` (e.g. lower for ablations; too low ⇒ DSS #485).
    """
    dss.Text.Command("set mode=snapshot")
    dss.Text.Command("set maxiterations=20")
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    dss.Solution.Mode(1)


def reassert_snapshot_before_each_solve() -> None:
    """Re-apply snapshot mode and solver caps before ``Solution.Solve()`` each timestep."""
    dss.Text.Command("set mode=snapshot")
    dss.Text.Command("set maxiterations=20")
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    dss.Solution.Mode(1)
