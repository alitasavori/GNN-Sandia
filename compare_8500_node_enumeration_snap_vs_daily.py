"""
Minimal experiment: same IEEE 8500 feeder, two compile paths as in GNN2.
Compares inj.get_all_bus_phase_nodes() after snapshot vs daily setup.

Run: python compare_8500_node_enumeration_snap_vs_daily.py

If on-disk gnn_node_index_master.csv counts differ from the live compile, the CSVs
were generated under a different OpenDSS build or feeder revision — regenerate datasets.
"""
from __future__ import annotations

import os
import sys

import opendssdirect as dss
import pandas as pd

import run_injection_dataset as inj
import run_loadtype_dataset_8500 as lt8500
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RUN_DSS_DAILY = REPO_ROOT / "8500-node" / "Run_8500Node_Daily_5min.dss"
MASTER_8500 = REPO_ROOT / "8500-node" / "Master.dss"


def nodes_snapshot() -> set[str]:
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{os.path.abspath(str(MASTER_8500))}"')
    dss.Solution.Mode(1)
    inj._apply_voltage_bases()
    node_names, _, _, _ = inj.get_all_bus_phase_nodes()
    return set(node_names)


def nodes_daily_pipeline() -> set[str]:
    """Matches run_daily_aggregate_dataset_8500._compile_8500_daily_setup + node list."""
    if not RUN_DSS_DAILY.is_file():
        raise FileNotFoundError(RUN_DSS_DAILY)
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{os.path.abspath(str(RUN_DSS_DAILY))}"')
    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=5m")
    dss.Text.Command("set number=1")
    dss.Text.Command("set maxiterations=30")
    dss.Text.Command("set maxcontroliter=20000")
    node_names, _, _, _ = inj.get_all_bus_phase_nodes()
    return set(node_names)


def nodes_master_then_only_daily_mode() -> set[str]:
    """Isolate: load Master like snapshot, then only switch to daily mode (no Run_*.dss)."""
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{os.path.abspath(str(MASTER_8500))}"')
    dss.Solution.Mode(1)
    inj._apply_voltage_bases()
    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=5m")
    dss.Text.Command("set number=1")
    node_names, _, _, _ = inj.get_all_bus_phase_nodes()
    return set(node_names)


def main() -> int:
    if not MASTER_8500.is_file():
        print("Missing Master.dss:", MASTER_8500, file=sys.stderr)
        return 1

    print("[0] Exact project helper compile_8500() …")
    lt8500.compile_8500()
    s0, _, _, _ = inj.get_all_bus_phase_nodes()
    print(f"    N = {len(s0)}")

    print("[1] Snapshot path (same as [0]: Master.dss + snap + voltage bases) …")
    s = nodes_snapshot()
    print(f"    N = {len(s)}")

    print("[2] Daily pipeline path (Run_8500Node_Daily_5min.dss + set mode=daily …) …")
    d = nodes_daily_pipeline()
    print(f"    N = {len(d)}")

    print("[3] Master.dss snapshot, then ONLY set mode=daily (no loadshapes on loads) …")
    m = nodes_master_then_only_daily_mode()
    print(f"    N = {len(m)}")

    only_s = sorted(s - d)
    only_d = sorted(d - s)
    print()
    print("--- Diff snapshot vs daily pipeline ---")
    print(f"Only in snapshot (not in daily): {len(only_s)}")
    for x in only_s:
        print(f"  {x}")
    print(f"Only in daily (not in snapshot): {len(only_d)}")

    print()
    print("--- Diff snapshot vs [Master + daily mode only] ---")
    om = sorted(s - m)
    mo = sorted(m - s)
    print(f"Only in snapshot (not in master+dailyMode): {len(om)}")
    for x in om[:20]:
        print(f"  {x}")
    if len(om) > 20:
        print(f"  … +{len(om) - 20} more")
    print(f"Only in master+dailyMode (not snapshot): {len(mo)}")

    snap_csv = REPO_ROOT / "datasets_gnn2" / "loadtype_8500" / "gnn_node_index_master.csv"
    daily_csv = REPO_ROOT / "datasets_gnn2" / "loadtype_8500_dailyagg" / "gnn_node_index_master.csv"
    print()
    print("--- On-disk master CSV row counts (if present) ---")
    for label, p in (("snapshot dataset", snap_csv), ("dailyagg dataset", daily_csv)):
        if p.is_file():
            nrows = len(pd.read_csv(p))
            print(f"  {label}: {nrows} rows ({p.name})")
        else:
            print(f"  {label}: (missing) {p}")

    print()
    print("Conclusion:")
    if len(s) == len(d) == len(m):
        print(
            "  Live compile: snapshot, daily Run_*.dss, and Master+daily-mode-only "
            f"all report the same N={len(s)}. Any 8541 vs 8531 mismatch in old CSVs is "
            "from an older feeder/OpenDSS run, not from daily mode hiding nodes."
        )
    else:
        print("  Counts differ — see diffs above for which path changes N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
