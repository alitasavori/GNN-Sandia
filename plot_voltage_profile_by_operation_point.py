"""
Plot 24h voltage profile at a node for different operation points (P_BASE, Q_BASE, PV_BASE).
Uses the same load/PV time series (5minDayShape, IrradShape) for all points.
Run from repo root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opendssdirect as dss

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, NPTS, STEP_MIN,
    build_bus_to_phases_from_master_nodes,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
)

os.chdir(BASE_DIR)

# Node to plot (bus.phase, e.g. "840.1")
OBSERVED_NODE = "840.1"

# Operation points: (label, P_BASE, Q_BASE, PV_BASE)
OPERATION_POINTS = [
    ("Current", 1415.2, 835.2, 1000.0),
    ("Lower PV (800 kW)", 1415.2, 835.2, 800.0),
    ("Higher load (1600/950)", 1600.0, 950.0, 1000.0),
    ("Lower net load (1200/700)", 1200.0, 700.0, 1000.0),
]


def run_voltage_profile(node, P_BASE, Q_BASE, PV_BASE, mL, mPV, loads_dss, dev_to_dss_load,
                        dev_to_busph_load, pv_dss, pv_to_dss, pv_to_busph, rng):
    """Run 288 timesteps and return (t_hours, vmag) for the given node."""
    t_hours = []
    vmag = []
    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng,
        )
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            t_hours.append(t * STEP_MIN / 60.0)
            vmag.append(np.nan)
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm, _ = vdict.get(node, (np.nan, 0.0))
        t_hours.append(t * STEP_MIN / 60.0)
        vmag.append(float(vm))
    return t_hours, vmag


def main(node=OBSERVED_NODE, operation_points=None, save_path=None):
    if operation_points is None:
        operation_points = OPERATION_POINTS

    dss_path = inj.compile_once()
    inj.setup_daily()

    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = read_profile_csv_two_col_noheader(resolve_csv_path(csvL_token, dss_path), npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(resolve_csv_path(csvPV_token, dss_path), npts=NPTS)

    # Get node list from loadtype dataset
    node_index_csv = os.path.join("gnn_samples_loadtype_full", "gnn_node_index_master.csv")
    if not os.path.exists(node_index_csv):
        raise FileNotFoundError(f"Missing {node_index_csv}. Run loadtype dataset generation first.")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    if node not in set(node_names_master):
        raise ValueError(f"Node '{node}' not in master list. Valid nodes: {node_names_master[:5]}...")

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(operation_points)))

    for i, (label, P_BASE, Q_BASE, PV_BASE) in enumerate(operation_points):
        print(f"  Running: {label} (P={P_BASE}, Q={Q_BASE}, PV={PV_BASE})...")
        inj.compile_once()
        inj.setup_daily()
        t_hours, vmag = run_voltage_profile(
            node, P_BASE, Q_BASE, PV_BASE,
            mL, mPV, loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, rng,
        )
        ax.plot(t_hours, vmag, label=label, color=colors[i], alpha=0.9)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"Voltage Profile @ {node} — Different Operation Points")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.5)
    ax.set_xlim(0, 24)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot voltage profile for different operation points")
    parser.add_argument("--node", default=OBSERVED_NODE, help=f"Node to plot (default: {OBSERVED_NODE})")
    parser.add_argument("--save", default="plots/voltage_by_operation_point.png", help="Output path")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    main(node=args.node, save_path=args.save)
