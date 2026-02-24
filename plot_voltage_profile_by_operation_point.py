"""
Plot 24h voltage profile at node(s) for different operation points (P_BASE, Q_BASE, PV_BASE).
Uses the same load/PV time series (5minDayShape, IrradShape) for all points.
Run from repo root. Explore nodes and operation points to find combinations without oscillations.
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

# Default nodes to plot (bus.phase)
DEFAULT_NODES = ["840.1", "844.1", "848.1", "836.1"]

# Operation points: (label, P_BASE, Q_BASE, PV_BASE) — expand to find oscillation-free combos
OPERATION_POINTS = [
    ("Current", 1415.2, 835.2, 1000.0),
    ("Lower PV (800 kW)", 1415.2, 835.2, 800.0),
    ("Higher load (1600/950)", 1600.0, 950.0, 1000.0),
    ("Lower net load (1200/700)", 1200.0, 700.0, 1000.0),
    ("Low PV + high load", 1600.0, 950.0, 600.0),
    ("High PV + low load", 1200.0, 700.0, 1200.0),
    ("Balanced mid", 1400.0, 820.0, 900.0),
]


def run_voltage_profile_multi_nodes(nodes, P_BASE, Q_BASE, PV_BASE, mL, mPV, loads_dss,
                                    dev_to_dss_load, dev_to_busph_load, pv_dss, pv_to_dss,
                                    pv_to_busph, rng):
    """Run 288 timesteps and return {node: (t_hours, vmag)} for the given nodes."""
    node_set = set(nodes)
    result = {n: ([], []) for n in nodes}
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
        t_hr = t * STEP_MIN / 60.0
        if not dss.Solution.Converged():
            for n in nodes:
                result[n][0].append(t_hr)
                result[n][1].append(np.nan)
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        for n in nodes:
            vm, _ = vdict.get(n, (np.nan, 0.0))
            result[n][0].append(t_hr)
            result[n][1].append(float(vm))
    return result


def main(nodes=None, operation_points=None, save_path=None, node=None):
    # Accept both 'node' and 'nodes' for compatibility
    if nodes is None and node is not None:
        nodes = node
    if nodes is None:
        nodes = DEFAULT_NODES
    if isinstance(nodes, str):
        nodes = [nodes]
    else:
        nodes = list(nodes)  # ensure list, not e.g. tuple
    if operation_points is None:
        operation_points = OPERATION_POINTS

    # Create output directory early (before any cwd changes)
    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dss_path = inj.compile_once()
    inj.setup_daily()

    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = read_profile_csv_two_col_noheader(resolve_csv_path(csvL_token, dss_path), npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(resolve_csv_path(csvPV_token, dss_path), npts=NPTS)

    node_index_csv = os.path.join("gnn_samples_loadtype_full", "gnn_node_index_master.csv")
    if not os.path.exists(node_index_csv):
        raise FileNotFoundError(f"Missing {node_index_csv}. Run loadtype dataset generation first.")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    valid = set(node_names_master)
    for n in nodes:
        if n not in valid:
            raise ValueError(f"Node '{n}' not in master list. Valid: {node_names_master[:8]}...")

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    n_nodes = len(nodes)
    n_ops = len(operation_points)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 4 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n_ops))

    for i, (label, P_BASE, Q_BASE, PV_BASE) in enumerate(operation_points):
        print(f"  Running: {label} (P={P_BASE}, Q={Q_BASE}, PV={PV_BASE})...")
        inj.compile_once()
        inj.setup_daily()
        node_data = run_voltage_profile_multi_nodes(
            nodes, P_BASE, Q_BASE, PV_BASE,
            mL, mPV, loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, rng,
        )
        for j, node in enumerate(nodes):
            t_hours, vmag = node_data[node]
            axes[j].plot(t_hours, vmag, label=label, color=colors[i], alpha=0.9)

    for j, node in enumerate(nodes):
        axes[j].set_ylabel("Voltage (pu)")
        axes[j].set_title(f"Node {node}")
        axes[j].legend(loc="best", fontsize=8)
        axes[j].grid(True, alpha=0.5)
        axes[j].set_xlim(0, 24)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("Voltage Profile — Different Nodes & Operation Points", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot voltage profile for different nodes and operation points")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, help="Nodes to plot (e.g. 840.1 844.1)")
    parser.add_argument("--save", default="plots/voltage_by_operation_point.png", help="Output path")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    main(nodes=args.nodes, save_path=args.save)
