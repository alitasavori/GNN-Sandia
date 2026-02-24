"""
Plot 24h voltage profile for the worst node (unidir vs bidir): OpenDSS vs unidir vs bidir.
Reads worst node from mae_per_node_unidir_vs_bidir.csv, or uses 820.1 if CSV missing.
Run from repo root. Requires: block_unidir.pt, block_bidir_unidir_compare.pt
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, NPTS, P_BASE, Q_BASE, PV_BASE, STEP_MIN,
    build_bus_to_phases_from_master_nodes, build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)

os.chdir(BASE_DIR)

OUT_DIR = "gnn_samples_loadtype_unidir"
OUTPUT_DIR = "gnn3_best7_output"
DIR_LOADTYPE = "gnn_samples_loadtype_full"
CKPT_PATH = os.path.join(OUTPUT_DIR, "block_unidir.pt")
CKPT_PATH_BIDIR = os.path.join(OUTPUT_DIR, "block_bidir_unidir_compare.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAE_CSV = os.path.join(OUTPUT_DIR, "mae_per_node_unidir_vs_bidir.csv")


def get_worst_node():
    """Return node name with largest MAE_unidir - MAE_bidir."""
    if os.path.exists(MAE_CSV):
        df = pd.read_csv(MAE_CSV)
        return str(df.iloc[0]["node"])
    return "820.1"


def run_24h_one_node(obs_node):
    """Run 24h profile for one node. Returns (t_hours, v_dss, v_uni, v_bidir)."""
    model_uni, static_uni = load_model_for_inference(CKPT_PATH, device=DEVICE)
    model_bidir, static_bidir = load_model_for_inference(CKPT_PATH_BIDIR, device=DEVICE)
    N = static_uni["N"]

    ei_uni = static_uni["edge_index"].to(DEVICE)
    ea_uni = static_uni["edge_attr"].to(DEVICE)
    eid_uni = static_uni["edge_id"].to(DEVICE)
    ei_bidir = static_bidir["edge_index"].to(DEVICE)
    ea_bidir = static_bidir["edge_attr"].to(DEVICE)
    eid_bidir = static_bidir["edge_id"].to(DEVICE)

    node_index_csv = os.path.join(OUT_DIR, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    obs_idx = node_to_idx[obs_node]

    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = read_profile_csv_two_col_noheader(resolve_csv_path(csvL_token, dss_path), npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(resolve_csv_path(csvPV_token, dss_path), npts=NPTS)

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)
    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)

    t_hours, v_dss, v_uni, v_bidir = [], [], [], []
    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng,
        )
        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            t_hours.append(t * STEP_MIN / 60.0)
            v_dss.append(np.nan)
            v_uni.append(np.nan)
            v_bidir.append(np.nan)
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm_dss, _ = vdict[obs_node]

        sum_p_load = float(sum(busphP_load.values()))
        sum_q_load = float(sum(busphQ_load.values()))
        sum_p_pv = float(sum(busphP_pv.values()))
        p_sys_balance = sum_p_load - sum_p_pv
        q_sys_balance = sum_q_load - sum(CAP_Q_KVAR.values())
        X = build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv, node_to_electrical_dist, p_sys_balance, q_sys_balance)
        x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        g_uni = Data(x=x_t, edge_index=ei_uni, edge_attr=ea_uni, edge_id=eid_uni, num_nodes=N)
        g_bidir = Data(x=x_t, edge_index=ei_bidir, edge_attr=ea_bidir, edge_id=eid_bidir, num_nodes=N)
        with torch.no_grad():
            vm_uni = float(model_uni(g_uni)[obs_idx, 0].item())
            vm_bidir = float(model_bidir(g_bidir)[obs_idx, 0].item())

        t_hours.append(t * STEP_MIN / 60.0)
        v_dss.append(float(vm_dss))
        v_uni.append(vm_uni)
        v_bidir.append(vm_bidir)

    vd = np.array(v_dss, dtype=np.float64)
    vu = np.array(v_uni, dtype=np.float64)
    vb = np.array(v_bidir, dtype=np.float64)
    ok = np.isfinite(vd)
    mae_uni = float(np.mean(np.abs(vd[ok] - vu[ok]))) if np.sum(ok) > 0 else np.nan
    mae_bidir = float(np.mean(np.abs(vd[ok] - vb[ok]))) if np.sum(ok) > 0 else np.nan
    return t_hours, v_dss, v_uni, v_bidir, mae_uni, mae_bidir


def main():
    obs_node = get_worst_node()
    print(f"Worst node: {obs_node}")
    t_hours, v_dss, v_uni, v_bidir, mae_uni, mae_bidir = run_24h_one_node(obs_node)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, v_dss, "b-", label="OpenDSS |V| (pu)", linewidth=2)
    ax.plot(t_hours, v_uni, color="orange", linestyle="--", label=f"GNN unidirectional (MAE={mae_uni:.4f})", linewidth=1.5)
    ax.plot(t_hours, v_bidir, "g:", label=f"GNN bidirectional (MAE={mae_bidir:.4f})", linewidth=1.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"24h voltage profile @ {obs_node} (worst node for unidirectional)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "overlay_24h_worst_node.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
