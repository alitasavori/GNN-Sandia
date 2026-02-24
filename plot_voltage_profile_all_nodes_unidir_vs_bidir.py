"""
Plot 24h voltage profile for all nodes: OpenDSS vs unidirectional vs bidirectional GNN.
Identifies nodes where unidirectional clearly fails (MAE_unidir >> MAE_bidir) and plots them.
Run from repo root. Requires: trained block_unidir.pt, block_bidir_unidir_compare.pt
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
TOP_N_PLOT = 9  # Plot voltage profiles for top N nodes where unidir fails worst


def run_24h_all_nodes():
    """Run 24h profile for all nodes. Returns (t_hours, node_names, V_dss, V_uni, V_bidir)."""
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

    V_dss = np.full((NPTS, N), np.nan)
    V_uni = np.full((NPTS, N), np.nan)
    V_bidir = np.full((NPTS, N), np.nan)

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
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        for i, n in enumerate(node_names_master):
            if n in vdict:
                V_dss[t, i], _ = vdict[n]

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
            y_uni = model_uni(g_uni)
            y_bidir = model_bidir(g_bidir)
        V_uni[t, :] = y_uni[:, 0].cpu().numpy()
        V_bidir[t, :] = y_bidir[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, node_names_master, V_dss, V_uni, V_bidir


def main():
    print("Running 24h profile for all nodes...")
    t_hours, node_names, V_dss, V_uni, V_bidir = run_24h_all_nodes()
    N = len(node_names)

    mae_uni = np.zeros(N)
    mae_bidir = np.zeros(N)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_uni[i] = np.mean(np.abs(V_dss[ok, i] - V_uni[ok, i]))
            mae_bidir[i] = np.mean(np.abs(V_dss[ok, i] - V_bidir[ok, i]))
        else:
            mae_uni[i] = np.nan
            mae_bidir[i] = np.nan

    mae_diff = mae_uni - mae_bidir
    order = np.argsort(-mae_diff)
    worst_unidir = order[:TOP_N_PLOT]

    df = pd.DataFrame({
        "node": node_names,
        "mae_unidir": mae_uni,
        "mae_bidir": mae_bidir,
        "mae_diff": mae_diff,
    }).sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_unidir_vs_bidir.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved MAE per node -> {csv_path}")
    print("\nTop 10 nodes where unidirectional fails worst (MAE_unidir - MAE_bidir):")
    for i, row in df.head(10).iterrows():
        print(f"  {row['node']}: unidir MAE={row['mae_unidir']:.4f} bidir MAE={row['mae_bidir']:.4f} diff={row['mae_diff']:.4f}")

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    idx_sorted = np.argsort(mae_diff)[::-1]
    x_pos = np.arange(N)
    ax1.bar(x_pos, mae_diff[idx_sorted], color="coral", alpha=0.8, label="MAE_unidir - MAE_bidir")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_xlabel("Node (sorted by unidir excess error)")
    ax1.set_ylabel("MAE unidir - MAE bidir (pu)")
    ax1.set_title("Nodes where unidirectional GNN fails vs bidirectional (positive = unidir worse)")
    ax1.legend()
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "mae_diff_per_node_unidir_minus_bidir.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart -> {p1}")

    ncol = 3
    nrow = (TOP_N_PLOT + ncol - 1) // ncol
    fig2, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True)
    axes = np.atleast_2d(axes)
    for k, idx in enumerate(worst_unidir):
        i, j = k // ncol, k % ncol
        ax = axes[i, j]
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS", linewidth=1.5)
        ax.plot(t_hours, V_uni[:, idx], "orange", linestyle="--", label="unidir", alpha=0.9)
        ax.plot(t_hours, V_bidir[:, idx], "g:", label="bidir", alpha=0.9)
        ax.set_title(f"{node_names[idx]} | unidir MAE={mae_uni[idx]:.4f} bidir={mae_bidir[idx]:.4f}")
        ax.set_ylabel("|V| (pu)")
        ax.grid(True)
        ax.legend(loc="lower left", fontsize=8)
    for k in range(len(worst_unidir), nrow * ncol):
        i, j = k // ncol, k % ncol
        axes[i, j].axis("off")
    axes[-1, 1].set_xlabel("Hour of day")
    plt.suptitle("Voltage profiles: nodes where unidirectional fails worst (24h)")
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "voltage_profiles_worst_unidir_nodes.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved voltage profiles -> {p2}")

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.scatter(mae_bidir, mae_uni, s=30, alpha=0.7)
    lims = [0, max(mae_uni.max(), mae_bidir.max()) * 1.05]
    ax3.plot(lims, lims, "k--", label="unidir = bidir")
    ax3.set_xlabel("MAE bidir (pu)")
    ax3.set_ylabel("MAE unidir (pu)")
    ax3.set_title("Per-node MAE: unidir vs bidir (points above diagonal = unidir worse)")
    for i in worst_unidir[:6]:
        ax3.annotate(node_names[i], (mae_bidir[i], mae_uni[i]), fontsize=8, alpha=0.9)
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "mae_scatter_unidir_vs_bidir.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter -> {p3}")
    print("Done.")


if __name__ == "__main__":
    main()
