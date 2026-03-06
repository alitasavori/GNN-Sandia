"""
Compare two GNN .pt models on a 24h daily voltage profile using the same average scaled
values as in dataset generation (new DSS). Produces:
  1) Daily voltage plots at a user-provided list of nodes (OpenDSS vs Model1 vs Model2).
  2) Daily voltage plots at the nodes where the two models differ the most (|MAE1 - MAE2|).
  3) A CSV and printed list of nodes ranked by how much the two models differ.

Scaled baseline (from run_injection_dataset.BASELINE, used in dataset generation):
  P_load_total_kw=849.12, Q_load_total_kvar=501.12, P_pv_total_kw=1400.0

Usage (from repo root):
  python compare_two_models_daily.py <ckpt1.pt> <ckpt2.pt> [--nodes 840.1 848.2 ...] [--top-k 5] [--output-dir DIR]
Example:
  python compare_two_models_daily.py models_gnn2/injection/best.pt models_gnn2/loadtype/best.pt --nodes 840.1 890.1 --top-k 5
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, NPTS, STEP_MIN,
    build_bus_to_phases_from_master_nodes,
    build_gnn_x_original,
    build_gnn_x_injection,
    build_gnn_x_loadtype,
    build_gnn_x_loadtype_per_type,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
from run_gnn3_best7_train import PFIdentityGNN

os.chdir(BASE_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR_LOADTYPE = os.path.join("datasets_gnn2", "loadtype")
DEFAULT_OUTPUT_DIR = "gnn3_best7_output"

# Use the same average scaled values as in dataset generation (run_injection_dataset.BASELINE)
P_BASE = inj.BASELINE["P_load_total_kw"]      # 849.12
Q_BASE = inj.BASELINE["Q_load_total_kvar"]    # 501.12
PV_BASE = inj.BASELINE["P_pv_total_kw"]       # 1400.0


def build_x_for_model(node_in_dim, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type,
                      P_grid, Q_grid, node_to_electrical_dist=None, p_sys_balance=None, q_sys_balance=None):
    """Build node features from node_in_dim (2=injection, 3=original, 10=loadtype per-type, 13=loadtype full)."""
    if node_in_dim == 2:
        return build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
    elif node_in_dim == 3:
        return build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv)
    elif node_in_dim == 10:
        return build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv)
    elif node_in_dim == 13:
        if node_to_electrical_dist is None or p_sys_balance is None or q_sys_balance is None:
            raise ValueError("node_to_electrical_dist, p_sys_balance, q_sys_balance required for node_in_dim=13")
        return build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                    node_to_electrical_dist, p_sys_balance, q_sys_balance)
    else:
        raise ValueError(f"Unsupported node_in_dim={node_in_dim}; use 2, 3, 10, or 13.")


def _resolve_node_list(ckpt_path):
    """Get node list from checkpoint's dataset dir or fallback to loadtype."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    dataset_dir = cfg.get("dataset", DIR_LOADTYPE)
    node_index_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(node_index_csv):
        node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    return master_df["node"].astype(str).tolist()


def run_24h_two_models(ckpt_1_path, ckpt_2_path, node_names_master, edge_csv_dist):
    """Run 24h profile for all nodes using P_BASE, Q_BASE, PV_BASE (scaled baseline). Returns (t_hours, V_dss, V_1, V_2)."""
    model_1, static_1 = load_model_for_inference(ckpt_1_path, device=DEVICE)
    model_2, static_2 = load_model_for_inference(ckpt_2_path, device=DEVICE)
    N = static_1["N"]
    dim_1 = int(static_1["config"]["node_in_dim"])
    dim_2 = int(static_2["config"]["node_in_dim"])

    ei = static_1["edge_index"].to(DEVICE)
    ea = static_1["edge_attr"].to(DEVICE)
    eid = static_1["edge_id"].to(DEVICE)

    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)

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

    V_dss = np.full((NPTS, N), np.nan)
    V_1 = np.full((NPTS, N), np.nan)
    V_2 = np.full((NPTS, N), np.nan)

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
        P_grid = sum_p_load - sum_p_pv
        Q_grid = sum_q_load - sum(CAP_Q_KVAR.values())
        p_sys_balance = sum_p_load - sum_p_pv
        q_sys_balance = sum_q_load - sum(CAP_Q_KVAR.values())

        kw = dict(node_names_master=node_names_master, busphP_load=busphP_load, busphQ_load=busphQ_load,
                  busphP_pv=busphP_pv, busph_per_type=busph_per_type, P_grid=P_grid, Q_grid=Q_grid,
                  node_to_electrical_dist=node_to_electrical_dist, p_sys_balance=p_sys_balance, q_sys_balance=q_sys_balance)
        X_1 = build_x_for_model(dim_1, **kw)
        X_2 = build_x_for_model(dim_2, **kw)

        x_1 = torch.tensor(X_1, dtype=torch.float32, device=DEVICE)
        x_2 = torch.tensor(X_2, dtype=torch.float32, device=DEVICE)
        g_1 = Data(x=x_1, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_2 = Data(x=x_2, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        with torch.no_grad():
            V_1[t, :] = model_1(g_1)[:, 0].cpu().numpy()
            V_2[t, :] = model_2(g_2)[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, V_dss, V_1, V_2


def main():
    parser = argparse.ArgumentParser(description="Compare two GNN models on 24h profile with scaled baseline (849.12/501.12/1400).")
    parser.add_argument("ckpt1", help="Path to first .pt checkpoint")
    parser.add_argument("ckpt2", help="Path to second .pt checkpoint")
    parser.add_argument("--nodes", nargs="*", default=[], help="Node names (e.g. 840.1 848.2) to plot daily profile")
    parser.add_argument("--top-k", type=int, default=5, help="Number of worst-difference nodes to plot (default 5)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and plots")
    parser.add_argument("--label1", default=None, help="Label for model 1 (default: basename of ckpt1)")
    parser.add_argument("--label2", default=None, help="Label for model 2 (default: basename of ckpt2)")
    args = parser.parse_args()

    ckpt_1_path = os.path.abspath(args.ckpt1)
    ckpt_2_path = os.path.abspath(args.ckpt2)
    for p in (ckpt_1_path, ckpt_2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    label_1 = args.label1 or os.path.splitext(os.path.basename(ckpt_1_path))[0]
    label_2 = args.label2 or os.path.splitext(os.path.basename(ckpt_2_path))[0]

    node_names_master = _resolve_node_list(ckpt_1_path)
    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join(os.path.dirname(ckpt_1_path), "..", "loadtype", "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join("datasets_gnn2", "loadtype", "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        raise FileNotFoundError("Could not find gnn_edges_phase_static.csv (e.g. in datasets_gnn2/loadtype).")

    print(f"Scaled baseline: P_load={P_BASE} kW, Q_load={Q_BASE} kVAR, P_pv={PV_BASE} kW (dataset-generation values)")
    print(f"Running 24h profile: {label_1} vs {label_2}...")
    t_hours, V_dss, V_1, V_2 = run_24h_two_models(ckpt_1_path, ckpt_2_path, node_names_master, edge_csv)
    N = len(node_names_master)

    mae_1 = np.full(N, np.nan)
    mae_2 = np.full(N, np.nan)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_1[i] = np.mean(np.abs(V_dss[ok, i] - V_1[ok, i]))
            mae_2[i] = np.mean(np.abs(V_dss[ok, i] - V_2[ok, i]))
    mae_diff = np.abs(mae_1 - mae_2)
    order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
    worst_indices = [int(idx) for idx in order if np.isfinite(mae_diff[idx])][: args.top_k]

    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    df = pd.DataFrame({
        "node": node_names_master,
        "mae_1": mae_1, "mae_2": mae_2, "mae_diff": mae_diff,
    }).sort_values("mae_diff", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{label_1}_vs_{label_2}".replace(" ", "_")
    csv_path = os.path.join(args.output_dir, f"compare_mae_per_node_{run_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    print(f"\nTop {args.top_k} nodes where the two models differ most (|MAE1 - MAE2|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: {label_1} MAE={mae_1[idx]:.4f} | {label_2} MAE={mae_2[idx]:.4f} | diff={mae_diff[idx]:.4f}")

    plot_nodes = list(args.nodes)
    for idx in worst_indices:
        n = node_names_master[idx]
        if n not in plot_nodes:
            plot_nodes.append(n)

    for idx in [node_to_idx[n] for n in plot_nodes if n in node_to_idx]:
        n = node_names_master[idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_1[:, idx], color="orange", linestyle="--", label=f"{label_1} (MAE={mae_1[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_2[:, idx], "g:", label=f"{label_2} (MAE={mae_2[idx]:.4f})", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {n} (scaled baseline {P_BASE:.0f}/{Q_BASE:.0f}/{PV_BASE:.0f} kW/kVAR/kW)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        fname = f"compare_24h_{run_name}_{n.replace('.', '_')}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
