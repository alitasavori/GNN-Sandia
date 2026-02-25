"""
Standalone evaluation: load two trained models, run 24h profile for all nodes,
find the node where the two models differ most (by |MAE_1 - MAE_2|), and plot
the voltage profile for that node (OpenDSS vs Model 1 vs Model 2).
Uses PV_SCALE=1.0 to stay in distribution with training.
Run from repo root. Requires: block_injection_features.pt, block_loadtype_per_type.pt
(Train with: %run run_injection_vs_loadtype_per_type.py)
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
    build_bus_to_phases_from_master_nodes, build_gnn_x_injection,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
try:
    from run_gnn3_overlay_7 import build_gnn_x_loadtype_per_type
except ImportError:
    def build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv):
        """Load-type per-type (10 feat): m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv."""
        X = np.zeros((len(node_names_master), 10), dtype=np.float32)
        for i, n in enumerate(node_names_master):
            bus, phs = n.split(".")
            ph = int(phs)
            m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
            m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
            m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
            m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
            m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
            m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
            m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
            m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))
            q_cap = float(CAP_Q_KVAR.get(bus, 0.0))
            p_pv = float(busphP_pv.get((bus, ph), 0.0))
            X[i, 0], X[i, 1] = m1_p, m1_q
            X[i, 2], X[i, 3] = m2_p, m2_q
            X[i, 4], X[i, 5] = m4_p, m4_q
            X[i, 6], X[i, 7] = m5_p, m5_q
            X[i, 8] = q_cap
            X[i, 9] = p_pv
        return X

os.chdir(BASE_DIR)

OUTPUT_DIR = "gnn3_best7_output"
DIR_LOADTYPE = "gnn_samples_loadtype_full"
CKPT_1 = os.path.join(OUTPUT_DIR, "block_injection_features.pt")
CKPT_2 = os.path.join(OUTPUT_DIR, "block_loadtype_per_type.pt")
LABEL_1 = "Injection (p_inj, q_inj)"
LABEL_2 = "Loadtype per-type (m1..m5, q_cap, p_pv)"
PV_SCALE = 1.0  # PV multiplier = 1
TOP_N_WORST = 5  # Number of worst nodes to plot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_x_for_model(node_in_dim, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type, P_grid, Q_grid):
    """Build node features based on node_in_dim (2=injection, 10=loadtype per-type)."""
    if node_in_dim == 2:
        return build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
    elif node_in_dim == 10:
        return build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv)
    else:
        raise ValueError(f"Unknown node_in_dim={node_in_dim}; expected 2 or 10.")


def run_24h_all_nodes(ckpt_1_path, ckpt_2_path):
    """Run 24h profile for all nodes. Returns (t_hours, node_names, V_dss, V_1, V_2)."""
    model_1, static_1 = load_model_for_inference(ckpt_1_path, device=DEVICE)
    model_2, static_2 = load_model_for_inference(ckpt_2_path, device=DEVICE)
    N = static_1["N"]
    dim_1 = int(static_1["config"]["node_in_dim"])
    dim_2 = int(static_2["config"]["node_in_dim"])

    ei = static_1["edge_index"].to(DEVICE)
    ea = static_1["edge_attr"].to(DEVICE)
    eid = static_1["edge_id"].to(DEVICE)

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
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

    V_dss = np.full((NPTS, N), np.nan)
    V_1 = np.full((NPTS, N), np.nan)
    V_2 = np.full((NPTS, N), np.nan)

    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE * PV_SCALE,
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

        X_1 = build_x_for_model(dim_1, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type, P_grid, Q_grid)
        X_2 = build_x_for_model(dim_2, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type, P_grid, Q_grid)

        x_1 = torch.tensor(X_1, dtype=torch.float32, device=DEVICE)
        x_2 = torch.tensor(X_2, dtype=torch.float32, device=DEVICE)
        g_1 = Data(x=x_1, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_2 = Data(x=x_2, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        with torch.no_grad():
            V_1[t, :] = model_1(g_1)[:, 0].cpu().numpy()
            V_2[t, :] = model_2(g_2)[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, node_names_master, V_dss, V_1, V_2


def main():
    if not os.path.exists(CKPT_1) or not os.path.exists(CKPT_2):
        raise FileNotFoundError(f"Missing checkpoints. Train first: {CKPT_1}, {CKPT_2}")

    print("Running 24h profile for all nodes (Injection vs Loadtype per-type, PV=1.0×)...")
    t_hours, node_names, V_dss, V_1, V_2 = run_24h_all_nodes(CKPT_1, CKPT_2)
    N = len(node_names)

    mae_1 = np.zeros(N)
    mae_2 = np.zeros(N)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_1[i] = np.mean(np.abs(V_dss[ok, i] - V_1[ok, i]))
            mae_2[i] = np.mean(np.abs(V_dss[ok, i] - V_2[ok, i]))
        else:
            mae_1[i] = np.nan
            mae_2[i] = np.nan

    mae_diff = np.abs(mae_1 - mae_2)
    order = np.argsort(-mae_diff)
    worst_indices = order[:TOP_N_WORST]

    df = pd.DataFrame({
        "node": node_names,
        "mae_1": mae_1, "mae_2": mae_2,
        "mae_diff": mae_diff,
    }).sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_injection_vs_loadtype_per_type.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top {TOP_N_WORST} worst nodes (|MAE_1 - MAE_2|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names[idx]}: {LABEL_1} MAE={mae_1[idx]:.4f} | {LABEL_2} MAE={mae_2[idx]:.4f}")
    print(f"Saved -> {csv_path}")

    for k, idx in enumerate(worst_indices):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_1[:, idx], color="orange", linestyle="--", label=f"{LABEL_1} (MAE={mae_1[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_2[:, idx], "g:", label=f"{LABEL_2} (MAE={mae_2[idx]:.4f})", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {node_names[idx]} (worst #{k+1}, PV={PV_SCALE:.1f}×)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"overlay_24h_injection_vs_loadtype_per_type_worst_{k+1}_{node_names[idx].replace('.', '_')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
