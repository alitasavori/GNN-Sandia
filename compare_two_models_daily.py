"""
Compare two GNN .pt models on a 24h daily voltage profile using the same average scaled
values as in dataset generation (new DSS). Produces:
  1) Daily voltage plots at a user-provided list of nodes (OpenDSS vs Model1 vs Model2).
  2) Daily voltage plots at the nodes where the two models differ the most (|MAE1 - MAE2|).
  3) A CSV and printed list of nodes ranked by how much the two models differ.

Metrics reported:
  - Training (test set): MAE/RMSE from the checkpoint (held-out test set at best_epoch).
  - This run (single 24h day): MAE/RMSE over the one day we plot (scaled baseline, all nodes).

The pickle (--save-results) stores the single-day comparison plus training metrics when
present in the checkpoint; it does not store full training history.

Scaled baseline (from run_injection_dataset.BASELINE, used in dataset generation):
  P_load_total_kw=849.12, Q_load_total_kvar=501.12, P_pv_total_kw=975.0

Usage (from repo root):
  python compare_two_models_daily.py <ckpt1.pt> <ckpt2.pt> [--nodes 840.1 848.2 ...] [--top-k 5] [--output-dir DIR]
  Use --top-k 0 to plot all nodes (89).
Example:
  python compare_two_models_daily.py models_gnn2/injection/best.pt models_gnn2/loadtype/best.pt --nodes 840.1 890.1 --top-k 5
"""
import argparse
import importlib
import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
if not hasattr(inj, "total_cap_q_kvar") or not hasattr(inj, "cap_q_kvar_per_node"):
    inj = importlib.reload(inj)
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, NPTS, STEP_MIN,
    build_gnn_x_original,
    build_gnn_x_injection,
    build_gnn_x_loadtype,
    build_gnn_x_loadtype_per_type,
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
PV_BASE = inj.BASELINE["P_pv_total_kw"]       # 975.0


def build_x_for_model(node_in_dim, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type,
                      P_grid, Q_grid, node_to_electrical_dist=None, p_sys_balance=None, q_sys_balance=None,
                      busphQ_pv=None):
    """Build node features from node_in_dim. Column order must match run_gnn3_best7_train feature_cols:
      node_in_dim=2: INJECTION_FEAT  [p_inj_kw, q_inj_kvar]
      node_in_dim=3: ORIGINAL_FEAT (legacy 3) [p_load_kw, q_load_kvar, p_pv_kw]
      node_in_dim=4: ORIGINAL_FEAT (4)       [p_load_kw, q_load_kvar, p_pv_kw, q_pv_kvar]
      node_in_dim=10: loadtype per-type (m1_p, m1_q, ..., q_cap, p_pv)
      node_in_dim=12: no-global LOADTYPE_FEAT [electrical_distance_ohm, m1_p_kw, m1_q_kvar, m2_p_kw, m2_q_kvar, m4_p_kw, m4_q_kvar, m5_p_kw, m5_q_kvar, q_cap_kvar, p_pv_kw, q_pv_kvar]
      node_in_dim=14: current LOADTYPE_FEAT [electrical_distance_ohm, m1_p_kw, m1_q_kvar, m2_p_kw, m2_q_kvar, m4_p_kw, m4_q_kvar, m5_p_kw, m5_q_kvar, q_cap_kvar, p_pv_kw, q_pv_kvar, p_sys_balance_kw, q_sys_balance_kvar]
      node_in_dim=15: no-global LOADTYPE_FEAT + 3-dim phase one-hot
      node_in_dim=17: current LOADTYPE_FEAT + 3-dim phase one-hot
    """
    if node_in_dim == 2:
        return build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid, busphQ_pv=busphQ_pv)
    elif node_in_dim == 3:
        return build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv, busphQ_pv=None)
    elif node_in_dim == 4:
        return build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv, busphQ_pv=busphQ_pv)
    elif node_in_dim == 10:
        return build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv)
    elif node_in_dim == 12:
        if node_to_electrical_dist is None or busphQ_pv is None:
            raise ValueError("node_to_electrical_dist and busphQ_pv required for node_in_dim=12")
        return build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                    node_to_electrical_dist, p_sys_balance, q_sys_balance,
                                    busphQ_pv=busphQ_pv, include_globals=False)
    elif node_in_dim == 14:
        if node_to_electrical_dist is None or p_sys_balance is None or q_sys_balance is None or busphQ_pv is None:
            raise ValueError("node_to_electrical_dist, p_sys_balance, q_sys_balance, busphQ_pv required for node_in_dim=14")
        return build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                    node_to_electrical_dist, p_sys_balance, q_sys_balance,
                                    busphQ_pv=busphQ_pv)
    elif node_in_dim == 15:
        if node_to_electrical_dist is None or busphQ_pv is None:
            raise ValueError("node_to_electrical_dist and busphQ_pv required for node_in_dim=15")
        X_12 = build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                    node_to_electrical_dist, p_sys_balance, q_sys_balance,
                                    busphQ_pv=busphQ_pv, include_globals=False)
        phase_map = np.array([int(n.split(".")[-1]) - 1 for n in node_names_master], dtype=np.int64)
        ph_oh = np.eye(3, dtype=np.float32)[phase_map]
        return np.concatenate([X_12, ph_oh], axis=-1)
    elif node_in_dim == 17:
        if node_to_electrical_dist is None or p_sys_balance is None or q_sys_balance is None or busphQ_pv is None:
            raise ValueError("node_to_electrical_dist, p_sys_balance, q_sys_balance, busphQ_pv required for node_in_dim=17")
        X_14 = build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                    node_to_electrical_dist, p_sys_balance, q_sys_balance,
                                    busphQ_pv=busphQ_pv)
        phase_map = np.array([int(n.split(".")[-1]) - 1 for n in node_names_master], dtype=np.int64)
        ph_oh = np.eye(3, dtype=np.float32)[phase_map]
        return np.concatenate([X_14, ph_oh], axis=-1)
    else:
        raise ValueError(
            f"Unsupported node_in_dim={node_in_dim}. "
            "Only the strict load-type inference workflow is supported now: "
            "use 12, 14, 15, or 17 for load-type checkpoints."
        )


def _resolve_node_list(ckpt_path, expected_n=89):
    """Get the 89-node list in the same order as training (excludes upstream buses)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    dataset_dir = cfg.get("dataset", DIR_LOADTYPE)
    node_csv = os.path.join(dataset_dir, "gnn_node_features_and_targets.csv")
    master_path = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master_path):
        master_path = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    if not os.path.exists(node_csv):
        node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    if not os.path.exists(node_csv) or not os.path.exists(master_path):
        raise FileNotFoundError(
            f"Need {node_csv} and {master_path} to resolve 89-node list. Run dataset generation first."
        )
    df_n = pd.read_csv(node_csv)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    kept_node_ids = sorted(df_n["node_idx"].unique())
    N = len(kept_node_ids)
    if N != expected_n:
        raise RuntimeError(f"Expected {expected_n} nodes per sample, got {N}. Check dataset.")
    master_df = pd.read_csv(master_path)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    old_to_name = master_df.set_index("node_idx")["node"].astype(str).to_dict()
    return [old_to_name[old] for old in kept_node_ids]


def _get_full_master_node_list(dataset_dir):
    """Get the full 95-node list from gnn_node_index_master.csv (includes sourcebus, 800).
    Used for electrical distance computation so source_nodes is correct (matches dataset generation)."""
    master_path = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master_path):
        master_path = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Need {master_path} for full node list. Run dataset generation first.")
    master_df = pd.read_csv(master_path)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx")
    return master_df["node"].astype(str).tolist()


def run_24h_two_models(ckpt_1_path, ckpt_2_path, node_names_master, edge_csv_dist, dataset_dir=None):
    """Run 24h profile for all nodes using P_BASE, Q_BASE, PV_BASE (scaled baseline). Returns (t_hours, V_dss, V_1, V_2)."""
    model_1, static_1 = load_model_for_inference(ckpt_1_path, device=DEVICE)
    model_2, static_2 = load_model_for_inference(ckpt_2_path, device=DEVICE)
    N = static_1["N"]
    dim_1 = int(static_1["config"]["node_in_dim"])
    dim_2 = int(static_2["config"]["node_in_dim"])

    ei_1 = static_1["edge_index"].to(DEVICE)
    ea_1 = static_1["edge_attr"].to(DEVICE)
    eid_1 = static_1["edge_id"].to(DEVICE)
    ei_2 = static_2["edge_index"].to(DEVICE)
    ea_2 = static_2["edge_attr"].to(DEVICE)
    eid_2 = static_2["edge_id"].to(DEVICE)

    # Use full 95-node master list for electrical distance (includes sourcebus, 800) so source_nodes is correct.
    # Matches dataset generation: run_loadtype_dataset uses get_all_bus_phase_nodes() (95 nodes) for this.
    if dataset_dir is None:
        dataset_dir = os.path.dirname(edge_csv_dist)
    full_node_list = _get_full_master_node_list(dataset_dir)
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_node_list, edge_csv_dist)

    dss_path = inj.compile_once()
    inj.setup_daily()
    # Use same CSV resolution and profile reader as dataset generation (run_loadtype_dataset)
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    # Use same bus_to_phases as dataset generation (from get_all_bus_phase_nodes)
    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
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
        busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        # Use same voltage reader as dataset generation (get_all_node_voltage_pu_and_angle_filtered)
        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        for i in range(len(node_names_master)):
            V_dss[t, i] = float(vmag_m[i])

        sum_p_load = float(sum(busphP_load.values()))
        sum_q_load = float(sum(busphQ_load.values()))
        sum_p_pv_actual = float(sum(busphP_pv_actual.values()))
        sum_q_pv_actual = float(sum(busphQ_pv_actual.values()))
        sum_q_cap = inj.total_cap_q_kvar(node_names_master)
        pwr = inj.dss.Circuit.TotalPower()
        P_grid = -float(pwr[0])
        Q_grid = -float(pwr[1])
        p_sys_balance = sum_p_load - sum_p_pv_actual
        q_sys_balance = sum_q_load - sum_q_pv_actual - sum_q_cap

        kw = dict(node_names_master=node_names_master, busphP_load=busphP_load, busphQ_load=busphQ_load,
                  busphP_pv=busphP_pv_actual, busphQ_pv=busphQ_pv_actual, busph_per_type=busph_per_type,
                  P_grid=P_grid, Q_grid=Q_grid, node_to_electrical_dist=node_to_electrical_dist,
                  p_sys_balance=p_sys_balance, q_sys_balance=q_sys_balance)
        X_1 = build_x_for_model(dim_1, **kw)
        X_2 = build_x_for_model(dim_2, **kw)

        x_1 = torch.tensor(X_1, dtype=torch.float32, device=DEVICE)
        x_2 = torch.tensor(X_2, dtype=torch.float32, device=DEVICE)
        g_1 = Data(x=x_1, edge_index=ei_1, edge_attr=ea_1, edge_id=eid_1, num_nodes=N)
        g_2 = Data(x=x_2, edge_index=ei_2, edge_attr=ea_2, edge_id=eid_2, num_nodes=N)
        with torch.no_grad():
            V_1[t, :] = model_1(g_1)[:, 0].cpu().numpy()
            V_2[t, :] = model_2(g_2)[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, V_dss, V_1, V_2


def run_one_step_and_return_features(ckpt_1_path, ckpt_2_path, node_names_master, edge_csv_dist, dataset_dir, t=0):
    """Run a single timestep t, build features for both models; return (X_1, X_2, vmag_row, dim_1, dim_2) for debugging. Does not load models."""
    full_node_list = _get_full_master_node_list(dataset_dir or os.path.dirname(edge_csv_dist))
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_node_list, edge_csv_dist)
    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)
    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)
    inj.set_time_index(t)
    _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
        P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
        mL_t=float(mL[t]), mPV_t=float(mPV[t]),
        loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
        sigma_load=0.0, sigma_pv=0.0, rng=rng,
    )
    inj.dss.Solution.Solve()
    busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
    vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
    sum_p_load = float(sum(busphP_load.values()))
    sum_q_load = float(sum(busphQ_load.values()))
    sum_p_pv_actual = float(sum(busphP_pv_actual.values()))
    sum_q_pv_actual = float(sum(busphQ_pv_actual.values()))
    sum_q_cap = inj.total_cap_q_kvar(node_names_master)
    pwr = inj.dss.Circuit.TotalPower()
    P_grid = -float(pwr[0])
    Q_grid = -float(pwr[1])
    p_sys_balance = sum_p_load - sum_p_pv_actual
    q_sys_balance = sum_q_load - sum_q_pv_actual - sum_q_cap
    kw = dict(node_names_master=node_names_master, busphP_load=busphP_load, busphQ_load=busphQ_load,
              busphP_pv=busphP_pv_actual, busphQ_pv=busphQ_pv_actual, busph_per_type=busph_per_type,
              P_grid=P_grid, Q_grid=Q_grid, node_to_electrical_dist=node_to_electrical_dist,
              p_sys_balance=p_sys_balance, q_sys_balance=q_sys_balance)
    ckpt_1 = torch.load(ckpt_1_path, map_location="cpu", weights_only=False)
    ckpt_2 = torch.load(ckpt_2_path, map_location="cpu", weights_only=False)
    dim_1 = int(ckpt_1.get("config", {}).get("node_in_dim", 2))
    dim_2 = int(ckpt_2.get("config", {}).get("node_in_dim", 2))
    X_1 = build_x_for_model(dim_1, **kw)
    X_2 = build_x_for_model(dim_2, **kw)
    return X_1, X_2, np.array([float(x) for x in vmag_m]), dim_1, dim_2


def run_comparison_for_notebook(ckpt_1_path, ckpt_2_path, nodes_to_plot=None, top_k=5):
    """
    Run 24h comparison and return data for notebook use (no PNG/CSV saved).
    Returns: t_hours, node_names, V_dss, V_1, V_2, df_mae, label_1, label_2, worst_indices, summary
    where summary is a dict with mae_1, rmse_1, mae_2, rmse_2, n_points (this run), and train_mae_1, train_rmse_1, train_epoch_1, train_mae_2, train_rmse_2, train_epoch_2 (from checkpoint when present).
    """
    nodes_to_plot = nodes_to_plot or []
    ckpt_1_path = os.path.abspath(ckpt_1_path)
    ckpt_2_path = os.path.abspath(ckpt_2_path)
    for p in (ckpt_1_path, ckpt_2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    label_1 = os.path.splitext(os.path.basename(ckpt_1_path))[0]
    label_2 = os.path.splitext(os.path.basename(ckpt_2_path))[0]

    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join(os.path.dirname(ckpt_1_path), "..", "loadtype", "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join("datasets_gnn2", "loadtype", "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        raise FileNotFoundError("Could not find gnn_edges_phase_static.csv (e.g. in datasets_gnn2/loadtype).")

    node_names_master = _resolve_node_list(ckpt_1_path)
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
    n_worst = N if top_k == 0 else max(1, top_k)
    worst_indices = [int(idx) for idx in order if np.isfinite(mae_diff[idx])][:n_worst]

    mask = np.isfinite(V_dss)
    global_mae_1 = np.mean(np.abs(V_dss[mask] - V_1[mask]))
    global_rmse_1 = np.sqrt(np.mean((V_dss[mask] - V_1[mask]) ** 2))
    global_mae_2 = np.mean(np.abs(V_dss[mask] - V_2[mask]))
    global_rmse_2 = np.sqrt(np.mean((V_dss[mask] - V_2[mask]) ** 2))

    ckpt_1 = torch.load(ckpt_1_path, map_location="cpu", weights_only=False)
    ckpt_2 = torch.load(ckpt_2_path, map_location="cpu", weights_only=False)
    summary = {
        "mae_1": global_mae_1, "rmse_1": global_rmse_1,
        "mae_2": global_mae_2, "rmse_2": global_rmse_2,
        "n_points": int(np.sum(mask)),
        "train_mae_1": ckpt_1.get("best_mae"), "train_rmse_1": ckpt_1.get("best_rmse"), "train_epoch_1": ckpt_1.get("best_epoch"),
        "train_mae_2": ckpt_2.get("best_mae"), "train_rmse_2": ckpt_2.get("best_rmse"), "train_epoch_2": ckpt_2.get("best_epoch"),
    }

    df_mae = pd.DataFrame({
        "node": node_names_master,
        "mae_1": mae_1, "mae_2": mae_2, "mae_diff": mae_diff,
    }).sort_values("mae_diff", ascending=False)

    return t_hours, node_names_master, V_dss, V_1, V_2, df_mae, label_1, label_2, worst_indices, summary


def main():
    parser = argparse.ArgumentParser(description="Compare two GNN models on 24h profile with scaled baseline (849.12/501.12/975).")
    parser.add_argument("ckpt1", help="Path to first .pt checkpoint")
    parser.add_argument("ckpt2", help="Path to second .pt checkpoint")
    parser.add_argument("--nodes", nargs="*", default=[], help="Node names (e.g. 840.1 848.2) to plot daily profile")
    parser.add_argument("--top-k", type=int, default=5, help="Number of worst-difference nodes to plot (default 5). Use 0 for all nodes.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and plots")
    parser.add_argument("--label1", default=None, help="Label for model 1 (default: basename of ckpt1)")
    parser.add_argument("--label2", default=None, help="Label for model 2 (default: basename of ckpt2)")
    parser.add_argument("--save-results", metavar="PATH", help="Save results to pickle for notebook loading (avoids torch import in notebook)")
    parser.add_argument("--no-save-png", action="store_true", help="Do not save PNG plots (for notebook: load pickle and plot inline)")
    parser.add_argument("--debug-features", action="store_true", help="Dump built features at t=0 to CSV in output-dir for inspection")
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
    dataset_dir = os.path.dirname(edge_csv)

    # Training metrics (from checkpoint: held-out test set at best_epoch)
    ckpt_1 = torch.load(ckpt_1_path, map_location="cpu", weights_only=False)
    ckpt_2 = torch.load(ckpt_2_path, map_location="cpu", weights_only=False)
    train_mae_1 = ckpt_1.get("best_mae")
    train_rmse_1 = ckpt_1.get("best_rmse")
    train_epoch_1 = ckpt_1.get("best_epoch")
    train_mae_2 = ckpt_2.get("best_mae")
    train_rmse_2 = ckpt_2.get("best_rmse")
    train_epoch_2 = ckpt_2.get("best_epoch")

    if args.debug_features:
        print("Debug: building features at t=0 and saving to CSV...")
        X_1, X_2, vmag_row, dim_1, dim_2 = run_one_step_and_return_features(
            ckpt_1_path, ckpt_2_path, node_names_master, edge_csv, dataset_dir, t=0
        )
        os.makedirs(args.output_dir, exist_ok=True)
        run_name = f"{label_1}_vs_{label_2}".replace(" ", "_")
        debug_path = os.path.join(args.output_dir, f"debug_features_t0_{run_name}.csv")
        cols_1 = [f"m1_f{j}" for j in range(dim_1)]
        cols_2 = [f"m2_f{j}" for j in range(dim_2)]
        df_debug = pd.DataFrame({"node": node_names_master, "V_dss_pu": vmag_row})
        for j in range(dim_1):
            df_debug[cols_1[j]] = X_1[:, j]
        for j in range(dim_2):
            df_debug[cols_2[j]] = X_2[:, j]
        df_debug.to_csv(debug_path, index=False)
        print(f"Saved -> {debug_path}")
        return

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
    n_worst = N if args.top_k == 0 else args.top_k
    worst_indices = [int(idx) for idx in order if np.isfinite(mae_diff[idx])][:max(1, n_worst)]

    # Overall metrics (whole day, all nodes): use only points where OpenDSS voltage is finite
    mask = np.isfinite(V_dss)
    n_pts = int(np.sum(mask))
    global_mae_1 = np.mean(np.abs(V_dss[mask] - V_1[mask]))
    global_rmse_1 = np.sqrt(np.mean((V_dss[mask] - V_1[mask]) ** 2))
    global_mae_2 = np.mean(np.abs(V_dss[mask] - V_2[mask]))
    global_rmse_2 = np.sqrt(np.mean((V_dss[mask] - V_2[mask]) ** 2))

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

    print(f"\nTraining (test set, from checkpoint):")
    if train_mae_1 is not None and train_rmse_1 is not None:
        ep1 = f" (epoch {train_epoch_1})" if train_epoch_1 is not None else ""
        print(f"  {label_1}:  MAE = {float(train_mae_1):.5f} pu   RMSE = {float(train_rmse_1):.5f} pu{ep1}")
    else:
        print(f"  {label_1}:  (not in checkpoint)")
    if train_mae_2 is not None and train_rmse_2 is not None:
        ep2 = f" (epoch {train_epoch_2})" if train_epoch_2 is not None else ""
        print(f"  {label_2}:  MAE = {float(train_mae_2):.5f} pu   RMSE = {float(train_rmse_2):.5f} pu{ep2}")
    else:
        print(f"  {label_2}:  (not in checkpoint)")

    print(f"\nThis run (single 24h day, all nodes, n={n_pts} points):")
    print(f"  {label_1}:  MAE = {global_mae_1:.5f} pu   RMSE = {global_rmse_1:.5f} pu")
    print(f"  {label_2}:  MAE = {global_mae_2:.5f} pu   RMSE = {global_rmse_2:.5f} pu")
    summary_path = os.path.join(args.output_dir, f"compare_summary_{run_name}.csv")
    summary_row = {
        "label_1": label_1, "label_2": label_2,
        "train_mae_1": train_mae_1, "train_rmse_1": train_rmse_1, "train_epoch_1": train_epoch_1,
        "train_mae_2": train_mae_2, "train_rmse_2": train_rmse_2, "train_epoch_2": train_epoch_2,
        "mae_1": global_mae_1, "rmse_1": global_rmse_1,
        "mae_2": global_mae_2, "rmse_2": global_rmse_2,
        "n_points": n_pts,
    }
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")

    if args.save_results:
        summary_dict = {
            "mae_1": global_mae_1, "rmse_1": global_rmse_1,
            "mae_2": global_mae_2, "rmse_2": global_rmse_2,
            "n_points": n_pts,
            "train_mae_1": train_mae_1, "train_rmse_1": train_rmse_1, "train_epoch_1": train_epoch_1,
            "train_mae_2": train_mae_2, "train_rmse_2": train_rmse_2, "train_epoch_2": train_epoch_2,
        }
        with open(args.save_results, "wb") as f:
            pickle.dump((t_hours, node_names_master, V_dss, V_1, V_2, df, label_1, label_2, worst_indices, summary_dict), f)
        print(f"Saved results -> {args.save_results}")

    print(f"\nTop {len(worst_indices)} nodes where the two models differ most (|MAE1 - MAE2|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: {label_1} MAE={mae_1[idx]:.4f} | {label_2} MAE={mae_2[idx]:.4f} | diff={mae_diff[idx]:.4f}")

    if not args.no_save_png:
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
