"""
Standalone evaluation: load two trained models, run 24h profile for all nodes,
find the node where the two models differ most (by |MAE_1 - MAE_2|), and plot
the voltage profile for that node (OpenDSS vs Model 1 vs Model 2).
Uses PV_SCALE=1.0 to stay in distribution with training.
Run from repo root. Set PRESET or pass as argv[1]:
  injection_vs_loadtype | loadtype_full_vs_per_type | phase_onehot_vs_subgraph | 14feat_vs_phase_a
For phase_onehot_vs_subgraph, 14feat_vs_phase_a: model 2 is phase A only; worst nodes restricted to phase A.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, NPTS, P_BASE, Q_BASE, PV_BASE, STEP_MIN,
    build_bus_to_phases_from_master_nodes, build_gnn_x_injection, build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
from run_deltav_dataset import _apply_snapshot_zero_pv
try:
    from gnn_narrow_exploration import load_phase_mapping
except ImportError:
    load_phase_mapping = None
from run_gnn3_best7_train import PFIdentityGNN
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


def get_preset_config(preset):
    """Return (CKPT_1, CKPT_2, LABEL_1, LABEL_2) for given preset."""
    if preset == "injection_vs_loadtype":
        return (
            os.path.join(OUTPUT_DIR, "block_injection_features.pt"),
            os.path.join(OUTPUT_DIR, "block_loadtype_per_type.pt"),
            "Injection (p_inj, q_inj)",
            "Loadtype per-type (m1..m5, q_cap, p_pv)",
        )
    elif preset == "loadtype_full_vs_per_type":
        return (
            os.path.join(OUTPUT_DIR, "block_loadtype_full.pt"),
            os.path.join(OUTPUT_DIR, "block_loadtype_per_type.pt"),
            "Loadtype full (elec_dist + m1..m5 + p_sys, q_sys)",
            "Loadtype per-type (m1..m5, q_cap, p_pv only)",
        )
    elif preset == "phase_onehot_vs_subgraph":
        return (
            os.path.join(OUTPUT_DIR, "block_phase_onehot.pt"),
            os.path.join(OUTPUT_DIR, "block_phase_subgraph.pt"),
            "Phase one-hot (13+3 feat)",
            "Phase A only (13 feat)",
        )
    elif preset == "14feat_vs_phase_a":
        return (
            os.path.join(OUTPUT_DIR, "block_14feat_vmagzero.pt"),
            os.path.join(OUTPUT_DIR, "block_14feat_phase_a.pt"),
            "14 feat (loadtype+vmag_zero)",
            "Phase A only (13 feat)",
        )
    else:
        raise ValueError(f"Unknown PRESET={preset}; use injection_vs_loadtype | loadtype_full_vs_per_type | phase_onehot_vs_subgraph | 14feat_vs_phase_a")


# Preset: "injection_vs_loadtype" | "loadtype_full_vs_per_type" (override via argv[1])
PRESET = sys.argv[1] if len(sys.argv) > 1 else "loadtype_full_vs_per_type"
CKPT_1, CKPT_2, LABEL_1, LABEL_2 = get_preset_config(PRESET)
PV_SCALE = 1.0  # PV multiplier = 1
TOP_N_WORST = 5  # Number of worst nodes to plot
EXTRA_NODES = ["812.1"] if PRESET == "loadtype_full_vs_per_type" else []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_x_for_model(node_in_dim, node_names_master, busphP_load, busphQ_load, busphP_pv, busph_per_type,
                      P_grid, Q_grid, node_to_electrical_dist=None, p_sys_balance=None, q_sys_balance=None):
    """Build node features based on node_in_dim (2=injection, 10=loadtype per-type, 13=loadtype full)."""
    if node_in_dim == 2:
        return build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
    elif node_in_dim == 10:
        return build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv)
    elif node_in_dim == 13:
        if node_to_electrical_dist is None or p_sys_balance is None or q_sys_balance is None:
            raise ValueError("node_to_electrical_dist, p_sys_balance, q_sys_balance required for node_in_dim=13")
        return build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                   node_to_electrical_dist, p_sys_balance, q_sys_balance)
    else:
        raise ValueError(f"Unknown node_in_dim={node_in_dim}; expected 2, 10, or 13.")


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

    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
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
    return t_hours, node_names_master, V_dss, V_1, V_2


def run_24h_phase_onehot_vs_subgraph(ckpt_1_path, ckpt_2_path):
    """Run 24h for phase one-hot vs phase A only. Model 2 predicts only phase A nodes. Returns (t_hours, node_names, V_dss, V_1, V_2, phase_a_indices)."""
    if load_phase_mapping is None:
        raise ImportError("phase_onehot_vs_subgraph requires gnn_narrow_exploration (load_phase_mapping)")

    ckpt_oh = torch.load(ckpt_1_path, map_location="cpu")
    ckpt_sg = torch.load(ckpt_2_path, map_location="cpu")
    cfg_oh, cfg_sg = ckpt_oh["config"], ckpt_sg["config"]
    N = int(cfg_oh["N"])
    phase_a_indices = ckpt_sg["phase_a_indices"]
    N_phase_a = len(phase_a_indices)

    model_1 = load_model_for_inference(ckpt_1_path, device=DEVICE)[0]
    ei_sg = ckpt_sg["edge_index"].to(DEVICE)
    ea_sg = ckpt_sg["edge_attr"].to(DEVICE)
    eid_sg = ckpt_sg["edge_id"].to(DEVICE)
    model_2 = PFIdentityGNN(num_nodes=N_phase_a, num_edges=int(cfg_sg["E"]), node_in_dim=13, edge_in_dim=2, out_dim=1,
                             node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_2.load_state_dict(ckpt_sg["state_dict"])
    model_2.eval()

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)
    phase_map = load_phase_mapping(DIR_LOADTYPE)
    ph_oh = F.one_hot(phase_map, num_classes=3).float().numpy()

    ei = ckpt_oh["edge_index"].to(DEVICE)
    ea = ckpt_oh["edge_attr"].to(DEVICE)
    eid = ckpt_oh["edge_id"].to(DEVICE)

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
        p_sys_balance = sum_p_load - sum_p_pv
        q_sys_balance = sum_q_load - sum(CAP_Q_KVAR.values())

        X = build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                 node_to_electrical_dist, p_sys_balance, q_sys_balance)
        X_oh = np.concatenate([X, ph_oh], axis=-1).astype(np.float32)
        x_1 = torch.tensor(X_oh, dtype=torch.float32, device=DEVICE)
        x_2_phase_a = torch.tensor(X[phase_a_indices, :], dtype=torch.float32, device=DEVICE)

        g_1 = Data(x=x_1, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_2 = Data(x=x_2_phase_a, edge_index=ei_sg, edge_attr=ea_sg, edge_id=eid_sg, num_nodes=N_phase_a)

        with torch.no_grad():
            V_1[t, :] = model_1(g_1)[:, 0].cpu().numpy()
            pred_2 = model_2(g_2)[:, 0].cpu().numpy()
            for j, gidx in enumerate(phase_a_indices):
                V_2[t, gidx] = pred_2[j]

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, node_names_master, V_dss, V_1, V_2, phase_a_indices


def run_24h_14feat_vs_phase_a(ckpt_1_path, ckpt_2_path):
    """Run 24h for 14feat (loadtype+vmag_zero) vs phase A only. Returns (t_hours, node_names, V_dss, V_1, V_2, phase_a_indices)."""
    if load_phase_mapping is None:
        raise ImportError("14feat_vs_phase_a requires gnn_narrow_exploration (load_phase_mapping)")

    ckpt_14 = torch.load(ckpt_1_path, map_location="cpu")
    ckpt_ph = torch.load(ckpt_2_path, map_location="cpu")
    cfg_14, cfg_ph = ckpt_14["config"], ckpt_ph["config"]
    N = int(cfg_14["N"])
    phase_a_indices = ckpt_ph["phase_a_indices"]
    N_phase_a = len(phase_a_indices)

    model_1 = PFIdentityGNN(num_nodes=N, num_edges=int(cfg_14["E"]), node_in_dim=14, edge_in_dim=2, out_dim=1,
                            node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_1.load_state_dict(ckpt_14["state_dict"])
    model_1.eval()

    ei_ph = ckpt_ph["edge_index"].to(DEVICE)
    ea_ph = ckpt_ph["edge_attr"].to(DEVICE)
    eid_ph = ckpt_ph["edge_id"].to(DEVICE)
    model_2 = PFIdentityGNN(num_nodes=N_phase_a, num_edges=int(cfg_ph["E"]), node_in_dim=13, edge_in_dim=2, out_dim=1,
                            node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_2.load_state_dict(ckpt_ph["state_dict"])
    model_2.eval()

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)

    ei = ckpt_14["edge_index"].to(DEVICE)
    ea = ckpt_14["edge_attr"].to(DEVICE)
    eid = ckpt_14["edge_id"].to(DEVICE)

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

    vmag_zero_precomputed = []
    for t in range(NPTS):
        inj.set_time_index(t)
        _apply_snapshot_zero_pv(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, mL_t=float(mL[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, rng=rng,
        )
        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            vmag_zero_precomputed.append(np.full(N, np.nan, dtype=np.float32))
            continue
        vdict_z = get_all_node_voltage_pu_and_angle_dict()
        vmag_z = np.array([float(vdict_z.get(n, (np.nan, 0))[0]) for n in node_names_master], dtype=np.float32)
        vmag_zero_precomputed.append(vmag_z)
    inj.compile_once()
    inj.setup_daily()

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
        p_sys_balance = sum_p_load - sum_p_pv
        q_sys_balance = sum_q_load - sum(CAP_Q_KVAR.values())

        X = build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv,
                                 node_to_electrical_dist, p_sys_balance, q_sys_balance)
        vmag_zero = vmag_zero_precomputed[t]
        X_14 = np.concatenate([X, vmag_zero[:, None]], axis=-1).astype(np.float32)
        x_1 = torch.tensor(X_14, dtype=torch.float32, device=DEVICE)
        x_2_phase_a = torch.tensor(X[phase_a_indices, :], dtype=torch.float32, device=DEVICE)

        g_1 = Data(x=x_1, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_2 = Data(x=x_2_phase_a, edge_index=ei_ph, edge_attr=ea_ph, edge_id=eid_ph, num_nodes=N_phase_a)

        with torch.no_grad():
            V_1[t, :] = model_1(g_1)[:, 0].cpu().numpy()
            pred_2 = model_2(g_2)[:, 0].cpu().numpy()
            for j, gidx in enumerate(phase_a_indices):
                V_2[t, gidx] = pred_2[j]

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    return t_hours, node_names_master, V_dss, V_1, V_2, phase_a_indices


def main():
    if not os.path.exists(CKPT_1) or not os.path.exists(CKPT_2):
        raise FileNotFoundError(f"Missing checkpoints. Train first: {CKPT_1}, {CKPT_2}")

    print(f"Running 24h profile for all nodes ({LABEL_1} vs {LABEL_2}, PV=1.0×)...")
    if PRESET == "phase_onehot_vs_subgraph":
        t_hours, node_names, V_dss, V_1, V_2, phase_a_indices = run_24h_phase_onehot_vs_subgraph(CKPT_1, CKPT_2)
    elif PRESET == "14feat_vs_phase_a":
        t_hours, node_names, V_dss, V_1, V_2, phase_a_indices = run_24h_14feat_vs_phase_a(CKPT_1, CKPT_2)
    else:
        t_hours, node_names, V_dss, V_1, V_2 = run_24h_all_nodes(CKPT_1, CKPT_2)
        phase_a_indices = None
    N = len(node_names)

    mae_1 = np.full(N, np.nan)
    mae_2 = np.full(N, np.nan)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_1[i] = np.mean(np.abs(V_dss[ok, i] - V_1[ok, i]))
            if phase_a_indices is None or i in phase_a_indices:
                if phase_a_indices is None or np.any(np.isfinite(V_2[ok, i])):
                    mae_2[i] = np.mean(np.abs(V_dss[ok, i] - V_2[ok, i]))

    mae_diff = np.full(N, np.nan)
    if phase_a_indices is not None:
        phase_a_set = set(phase_a_indices)
        for i in phase_a_indices:
            if np.isfinite(mae_1[i]) and np.isfinite(mae_2[i]):
                mae_diff[i] = np.abs(mae_1[i] - mae_2[i])
        order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
        worst_indices = [idx for idx in order if idx in phase_a_set and np.isfinite(mae_diff[idx])][:TOP_N_WORST]
    else:
        mae_diff = np.abs(mae_1 - mae_2)
        order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
        worst_indices = list(order[:TOP_N_WORST])
    node_to_idx = {n: i for i, n in enumerate(node_names)}
    for en in EXTRA_NODES:
        if en in node_to_idx:
            idx = node_to_idx[en]
            if idx not in worst_indices:
                worst_indices.append(idx)

    df = pd.DataFrame({
        "node": node_names,
        "mae_1": mae_1, "mae_2": mae_2,
        "mae_diff": mae_diff,
    }).sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, f"mae_per_node_{PRESET}.csv")
    df.to_csv(csv_path, index=False)
    suffix_phase = " (phase A only)" if PRESET in ("phase_onehot_vs_subgraph", "14feat_vs_phase_a") else ""
    print(f"Top {TOP_N_WORST} worst nodes (|MAE_1 - MAE_2|){suffix_phase}:")
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
        suffix = f"worst #{k+1}" if k < TOP_N_WORST else "requested"
        ax.set_title(f"24h voltage @ {node_names[idx]} ({suffix}, PV={PV_SCALE:.1f}×)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        fname = f"overlay_24h_{PRESET}_worst_{k+1}_{node_names[idx].replace('.', '_')}.png" if k < TOP_N_WORST else f"overlay_24h_{PRESET}_{node_names[idx].replace('.', '_')}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
