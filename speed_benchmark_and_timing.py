# speed_benchmark_and_timing.py
# - Benchmarks OpenDSS + multiple GNN models over 24h (288 steps)
# - Reports device, OpenDSS wall/solve times, avg iterations
# - Reports per-model total forward time and mean ms/step
# - For each model, prints a detailed per-step timing table (OpenDSS vs GNN)

import os
import sys
import time
import pathlib

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

import opendssdirect as dss
import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_deltav_dataset import _apply_snapshot_zero_pv

# Ensure project root
ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_gnn3_overlay_7 import (
    load_model_for_inference,
    build_bus_to_phases_from_master_nodes,
    build_gnn_x_original,
    build_gnn_x_injection,
    build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss,
    resolve_csv_path,
    read_profile_csv_two_col_noheader,
    _parse_phase_from_node_name,
)
from compare_two_models_daily import build_x_for_model

# ---- Configure checkpoints here (variable length) ----
CKPT_PATHS = [
    r"C:\Users\alita\OneDrive\Desktop\GNN2\gnn2_architecture_search\loadtype\best.pt",
    r"C:\Users\alita\OneDrive\Desktop\GNN2\gnn2_architecture_search\original\best.pt",
]

# ---- Constants (must match dataset generation) ----
DATA_ROOT = ROOT / "datasets_gnn2" / "loadtype"
node_csv = DATA_ROOT / "gnn_node_features_and_targets.csv"
master_csv = DATA_ROOT / "gnn_node_index_master.csv"
edge_csv = DATA_ROOT / "gnn_edges_phase_static.csv"

NPTS = inj.NPTS
STEP_MIN = 5
P_BASE = float(inj.BASELINE["P_load_total_kw"])
Q_BASE = float(inj.BASELINE["Q_load_total_kvar"])
PV_BASE = float(inj.BASELINE["P_pv_total_kw"])
OBS_NODE_TIMING = "816.1"  # same as timing comparison script


def resolve_node_lists_and_dist():
    # 89-node list (training order)
    df_n = pd.read_csv(node_csv)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    kept_node_ids = sorted(df_n["node_idx"].unique())

    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    old_to_name = master_df.set_index("node_idx")["node"].astype(str).to_dict()
    node_names_89 = [old_to_name[old] for old in kept_node_ids]

    # 95-node master list for electrical distance
    full_node_list = list(master_df.sort_values("node_idx")["node"].astype(str))
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_node_list, str(edge_csv))
    return node_names_89, node_to_electrical_dist


def benchmark_24h_speed(node_names_master, node_to_electrical_dist, device, ckpt_paths):
    N = len(node_names_master)
    print("Nodes: N=", N)

    # OpenDSS setup
    inj.compile_once()
    inj.setup_daily()

    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvL_token, dss_path), npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvPV_token, dss_path), npts=inj.NPTS, debug=False)

    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    # ---- Task A: OpenDSS daily solve time + iterations ----
    t0 = time.perf_counter()
    open_dss_solve_s = 0.0
    open_dss_ctrl_iters = []
    open_dss_pf_iters = []

    for t in range(inj.NPTS):
        inj.set_time_index(t)
        _r, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng,
        )
        t_s = time.perf_counter()
        inj.dss.Solution.Solve()
        open_dss_solve_s += time.perf_counter() - t_s
        if inj.dss.Solution.Converged():
            try:
                open_dss_ctrl_iters.append(int(inj.dss.Solution.ControlIterations()))
            except Exception:
                pass
            try:
                open_dss_pf_iters.append(int(inj.dss.Solution.Iterations()))
            except Exception:
                pass

    open_dss_total_s = time.perf_counter() - t0
    print("\nOpenDSS (24h):")
    print("  total wall time: %.4f s | mean/step: %.3f ms" % (open_dss_total_s, 1000*open_dss_total_s/inj.NPTS))
    print("  solve-only time: %.4f s | mean/step: %.3f ms" % (open_dss_solve_s, 1000*open_dss_solve_s/inj.NPTS))

    if len(open_dss_ctrl_iters) > 0:
        print("  avg ControlIterations (converged): %.2f over %d steps" % (float(np.mean(open_dss_ctrl_iters)), len(open_dss_ctrl_iters)))
    else:
        print("  avg ControlIterations (converged): n/a")
    if len(open_dss_pf_iters) > 0:
        print("  avg PF Iterations (converged): %.2f over %d steps" % (float(np.mean(open_dss_pf_iters)), len(open_dss_pf_iters)))
    else:
        print("  avg PF Iterations (converged): n/a")

    # ---- Precompute per-timestep OpenDSS-derived quantities needed for features ----
    busph_ctx = [None] * inj.NPTS

    inj.compile_once()
    inj.setup_daily()
    for t in range(inj.NPTS):
        inj.set_time_index(t)
        _r, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
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
        pwr = inj.dss.Circuit.TotalPower()
        P_grid = -float(pwr[0])
        Q_grid = -float(pwr[1])
        sum_p_load = float(sum(busphP_load.values()))
        sum_q_load = float(sum(busphQ_load.values()))
        sum_p_pv_actual = float(sum(busphP_pv_actual.values()))
        sum_q_pv_actual = float(sum(busphQ_pv_actual.values()))
        sum_q_cap = float(inj.total_cap_q_kvar(node_names_master))
        p_sys_balance = sum_p_load - sum_p_pv_actual
        q_sys_balance = sum_q_load + sum_q_pv_actual - sum_q_cap
        busph_ctx[t] = (
            busphP_load,
            busphQ_load,
            busphP_pv_actual,
            busph_per_type,
            busphQ_pv_actual,
            P_grid,
            Q_grid,
            p_sys_balance,
            q_sys_balance,
        )

    # ---- Task B: Each model daily voltages + time ----
    results = []
    for ckpt_path in ckpt_paths:
        model, static = load_model_for_inference(ckpt_path, device=device)
        cfg = static["config"]
        node_in_dim = int(cfg.get("node_in_dim", 2))
        edge_index = static["edge_index"].to(device)
        edge_attr = static["edge_attr"].to(device)
        edge_id = static["edge_id"].to(device)
        t_model = 0.0

        # Warmup (GPU)
        if device.type == "cuda":
            with torch.no_grad():
                x0 = torch.zeros((N, node_in_dim), dtype=torch.float32, device=device)
                g0 = Data(x=x0, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N)
                _ = model(g0)
                torch.cuda.synchronize()

        for t in range(inj.NPTS):
            ctx = busph_ctx[t]
            if ctx is None:
                continue
            (
                busphP_load,
                busphQ_load,
                busphP_pv_actual,
                busph_per_type,
                busphQ_pv_actual,
                P_grid,
                Q_grid,
                p_sys_balance,
                q_sys_balance,
            ) = ctx

            X = build_x_for_model(
                node_in_dim,
                node_names_master=node_names_master,
                busphP_load=busphP_load,
                busphQ_load=busphQ_load,
                busphP_pv=busphP_pv_actual,
                busph_per_type=busph_per_type,
                P_grid=P_grid,
                Q_grid=Q_grid,
                node_to_electrical_dist=node_to_electrical_dist,
                p_sys_balance=p_sys_balance,
                q_sys_balance=q_sys_balance,
                busphQ_pv=busphQ_pv_actual,
            )
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            g = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0m = time.perf_counter()
            with torch.no_grad():
                _ = model(g)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model += time.perf_counter() - t0m

        results.append((ckpt_path, t_model))
        print("\nModel:", os.path.basename(ckpt_path))
        print("  total forward time: %.4f s | mean/step: %.3f ms" % (t_model, 1000 * t_model / inj.NPTS))

    print("\nSummary:")
    print("  Device:", device)
    print("  OpenDSS wall total: %.4f s | mean/step: %.3f ms" % (open_dss_total_s, 1000 * open_dss_total_s / inj.NPTS))
    print("  OpenDSS solve total: %.4f s | mean/step: %.3f ms" % (open_dss_solve_s, 1000 * open_dss_solve_s / inj.NPTS))
    for ckpt_path, t_model in results:
        print("  %s: total=%.4fs | mean/step=%.3fms" % (os.path.basename(ckpt_path), t_model, 1000 * t_model / inj.NPTS))

    return results


def _print_timing_table(block_id, device_name, dss_steps, gnn_steps, is_deltav):
    print()
    print("=" * 72)
    print(f"BLOCK {block_id} | {device_name} | {NPTS} steps")
    print("=" * 72)

    open_dss_keys = ["1_set_time_index", "5_apply_snapshot_full", "6_solve_full", "7_get_voltage_full"]
    print("\nOpenDSS (profile only: set_time, apply_full, solve, get_voltage):")
    for k in open_dss_keys:
        v = dss_steps[k]
        print(f"  {k:30s}: {v*1000:8.2f} ms  ({v:.4f}s)")
    dss_total = sum(dss_steps[k] for k in open_dss_keys)
    print(f"  {'TOTAL':30s}: {dss_total*1000:8.2f} ms  ({dss_total:.4f}s)")

    zero_pv_time = 0.0
    if is_deltav:
        zero_pv_time = (
            dss_steps["2_apply_snapshot_zero_pv"]
            + dss_steps["3_solve_zero_pv"]
            + dss_steps["4_get_voltage_zero_pv"]
        )
        print("\nGNN (includes OpenDSS zero-PV for vmag_zero + GNN steps):")
    else:
        print("\nGNN (per-step times):")
    gnn_total = zero_pv_time + sum(gnn_steps.values())
    if is_deltav:
        pct_z = 100.0 * zero_pv_time / gnn_total if gnn_total > 0 else 0
        print(f"  {'0_dss_zero_pv (for GNN input)':30s}: {zero_pv_time*1000:8.2f} ms  ({zero_pv_time:.4f}s)  ({pct_z:.1f}%)")
    for k, v in gnn_steps.items():
        pct = 100.0 * v / gnn_total if gnn_total > 0 else 0
        print(f"  {k:30s}: {v*1000:8.2f} ms  ({v:.4f}s)  ({pct:.1f}%)")
    print(f"  {'TOTAL':30s}: {gnn_total*1000:8.2f} ms  ({gnn_total:.4f}s)")

    print("\nSummary (comparable: OpenDSS profile vs GNN pipeline):")
    print(f"  OpenDSS total: {dss_total*1000:.2f} ms  |  GNN total: {gnn_total*1000:.2f} ms")
    print(f"  GNN/OpenDSS ratio: {gnn_total/max(dss_total,1e-9):.2f}x")
    print("=" * 72)


def per_step_timing_local(ckpt_path, device=None, use_batched_gnn=True, pv_scale=1.0):
    """Per-step timing for one checkpoint, using the same 89-node list as the main benchmark."""
    ckpt_path = str(ckpt_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For dataset-type branching below; on this machine we always want to
    # resolve to the local datasets_gnn2 paths.
    dataset_dir = str(DATA_ROOT)

    # 89-node list (training order) from features CSV
    df_n = pd.read_csv(node_csv)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    kept_node_ids = sorted(df_n["node_idx"].unique())

    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    old_to_name = master_df.set_index("node_idx")["node"].astype(str).to_dict()
    node_names_master = [old_to_name[old] for old in kept_node_ids]  # 89 nodes
    N_expected = len(node_names_master)

    model, static = load_model_for_inference(ckpt_path, device=device)
    cfg = static["config"]
    node_in_dim = int(cfg.get("node_in_dim", 3))
    target_col = cfg.get("target_col", "vmag_pu")
    use_phase_onehot = bool(cfg.get("use_phase_onehot", False))
    is_deltav = target_col == "vmag_delta_pu"

    # Infer dataset kind from saved cfg["dataset"]
    dataset_raw = str(cfg.get("dataset", "")).lower()
    if "original" in dataset_raw and "loadtype" not in dataset_raw and "injection" not in dataset_raw:
        dataset_kind = "original"
    elif "injection" in dataset_raw:
        dataset_kind = "injection"
    else:
        dataset_kind = "loadtype"

    if int(static["N"]) != N_expected:
        raise RuntimeError(f"Model N={static['N']} != 89-node list length {N_expected}.")
    if OBS_NODE_TIMING not in set(node_names_master):
        raise RuntimeError(f"observed_node='{OBS_NODE_TIMING}' not in 89-node list.")
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    obs_idx = node_to_idx[OBS_NODE_TIMING]

    # OpenDSS setup (same as dataset gen)
    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = read_profile_csv_two_col_noheader(resolve_csv_path(csvL_token, dss_path), npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(resolve_csv_path(csvPV_token, dss_path), npts=NPTS)

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng_det = np.random.default_rng(0)
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, str(edge_csv))

    if use_phase_onehot:
        phase_map = np.array([_parse_phase_from_node_name(n) for n in node_names_master], dtype=np.int64)
        ph_oh = np.eye(3, dtype=np.float32)[phase_map]
    else:
        ph_oh = None

    edge_index = static["edge_index"].to(device)
    edge_attr = static["edge_attr"].to(device)
    edge_id = static["edge_id"].to(device)

    dss_steps = {
        "1_set_time_index": 0.0,
        "2_apply_snapshot_zero_pv": 0.0,
        "3_solve_zero_pv": 0.0,
        "4_get_voltage_zero_pv": 0.0,
        "5_apply_snapshot_full": 0.0,
        "6_solve_full": 0.0,
        "7_get_voltage_full": 0.0,
    }
    gnn_steps = {
        "1_build_gnn_x": 0.0,
        "2_tensor_data_creation": 0.0,
        "3_model_forward": 0.0,
    }
    use_cuda_timer = device.type == "cuda" and torch.cuda.is_available()
    vmag_gnn = [np.nan] * NPTS
    batched_X_list = []
    batched_t_list = []

    # Delta-V prepass
    vmag_zero_precomputed = None
    if is_deltav:
        inj.compile_once()
        inj.setup_daily()
        vmag_zero_precomputed = []
        for t in range(NPTS):
            inj.set_time_index(t)
            t0 = time.perf_counter()
            _apply_snapshot_zero_pv(
                P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, mL_t=float(mL[t]),
                loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
                sigma_load=0.0, rng=rng_det,
            )
            dss_steps["2_apply_snapshot_zero_pv"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            dss.Solution.Solve()
            dss_steps["3_solve_zero_pv"] += time.perf_counter() - t0
            if not dss.Solution.Converged():
                vmag_zero_precomputed.append(np.full(len(node_names_master), np.nan, dtype=np.float32))
                continue
            t0 = time.perf_counter()
            vdict_z = get_all_node_voltage_pu_and_angle_dict()
            vmag_z = np.array([float(vdict_z.get(n, (np.nan, 0))[0]) for n in node_names_master], dtype=np.float32)
            dss_steps["4_get_voltage_zero_pv"] += time.perf_counter() - t0
            vmag_zero_precomputed.append(vmag_z)
        inj.compile_once()
        inj.setup_daily()

    pv_nominal = PV_BASE * float(pv_scale)

    for t in range(NPTS):
        t0 = time.perf_counter()
        inj.set_time_index(t)
        dss_steps["1_set_time_index"] += time.perf_counter() - t0

        vmag_zero = vmag_zero_precomputed[t] if is_deltav else None
        if is_deltav and not np.isfinite(vmag_zero).all():
            continue

        t0 = time.perf_counter()
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=pv_nominal,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng_det,
        )
        dss_steps["5_apply_snapshot_full"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        dss.Solution.Solve()
        dss_steps["6_solve_full"] += time.perf_counter() - t0
        if not dss.Solution.Converged():
            continue

        t0 = time.perf_counter()
        vdict = get_all_node_voltage_pu_and_angle_dict()
        dss_steps["7_get_voltage_full"] += time.perf_counter() - t0

        # Build X
        t0 = time.perf_counter()
        if dataset_kind == "original":
            busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
            X = build_gnn_x_original(
                node_names_master,
                busphP_load,
                busphQ_load,
                busphP_pv_actual,
                busphQ_pv=busphQ_pv_actual if node_in_dim == 4 else None,
            )
        elif dataset_kind == "injection":
            pwr = dss.Circuit.TotalPower()
            P_grid = -float(pwr[0])
            Q_grid = -float(pwr[1])
            X = build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
        else:
            busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
            sum_p_load = float(sum(busphP_load.values()))
            sum_q_load = float(sum(busphQ_load.values()))
            sum_p_pv = float(sum(busphP_pv_actual.values()))
            sum_q_pv = float(sum(busphQ_pv_actual.values()))
            p_sys_balance = sum_p_load - sum_p_pv
            q_sys_balance = sum_q_load + sum_q_pv - inj.total_cap_q_kvar(node_names_master)
            X = build_gnn_x_loadtype(
                node_names_master, busph_per_type, busphP_pv_actual,
                node_to_electrical_dist, p_sys_balance, q_sys_balance,
                busphQ_pv=busphQ_pv_actual,
            )
        if is_deltav:
            X = np.concatenate([X, vmag_zero[:, None]], axis=-1)
        if use_phase_onehot and ph_oh is not None:
            X = np.concatenate([X, ph_oh], axis=-1)
        gnn_steps["1_build_gnn_x"] += time.perf_counter() - t0

        if use_batched_gnn:
            batched_X_list.append(X.copy())
            batched_t_list.append(t)
        else:
            t0 = time.perf_counter()
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            g = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N_expected)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gnn_steps["2_tensor_data_creation"] += time.perf_counter() - t0

            if use_cuda_timer:
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(g)
                end.record()
                torch.cuda.synchronize()
                gnn_steps["3_model_forward"] += float(start.elapsed_time(end)) / 1000.0
            else:
                t0 = time.perf_counter()
                _ = model(g)
                gnn_steps["3_model_forward"] += time.perf_counter() - t0

    if use_batched_gnn and batched_X_list:
        # Vectorize tensor creation: stack once, then use tensor views per graph.
        # This reduces Python overhead and repeated host->device transfers in the batched path.
        t0 = time.perf_counter()
        X_stack = np.stack(batched_X_list, axis=0)  # (B, N, F)
        x_big = torch.from_numpy(X_stack).to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        data_list = [
            Data(
                x=x_big[i],
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_id=edge_id,
                num_nodes=N_expected,
            )
            for i in range(x_big.shape[0])
        ]
        batch = Batch.from_data_list(data_list)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gnn_steps["2_tensor_data_creation"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_cuda_timer:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        _ = model(batch)
        if use_cuda_timer:
            end.record()
            torch.cuda.synchronize()
            gnn_steps["3_model_forward"] += float(start.elapsed_time(end)) / 1000.0
        else:
            gnn_steps["3_model_forward"] += time.perf_counter() - t0

    _print_timing_table(1, str(device).upper(), dss_steps, gnn_steps, is_deltav)
    return dss_steps, gnn_steps, is_deltav


if __name__ == "__main__":
    if len(CKPT_PATHS) == 0:
        raise SystemExit("Set CKPT_PATHS to one or more .pt checkpoints in this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| torch.cuda.is_available()=", torch.cuda.is_available())

    node_names_master, node_to_electrical_dist = resolve_node_lists_and_dist()
    results = benchmark_24h_speed(node_names_master, node_to_electrical_dist, device, CKPT_PATHS)

    print("\n\n=== DETAILED PER-STEP TIMING (one model at a time) ===")
    for ckpt in CKPT_PATHS:
        print("\n############################")
        print("Detailed timing for:", ckpt)
        per_step_timing_local(ckpt, device=device, use_batched_gnn=True, pv_scale=1.0)