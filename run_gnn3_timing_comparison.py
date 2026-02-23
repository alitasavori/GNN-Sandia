"""
Timing comparison: per-step breakdown for OpenDSS vs GNN.
Comparable totals:
  - OpenDSS: only what it needs for its profile (apply_full, solve_full, get_voltage).
             Excludes zero-PV solve for Delta-V (that's GNN-only overhead).
  - GNN: for Delta-V, includes zero-PV OpenDSS solve + GNN steps (GNN needs vmag_zero).
         For non-Delta-V, just GNN steps.
Runs once on CPU and once on GPU. No plots. Run from repo root.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_deltav_dataset import _apply_snapshot_zero_pv
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, DIR_LOADTYPE, OUTPUT_DIR, NPTS, P_BASE, Q_BASE, PV_BASE,
    build_bus_to_phases_from_master_nodes, build_gnn_x_original, build_gnn_x_injection,
    build_gnn_x_loadtype, get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference, _parse_phase_from_node_name,
)

os.chdir(BASE_DIR)

def timing_one_block_detailed(ckpt_path, device, block_id):
    """Returns dict of step name -> total seconds."""
    model, static = load_model_for_inference(ckpt_path, device=device)
    cfg = static["config"]
    target_col = cfg.get("target_col", "vmag_pu")
    use_phase_onehot = bool(cfg.get("use_phase_onehot", False))
    dataset_dir = cfg.get("dataset", DIR_LOADTYPE)
    is_deltav = target_col == "vmag_delta_pu"
    N_expected = static["N"]

    node_index_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(node_index_csv):
        raise FileNotFoundError(f"Missing {node_index_csv}")
    master_df = pd.read_csv(node_index_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx").reset_index(drop=True)
    node_names_master = master_df["node"].astype(str).tolist()
    if len(node_names_master) != N_expected:
        raise RuntimeError(f"MASTER node count {len(node_names_master)} != model expects {N_expected}.")

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
    edge_csv = os.path.join(dataset_dir, "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv)

    phase_map = None
    if use_phase_onehot:
        phase_map = np.array([_parse_phase_from_node_name(n) for n in node_names_master], dtype=np.int64)
        ph_oh = np.eye(3, dtype=np.float32)[phase_map]

    edge_index = static["edge_index"].to(device)
    edge_attr = static["edge_attr"].to(device)
    edge_id = static["edge_id"].to(device)

    # Accumulators for each step (seconds)
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

    for t in range(NPTS):
        # --- OpenDSS step 1 ---
        t0 = time.perf_counter()
        inj.set_time_index(t)
        dss_steps["1_set_time_index"] += time.perf_counter() - t0

        vmag_zero = None
        if is_deltav:
            # OpenDSS steps 2, 3, 4 (zero-PV)
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
                continue

            t0 = time.perf_counter()
            vdict_z = get_all_node_voltage_pu_and_angle_dict()
            vmag_zero = np.array([float(vdict_z.get(n, (np.nan, 0))[0]) for n in node_names_master], dtype=np.float32)
            dss_steps["4_get_voltage_zero_pv"] += time.perf_counter() - t0

        # OpenDSS steps 5, 6, 7 (full)
        t0 = time.perf_counter()
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
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

        # --- GNN step 1: build_gnn_x ---
        t0 = time.perf_counter()
        if dataset_dir == "gnn_samples_out":
            X = build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv)
        elif dataset_dir == "gnn_samples_inj_full":
            pwr = dss.Circuit.TotalPower()
            P_grid = -float(pwr[0])
            Q_grid = -float(pwr[1])
            X = build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
        else:
            sum_p_load = float(sum(busphP_load.values()))
            sum_q_load = float(sum(busphQ_load.values()))
            sum_p_pv = float(sum(busphP_pv.values()))
            sum_q_cap = float(sum(CAP_Q_KVAR.values()))
            p_sys_balance = sum_p_load - sum_p_pv
            q_sys_balance = sum_q_load - sum_q_cap
            X = build_gnn_x_loadtype(
                node_names_master, busph_per_type, busphP_pv,
                node_to_electrical_dist, p_sys_balance, q_sys_balance,
            )
        if is_deltav:
            X = np.concatenate([X, vmag_zero[:, None]], axis=-1)
        if use_phase_onehot:
            X = np.concatenate([X, ph_oh], axis=-1)
        gnn_steps["1_build_gnn_x"] += time.perf_counter() - t0

        # --- GNN step 2: tensor + Data creation ---
        t0 = time.perf_counter()
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        g = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N_expected)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gnn_steps["2_tensor_data_creation"] += time.perf_counter() - t0

        # --- GNN step 3: model forward ---
        if use_cuda_timer:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yhat = model(g)
            end.record()
            torch.cuda.synchronize()
            gnn_steps["3_model_forward"] += float(start.elapsed_time(end)) / 1000.0
        else:
            t0 = time.perf_counter()
            yhat = model(g)
            gnn_steps["3_model_forward"] += time.perf_counter() - t0

    return dss_steps, gnn_steps, is_deltav


def print_timing(block_id, device_name, dss_steps, gnn_steps, is_deltav):
    """Print per-step timing. OpenDSS = profile only. GNN = includes zero-PV for Delta-V."""
    print()
    print("=" * 72)
    print(f"BLOCK {block_id} | {device_name} | 288 steps")
    print("=" * 72)

    # OpenDSS: only what it needs for its profile (exclude zero-PV for Delta-V)
    open_dss_keys = ["1_set_time_index", "5_apply_snapshot_full", "6_solve_full", "7_get_voltage_full"]
    print("\nOpenDSS (profile only: set_time, apply_full, solve, get_voltage):")
    for k in open_dss_keys:
        v = dss_steps[k]
        print(f"  {k:30s}: {v*1000:8.2f} ms  ({v:.4f}s)")
    dss_total = sum(dss_steps[k] for k in open_dss_keys)
    print(f"  {'TOTAL':30s}: {dss_total*1000:8.2f} ms  ({dss_total:.4f}s)")

    # GNN: for Delta-V, add zero-PV OpenDSS (GNN needs vmag_zero); else just GNN steps
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


def main():
    print("\n" + "=" * 72)
    print("TIMING: Comparable OpenDSS vs GNN - All 7 blocks")
    print("  OpenDSS: profile only (set_time, apply_full, solve, get_voltage)")
    print("  GNN: Delta-V = zero-PV solve + GNN steps; non-Delta-V = GNN steps only")
    print("=" * 72)

    for block_id in range(1, 8):
        ckpt_path = os.path.join(OUTPUT_DIR, f"block{block_id}.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n>>> Block {block_id}: SKIP (checkpoint not found)")
            continue

        # Run on CPU
        print(f"\n>>> Block {block_id} on CPU...")
        dss_cpu, gnn_cpu, is_deltav = timing_one_block_detailed(ckpt_path, torch.device("cpu"), block_id)
        print_timing(block_id, "CPU", dss_cpu, gnn_cpu, is_deltav)

        # Run on GPU if available
        if torch.cuda.is_available():
            print(f"\n>>> Block {block_id} on GPU...")
            dss_gpu, gnn_gpu, _ = timing_one_block_detailed(ckpt_path, torch.device("cuda"), block_id)
            print_timing(block_id, "GPU", dss_gpu, gnn_gpu, is_deltav)
        else:
            print(f"\n>>> Block {block_id}: GPU not available, skipping GPU run.")


if __name__ == "__main__":
    main()
