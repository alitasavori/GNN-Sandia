"""
Pipeline: train Original and Injection models from existing datasets (no dataset regeneration),
then compare with a 24h voltage profile (OpenDSS + both models) on in-distribution baseline.

- Uses datasets_gnn2/original and datasets_gnn2/injection (must already exist).
- Trains two models (same architecture choices as run_gnn3_best7_train blocks 1 & 2).
- Runs 24h at inj.BASELINE (849.12, 501.12, 1400), sigma=0, same DSS → in-distribution.
- Plots three profiles: OpenDSS, Original-features model, Injection-features model.

Run from repo root or notebook:
  %run run_pipeline_original_vs_injection.py

To only plot (no retraining): set SKIP_TRAIN = True and run, or from notebook:
  from run_pipeline_original_vs_injection import plot_only
  plot_only()
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import opendssdirect as dss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import run_injection_dataset as inj
from run_gnn3_best7_train import (
    PFIdentityGNN,
    SEED,
    BASE_DIR,
    DATA_FRAC,
    TEST_FRAC,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    MIN_EPOCHS_BEFORE_STOP,
    MIN_DELTA,
    ORIGINAL_FEAT,
    INJECTION_FEAT,
    train_one as gnn3_train_one,
    _parse_phase_from_node_name,
)
from run_gnn3_overlay_7 import (
    build_bus_to_phases_from_master_nodes,
    build_gnn_x_original,
    build_gnn_x_injection,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss,
    resolve_csv_path,
    read_profile_csv_two_col_noheader,
    load_model_for_inference,
)

os.chdir(BASE_DIR)

# Output and paths
PIPELINE_OUTPUT = "pipeline_original_vs_injection_output"
DIR_ORIGINAL = os.path.join("datasets_gnn2", "original")
DIR_INJECTION = os.path.join("datasets_gnn2", "injection")
CKPT_ORIGINAL = os.path.join(PIPELINE_OUTPUT, "ckpt_original.pt")
CKPT_INJECTION = os.path.join(PIPELINE_OUTPUT, "ckpt_injection.pt")

NPTS = 288
STEP_MIN = 5
# In-distribution: same baseline as dataset generation and summarize_original_dataset
P_BASE = float(inj.BASELINE["P_load_total_kw"])
Q_BASE = float(inj.BASELINE["Q_load_total_kvar"])
PV_BASE = float(inj.BASELINE["P_pv_total_kw"])
OBSERVED_NODE = "840.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set True to only run 24h comparison (use existing checkpoints)
SKIP_TRAIN = False

os.makedirs(PIPELINE_OUTPUT, exist_ok=True)


def train_original():
    """Train model on original dataset (4 features). Same config as run_gnn3_best7_train block 1."""
    print("\n>>> Training Original model (datasets_gnn2/original, 4 feat, medium arch)...")
    # block_id=1, medium: n_emb=8, e_emb=4, h_dim=32, n_layers=4, use_norm=False, early_stop=True
    save_path = gnn3_train_one(
        1, "medium", DIR_ORIGINAL,
        ORIGINAL_FEAT, "vmag_pu",
        8, 4, 32, 4, False, False,
        early_stop=True,
    )
    if save_path is None:
        raise FileNotFoundError("Original training failed. Ensure datasets_gnn2/original exists.")
    # Rename block1.pt -> ckpt_original.pt
    block_path = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block1.pt")
    ckpt_dest = os.path.join(BASE_DIR, CKPT_ORIGINAL)
    if os.path.exists(block_path):
        if os.path.exists(ckpt_dest):
            os.remove(ckpt_dest)
        os.rename(block_path, ckpt_dest)
        print(f"  [SAVED] {ckpt_dest}")
    return ckpt_dest


def train_injection():
    """Train model on injection dataset (2 features). Same config as run_gnn3_best7_train block 2."""
    print("\n>>> Training Injection model (datasets_gnn2/injection, 2 feat, deep arch)...")
    save_path = gnn3_train_one(
        2, "deep", DIR_INJECTION,
        INJECTION_FEAT, "vmag_pu",
        16, 8, 64, 4, False, False,
        early_stop=True,
    )
    if save_path is None:
        raise FileNotFoundError("Injection training failed. Ensure datasets_gnn2/injection exists.")
    block_path = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block2.pt")
    ckpt_dest = os.path.join(BASE_DIR, CKPT_INJECTION)
    if os.path.exists(block_path):
        if os.path.exists(ckpt_dest):
            os.remove(ckpt_dest)
        os.rename(block_path, ckpt_dest)
        print(f"  [SAVED] {ckpt_dest}")
    return ckpt_dest


def _master_nodes_89(dataset_dir):
    """Return list of 89 node names (non-upstream) in same order as training (sorted by node_idx)."""
    master_path = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    df = pd.read_csv(master_path)
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="raise").astype(int)
    df = df.sort_values("node_idx").reset_index(drop=True)
    names = []
    for _, row in df.iterrows():
        bus = str(row["node"]).split(".")[0]
        if bus in inj.EXCLUDED_UPSTREAM_BUSES:
            continue
        names.append(str(row["node"]))
    return names


def evaluate_three_profile_24h(ckpt_orig, ckpt_inj, device=None):
    """
    Run 24h voltage profile: OpenDSS + Original model + Injection model.
    Uses BASELINE (849.12, 501.12, 1400), sigma=0 → in-distribution.
    Same DSS as dataset generation. Returns (t_hours, v_dss, v_orig, v_inj), MAEs.
    """
    if device is None:
        device = DEVICE
    model_orig, static_orig = load_model_for_inference(ckpt_orig, device=device)
    model_inj, static_inj = load_model_for_inference(ckpt_inj, device=device)
    N = static_orig["N"]
    ei = static_orig["edge_index"].to(device)
    ea = static_orig["edge_attr"].to(device)
    eid = static_orig["edge_id"].to(device)

    node_names_master = _master_nodes_89(DIR_ORIGINAL)
    if len(node_names_master) != N:
        raise RuntimeError(f"Node count mismatch: master {len(node_names_master)} vs model N={N}")
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}

    dss_path = inj.compile_once()
    inj.setup_daily()
    try:
        dss.Text.Command("set maxcontroliter=20000")
    except Exception:
        pass
    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = resolve_csv_path(csvL_token, dss_path)
    csvPV = resolve_csv_path(csvPV_token, dss_path)
    mL = read_profile_csv_two_col_noheader(csvL, npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(csvPV, npts=NPTS)

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    t_hours = []
    rows_dss, rows_orig, rows_inj = [], [], []
    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, _, _ = inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_BASE,
            Q_load_total_kvar=Q_BASE,
            P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]),
            mPV_t=float(mPV[t]),
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=0.0,
            sigma_pv=0.0,
            rng=rng,
        )
        try:
            dss.Solution.Solve()
        except Exception:
            pass
        if not dss.Solution.Converged():
            t_hours.append(t * STEP_MIN / 60.0)
            rows_dss.append(np.full(N, np.nan))
            rows_orig.append(np.full(N, np.nan))
            rows_inj.append(np.full(N, np.nan))
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm_dss_row = np.array([float(vdict.get(n, (np.nan, 0))[0]) for n in node_names_master], dtype=np.float64)

        busphP_pv, busphQ_pv = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        X_orig = build_gnn_x_original(
            node_names_master, busphP_load, busphQ_load, busphP_pv, busphQ_pv=busphQ_pv
        )
        pwr = dss.Circuit.TotalPower()
        P_grid = -float(pwr[0])
        Q_grid = -float(pwr[1])
        X_inj = build_gnn_x_injection(
            node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid,
            busphQ_pv=busphQ_pv,
        )

        x_orig = torch.tensor(X_orig, dtype=torch.float32, device=device)
        x_inj = torch.tensor(X_inj, dtype=torch.float32, device=device)
        g_orig = Data(x=x_orig, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_inj = Data(x=x_inj, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        with torch.no_grad():
            vo_row = model_orig(g_orig)[:, 0].cpu().numpy().astype(np.float64)
            vi_row = model_inj(g_inj)[:, 0].cpu().numpy().astype(np.float64)

        t_hours.append(t * STEP_MIN / 60.0)
        rows_dss.append(vm_dss_row)
        rows_orig.append(vo_row)
        rows_inj.append(vi_row)

    t_hours = np.array(t_hours)
    V_dss = np.array(rows_dss, dtype=np.float64)   # (NPTS, N)
    V_orig = np.array(rows_orig, dtype=np.float64)
    V_inj = np.array(rows_inj, dtype=np.float64)

    # MAE per node (over time)
    mae_orig_per_node = np.nanmean(np.abs(V_dss - V_orig), axis=0)
    mae_inj_per_node = np.nanmean(np.abs(V_dss - V_inj), axis=0)

    out_dir = os.path.join(BASE_DIR, PIPELINE_OUTPUT)
    all_nodes_dir = os.path.join(out_dir, "all_nodes")
    os.makedirs(all_nodes_dir, exist_ok=True)

    # One plot per node
    for idx in range(N):
        n = node_names_master[idx]
        m1, m2 = float(mae_orig_per_node[idx]), float(mae_inj_per_node[idx])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_orig[:, idx], color="orange", linestyle="--", label=f"Original MAE={m1:.4f}", linewidth=1.5)
        ax.plot(t_hours, V_inj[:, idx], "g:", label=f"Injection MAE={m2:.4f}", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h @ {n} — in-distribution baseline")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        safe_name = n.replace(".", "_")
        fig.savefig(os.path.join(all_nodes_dir, f"three_profile_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Summary plot at default node
    obs_idx = node_to_idx.get(OBSERVED_NODE, 0)
    mae_orig = float(np.nanmean(mae_orig_per_node))
    mae_inj = float(np.nanmean(mae_inj_per_node))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, V_dss[:, obs_idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
    ax.plot(t_hours, V_orig[:, obs_idx], color="orange", linestyle="--", label=f"Original (4 feat) MAE={mae_orig_per_node[obs_idx]:.4f}", linewidth=1.5)
    ax.plot(t_hours, V_inj[:, obs_idx], "g:", label=f"Injection (2 feat) MAE={mae_inj_per_node[obs_idx]:.4f}", linewidth=1.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"24h @ {OBSERVED_NODE} — in-distribution baseline ({P_BASE:.0f}/{Q_BASE:.0f}/{PV_BASE:.0f} kW/kVAR/kW)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "three_profile_24h_in_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [saved] {out_dir}/three_profile_24h_in_distribution.png (summary @ {OBSERVED_NODE})")
    print(f"  [saved] {all_nodes_dir}/ — one plot per node ({N} PNGs)")
    print(f"  Overall: Original MAE={mae_orig:.6f} | Injection MAE={mae_inj:.6f}")
    return (t_hours, node_names_master, V_dss, V_orig, V_inj, mae_orig_per_node, mae_inj_per_node)


def main():
    print("=" * 70)
    print("PIPELINE: Original vs Injection (existing datasets, train + 24h in-distribution compare)")
    print("=" * 70)
    print(f"  DSS: {os.path.abspath(inj.DSS_FILE)}")
    print(f"  Baseline (in-distribution): P={P_BASE:.2f} Q={Q_BASE:.2f} PV={PV_BASE:.2f} kW/kVAR/kW, sigma=0")
    if not os.path.exists(os.path.join(DIR_ORIGINAL, "gnn_node_features_and_targets.csv")):
        raise FileNotFoundError(f"Missing {DIR_ORIGINAL}. Generate original dataset first.")
    if not os.path.exists(os.path.join(DIR_INJECTION, "gnn_node_features_and_targets.csv")):
        raise FileNotFoundError(f"Missing {DIR_INJECTION}. Generate injection dataset first.")

    if not SKIP_TRAIN:
        # Train (saves block1.pt / block2.pt in PIPELINE_OUTPUT, then we rename to ckpt_*.pt)
        import run_gnn3_best7_train as gnn3
        old_out = gnn3.OUTPUT_DIR
        gnn3.OUTPUT_DIR = os.path.join(BASE_DIR, PIPELINE_OUTPUT)
        try:
            train_original()
            train_injection()
        finally:
            gnn3.OUTPUT_DIR = old_out
    else:
        print("\n  [SKIP_TRAIN=True] Using existing checkpoints.")

    ckpt_orig = os.path.join(BASE_DIR, CKPT_ORIGINAL)
    ckpt_inj = os.path.join(BASE_DIR, CKPT_INJECTION)
    if not os.path.exists(ckpt_orig):
        ckpt_orig = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block1.pt")
    if not os.path.exists(ckpt_inj):
        ckpt_inj = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block2.pt")

    print("\n>>> Evaluating 24h three-profile (OpenDSS + Original + Injection) @", OBSERVED_NODE)
    evaluate_three_profile_24h(ckpt_orig, ckpt_inj)
    print("=" * 70)


def plot_only(ckpt_orig=None, ckpt_inj=None):
    """Run 24h three-profile comparison and save plot only (no training).
    Uses existing checkpoints from pipeline_original_vs_injection_output/ if paths not given.
    Example (notebook): from run_pipeline_original_vs_injection import plot_only; plot_only()
    """
    ckpt_orig = ckpt_orig or os.path.join(BASE_DIR, CKPT_ORIGINAL)
    ckpt_inj = ckpt_inj or os.path.join(BASE_DIR, CKPT_INJECTION)
    if not os.path.exists(ckpt_orig):
        ckpt_orig = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block1.pt")
    if not os.path.exists(ckpt_inj):
        ckpt_inj = os.path.join(BASE_DIR, PIPELINE_OUTPUT, "block2.pt")
    if not os.path.exists(ckpt_orig) or not os.path.exists(ckpt_inj):
        raise FileNotFoundError(
            f"Checkpoints not found. Train first (SKIP_TRAIN=False) or pass ckpt_orig=..., ckpt_inj=..."
        )
    print(">>> 24h three-profile (OpenDSS + Original + Injection) @", OBSERVED_NODE)
    evaluate_three_profile_24h(ckpt_orig, ckpt_inj)


if __name__ == "__main__":
    main()
