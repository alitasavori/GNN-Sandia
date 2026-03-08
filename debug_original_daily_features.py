"""
Debug script for ORIGINAL GNN: compare training vs daily-profile features
and predictions at a chosen node.

Goals:
- Check whether the daily 24h baseline trajectory sits in the same feature
  region as the training data for the ORIGINAL dataset model.
- Compare:
    (a) Training distribution of node features at one node.
    (b) Daily-profile distribution of node features at that node (loadtype
        snapshot generator used by voltage_profile_overlay_24h).
- Compare model predictions vs OpenDSS on:
    (a) A random subset of training snapshots.
    (b) The 24h baseline day.

Usage (from repo root):

  python debug_original_daily_features.py models_gnn2/original/light_xwide_emb_depth4_best.pt 840.1

If obs_node is omitted, defaults to '840.1'.

Outputs:
- Prints summary stats (mean/std/min/max) for each input feature at the
  chosen node, for both training and 24h daily profile.
- Prints MAE/RMSE for:
    - training subset vs OpenDSS (per-snapshot)
    - 24h overlay vs OpenDSS at obs_node
- Writes CSVs under debug_original/:
    - train_features_<node>.csv
    - daily_features_<node>.csv
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR,
    DIR_LOADTYPE,
    NPTS,
    P_BASE,
    Q_BASE,
    PV_BASE,
    STEP_MIN,
    build_bus_to_phases_from_master_nodes,
    build_gnn_x_original,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss,
    resolve_csv_path,
    read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
from run_gnn3_best7_train import ORIGINAL_FEAT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR_ORIGINAL = os.path.join(BASE_DIR, "datasets_gnn2", "original")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_original")
os.makedirs(DEBUG_DIR, exist_ok=True)


def _load_training_features_for_node(
    obs_node: str,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ORIGINAL dataset and return features + targets for the chosen node
    over a subset of samples.

    Returns:
      feats_train: (S, D) array of node features (columns = ORIGINAL_FEAT)
      vmag_train: (S,) array of target |V|_pu at obs_node
      t_ids:      (S,) sample_ids used
      all_feat_names: list of feature names in same order as feats_train columns
      node_names_master: list of node names (for debugging if needed)
    """
    edge_csv = os.path.join(DIR_ORIGINAL, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DIR_ORIGINAL, "gnn_node_features_and_targets.csv")
    master_csv = os.path.join(DIR_ORIGINAL, "gnn_node_index_master.csv")
    if not (os.path.exists(edge_csv) and os.path.exists(node_csv) and os.path.exists(master_csv)):
        raise FileNotFoundError("Original dataset files not found; run run_original_dataset.py first.")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    master_df = pd.read_csv(master_csv)

    # Numeric casting and basic cleaning (mirrors run_gnn3_best7_train logic for original).
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()

    for c in ["sample_id", "node_idx"] + ORIGINAL_FEAT + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_id", "node_idx"] + ORIGINAL_FEAT + ["vmag_pu"]).copy()

    df_n["sample_id"] = df_n["sample_id"].astype(int)
    df_n["node_idx"] = df_n["node_idx"].astype(int)

    # Determine kept node set and remap node_idx to 0..N-1 (same as training).
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    N = int(counts.mode()[0])

    kept_node_ids = sorted(df_n["node_idx"].unique())
    old_to_new = {old: new for new, old in enumerate(kept_node_ids)}
    df_n["node_idx"] = df_n["node_idx"].map(old_to_new)

    # Restrict edges to kept nodes and remap.
    df_e = df_e[df_e["u_idx"].isin(kept_node_ids) & df_e["v_idx"].isin(kept_node_ids)].copy()
    df_e["u_idx"] = df_e["u_idx"].map(old_to_new)
    df_e["v_idx"] = df_e["v_idx"].map(old_to_new)
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)

    # Map new node_idx -> node name so we can locate obs_node index.
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df[master_df["node_idx"].isin(kept_node_ids)].copy()
    master_df["new_idx"] = master_df["node_idx"].map(old_to_new)
    master_df = master_df.sort_values("new_idx").reset_index(drop=True)
    node_names_master = master_df["node"].astype(str).tolist()
    node_to_new = {n: int(i) for i, n in enumerate(node_names_master)}
    if obs_node not in node_to_new:
        raise RuntimeError(f"obs_node='{obs_node}' not found in original node index master.")
    obs_idx = node_to_new[obs_node]

    # Keep only complete snapshots (N nodes).
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    # Subsample snapshots for inspection (max_samples).
    all_ids = df_n["sample_id"].unique()
    rng = np.random.default_rng(20260308)
    if len(all_ids) > max_samples:
        keep_ids = rng.choice(all_ids, size=max_samples, replace=False)
    else:
        keep_ids = all_ids
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()

    # Reshape to (S, N, D).
    X_all = df_n[ORIGINAL_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N)
    sample_ids = df_n["sample_id"].unique()

    feats_train = X_all[:, obs_idx, :]  # (S, D)
    vmag_train = Y_all[:, obs_idx]      # (S,)

    return feats_train, vmag_train, sample_ids, np.array(ORIGINAL_FEAT), np.array(node_names_master)


def _load_daily_features_for_node(
    ckpt_path: str,
    obs_node: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run 24h baseline daily profile (like voltage_profile_overlay_24h) and
    collect features + DSS & GNN voltages at obs_node over all timesteps.

    Returns:
      feats_day: (T, D) features at obs_node (columns = ORIGINAL_FEAT subset)
      v_dss:     (T,) DSS |V|_pu at obs_node
      v_gnn:     (T,) GNN prediction at obs_node
    """
    model, static = load_model_for_inference(ckpt_path, device=DEVICE)
    cfg = static["config"]
    node_in_dim = int(cfg.get("node_in_dim", len(ORIGINAL_FEAT)))
    dataset_dir = cfg.get("dataset", DIR_ORIGINAL)

    # Use the same MASTER as overlay; for original, dataset_dir should be DIR_ORIGINAL.
    master_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Missing {master_csv}")
    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx").reset_index(drop=True)
    node_names_master = master_df["node"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    if obs_node not in node_to_idx:
        raise RuntimeError(f"obs_node='{obs_node}' not in MASTER.")
    obs_idx = node_to_idx[obs_node]
    N_expected = static["N"]
    if len(node_names_master) != N_expected:
        raise RuntimeError(f"MASTER node count {len(node_names_master)} != model expects {N_expected}.")

    edge_index = static["edge_index"].to(DEVICE)
    edge_attr = static["edge_attr"].to(DEVICE)
    edge_id = static["edge_id"].to(DEVICE)

    # Daily profiles for baseline.
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

    feats_day_list = []
    v_dss_list = []
    v_gnn_list = []

    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, _ = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng_det,
        )

        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            feats_day_list.append(np.full(node_in_dim, np.nan, dtype=np.float32))
            v_dss_list.append(np.nan)
            v_gnn_list.append(np.nan)
            continue

        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm_dss, _ = vdict[obs_node]

        # Build features for ORIGINAL model (3 or 4 dim).
        X = build_gnn_x_original(
            node_names_master,
            busphP_load,
            busphQ_load,
            busphP_pv,
            busphQ_pv=busphQ_pv if node_in_dim == 4 else None,
        )
        if X.shape[1] != node_in_dim:
            raise RuntimeError(f"Expected node_in_dim={node_in_dim}, got X.shape[1]={X.shape[1]}")

        x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        g = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N_expected)
        with torch.no_grad():
            yhat = model(g)
        vm_gnn = float(yhat[obs_idx, 0].item())

        feats_day_list.append(X[obs_idx, :].astype(np.float32))
        v_dss_list.append(float(vm_dss))
        v_gnn_list.append(vm_gnn)

    feats_day = np.stack(feats_day_list, axis=0)
    v_dss = np.array(v_dss_list, dtype=np.float64)
    v_gnn = np.array(v_gnn_list, dtype=np.float64)
    return feats_day, v_dss, v_gnn


def _print_stats(label: str, feats: np.ndarray, feat_names: np.ndarray) -> None:
    print(f"\n[{label}] feature stats at node:")
    for j, name in enumerate(feat_names):
        col = feats[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            print(f"  {name}: all NaN")
            continue
        vals = col[finite]
        print(
            f"  {name:12s}: min={vals.min():.4f}  max={vals.max():.4f}  "
            f"mean={vals.mean():.4f}  std={vals.std():.4f}  (n={len(vals)})"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python debug_original_daily_features.py <ckpt_path> [obs_node]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    obs_node = sys.argv[2] if len(sys.argv) >= 3 else "840.1"

    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 70)
    print(f"DEBUG ORIGINAL DAILY FEATURES @ {obs_node}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    # 1) Training distribution for obs_node.
    feats_train, v_train, sample_ids, feat_names, node_names_master = _load_training_features_for_node(obs_node)
    print(f"\nTraining snapshots used: {len(sample_ids)} samples; feature_dim={feats_train.shape[1]}")
    _print_stats("TRAINING", feats_train, feat_names)

    # 2) Daily baseline distribution for obs_node.
    feats_day, v_dss_day, v_gnn_day = _load_daily_features_for_node(ckpt_path, obs_node)
    print(f"\nDaily baseline timesteps: {feats_day.shape[0]} (NPTS={NPTS})")
    _print_stats("DAILY 24H", feats_day, feat_names[: feats_day.shape[1]])

    # 3) MAE/RMSE on daily 24h @ obs_node.
    ok = np.isfinite(v_dss_day) & np.isfinite(v_gnn_day)
    if np.any(ok):
        mae_day = float(np.mean(np.abs(v_dss_day[ok] - v_gnn_day[ok])))
        rmse_day = float(np.sqrt(np.mean((v_dss_day[ok] - v_gnn_day[ok]) ** 2)))
        print(f"\nDaily 24h @ {obs_node}: MAE={mae_day:.6f} pu | RMSE={rmse_day:.6f} pu (n={ok.sum()} points)")
    else:
        print("\nDaily 24h: all NaN (no converged timesteps).")

    # 4) Save CSVs for inspection in notebook.
    node_safe = obs_node.replace(".", "_")
    df_train = pd.DataFrame(feats_train, columns=feat_names)
    df_train["vmag_pu"] = v_train
    df_train["sample_id"] = sample_ids
    train_csv = os.path.join(DEBUG_DIR, f"train_features_{node_safe}.csv")
    df_train.to_csv(train_csv, index=False)

    df_day = pd.DataFrame(feats_day, columns=feat_names[: feats_day.shape[1]])
    df_day["vmag_dss_pu"] = v_dss_day
    df_day["vmag_gnn_pu"] = v_gnn_day
    df_day["t_hour"] = np.arange(len(df_day)) * STEP_MIN / 60.0
    day_csv = os.path.join(DEBUG_DIR, f"daily_features_{node_safe}.csv")
    df_day.to_csv(day_csv, index=False)

    print(f"\nSaved training features -> {train_csv}")
    print(f"Saved daily features    -> {day_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()

