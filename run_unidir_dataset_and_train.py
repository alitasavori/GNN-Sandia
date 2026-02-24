"""
Unidirectional vs bidirectional GNN: train same model on both edge types, compare 24h profile.
- Unidirectional: edges only upstream→downstream (~20k samples, 92 edges)
- Bidirectional: same ~20k samples from loadtype, full bidirectional edges (184 edges)
Both use light_emb_h96 (h=96, 2L). Evaluates 24h voltage profile vs OpenDSS at OBSERVED_NODE.
Run from repo root. Requires: gnn_samples_loadtype_full
"""
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, DIR_LOADTYPE, NPTS, P_BASE, Q_BASE, PV_BASE, STEP_MIN,
    build_bus_to_phases_from_master_nodes, build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
from run_gnn3_best7_train import PFIdentityGNN
import opendssdirect as dss

os.chdir(BASE_DIR)

OUT_DIR = "gnn_samples_loadtype_unidir"
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")
OUTPUT_DIR = "gnn3_best7_output"
CKPT_PATH = os.path.join(OUTPUT_DIR, "block_unidir.pt")
CKPT_PATH_BIDIR = os.path.join(OUTPUT_DIR, "block_bidir_unidir_compare.pt")
DIR_LOADTYPE = "gnn_samples_loadtype_full"
OBSERVED_NODE = "840.1"

TARGET_SAMPLES = 20000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
MIN_EPOCHS_BEFORE_STOP = 12
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOADTYPE_FEAT = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]


def _parse_phase_from_node_name(name):
    m = re.search(r"\.(\d+)$", str(name))
    return int(m.group(1)) if m else 1


def create_unidir_dataset():
    """Create unidirectional dataset: subset loadtype to ~20k samples, unidirectional edges."""
    src_dir = "gnn_samples_loadtype_full"
    src_node = os.path.join(src_dir, "gnn_node_features_and_targets.csv")
    src_edge = os.path.join(src_dir, "gnn_edges_phase_static.csv")
    src_index = os.path.join(src_dir, "gnn_node_index_master.csv")

    if not os.path.exists(src_node) or not os.path.exists(src_edge):
        raise FileNotFoundError(f"Missing {src_dir}. Run loadtype dataset generation first.")

    df_n = pd.read_csv(src_node)
    df_e = pd.read_csv(src_edge)
    df_idx = pd.read_csv(src_index)

    node_names_master = df_idx["node"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    N = len(node_names_master)

    node_to_dist = lt._compute_electrical_distance_from_source(node_names_master, src_edge)

    # Keep only edges from upstream (smaller dist) to downstream (larger dist)
    rows_uni = []
    seen_physical = set()
    for _, row in df_e.iterrows():
        a, b = str(row["from_node"]), str(row["to_node"])
        u_idx, v_idx = int(row["u_idx"]), int(row["v_idx"])
        dist_a = node_to_dist.get(a, float("inf"))
        dist_b = node_to_dist.get(b, float("inf"))
        key = tuple(sorted([a, b]))
        if key in seen_physical:
            continue
        seen_physical.add(key)
        if dist_a <= dist_b:
            up_idx, down_idx = u_idx, v_idx
            up_node, down_node = a, b
        else:
            up_idx, down_idx = v_idx, u_idx
            up_node, down_node = b, a
        rows_uni.append({
            "from_node": up_node, "to_node": down_node,
            "from_bus": row["from_bus"], "to_bus": row["to_bus"],
            "phase": row["phase"], "line_name": row["line_name"],
            "R_full": row["R_full"], "X_full": row["X_full"],
            "u_idx": up_idx, "v_idx": down_idx,
        })

    df_e_uni = pd.DataFrame(rows_uni)
    df_e_uni["edge_id"] = np.arange(len(df_e_uni), dtype=int)
    df_e_uni.to_csv(EDGE_CSV, index=False)
    df_idx.to_csv(NODE_INDEX_CSV, index=False)

    n_samples_want = max(1, TARGET_SAMPLES // N)
    all_ids = df_n["sample_id"].unique()
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=min(n_samples_want, len(all_ids)), replace=False)
    df_n_sub = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    df_n_sub.to_csv(NODE_CSV, index=False)
    print(f"[UNIDIR] Created {OUT_DIR}/ | samples={df_n_sub['sample_id'].nunique()} | edges={len(df_e_uni)} (was {len(df_e)})")
    return len(df_e_uni)


def train_unidir():
    """Train model on unidirectional dataset."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df_e = pd.read_csv(EDGE_CSV)
    df_n = pd.read_csv(NODE_CSV)
    required = {"sample_id", "node_idx", "vmag_pu"} | set(LOADTYPE_FEAT)
    if required - set(df_n.columns):
        raise ValueError(f"Missing columns: {required - set(df_n.columns)}")

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    for c in LOADTYPE_FEAT + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.dropna(subset=list(required))
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    N = int(df_n["node_idx"].max()) + 1
    E = len(df_e)
    S = df_n["sample_id"].nunique()
    X_all = df_n[LOADTYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                     edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in train_idx]
    test_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in test_idx]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    node_in_dim = X_all.shape[-1]
    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                         node_emb_dim=16, edge_emb_dim=8, h_dim=96, num_layers=2, use_norm=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_rmse, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(data), data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for data in test_loader:
                data = data.to(DEVICE)
                errs.append(((model(data) - data.y) ** 2).mean().sqrt().item())
            rmse = np.mean(errs)
        if rmse < best_rmse:
            best_rmse, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "N": N, "E": E, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
            "node_emb_dim": 16, "edge_emb_dim": 8, "h_dim": 96, "num_layers": 2, "use_norm": False,
            "target_col": "vmag_pu", "dataset": OUT_DIR, "use_phase_onehot": False,
        },
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, CKPT_PATH)
    print(f"  [SAVED] {CKPT_PATH}")
    return CKPT_PATH


def train_bidir():
    """Train same architecture on bidirectional loadtype dataset (~20k samples)."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    required = {"sample_id", "node_idx", "vmag_pu"} | set(LOADTYPE_FEAT)
    if required - set(df_n.columns):
        raise ValueError(f"Missing columns: {required - set(df_n.columns)}")

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_e["R_full"] = pd.to_numeric(df_e["R_full"], errors="coerce")
    df_e["X_full"] = pd.to_numeric(df_e["X_full"], errors="coerce")
    df_e = df_e.dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).reset_index(drop=True)
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)

    for c in LOADTYPE_FEAT + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.dropna(subset=list(required))
    N = int(df_n["node_idx"].max()) + 1
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    n_samples_want = max(1, TARGET_SAMPLES // N)
    all_ids = df_n["sample_id"].unique()
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=min(n_samples_want, len(all_ids)), replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    E = len(df_e)
    S = df_n["sample_id"].nunique()
    X_all = df_n[LOADTYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                     edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in train_idx]
    test_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in test_idx]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    node_in_dim = X_all.shape[-1]
    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                        node_emb_dim=16, edge_emb_dim=8, h_dim=96, num_layers=2, use_norm=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_rmse, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(data), data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for data in test_loader:
                data = data.to(DEVICE)
                errs.append(((model(data) - data.y) ** 2).mean().sqrt().item())
            rmse = np.mean(errs)
        if rmse < best_rmse:
            best_rmse, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "N": N, "E": E, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
            "node_emb_dim": 16, "edge_emb_dim": 8, "h_dim": 96, "num_layers": 2, "use_norm": False,
            "target_col": "vmag_pu", "dataset": DIR_LOADTYPE, "use_phase_onehot": False,
        },
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, CKPT_PATH_BIDIR)
    print(f"  [SAVED] {CKPT_PATH_BIDIR}")
    return CKPT_PATH_BIDIR


def evaluate_profile(obs_node=OBSERVED_NODE):
    """Run 24h profile: OpenDSS vs GNN unidirectional vs GNN bidirectional at obs_node."""
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

    t_hours, vmag_dss, vmag_uni, vmag_bidir = [], [], [], []
    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng,
        )
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            t_hours.append(t * STEP_MIN / 60.0)
            vmag_dss.append(np.nan)
            vmag_uni.append(np.nan)
            vmag_bidir.append(np.nan)
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
        vmag_dss.append(float(vm_dss))
        vmag_uni.append(vm_uni)
        vmag_bidir.append(vm_bidir)

    vd = np.array(vmag_dss, dtype=np.float64)
    vu = np.array(vmag_uni, dtype=np.float64)
    vb = np.array(vmag_bidir, dtype=np.float64)
    ok = np.isfinite(vd)
    mae_uni = float(np.mean(np.abs(vd[ok] - vu[ok]))) if np.sum(ok) > 0 else np.nan
    rmse_uni = float(np.sqrt(np.mean((vd[ok] - vu[ok]) ** 2))) if np.sum(ok) > 0 else np.nan
    mae_bidir = float(np.mean(np.abs(vd[ok] - vb[ok]))) if np.sum(ok) > 0 else np.nan
    rmse_bidir = float(np.sqrt(np.mean((vd[ok] - vb[ok]) ** 2))) if np.sum(ok) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, vmag_dss, label="OpenDSS |V| (pu)")
    ax.plot(t_hours, vmag_uni, label=f"GNN unidirectional |V| (MAE={mae_uni:.4f})")
    ax.plot(t_hours, vmag_bidir, label=f"GNN bidirectional |V| (MAE={mae_bidir:.4f})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"Unidirectional vs Bidirectional GNN vs OpenDSS @ {obs_node} (24h)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "overlay_24h_unidir_vs_bidir.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  [saved] {plot_path}")
    print(f"  @ {obs_node}: unidir MAE={mae_uni:.6f} RMSE={rmse_uni:.6f} | bidir MAE={mae_bidir:.6f} RMSE={rmse_bidir:.6f}")
    return mae_uni, rmse_uni, mae_bidir, rmse_bidir


def main():
    print("=" * 70)
    print("UNIDIRECTIONAL vs BIDIRECTIONAL: Dataset + Train + Evaluate")
    print("=" * 70)
    create_unidir_dataset()
    print("\n>>> Training on unidirectional edges...")
    train_unidir()
    print("\n>>> Training on bidirectional edges (same ~20k samples, same arch)...")
    train_bidir()
    print("\n>>> Evaluating 24h profile vs OpenDSS @", OBSERVED_NODE)
    evaluate_profile(OBSERVED_NODE)
    print("=" * 70)


if __name__ == "__main__":
    main()
