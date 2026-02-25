"""
Train medium architecture (best for original) on same 10k samples with:
  (A) Original features: p_load_kw, q_load_kvar, p_pv_kw
  (B) Injection features: p_inj_kw, q_inj_kvar
Derives both from loadtype dataset. Saves models, runs 24h profile vs OpenDSS.
Run from repo root. Requires: gnn_samples_loadtype_full
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, NPTS, P_BASE, Q_BASE, PV_BASE, STEP_MIN,
    build_bus_to_phases_from_master_nodes, build_gnn_x_original, build_gnn_x_injection,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
    load_model_for_inference,
)
from run_gnn3_best7_train import PFIdentityGNN

os.chdir(BASE_DIR)

DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
CKPT_ORIGINAL = os.path.join(OUTPUT_DIR, "block_original_features.pt")
CKPT_INJECTION = os.path.join(OUTPUT_DIR, "block_injection_features.pt")
OBSERVED_NODE = "840.1"
PV_SCALE = 1.65

TARGET_SAMPLES = 10000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_FEAT = ["p_load_kw", "q_load_kvar", "p_pv_kw"]
INJECTION_FEAT = ["p_inj_kw", "q_inj_kvar"]
LOADTYPE_COLS = ["m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar", "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw", "p_sys_balance_kw", "q_sys_balance_kvar"]


def derive_features_from_loadtype(df_n):
    """Derive original and injection features from loadtype columns."""
    df = df_n.copy()
    if "bus" not in df.columns and "node" in df.columns:
        df["bus"] = df["node"].astype(str).str.split(".").str[0]
    df["p_load_kw"] = df["m1_p_kw"] + df["m2_p_kw"] + df["m4_p_kw"] + df["m5_p_kw"]
    df["q_load_kvar"] = df["m1_q_kvar"] + df["m2_q_kvar"] + df["m4_q_kvar"] + df["m5_q_kvar"]
    P_grid = df.groupby("sample_id")["p_sys_balance_kw"].first().to_dict()
    Q_grid = df.groupby("sample_id")["q_sys_balance_kvar"].first().to_dict()
    P_grid_ph = {sid: P_grid[sid] / 3.0 for sid in P_grid}
    Q_grid_ph = {sid: Q_grid[sid] / 3.0 for sid in Q_grid}
    p_inj, q_inj = [], []
    for _, row in df.iterrows():
        sid, bus = row["sample_id"], str(row["bus"])
        if bus in ("sourcebus", "800"):
            p_inj.append(P_grid_ph[sid])
            q_inj.append(Q_grid_ph[sid])
        else:
            p_inj.append(float(row["p_pv_kw"]) - float(row["p_load_kw"]))
            q_inj.append(-float(row["q_load_kvar"]) + float(row["q_cap_kvar"]))
    df["p_inj_kw"] = p_inj
    df["q_inj_kvar"] = q_inj
    return df


def train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, node_in_dim, ckpt_path, name):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    S = X_all.shape[0]
    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                     edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in train_idx]
    test_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in test_idx]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                         node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
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
            errs = [((model(data.to(DEVICE)) - data.y) ** 2).mean().sqrt().item() for data in test_loader]
            rmse = np.mean(errs)
        if rmse < best_rmse:
            best_rmse, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{name}] Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"N": N, "E": E, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
                   "node_emb_dim": 8, "edge_emb_dim": 4, "h_dim": 32, "num_layers": 4, "use_norm": False,
                   "target_col": "vmag_pu", "dataset": DIR_LOADTYPE, "use_phase_onehot": False},
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def evaluate_profile():
    model_orig, static_orig = load_model_for_inference(CKPT_ORIGINAL, device=DEVICE)
    model_inj, static_inj = load_model_for_inference(CKPT_INJECTION, device=DEVICE)
    N = static_orig["N"]
    ei = static_orig["edge_index"].to(DEVICE)
    ea = static_orig["edge_attr"].to(DEVICE)
    eid = static_orig["edge_id"].to(DEVICE)

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    obs_idx = node_to_idx[OBSERVED_NODE]

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

    t_hours, v_dss, v_orig, v_inj = [], [], [], []
    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, _ = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE * PV_SCALE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng,
        )
        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            t_hours.append(t * STEP_MIN / 60.0)
            v_dss.append(np.nan)
            v_orig.append(np.nan)
            v_inj.append(np.nan)
            continue
        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm_dss, _ = vdict[OBSERVED_NODE]

        sum_p_load = float(sum(busphP_load.values()))
        sum_q_load = float(sum(busphQ_load.values()))
        sum_p_pv = float(sum(busphP_pv.values()))
        P_grid = sum_p_load - sum_p_pv
        Q_grid = sum_q_load - sum(CAP_Q_KVAR.values())

        X_orig = build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv)
        X_inj = build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)

        x_orig = torch.tensor(X_orig, dtype=torch.float32, device=DEVICE)
        x_inj = torch.tensor(X_inj, dtype=torch.float32, device=DEVICE)
        g_orig = Data(x=x_orig, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_inj = Data(x=x_inj, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        with torch.no_grad():
            v_orig.append(float(model_orig(g_orig)[obs_idx, 0].item()))
            v_inj.append(float(model_inj(g_inj)[obs_idx, 0].item()))

        t_hours.append(t * STEP_MIN / 60.0)
        v_dss.append(float(vm_dss))

    vd = np.array(v_dss, dtype=np.float64)
    vo = np.array(v_orig, dtype=np.float64)
    vi = np.array(v_inj, dtype=np.float64)
    ok = np.isfinite(vd)
    mae_orig = float(np.mean(np.abs(vd[ok] - vo[ok]))) if np.sum(ok) > 0 else np.nan
    mae_inj = float(np.mean(np.abs(vd[ok] - vi[ok]))) if np.sum(ok) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, v_dss, "b-", label="OpenDSS |V| (pu)", linewidth=2)
    ax.plot(t_hours, v_orig, color="orange", linestyle="--", label=f"Original feat (MAE={mae_orig:.4f})", linewidth=1.5)
    ax.plot(t_hours, v_inj, "g:", label=f"Injection feat (MAE={mae_inj:.4f})", linewidth=1.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"Original vs Injection features @ {OBSERVED_NODE} (24h, PV={PV_SCALE:.1f}×, medium arch)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "overlay_24h_original_vs_injection_features.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  [saved] {out_path}")
    print(f"  @ {OBSERVED_NODE}: original MAE={mae_orig:.6f} | injection MAE={mae_inj:.6f}")
    return mae_orig, mae_inj


def main():
    print("=" * 70)
    print("ORIGINAL vs INJECTION features: same data, same arch (medium), 10k samples")
    print("=" * 70)

    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    for c in LOADTYPE_COLS + ["vmag_pu", "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    if "bus" in df_n.columns:
        df_n["bus"] = df_n["bus"].astype(str)
    df_n = df_n.dropna(subset=LOADTYPE_COLS + ["vmag_pu"])
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    all_ids = df_n["sample_id"].unique()
    n_want = min(TARGET_SAMPLES, len(all_ids))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_want, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    df_n = derive_features_from_loadtype(df_n)

    N = int(df_n["node_idx"].max()) + 1
    E = len(df_e)
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    S = df_n["sample_id"].nunique()
    X_orig = df_n[ORIGINAL_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    X_inj = df_n[INJECTION_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    print(f"\n>>> Training on {S} samples (medium: 4L, h=32)...")
    train_one(X_orig, Y_all, edge_index, edge_attr, edge_id, N, E, 3, CKPT_ORIGINAL, "Original")
    train_one(X_inj, Y_all, edge_index, edge_attr, edge_id, N, E, 2, CKPT_INJECTION, "Injection")

    print("\n>>> Evaluating 24h profile vs OpenDSS @", OBSERVED_NODE)
    evaluate_profile()
    print("=" * 70)


if __name__ == "__main__":
    main()
