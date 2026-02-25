"""
Train two models on the same 10k samples from gnn_samples_deltav_full.
Both trained and evaluated on phase A only:
  (A) Phase A + vmag_zero: 14 features (13 loadtype + vmag_zero_pv_pu), phase A subgraph
  (B) Phase A only: 13 features (no vmag_zero), phase A subgraph
Same data. Worst-node selection among phase A nodes.
Requires: gnn_samples_deltav_full (run run_deltav_dataset.py first),
          gnn_samples_loadtype_full (for node index and edges).
Run from repo root.
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
    build_bus_to_phases_from_master_nodes, build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
)
from run_deltav_dataset import _apply_snapshot_zero_pv
from run_gnn3_best7_train import PFIdentityGNN
from gnn_narrow_exploration import load_phase_subgraph_edges, load_phase_mapping

os.chdir(BASE_DIR)

DIR_DELTAV = "gnn_samples_deltav_full"
DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
CKPT_14FEAT = os.path.join(OUTPUT_DIR, "block_14feat_vmagzero.pt")
CKPT_PHASE_A = os.path.join(OUTPUT_DIR, "block_14feat_phase_a.pt")

TARGET_SAMPLES = 10000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PHASE_A = 1
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOADTYPE_FULL_FEAT = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]
FEAT_14 = LOADTYPE_FULL_FEAT + ["vmag_zero_pv_pu"]


def build_phase_a_subgraph(phase_map_np, phase_edges, N):
    """Build phase A subgraph."""
    phase_a_mask = (phase_map_np == PHASE_A - 1)
    phase_a_indices = np.where(phase_a_mask)[0].tolist()
    N_phase_a = len(phase_a_indices)
    global_to_local = {g: l for l, g in enumerate(phase_a_indices)}

    ei_ph, ea_ph, eid_ph = phase_edges[PHASE_A - 1]
    if ei_ph.numel() == 0:
        return phase_a_indices, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)

    ei_np = ei_ph.numpy()
    keep = []
    for j in range(ei_np.shape[1]):
        u, v = int(ei_np[0, j]), int(ei_np[1, j])
        if u in global_to_local and v in global_to_local:
            keep.append(j)
    if len(keep) == 0:
        return phase_a_indices, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)

    u_local = np.array([global_to_local[int(ei_np[0, j])] for j in keep], dtype=np.int64)
    v_local = np.array([global_to_local[int(ei_np[1, j])] for j in keep], dtype=np.int64)
    ei_local = torch.tensor(np.stack([u_local, v_local]), dtype=torch.long)
    ea_local = ea_ph[torch.tensor(keep, dtype=torch.long)]
    eid_local = torch.arange(len(keep), dtype=torch.long)
    return phase_a_indices, ei_local, ea_local, eid_local


def train_phase_a_only(X_ph, Y_ph, ei_local, ea_local, eid_local, phase_a_indices, N_full, node_in_dim, ckpt_path, name):
    """Train single GNN on phase A nodes only. X_ph: (S, N_phase_a, node_in_dim), Y_ph: (S, N_phase_a, 1)."""
    N_phase_a = X_ph.shape[1]
    E_phase_a = ei_local.shape[1]
    if N_phase_a == 0 or E_phase_a == 0:
        raise RuntimeError("No phase A nodes or edges found.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    S = X_ph.shape[0]
    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_ds = [Data(x=torch.tensor(X_ph[k], dtype=torch.float), y=torch.tensor(Y_ph[k], dtype=torch.float),
                     edge_index=ei_local, edge_attr=ea_local, edge_id=eid_local, num_nodes=N_phase_a) for k in train_idx]
    test_ds = [Data(x=torch.tensor(X_ph[k], dtype=torch.float), y=torch.tensor(Y_ph[k], dtype=torch.float),
                    edge_index=ei_local, edge_attr=ea_local, edge_id=eid_local, num_nodes=N_phase_a) for k in test_idx]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(num_nodes=N_phase_a, num_edges=E_phase_a, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
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
        "config": {"N": N_phase_a, "E": E_phase_a, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
                   "node_emb_dim": 8, "edge_emb_dim": 4, "h_dim": 32, "num_layers": 4, "use_norm": False,
                   "target_col": "vmag_pu", "dataset": DIR_DELTAV, "phase_a_only": True},
        "edge_index": ei_local, "edge_attr": ea_local, "edge_id": eid_local,
        "phase_a_indices": phase_a_indices, "N_full": N_full,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def run_24h_and_plot():
    """Load both models, run 24h profile, find worst nodes (phase A only), plot."""
    ckpt_14 = torch.load(CKPT_14FEAT, map_location="cpu")
    ckpt_ph = torch.load(CKPT_PHASE_A, map_location="cpu")
    cfg_14, cfg_ph = ckpt_14["config"], ckpt_ph["config"]
    phase_a_indices = ckpt_ph["phase_a_indices"]
    N_phase_a = len(phase_a_indices)
    N = ckpt_ph["N_full"]

    ei_ph = ckpt_ph["edge_index"].to(DEVICE)
    ea_ph = ckpt_ph["edge_attr"].to(DEVICE)
    eid_ph = ckpt_ph["edge_id"].to(DEVICE)

    model_14 = PFIdentityGNN(num_nodes=N_phase_a, num_edges=int(cfg_14["E"]), node_in_dim=14, edge_in_dim=2, out_dim=1,
                             node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_14.load_state_dict(ckpt_14["state_dict"])
    model_14.eval()

    model_ph = PFIdentityGNN(num_nodes=N_phase_a, num_edges=int(cfg_ph["E"]), node_in_dim=13, edge_in_dim=2, out_dim=1,
                             node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_ph.load_state_dict(ckpt_ph["state_dict"])
    model_ph.eval()

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
    PV_SCALE = 1.0

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
    V_14 = np.full((NPTS, N), np.nan)
    V_ph = np.full((NPTS, N), np.nan)

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
        X_14_full = np.concatenate([X, vmag_zero[:, None]], axis=-1).astype(np.float32)
        x_14 = torch.tensor(X_14_full[phase_a_indices, :], dtype=torch.float32, device=DEVICE)
        x_ph = torch.tensor(X[phase_a_indices, :], dtype=torch.float32, device=DEVICE)

        g_14 = Data(x=x_14, edge_index=ei_ph, edge_attr=ea_ph, edge_id=eid_ph, num_nodes=N_phase_a)
        g_ph = Data(x=x_ph, edge_index=ei_ph, edge_attr=ea_ph, edge_id=eid_ph, num_nodes=N_phase_a)

        with torch.no_grad():
            pred_14 = model_14(g_14)[:, 0].cpu().numpy()
            pred_ph = model_ph(g_ph)[:, 0].cpu().numpy()
            for j, gidx in enumerate(phase_a_indices):
                V_14[t, gidx] = pred_14[j]
                V_ph[t, gidx] = pred_ph[j]

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    mae_14 = np.full(N, np.nan)
    mae_ph = np.full(N, np.nan)
    for i in phase_a_indices:
        ok = np.isfinite(V_dss[:, i]) & np.isfinite(V_14[:, i]) & np.isfinite(V_ph[:, i])
        if np.sum(ok) > 0:
            mae_14[i] = np.mean(np.abs(V_dss[ok, i] - V_14[ok, i]))
            mae_ph[i] = np.mean(np.abs(V_dss[ok, i] - V_ph[ok, i]))

    mae_diff = np.full(N, np.nan)
    for i in phase_a_indices:
        if np.isfinite(mae_14[i]) and np.isfinite(mae_ph[i]):
            mae_diff[i] = np.abs(mae_14[i] - mae_ph[i])
    order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
    worst_indices = [idx for idx in order if idx in phase_a_indices and np.isfinite(mae_diff[idx])][:5]

    df = pd.DataFrame({"node": node_names_master, "mae_14": mae_14, "mae_ph": mae_ph, "mae_diff": mae_diff})
    df = df.sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_14feat_vs_phase_a.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top 5 worst nodes (phase A only, |MAE_14 - MAE_ph|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: +vmag_zero MAE={mae_14[idx]:.4f} | 13feat MAE={mae_ph[idx]:.4f}")
    print(f"Saved -> {csv_path}")

    for k, idx in enumerate(worst_indices):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_14[:, idx], color="orange", linestyle="--",
                label=f"Phase A + vmag_zero (MAE={mae_14[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_ph[:, idx], "g:", label=f"Phase A only 13feat (MAE={mae_ph[idx]:.4f})", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {node_names_master[idx]} (worst #{k+1}, PV={PV_SCALE:.1f}×)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"overlay_24h_14feat_vs_phase_a_worst_{k+1}_{node_names_master[idx].replace('.', '_')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved -> {out_path}")


def main():
    print("=" * 70)
    print("PHASE A + vmag_zero vs PHASE A ONLY: both on phase A, same 10k samples from deltav")
    print("=" * 70)

    node_csv_deltav = os.path.join(DIR_DELTAV, "gnn_node_features_and_targets.csv")
    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    if not os.path.exists(node_csv_deltav):
        raise FileNotFoundError(f"Missing {DIR_DELTAV}. Run run_deltav_dataset.py first.")
    if not os.path.exists(edge_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    phase_map = load_phase_mapping(DIR_LOADTYPE)
    if phase_map is None:
        raise FileNotFoundError(f"Missing gnn_node_index_master.csv in {DIR_LOADTYPE}")
    phase_map_np = phase_map.cpu().numpy() if torch.is_tensor(phase_map) else np.array(phase_map)

    phase_edges = load_phase_subgraph_edges(edge_csv)
    if max(pe[2].numel() for pe in phase_edges) == 0:
        raise RuntimeError("No phase edges found. Check gnn_edges_phase_static.csv has 'phase' column.")

    df_n = pd.read_csv(node_csv_deltav)
    for c in FEAT_14 + ["vmag_delta_pu", "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n["vmag_pu"] = df_n["vmag_zero_pv_pu"] + df_n["vmag_delta_pu"]
    df_n = df_n.dropna(subset=FEAT_14 + ["vmag_pu"])

    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    all_ids = df_n["sample_id"].unique()
    n_want = min(TARGET_SAMPLES, len(all_ids))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_want, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    N = int(df_n["node_idx"].max()) + 1
    S = df_n["sample_id"].nunique()
    X_14 = df_n[FEAT_14].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    phase_a_indices, ei_local, ea_local, eid_local = build_phase_a_subgraph(phase_map_np, phase_edges, N)
    X_14_ph = X_14[:, phase_a_indices, :]  # (S, N_phase_a, 14)
    X_13_ph = X_14_ph[:, :, :13]           # (S, N_phase_a, 13)
    Y_ph = Y_all[:, phase_a_indices, :]    # (S, N_phase_a, 1)

    print(f"\n>>> Training on {S} samples, phase A only (medium: 4L, h=32)...")
    train_phase_a_only(X_14_ph, Y_ph, ei_local, ea_local, eid_local, phase_a_indices, N, 14, CKPT_14FEAT,
                       "Phase A + vmag_zero (14 feat)")
    train_phase_a_only(X_13_ph, Y_ph, ei_local, ea_local, eid_local, phase_a_indices, N, 13, CKPT_PHASE_A,
                       "Phase A only (13 feat)")

    print("\n>>> Running 24h profile and plotting 5 worst nodes (phase A only)...")
    run_24h_and_plot()
    print("=" * 70)


if __name__ == "__main__":
    main()
