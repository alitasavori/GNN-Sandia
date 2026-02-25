"""
Train two models on the same 10k samples from gnn_samples_loadtype_full:
  (A) Phase one-hot: single GNN, 16 features (13 loadtype + phase_onehot_1,2,3), all nodes
  (B) Phase A only: single GNN trained and evaluated on phase A nodes only (phases B/C ignored)
Same data, same base features. Comparison and worst-node selection only on phase A nodes.
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
    build_bus_to_phases_from_master_nodes, build_gnn_x_loadtype,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
)
from run_gnn3_best7_train import PFIdentityGNN
from gnn_narrow_exploration import load_phase_subgraph_edges, load_phase_mapping

os.chdir(BASE_DIR)

DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
CKPT_PHASE_ONEHOT = os.path.join(OUTPUT_DIR, "block_phase_onehot.pt")
CKPT_PHASE_SUBGRAPH = os.path.join(OUTPUT_DIR, "block_phase_subgraph.pt")

TARGET_SAMPLES = 10000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PHASE_A = 1  # Phase A = phase 1 in node names (e.g. 812.1)
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOADTYPE_FULL_FEAT = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]


def train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, node_in_dim, ckpt_path, name, use_phase_onehot=False):
    """Train standard PFIdentityGNN (phase one-hot or full graph)."""
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
                   "target_col": "vmag_pu", "dataset": DIR_LOADTYPE, "use_phase_onehot": use_phase_onehot,
                   "phase_a_only": False},
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def build_phase_a_subgraph(phase_map_np, phase_edges, N):
    """Build phase A subgraph: phase_A_indices, ei_local, ea, eid. Keep edges where both endpoints in phase A."""
    phase_a_mask = (phase_map_np == PHASE_A - 1)  # phase_map: 0=ph1, 1=ph2, 2=ph3
    phase_a_indices = np.where(phase_a_mask)[0].tolist()
    N_phase_a = len(phase_a_indices)
    global_to_local = {g: l for l, g in enumerate(phase_a_indices)}

    ei_ph, ea_ph, eid_ph = phase_edges[PHASE_A - 1]  # phase 1 = index 0
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


def train_phase_a_only(X_all, Y_all, phase_map_np, phase_edges, N, ckpt_path):
    """Train single GNN on phase A nodes only. Other phases ignored."""
    phase_a_indices, ei_local, ea_local, eid_local = build_phase_a_subgraph(phase_map_np, phase_edges, N)
    N_phase_a = len(phase_a_indices)
    E_phase_a = ei_local.shape[1]
    if N_phase_a == 0 or E_phase_a == 0:
        raise RuntimeError("No phase A nodes or edges found.")

    X_ph = X_all[:, phase_a_indices, :]  # (S, N_phase_a, 13)
    Y_ph = Y_all[:, phase_a_indices, :]  # (S, N_phase_a, 1)

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

    model = PFIdentityGNN(num_nodes=N_phase_a, num_edges=E_phase_a, node_in_dim=13, edge_in_dim=2, out_dim=1,
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
            print(f"    [Phase A only (13 feat)] Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"N": N_phase_a, "E": E_phase_a, "node_in_dim": 13, "edge_in_dim": 2, "out_dim": 1,
                   "node_emb_dim": 8, "edge_emb_dim": 4, "h_dim": 32, "num_layers": 4, "use_norm": False,
                   "target_col": "vmag_pu", "dataset": DIR_LOADTYPE, "phase_a_only": True},
        "edge_index": ei_local, "edge_attr": ea_local, "edge_id": eid_local,
        "phase_a_indices": phase_a_indices, "N_full": N,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def run_24h_and_plot():
    """Load both models, run 24h profile, find worst nodes (phase A only), plot."""
    ckpt_oh = torch.load(CKPT_PHASE_ONEHOT, map_location="cpu")
    ckpt_sg = torch.load(CKPT_PHASE_SUBGRAPH, map_location="cpu")
    cfg_oh, cfg_sg = ckpt_oh["config"], ckpt_sg["config"]
    N = int(cfg_oh["N"])
    phase_a_indices = ckpt_sg["phase_a_indices"]
    N_phase_a = len(phase_a_indices)

    model_oh = PFIdentityGNN(num_nodes=N, num_edges=int(cfg_oh["E"]), node_in_dim=int(cfg_oh["node_in_dim"]),
                             edge_in_dim=2, out_dim=1, node_emb_dim=8, edge_emb_dim=4,
                             h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_oh.load_state_dict(ckpt_oh["state_dict"])
    model_oh.eval()

    ei_sg = ckpt_sg["edge_index"].to(DEVICE)
    ea_sg = ckpt_sg["edge_attr"].to(DEVICE)
    eid_sg = ckpt_sg["edge_id"].to(DEVICE)
    model_sg = PFIdentityGNN(num_nodes=N_phase_a, num_edges=int(cfg_sg["E"]), node_in_dim=13, edge_in_dim=2, out_dim=1,
                             node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_sg.load_state_dict(ckpt_sg["state_dict"])
    model_sg.eval()

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)
    phase_map = load_phase_mapping(DIR_LOADTYPE).to(DEVICE)
    ph_oh = F.one_hot(phase_map, num_classes=3).float().cpu().numpy()

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
    PV_SCALE = 1.0

    V_dss = np.full((NPTS, N), np.nan)
    V_oh = np.full((NPTS, N), np.nan)
    V_sg = np.full((NPTS, N), np.nan)

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
        x_oh = torch.tensor(X_oh, dtype=torch.float32, device=DEVICE)
        x_sg_phase_a = torch.tensor(X[:, phase_a_indices, :], dtype=torch.float32, device=DEVICE)

        g_oh = Data(x=x_oh, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_sg = Data(x=x_sg_phase_a, edge_index=ei_sg, edge_attr=ea_sg, edge_id=eid_sg, num_nodes=N_phase_a)

        with torch.no_grad():
            V_oh[t, :] = model_oh(g_oh)[:, 0].cpu().numpy()
            pred_sg = model_sg(g_sg)[:, 0].cpu().numpy()
            for j, gidx in enumerate(phase_a_indices):
                V_sg[t, gidx] = pred_sg[j]

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    mae_oh = np.full(N, np.nan)
    mae_sg = np.full(N, np.nan)
    phase_a_set = set(phase_a_indices)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_oh[i] = np.mean(np.abs(V_dss[ok, i] - V_oh[ok, i]))
            if i in phase_a_set and np.any(np.isfinite(V_sg[ok, i])):
                mae_sg[i] = np.mean(np.abs(V_dss[ok, i] - V_sg[ok, i]))

    mae_diff = np.full(N, np.nan)
    for i in phase_a_indices:
        if np.isfinite(mae_oh[i]) and np.isfinite(mae_sg[i]):
            mae_diff[i] = np.abs(mae_oh[i] - mae_sg[i])
    order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
    worst_indices = [idx for idx in order if idx in phase_a_indices and np.isfinite(mae_diff[idx])][:5]

    df = pd.DataFrame({"node": node_names_master, "mae_oh": mae_oh, "mae_sg": mae_sg, "mae_diff": mae_diff})
    df = df.sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_phase_onehot_vs_subgraph.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top 5 worst nodes (phase A only, |MAE_oh - MAE_sg|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: one-hot MAE={mae_oh[idx]:.4f} | phase-A MAE={mae_sg[idx]:.4f}")
    print(f"Saved -> {csv_path}")

    for k, idx in enumerate(worst_indices):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_oh[:, idx], color="orange", linestyle="--",
                label=f"Phase one-hot (MAE={mae_oh[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_sg[:, idx], "g:", label=f"Phase A only (MAE={mae_sg[idx]:.4f})", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {node_names_master[idx]} (worst #{k+1}, PV={PV_SCALE:.1f}×)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"overlay_24h_phase_onehot_vs_subgraph_worst_{k+1}_{node_names_master[idx].replace('.', '_')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved -> {out_path}")


def main():
    print("=" * 70)
    print("PHASE ONE-HOT vs PHASE A ONLY: same 10k samples, phase A only for comparison")
    print("=" * 70)

    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    phase_map = load_phase_mapping(DIR_LOADTYPE)
    if phase_map is None:
        raise FileNotFoundError(f"Missing gnn_node_index_master.csv in {DIR_LOADTYPE}")
    phase_onehot = F.one_hot(phase_map, num_classes=3).numpy().astype(np.float32)

    phase_edges = load_phase_subgraph_edges(edge_csv)
    if max(pe[2].numel() for pe in phase_edges) == 0:
        raise RuntimeError("No phase edges found. Check gnn_edges_phase_static.csv has 'phase' column.")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    for c in LOADTYPE_FULL_FEAT + ["vmag_pu", "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.dropna(subset=LOADTYPE_FULL_FEAT + ["vmag_pu"])
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    all_ids = df_n["sample_id"].unique()
    n_want = min(TARGET_SAMPLES, len(all_ids))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_want, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    N = int(df_n["node_idx"].max()) + 1
    E = len(df_e)
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    S = df_n["sample_id"].nunique()
    X_full = df_n[LOADTYPE_FULL_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    X_oh = np.concatenate([X_full, np.broadcast_to(phase_onehot[None, :, :], (S, N, 3))], axis=-1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    phase_map_np = phase_map.cpu().numpy() if torch.is_tensor(phase_map) else np.array(phase_map)
    print(f"\n>>> Training on {S} samples (medium: 4L, h=32)...")
    train_one(X_oh, Y_all, edge_index, edge_attr, edge_id, N, E, 16, CKPT_PHASE_ONEHOT,
              "Phase one-hot (13+3 feat)", use_phase_onehot=True)
    train_phase_a_only(X_full, Y_all, phase_map_np, phase_edges, N, CKPT_PHASE_SUBGRAPH)

    print("\n>>> Running 24h profile and plotting 5 worst nodes (phase A only)...")
    run_24h_and_plot()
    print(">>> Or re-plot later: %run plot_two_models_worst_node_profile.py phase_onehot_vs_subgraph")
    print("=" * 70)


if __name__ == "__main__":
    main()
