"""
Train two models on the same 10k samples from gnn_samples_loadtype_full:
  (A) Phase one-hot: single GNN, 16 features (13 loadtype + phase_onehot_1,2,3)
  (B) Phase subgraph: 3 separate GNNs (one per phase), 13 features, no phase encoding
Same data, same base features. Saves models, runs 24h profile, plots worst nodes.
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
from gnn_narrow_exploration import (
    load_phase_subgraph_edges, load_phase_mapping, PhaseSubgraphGNN,
    _batch_phase_subgraph,
)

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
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOADTYPE_FULL_FEAT = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]


def train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, node_in_dim,
              ckpt_path, name, use_phase_onehot=False, phase_edges=None, phase_map=None):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    S = X_all.shape[0]
    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        out = []
        for k in idx:
            x = torch.tensor(X_all[k], dtype=torch.float)
            y = torch.tensor(Y_all[k], dtype=torch.float)
            if use_phase_subgraph and phase_edges is not None:
                d = Data(x=x, y=y, num_nodes=N, phase_idx=phase_map)
                for p, (ei, ea, eid) in enumerate(phase_edges):
                    d[f"edge_index_{p}"] = ei
                    d[f"edge_attr_{p}"] = ea
                    d[f"edge_id_{p}"] = eid
            else:
                d = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                         edge_id=edge_id, num_nodes=N)
            out.append(d)
        return out

    use_phase_subgraph = phase_edges is not None
    train_ds = make_ds(train_idx)
    test_ds = make_ds(test_idx)
    batch_sz = 1 if use_phase_subgraph else BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    if use_phase_subgraph:
        max_e = max(pe[2].numel() for pe in phase_edges)
        model = PhaseSubgraphGNN(num_nodes=N, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                                 node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4,
                                 max_edges_per_phase=max_e).to(DEVICE)
        batch_fn = lambda d: _batch_phase_subgraph(d, phase_edges)
    else:
        model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                              node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
        batch_fn = None

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_rmse, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            if batch_fn:
                data = batch_fn(data)
            opt.zero_grad()
            F.mse_loss(model(data), data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for data in test_loader:
                data = data.to(DEVICE)
                if batch_fn:
                    data = batch_fn(data)
                errs.append(((model(data) - data.y) ** 2).mean().sqrt().item())
            rmse = np.mean(errs)
        if rmse < best_rmse:
            best_rmse, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{name}] Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"N": N, "E": E if not use_phase_subgraph else 0, "node_in_dim": node_in_dim,
                   "edge_in_dim": 2, "out_dim": 1, "node_emb_dim": 8, "edge_emb_dim": 4,
                   "h_dim": 32, "num_layers": 4, "use_norm": False,
                   "target_col": "vmag_pu", "dataset": DIR_LOADTYPE,
                   "use_phase_onehot": use_phase_onehot, "use_phase_subgraph": use_phase_subgraph},
        "edge_index": edge_index if not use_phase_subgraph else None,
        "edge_attr": edge_attr if not use_phase_subgraph else None,
        "edge_id": edge_id if not use_phase_subgraph else None,
        "phase_edges": phase_edges if use_phase_subgraph else None,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def run_24h_and_plot():
    """Load both models, run 24h profile, find worst nodes, plot."""
    ckpt_oh = torch.load(CKPT_PHASE_ONEHOT, map_location="cpu")
    ckpt_sg = torch.load(CKPT_PHASE_SUBGRAPH, map_location="cpu")
    cfg_oh, cfg_sg = ckpt_oh["config"], ckpt_sg["config"]
    N = int(cfg_oh["N"])

    model_oh = PFIdentityGNN(num_nodes=N, num_edges=int(cfg_oh["E"]), node_in_dim=int(cfg_oh["node_in_dim"]),
                             edge_in_dim=2, out_dim=1, node_emb_dim=8, edge_emb_dim=4,
                             h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_oh.load_state_dict(ckpt_oh["state_dict"])
    model_oh.eval()

    phase_edges = ckpt_sg["phase_edges"]
    max_e = max(pe[2].numel() for pe in phase_edges)
    model_sg = PhaseSubgraphGNN(num_nodes=N, node_in_dim=int(cfg_sg["node_in_dim"]), edge_in_dim=2, out_dim=1,
                               node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4,
                               max_edges_per_phase=max_e).to(DEVICE)
    model_sg.load_state_dict(ckpt_sg["state_dict"])
    model_sg.eval()

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()
    edge_csv_dist = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv_dist)
    phase_map = load_phase_mapping(DIR_LOADTYPE).to(DEVICE)
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
        x_sg = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        g_oh = Data(x=x_oh, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)
        g_sg = Data(x=x_sg, num_nodes=N, phase_idx=phase_map)
        for p, (eip, eap, eidp) in enumerate(phase_edges):
            g_sg[f"edge_index_{p}"] = eip.to(DEVICE)
            g_sg[f"edge_attr_{p}"] = eap.to(DEVICE)
            g_sg[f"edge_id_{p}"] = eidp.to(DEVICE)

        with torch.no_grad():
            V_oh[t, :] = model_oh(g_oh)[:, 0].cpu().numpy()
            V_sg[t, :] = model_sg(g_sg)[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    mae_oh = np.zeros(N)
    mae_sg = np.zeros(N)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_oh[i] = np.mean(np.abs(V_dss[ok, i] - V_oh[ok, i]))
            mae_sg[i] = np.mean(np.abs(V_dss[ok, i] - V_sg[ok, i]))
        else:
            mae_oh[i] = mae_sg[i] = np.nan

    mae_diff = np.abs(mae_oh - mae_sg)
    order = np.argsort(-mae_diff)
    worst_indices = order[:5]

    df = pd.DataFrame({"node": node_names_master, "mae_oh": mae_oh, "mae_sg": mae_sg, "mae_diff": mae_diff})
    df = df.sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_phase_onehot_vs_subgraph.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top 5 worst nodes (|MAE_oh - MAE_sg|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: one-hot MAE={mae_oh[idx]:.4f} | subgraph MAE={mae_sg[idx]:.4f}")
    print(f"Saved -> {csv_path}")

    for k, idx in enumerate(worst_indices):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_oh[:, idx], color="orange", linestyle="--",
                label=f"Phase one-hot (MAE={mae_oh[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_sg[:, idx], "g:", label=f"Phase subgraph (MAE={mae_sg[idx]:.4f})", linewidth=1.5)
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
    print("PHASE ONE-HOT vs PHASE SUBGRAPH: same 10k samples, same base features")
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

    print(f"\n>>> Training on {S} samples (medium: 4L, h=32)...")
    train_one(X_oh, Y_all, edge_index, edge_attr, edge_id, N, E, 16, CKPT_PHASE_ONEHOT,
              "Phase one-hot (13+3 feat)", use_phase_onehot=True)
    train_one(X_full, Y_all, edge_index, edge_attr, edge_id, N, E, 13, CKPT_PHASE_SUBGRAPH,
              "Phase subgraph (3 GNNs, 13 feat)", phase_edges=phase_edges, phase_map=phase_map)

    print("\n>>> Running 24h profile and plotting worst nodes...")
    run_24h_and_plot()
    print("=" * 70)


if __name__ == "__main__":
    main()
