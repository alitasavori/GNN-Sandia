"""
Train two models on the same 10k samples from gnn_samples_loadtype_full.
Both use 10 features (loadtype per-type): m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv.
Both trained and evaluated on all phases (full graph):
  (A) With node and edge embedding vectors (PFIdentityGNN)
  (B) Without node and edge embedding vectors (PFIdentityGNNNoEmb)
Same data, same graph. Find 5 worst nodes, plot vs OpenDSS.
Run from repo root. Requires: gnn_samples_loadtype_full
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_overlay_7 import (
    BASE_DIR, CAP_Q_KVAR, NPTS, P_BASE, Q_BASE, PV_BASE, STEP_MIN,
    build_bus_to_phases_from_master_nodes, build_gnn_x_loadtype_per_type,
    get_all_node_voltage_pu_and_angle_dict,
    find_loadshape_csv_in_dss, resolve_csv_path, read_profile_csv_two_col_noheader,
)
from run_gnn3_best7_train import PFIdentityGNN

os.chdir(BASE_DIR)

DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
CKPT_WITH_EMB = os.path.join(OUTPUT_DIR, "block_10feat_with_emb.pt")
CKPT_NO_EMB = os.path.join(OUTPUT_DIR, "block_10feat_no_emb.pt")

TARGET_SAMPLES = 10000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOADTYPE_PER_TYPE_FEAT = [
    "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar",
    "q_cap_kvar", "p_pv_kw",
]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeIdentityMPNoEmb(MessagePassing):
    """Message passing without edge embeddings: MLP(h_j, edge_attr) only."""
    def __init__(self, h_dim, edge_feat_dim):
        super().__init__(aggr="add")
        self.psi = MLP(in_dim=h_dim + edge_feat_dim, out_dim=h_dim, hidden=h_dim)

    def forward(self, h, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr)

    def message(self, h_j, edge_attr):
        return self.psi(torch.cat([h_j, edge_attr], dim=-1))


class PFIdentityGNNNoEmb(nn.Module):
    """GNN without node or edge embeddings. Uses only input features and edge attributes."""
    def __init__(self, num_nodes, num_edges, node_in_dim, edge_in_dim=2, out_dim=1,
                 h_dim=64, num_layers=4):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.phi0 = MLP(in_dim=node_in_dim, out_dim=h_dim, hidden=h_dim)
        self.mps = nn.ModuleList([EdgeIdentityMPNoEmb(h_dim, edge_in_dim) for _ in range(num_layers)])
        self.updates = nn.ModuleList([MLP(in_dim=h_dim + h_dim, out_dim=h_dim, hidden=h_dim) for _ in range(num_layers)])
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.phi0(x)
        for mp, upd in zip(self.mps, self.updates):
            m = mp(h=h, edge_index=edge_index, edge_attr=edge_attr)
            h = upd(torch.cat([h, m], dim=-1))
        return self.readout(h)


def train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, model, ckpt_path, name, with_embeddings):
    """Train model, save checkpoint."""
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

    model = model.to(DEVICE)
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
        "config": {"N": N, "E": E, "node_in_dim": 10, "edge_in_dim": 2, "out_dim": 1,
                   "h_dim": 32, "num_layers": 4, "target_col": "vmag_pu", "dataset": DIR_LOADTYPE,
                   "with_embeddings": model.__class__.__name__ == "PFIdentityGNN"},
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def run_24h_and_plot():
    """Load both models, run 24h profile, find worst 5 nodes, plot."""
    ckpt_emb = torch.load(CKPT_WITH_EMB, map_location="cpu")
    ckpt_no = torch.load(CKPT_NO_EMB, map_location="cpu")
    cfg_emb, cfg_no = ckpt_emb["config"], ckpt_no["config"]
    N = int(cfg_emb["N"])
    E = int(cfg_emb["E"])

    model_emb = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=10, edge_in_dim=2, out_dim=1,
                              node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    model_emb.load_state_dict(ckpt_emb["state_dict"])
    model_emb.eval()

    model_no = PFIdentityGNNNoEmb(num_nodes=N, num_edges=E, node_in_dim=10, edge_in_dim=2, out_dim=1,
                                  h_dim=32, num_layers=4).to(DEVICE)
    model_no.load_state_dict(ckpt_no["state_dict"])
    model_no.eval()

    node_index_csv = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
    master_df = pd.read_csv(node_index_csv)
    node_names_master = master_df["node"].astype(str).tolist()

    ei = ckpt_emb["edge_index"].to(DEVICE)
    ea = ckpt_emb["edge_attr"].to(DEVICE)
    eid = ckpt_emb["edge_id"].to(DEVICE)

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
    V_emb = np.full((NPTS, N), np.nan)
    V_no = np.full((NPTS, N), np.nan)

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

        X = build_gnn_x_loadtype_per_type(node_names_master, busph_per_type, busphP_pv)
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        g = Data(x=x, edge_index=ei, edge_attr=ea, edge_id=eid, num_nodes=N)

        with torch.no_grad():
            V_emb[t, :] = model_emb(g)[:, 0].cpu().numpy()
            V_no[t, :] = model_no(g)[:, 0].cpu().numpy()

    t_hours = np.arange(NPTS) * STEP_MIN / 60.0
    mae_emb = np.full(N, np.nan)
    mae_no = np.full(N, np.nan)
    for i in range(N):
        ok = np.isfinite(V_dss[:, i])
        if np.sum(ok) > 0:
            mae_emb[i] = np.mean(np.abs(V_dss[ok, i] - V_emb[ok, i]))
            mae_no[i] = np.mean(np.abs(V_dss[ok, i] - V_no[ok, i]))

    mae_diff = np.abs(mae_emb - mae_no)
    order = np.argsort(-np.nan_to_num(mae_diff, nan=-np.inf))
    worst_indices = list(order[:5])

    df = pd.DataFrame({"node": node_names_master, "mae_emb": mae_emb, "mae_no": mae_no, "mae_diff": mae_diff})
    df = df.sort_values("mae_diff", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "mae_per_node_emb_vs_no_emb.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top 5 worst nodes (|MAE_emb - MAE_no|):")
    for k, idx in enumerate(worst_indices):
        print(f"  {k+1}. {node_names_master[idx]}: with_emb MAE={mae_emb[idx]:.4f} | no_emb MAE={mae_no[idx]:.4f}")
    print(f"Saved -> {csv_path}")

    for k, idx in enumerate(worst_indices):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, V_dss[:, idx], "b-", label="OpenDSS |V| (pu)", linewidth=2)
        ax.plot(t_hours, V_emb[:, idx], color="orange", linestyle="--",
                label=f"With embeddings (MAE={mae_emb[idx]:.4f})", linewidth=1.5)
        ax.plot(t_hours, V_no[:, idx], "g:", label=f"No embeddings (MAE={mae_no[idx]:.4f})", linewidth=1.5)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {node_names_master[idx]} (worst #{k+1}, PV={PV_SCALE:.1f}×)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"overlay_24h_emb_vs_no_emb_worst_{k+1}_{node_names_master[idx].replace('.', '_')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved -> {out_path}")


def main():
    print("=" * 70)
    print("WITH vs WITHOUT EMBEDDINGS: same 10k samples, 10 feat (loadtype per-type), all phases")
    print("=" * 70)

    node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    if not os.path.exists(node_csv) or not os.path.exists(edge_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    df_n = pd.read_csv(node_csv)
    for c in LOADTYPE_PER_TYPE_FEAT + ["vmag_pu", "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.dropna(subset=LOADTYPE_PER_TYPE_FEAT + ["vmag_pu"])

    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    all_ids = df_n["sample_id"].unique()
    n_want = min(TARGET_SAMPLES, len(all_ids))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_want, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    N = int(df_n["node_idx"].max()) + 1
    df_e = pd.read_csv(edge_csv)
    E = len(df_e)
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    S = df_n["sample_id"].nunique()
    X_all = df_n[LOADTYPE_PER_TYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    print(f"\n>>> Training on {S} samples (medium: 4L, h=32)...")
    model_emb = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=10, edge_in_dim=2, out_dim=1,
                              node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False)
    train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, model_emb, CKPT_WITH_EMB,
              "10 feat with node+edge embeddings", with_embeddings=True)

    model_no = PFIdentityGNNNoEmb(num_nodes=N, num_edges=E, node_in_dim=10, edge_in_dim=2, out_dim=1,
                                  h_dim=32, num_layers=4)
    train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, model_no, CKPT_NO_EMB,
              "10 feat without embeddings", with_embeddings=False)

    print("\n>>> Running 24h profile and plotting 5 worst nodes...")
    run_24h_and_plot()
    print("=" * 70)


if __name__ == "__main__":
    main()
