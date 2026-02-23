"""
Train the 7 best models from GNN3 exploration (one per block).
Saves checkpoints to gnn3_best7_checkpoints/ for use in the 24h overlay.
Run from repo root. Requires: gnn_samples_loadtype_full, gnn_samples_deltav_full, gnn_samples_deltav_5x_full.
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

SEED = 20260130
DATA_FRAC = 1.0
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
MIN_EPOCHS_BEFORE_STOP = 12
MIN_DELTA = 1e-6
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "gnn3_best7_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 7 best models: (block_id, name, out_dir, feature_cols, target_col, n_emb, e_emb, h_dim, n_layers, use_norm, use_phase_onehot)
LOADTYPE_FEAT = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]
DELTAV_FEAT = LOADTYPE_FEAT + ["vmag_zero_pv_pu"]

MODELS = [
    (1, "light_xwide", "gnn_samples_loadtype_full", LOADTYPE_FEAT, "vmag_pu", 8, 4, 128, 2, False, False),
    (2, "light_emb_h96", "gnn_samples_loadtype_full", LOADTYPE_FEAT, "vmag_pu", 16, 8, 96, 2, False, False),
    (3, "light_xwide_emb_depth3", "gnn_samples_loadtype_full", LOADTYPE_FEAT, "vmag_pu", 16, 8, 128, 3, False, False),
    (4, "light_emb_h96_phase_onehot_depth3", "gnn_samples_loadtype_full", LOADTYPE_FEAT, "vmag_pu", 16, 8, 96, 3, False, True),
    (5, "light_emb_h96_phase_onehot_depth3_h112", "gnn_samples_loadtype_full", LOADTYPE_FEAT, "vmag_pu", 16, 8, 112, 3, False, True),
    (6, "light_xwide_emb_phase_onehot", "gnn_samples_deltav_full", DELTAV_FEAT, "vmag_delta_pu", 16, 8, 128, 2, False, True),
    (7, "light_xwide_emb_phase_onehot", "gnn_samples_deltav_5x_full", DELTAV_FEAT, "vmag_delta_pu", 16, 8, 128, 2, False, True),
]


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_phase_from_node_name(name):
    m = re.search(r"\.(\d+)$", str(name))
    return int(m.group(1)) if m else 1


def load_phase_mapping(out_dir):
    master = os.path.join(out_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master):
        return None
    df = pd.read_csv(master)
    phase = np.array([_parse_phase_from_node_name(n) - 1 for n in df["node"]], dtype=np.int64)
    return torch.tensor(phase, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EdgeIdentityMP(MessagePassing):
    def __init__(self, h_dim, edge_feat_dim, edge_emb_dim):
        super().__init__(aggr="add")
        self.psi = MLP(in_dim=h_dim + edge_feat_dim + edge_emb_dim, out_dim=h_dim, hidden=h_dim)

    def forward(self, h, edge_index, edge_attr, edge_emb):
        return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr, edge_emb=edge_emb)

    def message(self, h_j, edge_attr, edge_emb):
        return self.psi(torch.cat([h_j, edge_attr, edge_emb], dim=-1))


class PFIdentityGNN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_in_dim, edge_in_dim=2, out_dim=1,
                 node_emb_dim=8, edge_emb_dim=4, h_dim=64, num_layers=4, use_norm=False):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(num_edges, edge_emb_dim)
        self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim)
        self.use_norm = use_norm
        if use_norm:
            self.node_norm = nn.LayerNorm(node_in_dim + node_emb_dim)
            self.edge_norm = nn.LayerNorm(edge_in_dim)
        self.mps = nn.ModuleList([EdgeIdentityMP(h_dim, edge_in_dim, edge_emb_dim) for _ in range(num_layers)])
        self.updates = nn.ModuleList([MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim) for _ in range(num_layers)])
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def forward(self, data):
        x, edge_index, edge_attr, edge_id = data.x, data.edge_index, data.edge_attr, data.edge_id
        node_ids = self._local_node_ids_from_ptr(data.ptr.to(x.device)) if hasattr(data, "ptr") and data.ptr is not None else torch.arange(data.num_nodes, device=x.device)
        z = self.node_emb(node_ids)
        inp = torch.cat([x, z], dim=-1)
        if self.use_norm:
            inp = self.node_norm(inp)
            edge_attr = self.edge_norm(edge_attr)
        h = self.phi0(inp)
        r = self.edge_emb(edge_id)
        for mp, upd in zip(self.mps, self.updates):
            m = mp(h=h, edge_index=edge_index, edge_attr=edge_attr, edge_emb=r)
            h = upd(torch.cat([h, m, z], dim=-1))
        return self.readout(h)

    def _local_node_ids_from_ptr(self, ptr):
        ids = []
        for g in range(ptr.numel() - 1):
            n0, n1 = int(ptr[g].item()), int(ptr[g + 1].item())
            n = n1 - n0
            if n != self.num_nodes:
                raise RuntimeError(f"Batch graph {g} has {n} nodes != expected {self.num_nodes}.")
            ids.append(torch.arange(n, device=ptr.device))
        return torch.cat(ids, dim=0)


def compute_metrics(yhat, ytrue):
    err = (yhat - ytrue).squeeze(-1)
    return err.abs().mean(), torch.sqrt((err ** 2).mean())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae_sum = torch.tensor(0.0, device=DEVICE)
    rmse_sum = torch.tensor(0.0, device=DEVICE)
    n_batches = 0
    for data in loader:
        data = data.to(DEVICE)
        mae, rmse = compute_metrics(model(data), data.y)
        mae_sum += mae
        rmse_sum += rmse
        n_batches += 1
    return (mae_sum / max(1, n_batches)).item(), (rmse_sum / max(1, n_batches)).item()


def train_one(block_id, cfg_name, out_dir, feature_cols, target_col, n_emb, e_emb, h_dim, n_layers, use_norm, use_phase_onehot):
    seed_all(SEED)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        print(f"  [SKIP] Block {block_id}: Missing {out_dir}. Run dataset generation first.")
        return None
    required = {"sample_id", "node_idx", target_col} | set(feature_cols)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    if required - set(df_n.columns):
        print(f"  [SKIP] Block {block_id}: Missing columns {required - set(df_n.columns)}")
        return None

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + [target_col]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)

    E = int(len(df_e))
    N = int(df_n["node_idx"].max()) + 1
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    all_ids = df_n["sample_id"].unique()
    n_keep = max(1, int(len(all_ids) * DATA_FRAC))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_keep, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()
    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n[target_col].to_numpy(dtype=np.float32).reshape(S, N, 1)

    if use_phase_onehot:
        phase_map = load_phase_mapping(out_dir)
        if phase_map is None:
            print(f"  [SKIP] Block {block_id}: No phase mapping")
            return None
        ph_oh = F.one_hot(phase_map, num_classes=3).numpy().astype(np.float32)
        X_all = np.concatenate([X_all, np.broadcast_to(ph_oh[None, :, :], (S, N, 3))], axis=-1)

    node_in_dim = X_all.shape[-1]
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        return [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in idx]

    train_loader = DataLoader(make_ds(train_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(make_ds(test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                          node_emb_dim=n_emb, edge_emb_dim=e_emb, h_dim=h_dim, num_layers=n_layers,
                          use_norm=use_norm).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_test, best_state, best_epoch, best_mae, best_rmse = float("inf"), None, None, None, None
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        mae_t, rmse_t = evaluate(model, test_loader)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test, best_state, best_epoch = rmse_t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
            best_mae, best_rmse, patience_left = mae_t, rmse_t, EARLY_STOP_PATIENCE
        else:
            if epoch >= MIN_EPOCHS_BEFORE_STOP:
                patience_left -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}")
        if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break

    if best_state is None:
        return None
    model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(model, test_loader)

    ckpt_path = os.path.join(OUTPUT_DIR, f"block{block_id}.pt")
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "N": N, "E": E, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
            "node_emb_dim": n_emb, "edge_emb_dim": e_emb, "h_dim": h_dim, "num_layers": n_layers,
            "use_norm": use_norm, "target_col": target_col, "dataset": out_dir,
            "use_phase_onehot": use_phase_onehot,
        },
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_id": edge_id,
        "best_mae": best_mae, "best_rmse": best_rmse, "best_epoch": best_epoch,
    }
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] Block {block_id} -> {ckpt_path} | MAE={mae_f:.6f} RMSE={rmse_f:.6f}")
    return ckpt_path


def main():
    print("=" * 70)
    print("GNN3 BEST 7: Train and save checkpoints for overlay")
    print("=" * 70)
    for tup in MODELS:
        block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph = tup
        print(f"\n>>> Block {block_id}: {name} + {out_dir} (target={target})")
        train_one(block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph)
    print("\n" + "=" * 70)
    print("Done. Checkpoints in", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
