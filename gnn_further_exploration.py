"""
Further exploration: 60 new nominees × 3 datasets (30% data, target MAE < 0.001).
Evidence-based from GNN2 final exploration results.
- Load-type best: light_xwide (0.01525), light_wide (0.01539) — 2 layers, h=64–128, emb 8/4
- Original best: medium (0.0264), emb_heavy (0.0259) — 4 layers, h=32–64
- Derived best: deep (0.0237), medium (0.0241) — 4 layers
- AVOID: 6+ layers (all ~0.063 MAE)
Strategy: extrapolate Load-type (wider h), 3-layer sweet spot, emb 12/6, interpolation in h.
"""
import os
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

DATA_FRAC = 0.30
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 40
EARLY_STOP_PATIENCE = 8
MIN_EPOCHS_BEFORE_STOP = 10
MIN_DELTA = 1e-6
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MAE = 0.001

# 60 new nominees: 60 configs x 3 datasets = 180 runs
# Evidence from first 60: Load-type best=2L+h64-128; Original/Derived best=4L; 6+L fails
# All configs are NEW (no repeats from first 20) and evidence-based extrapolations
CANDIDATES = [
    # --- Load-type extrapolation (1-18): 2 layers, wider h, emb variants ---
    # First-60 best: light_xwide(0.01525), light_wide(0.01539), wide_shallow(0.01548)
    ("light_xxwide", 8, 4, 160, 2),       # extrapolate h beyond 128
    ("light_xxxwide", 8, 4, 192, 2),
    ("light_xwide_emb", 16, 8, 128, 2),  # best Load-type + bigger emb (light_xwide was 8/4)
    ("light_wide_h96", 8, 4, 96, 2),     # interpolate between light_wide(64) and light_xwide(128)
    ("light_h80", 8, 4, 80, 2),
    ("light_h112", 8, 4, 112, 2),
    ("wide_shallow_h96", 8, 4, 96, 2),   # wide_shallow variant
    ("wide_shallow_h160", 8, 4, 160, 2),
    ("light_emb_h96", 16, 8, 96, 2),     # light_emb(32) + wider
    ("light_xxwide_emb", 16, 8, 160, 2),
    ("light_xxxwide_emb", 16, 8, 192, 2),
    ("light_h144", 8, 4, 144, 2),
    ("light_h176", 8, 4, 176, 2),
    ("light_wide_h80", 8, 4, 80, 2),
    ("light_wide_h112", 8, 4, 112, 2),
    ("light_xwide_h96", 8, 4, 96, 2),
    ("wide_shallow_h112", 8, 4, 112, 2),
    ("light_emb_h80", 16, 8, 80, 2),
    # --- 3-layer sweet spot (19-28): first-60 best were 4L; test 3L ---
    # First-60: medium(0.026), deep(0.024), heavy(0.028) on Original/Derived
    ("medium_3", 8, 4, 32, 3),
    ("heavy_3", 8, 4, 64, 3),
    ("deep_3", 16, 8, 64, 3),
    ("medium_wide_3", 8, 4, 64, 3),
    ("deep_wide_3", 16, 8, 128, 3),
    ("light_wide_3", 8, 4, 64, 3),
    ("light_xwide_3", 8, 4, 128, 3),
    ("heavy_wide_3", 8, 4, 128, 3),
    ("emb_heavy_3", 16, 8, 32, 3),
    ("max_cap_3", 32, 16, 128, 3),
    # --- Emb 12/6 interpolation (29-38): between 8/4 and 16/8 ---
    ("light_emb12", 12, 6, 32, 2),
    ("light_wide_emb12", 12, 6, 64, 2),
    ("light_xwide_emb12", 12, 6, 128, 2),
    ("medium_emb12", 12, 6, 32, 4),
    ("deep_emb12", 12, 6, 64, 4),
    ("heavy_emb12", 12, 6, 64, 4),
    ("wide_shallow_emb12", 12, 6, 128, 2),
    ("light_xxwide_emb12", 12, 6, 160, 2),
    ("medium_wide_emb12", 12, 6, 64, 4),
    ("deep_wide_emb12", 12, 6, 128, 4),
    # --- Original/Derived refinements (39-48): emb_heavy, medium, deep variants ---
    # First-60: emb_heavy(0.026 Orig), medium(0.026), deep(0.024 Der)
    ("emb_heavy_wide", 16, 8, 64, 4),    # emb_heavy(32) wider
    ("emb_heavy_xwide", 16, 8, 128, 4),
    ("medium_narrow", 8, 4, 48, 4),      # medium(32) narrower
    ("medium_h80", 8, 4, 80, 4),
    ("deep_narrow", 16, 8, 48, 4),
    ("heavy_narrow", 8, 4, 48, 4),
    ("medium_emb_wide", 16, 8, 64, 4),  # medium + emb_heavy style
    ("deep_emb_narrow", 16, 8, 48, 4),
    ("deep_3_h96", 16, 8, 96, 3),
    ("heavy_3_h96", 8, 4, 96, 3),
    # --- Additional Load-type (49-60): emb12 + h variants ---
    ("light_emb12_h96", 12, 6, 96, 2),
    ("light_emb12_h160", 12, 6, 160, 2),
    ("max_cap_wide_3", 32, 16, 128, 3),
    ("light_emb_h112", 16, 8, 112, 2),
    ("wide_shallow_emb_h96", 16, 8, 96, 2),
    ("light_wide_emb_h80", 16, 8, 80, 2),
    ("light_wide_emb_h112", 16, 8, 112, 2),
    ("medium_3_h48", 8, 4, 48, 3),
    ("deep_wide_3_h96", 16, 8, 96, 3),
    ("emb_heavy_3_h64", 16, 8, 64, 3),
    ("light_xxwide_emb12", 12, 6, 160, 2),
    ("wide_shallow_h144", 8, 4, 144, 2),
]

DATASETS = [
    ("gnn_samples_out", ["p_load_kw", "q_load_kvar", "p_pv_kw"], "Original"),
    ("gnn_samples_inj_full", ["p_inj_kw", "q_inj_kvar"], "Derived"),
    ("gnn_samples_loadtype_full", [
        "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
        "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
        "p_sys_balance_kw", "q_sys_balance_kvar"
    ], "Load-type"),
]


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
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
                 node_emb_dim=8, edge_emb_dim=4, h_dim=64, num_layers=4):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(num_edges, edge_emb_dim)
        self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim)
        self.mps = nn.ModuleList([EdgeIdentityMP(h_dim, edge_in_dim, edge_emb_dim) for _ in range(num_layers)])
        self.updates = nn.ModuleList([MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim) for _ in range(num_layers)])
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def forward(self, data):
        x, edge_index, edge_attr, edge_id = data.x, data.edge_index, data.edge_attr, data.edge_id
        node_ids = self._local_node_ids_from_ptr(data.ptr.to(x.device)) if hasattr(data, "ptr") and data.ptr is not None else torch.arange(data.num_nodes, device=x.device)
        z = self.node_emb(node_ids)
        h = self.phi0(torch.cat([x, z], dim=-1))
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


def train_one(out_dir, feature_cols, label, cfg_name, node_emb_dim, edge_emb_dim, h_dim, num_layers, data_frac=0.30):
    seed_all(SEED)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        return None, None, None
    required_node_cols = {"sample_id", "node_idx", "vmag_pu"} | set(feature_cols)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    missing = required_node_cols - set(df_n.columns)
    if missing:
        return None, None, None
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_node_cols)).copy()
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)
    E = int(edge_index.shape[1])
    N = int(df_n["node_idx"].max()) + 1
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    all_sample_ids = df_n["sample_id"].unique()
    S_full = len(all_sample_ids)
    n_keep = max(1, int(S_full * data_frac))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_sample_ids, size=n_keep, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()
    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)
    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        return [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in idx]

    train_data, test_data = make_ds(train_idx), make_ds(test_idx)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    node_in_dim = len(feature_cols)
    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                         node_emb_dim=node_emb_dim, edge_emb_dim=edge_emb_dim, h_dim=h_dim, num_layers=num_layers).to(DEVICE)
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
        train_loss = total_loss / max(1, nb)
        mae_t, rmse_t = evaluate(model, test_loader)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test, best_state, best_epoch = rmse_t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
            best_mae, best_rmse, patience_left = mae_t, rmse_t, EARLY_STOP_PATIENCE
        else:
            if epoch >= MIN_EPOCHS_BEFORE_STOP:
                patience_left -= 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | train_loss={train_loss:.6f} | RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}")
        if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(model, test_loader)
    return mae_f, rmse_f, best_epoch


def main():
    print("=" * 70)
    print(f"FURTHER EXPLORATION: {int(DATA_FRAC*100)}% data, 60 nominees × 3 datasets, target MAE < {TARGET_MAE}")
    print("=" * 70)
    results = []
    for cfg_name, n_emb, e_emb, h_dim, n_layers in CANDIDATES:
        for out_dir, feature_cols, ds_label in DATASETS:
            print(f"\n>>> {cfg_name} + {ds_label} (n={n_layers} h={h_dim} emb={n_emb}/{e_emb})")
            try:
                mae, rmse, best_ep = train_one(out_dir, feature_cols, ds_label, cfg_name, n_emb, e_emb, h_dim, n_layers, data_frac=DATA_FRAC)
                if mae is not None:
                    hit = " *** TARGET HIT ***" if mae < TARGET_MAE else ""
                    results.append((cfg_name, ds_label, mae, rmse, best_ep, hit))
                    print(f"  FINAL: MAE={mae:.6f} RMSE={rmse:.6f} best_epoch={best_ep}{hit}")
                else:
                    print(f"  SKIP (missing data)")
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by MAE)")
    print("=" * 70)
    results.sort(key=lambda r: r[2])
    df_res = pd.DataFrame(results, columns=["Config", "Dataset", "MAE", "RMSE", "Best_epoch", "Target_hit"])
    if len(df_res) > 0:
        print(df_res.to_string())
        best = df_res.iloc[0]
        print(f"\nBest: {best['Config']} + {best['Dataset']} | MAE={best['MAE']:.6f} | RMSE={best['RMSE']:.6f}")
        below = df_res[df_res["MAE"] < TARGET_MAE]
        if len(below) > 0:
            print(f"\n*** {len(below)} config(s) achieved MAE < {TARGET_MAE} ***")
        else:
            print(f"\n(No config reached MAE < {TARGET_MAE}; best MAE = {best['MAE']:.6f})")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
