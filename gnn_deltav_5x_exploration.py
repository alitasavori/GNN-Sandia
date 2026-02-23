"""
Delta-V 5× exploration: best architectures on fifth dataset (5× PV scaling).
Uses gnn_samples_deltav_5x_full with target vmag_delta_pu (larger delta-V than baseline).
Features: Load-type (13) + vmag_zero_pv_pu = 14 features.
30% data, MSE loss, no dropout.
Reports target mean/std and compares RMSE to mean for relative performance.
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

DATA_FRAC = 0.30
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
MIN_EPOCHS_BEFORE_STOP = 12
MIN_DELTA = 1e-6
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "gnn_samples_deltav_5x_full"
FEATURE_COLS = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
    "p_sys_balance_kw", "q_sys_balance_kvar", "vmag_zero_pv_pu",
]
TARGET_COL = "vmag_delta_pu"

CANDIDATES = [
    ("light_emb_h96_phase_onehot_depth3", 16, 8, 96, 3, False, True),
    ("light_xwide_emb_phase_onehot_depth3_h160", 16, 8, 160, 3, False, True),
    ("light_xwide_emb_depth3", 16, 8, 128, 3, False, False),
    ("light_emb_h96_phase_onehot", 16, 8, 96, 2, False, True),
    ("light_xwide_emb_phase_onehot", 16, 8, 128, 2, False, True),
    ("light_emb_h96_norm", 16, 8, 96, 2, True, False),
    ("light_emb_h96_depth3", 16, 8, 96, 3, False, False),
    ("wide_shallow_h160_depth3", 8, 4, 160, 3, False, False),
    ("light_xwide_emb_norm", 16, 8, 128, 2, True, False),
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


def report_target_stats(out_dir, target_col=TARGET_COL):
    """Report mean, std, min, max of target for comparison with model performance."""
    nc = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(nc):
        return None
    df = pd.read_csv(nc)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    y = df[target_col].dropna()
    if len(y) == 0:
        return None
    return {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "n": len(y),
        "n_samples": int(df["sample_id"].nunique()),
    }


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


def train_one(out_dir, feature_cols, target_col, cfg_name, n_emb, e_emb, h_dim, n_layers,
              use_norm, use_phase_onehot, data_frac=0.30):
    seed_all(SEED)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        print(f"  [SKIP] Missing {out_dir}. Run GNN2 fifth dataset block first.")
        return None, None, None
    required_node_cols = {"sample_id", "node_idx", target_col} | set(feature_cols)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    if required_node_cols - set(df_n.columns):
        missing = required_node_cols - set(df_n.columns)
        print(f"  [SKIP] Missing columns: {missing}")
        return None, None, None

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + [target_col]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_node_cols)).copy()
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)

    E = int(len(df_e))
    N = int(df_n["node_idx"].max()) + 1
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    all_sample_ids = df_n["sample_id"].unique()
    n_keep = max(1, int(len(all_sample_ids) * data_frac))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_sample_ids, size=n_keep, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()
    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n[target_col].to_numpy(dtype=np.float32).reshape(S, N, 1)

    if use_phase_onehot:
        phase_map = load_phase_mapping(out_dir)
        if phase_map is None:
            print(f"  [SKIP] No phase mapping in {out_dir}")
            return None, None, None
        phase_onehot = F.one_hot(phase_map, num_classes=3).numpy().astype(np.float32)
        X_all = np.concatenate([X_all, np.broadcast_to(phase_onehot[None, :, :], (S, N, 3))], axis=-1)

    node_in_dim = X_all.shape[-1]
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        out = []
        for k in idx:
            x = torch.tensor(X_all[k], dtype=torch.float)
            y = torch.tensor(Y_all[k], dtype=torch.float)
            d = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N)
            out.append(d)
        return out

    train_data, test_data = make_ds(train_idx), make_ds(test_idx)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                         node_emb_dim=n_emb, edge_emb_dim=e_emb, h_dim=h_dim, num_layers=n_layers,
                         use_norm=use_norm).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    loss_fn = F.mse_loss

    best_test, best_state, best_epoch = float("inf"), None, None
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(data), data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        train_loss = total_loss / max(1, nb)
        mae_t, rmse_t = evaluate(model, test_loader)
        scheduler.step(rmse_t)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test, best_state, best_epoch = rmse_t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
            patience_left = EARLY_STOP_PATIENCE
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
    print("DELTA-V 5× EXPLORATION: Best Load-type configs on fifth dataset (vmag_delta_pu, 5× PV)")
    print(f"Dataset: {OUT_DIR} | Features: Load-type + vmag_zero_pv_pu | Target: {TARGET_COL}")
    print(f"30% data | {len(CANDIDATES)} configs")
    print(f"Working dir: {os.path.abspath('.')}")
    ec = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
    nc = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
    print(f"  {OUT_DIR}: {'OK' if os.path.exists(ec) and os.path.exists(nc) else 'MISSING'} (edge={os.path.exists(ec)}, node={os.path.exists(nc)})")
    print("=" * 70)

    # Report target stats (mean, std) for comparison
    stats = report_target_stats(OUT_DIR)
    if stats:
        print("\n--- Target (vmag_delta_pu) statistics ---")
        print(f"  mean = {stats['mean']:.6f}")
        print(f"  std  = {stats['std']:.6f}")
        print(f"  min  = {stats['min']:.6f}")
        print(f"  max  = {stats['max']:.6f}")
        print(f"  n_node_rows = {stats['n']:,} | n_samples = {stats['n_samples']:,}")
        print("  (Compare RMSE to mean/std for relative performance)")
        print("-" * 70)
    else:
        print("  [SKIP] Cannot compute target stats (missing data)")
        return

    results = []
    for cfg in CANDIDATES:
        cfg_name, n_emb, e_emb, h_dim, n_layers, use_norm, use_phase_onehot = cfg
        print(f"\n>>> {cfg_name} + Delta-V 5× (n={n_layers} h={h_dim} norm={use_norm} phase_oh={use_phase_onehot})")
        try:
            mae, rmse, best_ep = train_one(OUT_DIR, FEATURE_COLS, TARGET_COL, cfg_name, n_emb, e_emb, h_dim, n_layers,
                                          use_norm, use_phase_onehot, data_frac=DATA_FRAC)
            if mae is not None:
                results.append((cfg_name, mae, rmse, best_ep))
                print(f"  FINAL: MAE={mae:.6f} RMSE={rmse:.6f} best_epoch={best_ep}")
            else:
                print(f"  SKIP (missing data)")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by MAE)")
    print("=" * 70)
    results.sort(key=lambda r: r[1])
    df_res = pd.DataFrame(results, columns=["Config", "MAE", "RMSE", "Best_epoch"])
    if len(df_res) > 0:
        print(df_res.to_string())
        best = df_res.iloc[0]
        print(f"\nBest: {best['Config']} + Delta-V 5× | MAE={best['MAE']:.6f} | RMSE={best['RMSE']:.6f}")

        # Compare to target mean
        if stats:
            mean_abs = abs(stats["mean"])
            std_val = stats["std"]
            best_rmse = best["RMSE"]
            print("\n--- Performance vs target scale ---")
            print(f"  Target mean = {stats['mean']:.6f} | std = {std_val:.6f}")
            print(f"  Best RMSE  = {best_rmse:.6f}")
            if mean_abs > 1e-9:
                print(f"  RMSE/|mean| = {best_rmse / mean_abs:.2%}  (lower is better)")
            if std_val > 1e-9:
                print(f"  RMSE/std   = {best_rmse / std_val:.2%}  (lower is better; <100% beats mean predictor)")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
