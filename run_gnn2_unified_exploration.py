"""
Unified exploration: multiple architectures × all 3 GNN2 datasets.

- Datasets (new unified layout):
    * datasets_gnn2/original   : Original (3 features: p_load, q_load, p_pv)
    * datasets_gnn2/injection  : Injection (2 features: p_inj, q_inj)
    * datasets_gnn2/loadtype   : Load-type (13 features)
- For each architecture config (light/medium/heavy/deep), trains on each dataset.
- Early stopping: patience 12, min 15 epochs, max 60.
- Saves checkpoints:
    * Per-dataset (for backward compatibility): <dataset_dir>/checkpoints/pf_identity_gnn_unified_<cfg>.pt
    * Unified model root: models_gnn2/unified_exploration/<dataset_slug>/<config_slug>/model.pt
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

# Unified model root for this exploration
MODEL_ROOT = os.path.join(BASE_DIR, "models_gnn2", "unified_exploration")
os.makedirs(MODEL_ROOT, exist_ok=True)


# Architecture configs to explore (name, node_emb, edge_emb, hidden, num_layers)
CONFIGS = [
    ("light", 8, 4, 32, 2),
    ("medium", 8, 4, 32, 4),
    ("heavy", 8, 4, 64, 4),
    ("deep", 16, 8, 64, 4),
]

SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 60          # max cap — allows full convergence
EARLY_STOP_PATIENCE = 12
MIN_EPOCHS_BEFORE_STOP = 15
MIN_DELTA = 1e-6
LR = 1e-3
WEIGHT_DECAY = 1e-5
NODE_EMB_DIM = 8
EDGE_EMB_DIM = 4
HIDDEN = 32
NUM_LAYERS = 4
DROPOUT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use unified dataset directories
DATASETS = [
    (os.path.join("datasets_gnn2", "original"), ["p_load_kw", "q_load_kvar", "p_pv_kw"], "Original (3 feat)"),
    (os.path.join("datasets_gnn2", "injection"), ["p_inj_kw", "q_inj_kvar"], "Derived (2 feat)"),
    (os.path.join("datasets_gnn2", "loadtype"), [
        "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
        "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
        "p_sys_balance_kw", "q_sys_balance_kvar"
    ], "Load-type (13 feat)"),
]


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout and dropout > 0:
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
    def __init__(
        self,
        num_nodes,
        num_edges,
        node_in_dim,
        edge_in_dim=2,
        out_dim=1,
        node_emb_dim=NODE_EMB_DIM,
        edge_emb_dim=EDGE_EMB_DIM,
        h_dim=HIDDEN,
        num_layers=NUM_LAYERS,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(num_edges, edge_emb_dim)
        self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim)
        self.mps = nn.ModuleList(
            [EdgeIdentityMP(h_dim, edge_in_dim, edge_emb_dim) for _ in range(num_layers)]
        )
        self.updates = nn.ModuleList(
            [MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim) for _ in range(num_layers)]
        )
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def _local_node_ids_from_ptr(self, ptr):
        ids = []
        for g in range(ptr.numel() - 1):
            n0, n1 = int(ptr[g].item()), int(ptr[g + 1].item())
            ids.append(torch.arange(n1 - n0, device=ptr.device))
        return torch.cat(ids, dim=0)

    def forward(self, data):
        x, edge_index, edge_attr, edge_id = data.x, data.edge_index, data.edge_attr, data.edge_id
        node_ids = (
            self._local_node_ids_from_ptr(data.ptr.to(x.device))
            if hasattr(data, "ptr") and data.ptr is not None
            else torch.arange(data.num_nodes, device=x.device)
        )
        z = self.node_emb(node_ids)
        h = self.phi0(torch.cat([x, z], dim=-1))
        r = self.edge_emb(edge_id)
        for mp, upd in zip(self.mps, self.updates):
            m = mp(h=h, edge_index=edge_index, edge_attr=edge_attr, edge_emb=r)
            h = upd(torch.cat([h, m, z], dim=-1))
        return self.readout(h)


def compute_metrics(yhat, ytrue):
    err = (yhat - ytrue).squeeze(-1)
    return err.abs().mean(), torch.sqrt((err**2).mean())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae_sum = rmse_sum = torch.tensor(0.0, device=DEVICE)
    n_batches = 0
    for data in loader:
        data = data.to(DEVICE)
        mae, rmse = compute_metrics(model(data), data.y)
        mae_sum += mae
        rmse_sum += rmse
        n_batches += 1
    return (mae_sum / max(1, n_batches)).item(), (rmse_sum / max(1, n_batches)).item()


def train_one_dataset(out_dir, feature_cols, label, cfg_name, node_emb_dim, edge_emb_dim, h_dim, num_layers):
    seed_all(SEED)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        print(f"[SKIP] {label}: missing CSVs in {out_dir}")
        return None, None, None, None

    required_node_cols = {"sample_id", "node_idx", "vmag_pu"} | set(feature_cols)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    missing = required_node_cols - set(df_n.columns)
    if missing:
        print(f"[SKIP] {label}: missing cols {missing}")
        return None, None, None, None

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)

    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")

    df_e = (
        df_e.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["u_idx", "v_idx", "R_full", "X_full"])
        .copy()
    )
    df_n = (
        df_n.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(required_node_cols))
        .copy()
    )
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
    df_n = (
        df_n[df_n["sample_id"].isin(good_ids)]
        .copy()
        .sort_values(["sample_id", "node_idx"])
        .reset_index(drop=True)
    )

    S = df_n["sample_id"].nunique()
    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        return [
            Data(
                x=torch.tensor(X_all[k], dtype=torch.float),
                y=torch.tensor(Y_all[k], dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_id=edge_id,
                num_nodes=N,
            )
            for k in idx
        ]

    train_data, test_data = make_ds(train_idx), make_ds(test_idx)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    node_in_dim = len(feature_cols)
    model = PFIdentityGNN(
        num_nodes=N,
        num_edges=E,
        node_in_dim=node_in_dim,
        edge_in_dim=2,
        out_dim=1,
        node_emb_dim=node_emb_dim,
        edge_emb_dim=edge_emb_dim,
        h_dim=h_dim,
        num_layers=num_layers,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test, best_state, best_epoch, best_mae, best_rmse = float("inf"), None, None, None, None
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += float(loss.item())

        mae_t, rmse_t = evaluate(model, test_loader)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test = rmse_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_mae, best_rmse, patience_left = mae_t, rmse_t, EARLY_STOP_PATIENCE
        else:
            if epoch >= MIN_EPOCHS_BEFORE_STOP:
                patience_left -= 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{cfg_name}] {label} | Epoch {epoch:02d} "
                f"| RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}"
            )
        if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    mae_f, rmse_f = evaluate(model, test_loader)

    # Per-node mean abs error for worst-node analysis (batched — same speed as evaluate)
    node_err_sum = np.zeros(N)
    node_cnt = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            pred = model(data)
            err = (pred - data.y).abs().squeeze(-1)  # [total_nodes]
            B = data.num_graphs
            err_2d = err.reshape(B, N).cpu().numpy()
            node_err_sum += err_2d.sum(axis=0)
            node_cnt += B
    mean_abs_err = node_err_sum / max(1, node_cnt)

    node_idx_to_name = {}
    if "node" in df_n.columns:
        for _, row in df_n[["node_idx", "node"]].drop_duplicates("node_idx").iterrows():
            node_idx_to_name[int(row["node_idx"])] = str(row["node"])

    top_k = 30
    worst_idx = np.argsort(-mean_abs_err)[:top_k]
    print(f"  Top {top_k} worst nodes (by mean |V| error):")
    for r, idx in enumerate(worst_idx, 1):
        name = node_idx_to_name.get(idx, f"node_{idx}")
        print(f"    {r:2d}. {name:12s}  mean_|err|={mean_abs_err[idx]:.6f} pu")

    # Per-dataset checkpoint (backwards compatible with older scripts)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"pf_identity_gnn_unified_{cfg_name}.pt")
    node_idx_to_name_list = [node_idx_to_name.get(i, f"node_{i}") for i in range(N)]
    ckpt = {
        "config": dict(
            node_in_dim=node_in_dim,
            edge_in_dim=2,
            out_dim=1,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            h_dim=h_dim,
            num_layers=num_layers,
            N=N,
            E=E,
        ),
        "state_dict": model.state_dict(),
        "best_score": float(best_test) if best_test != float("inf") else float(rmse_f),
        "mae_test": float(mae_f),
        "rmse_test": float(rmse_f),
        "edge_index": edge_index.cpu(),
        "edge_attr": edge_attr.cpu(),
        "edge_id": edge_id.cpu(),
        "node_idx_to_name": node_idx_to_name_list,
    }
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")

    # Unified model root checkpoint (for later profile comparison scripts)
    ds_slug = str(label).split("(")[0].strip().lower().replace(" ", "_")
    cfg_slug = str(cfg_name).lower()
    unified_dir = os.path.join(MODEL_ROOT, ds_slug, cfg_slug)
    os.makedirs(unified_dir, exist_ok=True)
    unified_ckpt_path = os.path.join(unified_dir, "model.pt")
    unified_ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "num_nodes": N,
            "num_edges": E,
            "node_in_dim": node_in_dim,
            "edge_in_dim": 2,
            "out_dim": 1,
            "node_emb_dim": node_emb_dim,
            "edge_emb_dim": edge_emb_dim,
            "h_dim": h_dim,
            "num_layers": num_layers,
            "dataset_dir": out_dir,
            "feature_cols": feature_cols,
            "target_col": "vmag_pu",
            "label": label,
        },
        "metrics": {
            "best_epoch": best_epoch,
            "best_mae": float(best_mae) if best_mae is not None else None,
            "best_rmse": float(best_rmse) if best_rmse is not None else None,
            "final_mae": float(mae_f),
            "final_rmse": float(rmse_f),
        },
    }
    torch.save(unified_ckpt, unified_ckpt_path)
    print(f"  [SAVED unified] {unified_ckpt_path}")

    return float(mae_f), float(rmse_f), best_epoch, unified_ckpt_path


def main():
    print("=" * 60)
    print("UNIFIED EXPLORATION: All configs × all 3 datasets")
    print(
        f"Early stop: patience={EARLY_STOP_PATIENCE} "
        f"min_epochs={MIN_EPOCHS_BEFORE_STOP} max_epochs={EPOCHS}"
    )
    print("=" * 60)

    results = []
    for cfg_name, n_emb, e_emb, h_dim, n_layers in CONFIGS:
        print(
            f"\n>>> Config: {cfg_name} (node_emb={n_emb} edge_emb={e_emb} "
            f"hidden={h_dim} layers={n_layers})"
        )
        for out_dir, feat_cols, label in DATASETS:
            print(f"\n--- {label} ({out_dir}) ---")
            mae, rmse, best_ep, ckpt_path = train_one_dataset(
                out_dir, feat_cols, label, cfg_name, n_emb, e_emb, h_dim, n_layers
            )
            if mae is not None:
                results.append(
                    {
                        "Config": cfg_name,
                        "Dataset": label,
                        "MAE (pu)": f"{mae:.6f}",
                        "RMSE (pu)": f"{rmse:.6f}",
                        "Best epoch": best_ep,
                        "Checkpoint": ckpt_path,
                    }
                )
                print(f"  FINAL: MAE={mae:.6f} RMSE={rmse:.6f} best_epoch={best_ep}")

    print("\n" + "=" * 60)
    print("SUMMARY (all config × dataset combinations)")
    print("=" * 60)
    if results:
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()

