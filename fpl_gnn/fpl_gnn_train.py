import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from run_gnn3_best7_train import PFIdentityGNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260303


def _load_fpl_residual_data(
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    return df_e, df_n, None


def train_fpl_gnn(
    out_dir: str = os.path.join("fpl_gnn", "gnn_samples_fpl_residual_full"),
    h_dim: int = 96,
    num_layers: int = 3,
    batch_size: int = 64,
    epochs: int = 50,
    test_frac: float = 0.2,
) -> PFIdentityGNN:
    """
    Train a PFIdentityGNN on FPL residual labels.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df_e, df_n, _ = _load_fpl_residual_data(out_dir)

    feature_cols = ["electrical_distance_ohm", "P_node_kw", "Q_node_kvar"]
    target_col = "vmag_pu_resid"
    required = {"sample_id", "node_idx", target_col} | set(feature_cols)
    if required - set(df_n.columns):
        raise RuntimeError(f"Missing columns: {required - set(df_n.columns)}")

    for c in ["u_idx", "v_idx", "R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + [target_col, "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")

    df_e = (
        df_e.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["u_idx", "v_idx", "R_full", "X_full"])
        .reset_index(drop=True)
    )
    df_n = (
        df_n.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(required))
        .reset_index(drop=True)
    )

    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    E = int(len(df_e))
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

    all_ids = df_n["sample_id"].unique()
    S = len(all_ids)

    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n[target_col].to_numpy(dtype=np.float32).reshape(S, N, 1)

    node_in_dim = X_all.shape[-1]
    edge_index = torch.tensor(
        df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long
    )
    edge_attr = torch.tensor(
        df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float32
    )
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(S)
    n_test = int(np.floor(test_frac * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        out = []
        for k in idx:
            x = torch.tensor(X_all[k], dtype=torch.float32)
            y = torch.tensor(Y_all[k], dtype=torch.float32)
            g = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_id=edge_id,
                num_nodes=N,
            )
            out.append(g)
        return out

    train_loader = DataLoader(make_ds(train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(make_ds(test_idx), batch_size=batch_size, shuffle=False)

    model = PFIdentityGNN(
        num_nodes=N,
        num_edges=E,
        node_in_dim=node_in_dim,
        edge_in_dim=2,
        out_dim=1,
        node_emb_dim=16,
        edge_emb_dim=8,
        h_dim=h_dim,
        num_layers=num_layers,
        use_norm=False,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def compute_metrics(yhat, ytrue):
        err = (yhat - ytrue).squeeze(-1)
        mae = err.abs().mean()
        rmse = torch.sqrt((err ** 2).mean())
        return mae, rmse

    @torch.no_grad()
    def evaluate(loader):
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
        return (mae_sum / max(1, n_batches)).item(), (
            rmse_sum / max(1, n_batches)
        ).item()

    best_rmse = float("inf")
    best_state = None
    patience = 10
    min_delta = 1e-6

    for epoch in range(1, epochs + 1):
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
        mae_t, rmse_t = evaluate(test_loader)
        if (best_rmse - rmse_t) > min_delta:
            best_rmse = rmse_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
        else:
            patience -= 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | RMSE={rmse_t:.6f} | best={best_rmse:.6f} | patience={patience}")
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(test_loader)
    print(f"Final: MAE={mae_f:.6f} RMSE={rmse_f:.6f}")
    return model


