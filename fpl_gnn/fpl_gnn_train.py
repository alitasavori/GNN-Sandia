"""
Train an MLP to predict FPL voltage residual from DER injection (delta_P, delta_Q) at all nodes.
Input: stacked [delta_P, delta_Q] for all bus-phases, shape (batch, 2*N).
Output: residual at each node, shape (batch, N).
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260303


class ResidualMLP(nn.Module):
    """MLP: input 2*N (delta_P, delta_Q all nodes), output N (residual per node)."""

    def __init__(self, input_dim: int, out_dim: int, hidden_dims: Tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_fpl_gnn(
    out_dir: str = os.path.join("fpl_gnn", "gnn_samples_fpl_residual_full"),
    hidden_dims: Tuple[int, ...] = (128, 128, 64),
    batch_size: int = 64,
    epochs: int = 50,
    test_frac: float = 0.2,
):
    """
    Train MLP on FPL residual: input = [delta_P, delta_Q] all nodes (2N), output = residual (N).
    Dataset CSV must have sample_id, node_idx, delta_P_kw, delta_Q_kvar, vmag_resid.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(node_csv):
        raise FileNotFoundError(f"Run dataset generation first. Missing {node_csv}")

    df_n = pd.read_csv(node_csv)
    for c in ["sample_id", "node_idx", "delta_P_kw", "delta_Q_kvar", "vmag_resid"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n = df_n.dropna(subset=["sample_id", "node_idx", "delta_P_kw", "delta_Q_kvar", "vmag_resid"])

    N = int(df_n["node_idx"].max()) + 1
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    S = df_n["sample_id"].nunique()
    # Each sample: N rows -> (delta_P, delta_Q) per node -> stack to (2*N,); vmag_resid -> (N,)
    delta_P = df_n["delta_P_kw"].to_numpy(dtype=np.float32).reshape(S, N)
    delta_Q = df_n["delta_Q_kvar"].to_numpy(dtype=np.float32).reshape(S, N)
    resid = df_n["vmag_resid"].to_numpy(dtype=np.float32).reshape(S, N)

    X_all = np.concatenate([delta_P, delta_Q], axis=1)  # (S, 2*N)
    Y_all = resid  # (S, N)

    input_dim = X_all.shape[1]
    out_dim = N

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(S)
    n_test = int(np.floor(test_frac * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_all[train_idx], dtype=torch.float32)
    X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
    Y_test = torch.tensor(Y_all[test_idx], dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ResidualMLP(input_dim=input_dim, out_dim=out_dim, hidden_dims=hidden_dims).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def compute_metrics(yhat, ytrue):
        err = yhat - ytrue
        mae = err.abs().mean().item()
        rmse = np.sqrt((err ** 2).mean().item())
        return mae, rmse

    @torch.no_grad()
    def evaluate(X, Y):
        model.eval()
        yh = model(X.to(DEVICE))
        return compute_metrics(yh.cpu().numpy(), Y.numpy())

    best_rmse = float("inf")
    best_state = None
    patience = 10
    min_delta = 1e-6

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        mae_t, rmse_t = evaluate(X_test, Y_test)
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
    mae_f, rmse_f = evaluate(X_test, Y_test)
    print(f"Final: MAE={mae_f:.6f} RMSE={rmse_f:.6f}")
    return model
