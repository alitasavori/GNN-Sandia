"""
Step 6 (IEEE 8500 checklist) — Train a GNN baseline.

Reads artifacts produced by earlier steps:
  - datasets_gnn2/loadtype_8500/dataset_tensors/X.pt        [S, N, 14]
  - datasets_gnn2/loadtype_8500/dataset_tensors/Y.pt        [S, N]    (vmag_pu)
  - datasets_gnn2/loadtype_8500/dataset_tensors/Y_angle.pt  [S, N]    (vang_deg, optional)
  - datasets_gnn2/loadtype_8500/graph_tensors/edge_index.pt  [2, E]
  - datasets_gnn2/loadtype_8500/graph_tensors/edge_attr.pt   [E, 4]    (R_full, X_full, length, phase)

Model:
  - Residual stack of GCNConv layers (hidden_dim H)
  - Readout Linear(H, 2) predicting (vmag_pu, vang_deg) when angle targets exist

Loss:
  - MSE(vmag) + angle_loss_weight * MSE(angle_deg) when Y_angle.pt exists
  - Otherwise: MSE(vmag) only

Node-type one-hot augmentation (best-effort from features and node names):
  - load bus, regulator bus, capacitor bus, source bus
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


REPO_ROOT = Path(__file__).resolve().parent

DATASET_DIR_DEFAULT = REPO_ROOT / "datasets_gnn2" / "loadtype_8500"
TENSOR_SUBDIR = "dataset_tensors"
GRAPH_SUBDIR = "graph_tensors"

# Must match assemble_dataset_tensors_8500.DEFAULT_FEATURE_COLS
LOADTYPE_FEAT_COLS: tuple[str, ...] = (
    "electrical_distance_ohm",
    "m1_p_kw",
    "m1_q_kvar",
    "m2_p_kw",
    "m2_q_kvar",
    "m4_p_kw",
    "m4_q_kvar",
    "m5_p_kw",
    "m5_q_kvar",
    "q_cap_kvar",
    "p_pv_kw",
    "q_pv_kvar",
    "p_sys_balance_kw",
    "q_sys_balance_kvar",
)


def _zscore_train_only(X: torch.Tensor, train_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X: [S, N, F]. z-score using mean/std over train samples and all nodes."""
    X_tr = X[train_idx]  # [S_tr, N, F]
    X_flat = X_tr.reshape(-1, X.shape[-1])  # [S_tr*N, F]
    mean = X_flat.mean(dim=0, keepdim=True)
    std = X_flat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (X - mean) / std, mean, std


def _build_node_type_onehot(X_raw: torch.Tensor, dataset_dir: Path, threshold: float = 1e-8) -> torch.Tensor:
    """
    Returns one-hot [N, 4] with columns:
      [load_bus, regulator_bus, capacitor_bus, source_bus]
    """
    S, N, F = X_raw.shape
    col_to_i = {c: i for i, c in enumerate(LOADTYPE_FEAT_COLS)}
    i_qcap = col_to_i["q_cap_kvar"]
    i_pv = col_to_i["p_pv_kw"]
    i_pv_q = col_to_i["q_pv_kvar"]
    load_idxs = [col_to_i[c] for c in LOADTYPE_FEAT_COLS if c.startswith("m") and (c.endswith("_kw") or c.endswith("_kvar"))]

    df = pd.read_csv(dataset_dir / "gnn_node_index_master.csv")
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="raise").astype(int)
    df = df.sort_values("node_idx").reset_index(drop=True)
    if len(df) != N:
        raise RuntimeError(f"node count mismatch: X N={N} but csv rows={len(df)}")

    node_names = df["node"].astype(str).tolist()
    is_source = torch.tensor([1.0 if str(n).lower().startswith("sourcebus") else 0.0 for n in node_names], dtype=torch.float32)

    X_abs = X_raw.abs()
    pv_activity = (X_abs[:, :, i_pv] + X_abs[:, :, i_pv_q]).mean(dim=0)  # [N]
    cap_activity = X_abs[:, :, i_qcap].mean(dim=0)  # [N]
    load_activity = X_abs[:, :, load_idxs].mean(dim=(0, 2)) if load_idxs else torch.zeros(N)

    is_pv = pv_activity > threshold
    is_cap = cap_activity > threshold
    is_load = load_activity > threshold

    # regulator = neither load nor PV nor cap nor source
    is_reg = (~is_load) & (~is_pv) & (~is_cap) & (~is_source.bool())

    onehot = torch.zeros((N, 4), dtype=torch.float32)
    onehot[:, 0] = is_load.float()
    onehot[:, 1] = is_reg.float()
    onehot[:, 2] = is_cap.float()
    onehot[:, 3] = is_source
    return onehot


class SimpleGCNLayer(nn.Module):
    """
    Minimal GCN-like layer implemented with torch-only scatter.

    edge_index format: [2, E] with edge_index[0] = src, edge_index[1] = dst.
    Uses symmetric normalization with degrees computed from (weighted) in-graph degrees.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [N, H]
        edge_index: torch.Tensor,  # [2, E]
        edge_weight: torch.Tensor | None = None,  # [E]
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        h = self.lin(x)  # [N, H]

        if edge_weight is None:
            w = torch.ones((edge_index.shape[1],), device=x.device, dtype=x.dtype)
        else:
            w = edge_weight.to(dtype=x.dtype)

        N = x.shape[0]
        deg_src = torch.zeros((N,), device=x.device, dtype=x.dtype).scatter_add_(0, src, w)
        deg_dst = torch.zeros((N,), device=x.device, dtype=x.dtype).scatter_add_(0, dst, w)

        norm = (deg_src[src] * deg_dst[dst]).clamp_min(1e-12).rsqrt()
        # norm_factor includes w (GCN uses w on each edge); if w==1 this reduces to symmetric norm.
        norm_factor = w * norm  # [E]

        msg = h[src] * norm_factor.unsqueeze(1)  # [E, H]
        out = torch.zeros_like(h)
        # index_add accumulates rows of out[dst] += msg
        out.index_add_(0, dst, msg)
        return out


class ResidualGCN8500(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SimpleGCNLayer(hidden_dim) for _ in range(num_layers)])
        # Magnitude-only output for speed and checklist alignment.
        self.readout = nn.Linear(hidden_dim, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.in_proj(x))
        for layer in self.layers:
            h_in = h
            h_new = layer(h, edge_index, edge_weight=edge_weight)
            h_new = F.relu(h_new)
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_in + h_new
        return self.readout(h).squeeze(-1)


@torch.no_grad()
def _eval_mae_vmag(
    model: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    X_in: torch.Tensor,
    Y_mag: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    maes_v = []
    for i in indices:
        x = X_in[i].to(device, non_blocking=True)
        pred = model(x, edge_index, edge_weight=edge_weight)
        # pred is [N] magnitude only
        maes_v.append((pred - Y_mag[i].to(device, non_blocking=True)).abs().mean().item())
    out = {"test_mae_vmag_pu": float(np.mean(maes_v))}
    return out


def train_gnn_8500(
    dataset_dir: str | os.PathLike | None = None,
    *,
    out_subdir: str = "gnn_8500_baseline",
    run_name: str = "res_gcn_h64_L3",
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 20260322,
    max_train_samples: int | None = 50,
    max_val_samples: int | None = 20,
    max_test_samples: int | None = 20,
    preload_to_device: bool = True,
) -> dict[str, float]:
    dset = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR_DEFAULT
    tdir = dset / TENSOR_SUBDIR
    gdir = dset / GRAPH_SUBDIR

    path_x = tdir / "X.pt"
    path_y = tdir / "Y.pt"
    path_ei = gdir / "edge_index.pt"
    path_ea = gdir / "edge_attr.pt"

    for p in (path_x, path_y, path_ei, path_ea):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    X_raw = torch.load(path_x, map_location="cpu").float()
    Y_mag = torch.load(path_y, map_location="cpu").float()
    # Angle targets intentionally ignored.
    Y_ang = None

    edge_index = torch.load(path_ei, map_location="cpu").long()
    edge_attr = torch.load(path_ea, map_location="cpu").float()
    if edge_attr.dim() == 2 and edge_attr.shape[1] >= 2 and edge_attr.shape[0] == edge_index.shape[1]:
        r = edge_attr[:, 0]
        x = edge_attr[:, 1]
        edge_weight = torch.sqrt(r * r + x * x).clamp_min(1e-9)
    else:
        edge_weight = None

    S, N, n_feat = X_raw.shape
    assert Y_mag.shape == (S, N), f"Y_mag shape mismatch: {tuple(Y_mag.shape)}"
    # Y_ang is always None in magnitude-only mode.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_gnn_8500] device={device}  S={S} N={N} F={n_feat} angle_target=OFF")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(S, generator=g).cpu().numpy()
    n_train = int(S * train_frac)
    n_val = int(S * val_frac)
    n_test = S - n_train - n_val
    if n_test < 1:
        raise RuntimeError(f"Not enough samples S={S}")
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    if max_train_samples is not None:
        idx_train = idx_train[: int(max_train_samples)]
    if max_val_samples is not None:
        idx_val = idx_val[: int(max_val_samples)]
    if max_test_samples is not None:
        idx_test = idx_test[: int(max_test_samples)]

    # Feature normalization (train-only)
    X_in_base, mean, std = _zscore_train_only(X_raw, idx_train)

    # Node-type features from raw X
    node_type = _build_node_type_onehot(X_raw, dset)  # [N,4]
    type_rep = node_type.unsqueeze(0).expand(S, N, 4)
    X_in = torch.cat([X_in_base, type_rep], dim=2)  # [S,N,F+4]
    in_dim = X_in.shape[-1]

    # Preload for speed
    if device.type == "cuda" and preload_to_device:
        X_in = X_in.to(device, non_blocking=True)
        Y_mag = Y_mag.to(device, non_blocking=True)
        if Y_ang is not None:
            Y_ang = Y_ang.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device, non_blocking=True)
    else:
        edge_index = edge_index.to(device, non_blocking=True)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device, non_blocking=True)

    model = ResidualGCN8500(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    out_dir = dset / out_subdir / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for i in idx_train:
            if device.type == "cuda" and preload_to_device:
                x = X_in[i]
                yv = Y_mag[i]
            else:
                x = X_in[i].to(device, non_blocking=True)
                yv = Y_mag[i].to(device, non_blocking=True)

            pred = model(x, edge_index, edge_weight=edge_weight)
            loss = F.mse_loss(pred, yv)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())

        # Validation MAE on |V|
        val_metrics = _eval_mae_vmag(
            model=model,
            edge_index=edge_index,
            edge_weight=edge_weight,
            X_in=X_in,
            Y_mag=Y_mag,
            indices=idx_val,
            device=device,
        )
        val_mae = float(val_metrics["test_mae_vmag_pu"])
        sched.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 1 == 0:
            print(f"  epoch {ep:3d}/{epochs}  train_loss={train_loss/max(1,len(idx_train)):.6f}  val_mae_|V|={val_mae:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _eval_mae_vmag(
        model=model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        X_in=X_in,
        Y_mag=Y_mag,
        indices=idx_test,
        device=device,
    )

    # Save checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "node_type_note": "computed from X features + sourcebus names",
        "use_angle": False,
    }
    torch.save(ckpt, out_dir / "gnn_8500.pt")

    meta = {
        "dataset_dir": str(dset.resolve()),
        "run_name": run_name,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "best_val_mae_vmag_pu": best_val_mae,
        "test_mae_vmag_pu": test_metrics["test_mae_vmag_pu"],
    }
    (out_dir / "gnn_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[train_gnn_8500] done. out_dir={out_dir}")
    print(f"  test MAE |V| (pu): {test_metrics['test_mae_vmag_pu']:.6f}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Train IEEE 8500 GNN baseline (vmag only).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-train-samples", type=int, default=50)
    p.add_argument("--max-val-samples", type=int, default=20)
    p.add_argument("--max-test-samples", type=int, default=20)
    args = p.parse_args()

    train_gnn_8500(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()

