"""
Train homogeneous GINE / GCN on IEEE 8500 MV node CSV + static line-edge catalog.

Default node file: hetero_mv_nodes_load_transformer_reg_tap_only.csv
  (15 features: p_load_kw, q_load_kvar, q_capacitor_bank + 12 FEEDER_REG* / VREG* columns)

Edges: hetero_mv_edge_catalog.csv — only rows with edge_type == 'line' (regulator edges omitted).

No other CSVs are read by this script (see train_metrics.json field extra_csvs_used).

Colab: pass --data_root /content/drive/MyDrive/.../loadtype_8500_dailyagg (or clone the repo and upload the two CSV folders).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv

# 15-D node features: 3 base + 12 regulator-phase products (tap-only or dist×tap — column names match)
BASE_FEAT_COLS: tuple[str, ...] = ("p_load_kw", "q_load_kvar", "q_capacitor_bank")
TAP_FEAT_COLS: tuple[str, ...] = (
    "FEEDER_REGA",
    "FEEDER_REGB",
    "FEEDER_REGC",
    "VREG2_A",
    "VREG2_B",
    "VREG2_C",
    "VREG3_A",
    "VREG3_B",
    "VREG3_C",
    "VREG4_A",
    "VREG4_B",
    "VREG4_C",
)
NODE_FEAT_COLS: tuple[str, ...] = BASE_FEAT_COLS + TAP_FEAT_COLS


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_old_to_new(node_indices: np.ndarray) -> dict[int, int]:
    uniq = np.unique(node_indices)
    return {int(o): i for i, o in enumerate(uniq)}


def _load_line_edges_supervised(
    edge_catalog_path: Path,
    old_to_new: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Line edges only; bidirectional; endpoints remapped to 0..N-1; drop edges with unknown nodes."""
    df = pd.read_csv(edge_catalog_path)
    df = df[df["edge_type"].astype(str).str.lower() == "line"].copy()
    u = df["u_idx"].astype(float).round().astype(int)
    v = df["v_idx"].astype(float).round().astype(int)
    src: list[int] = []
    dst: list[int] = []
    r_list: list[float] = []
    x_list: list[float] = []
    allowed = set(old_to_new.keys())
    for ui, vi, ri, xi in zip(u.tolist(), v.tolist(), df["R_full"], df["X_full"]):
        if ui not in allowed or vi not in allowed:
            continue
        iu, iv = old_to_new[ui], old_to_new[vi]
        rf, xf = float(ri), float(xi)
        src.extend([iu, iv])
        dst.extend([iv, iu])
        r_list.extend([rf, rf])
        x_list.extend([xf, xf])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.column_stack([r_list, x_list]), dtype=torch.float32)
    return edge_index, edge_attr


class HomoGINEDataset(Dataset):
    """One graph per timestep; same edge_index / edge_attr; x and y vary."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Data:
        return Data(
            x=self.x[idx],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y[idx],
        )


class HomoGINE(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, edge_attr))
            h = h + self.dropout(norm(h_msg))
        return self.output_head(h)


class HomoGCNRes(nn.Module):
    """Fast baseline: scalar edge weight from R,X."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden, 1)

    @staticmethod
    def edge_attr_to_weight(edge_attr: torch.Tensor) -> torch.Tensor:
        z = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2).clamp(min=1e-6)
        return 1.0 / z

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_weight = self.edge_attr_to_weight(edge_attr)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, edge_weight=edge_weight))
            h = h + self.dropout(norm(h_msg))
        return self.output_head(h)


def _load_and_stack_nodes(
    nodes_csv: Path,
    feat_cols: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, list[int], dict[int, int]]:
    """Returns x [S,N,F], y [S,N], node_order, sample_ids, old_to_new (global idx -> 0..N-1)."""
    df = pd.read_csv(nodes_csv)
    for c in ("sample_id", "node_idx", "vmag_pu", *feat_cols):
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in {nodes_csv}")

    sample_ids = sorted(df["sample_id"].unique().tolist())
    first = df[df["sample_id"] == sample_ids[0]].sort_values("node_idx")
    node_order = first["node_idx"].to_numpy()
    old_to_new = _build_old_to_new(node_order)
    N = len(node_order)

    S = len(sample_ids)
    F = len(feat_cols)
    x_np = np.zeros((S, N, F), dtype=np.float32)
    y_np = np.zeros((S, N), dtype=np.float32)

    for si, sid in enumerate(sample_ids):
        sub = df[df["sample_id"] == sid].sort_values("node_idx")
        if len(sub) != N:
            raise RuntimeError(f"sample_id={sid}: expected {N} rows, got {len(sub)}")
        if not np.array_equal(sub["node_idx"].to_numpy(), node_order):
            raise RuntimeError(f"sample_id={sid}: node_idx order differs from first sample")
        x_np[si] = sub[list(feat_cols)].to_numpy(dtype=np.float32)
        y_np[si] = sub["vmag_pu"].to_numpy(dtype=np.float32)

    return torch.from_numpy(x_np), torch.from_numpy(y_np), node_order, sample_ids, old_to_new


def _zscore_features_train(
    x: torch.Tensor, train_idx: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: [S,N,F] — normalize using train samples only."""
    xt = x[train_idx].reshape(-1, x.shape[-1])
    mean = xt.mean(dim=0, keepdim=True)
    std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
    use_compile: bool,
) -> float:
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_mae = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = torch.isfinite(batch.y)
            if mask.sum() == 0:
                continue
            loss = criterion(pred[mask], batch.y[mask])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_mae += (pred[mask] - batch.y[mask]).abs().sum().item()
            n_train += int(mask.sum().item())
        train_mae = train_mae / max(n_train, 1)

        model.eval()
        val_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                mask = torch.isfinite(batch.y)
                val_mae += (pred[mask] - batch.y[mask]).abs().sum().item()
                n_val += int(mask.sum().item())
        val_mae = val_mae / max(n_val, 1)

        scheduler.step(val_mae)

        if val_mae < best:
            best = val_mae
            bad = 0
            to_save = getattr(model, "_orig_mod", model)
            torch.save(to_save.state_dict(), checkpoint_path)
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | train_mae={train_mae:.6f} | val_mae={val_mae:.6f} | "
                f"best_val_mae={best:.6f} | patience {bad}/{patience}"
            )

        if bad >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HomoGINE from hetero MV CSVs.")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root folder containing 'Heterogenous GNN dataset/'. Default: <repo>/datasets_gnn2/loadtype_8500_dailyagg",
    )
    p.add_argument(
        "--nodes_csv",
        type=str,
        default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
        help="Relative to data_root unless absolute.",
    )
    p.add_argument(
        "--edge_catalog_csv",
        type=str,
        default="Heterogenous GNN dataset/edges/hetero_mv_edge_catalog.csv",
        help="Relative to data_root unless absolute.",
    )
    p.add_argument("--out_dir", type=str, default="checkpoints_homo_gine", help="Checkpoints + metrics JSON.")
    p.add_argument("--model", type=str, choices=("gine", "gcn"), default="gine")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_normalize", action="store_true", help="Disable z-score on node features (train stats).")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap model with torch.compile for training (PyTorch 2+). Checkpoint stores underlying weights.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Debug: only use first K samples after sort.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seeds(args.seed)

    repo = Path(__file__).resolve().parent
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = repo / "datasets_gnn2" / "loadtype_8500_dailyagg"

    nodes_path = Path(args.nodes_csv) if os.path.isabs(args.nodes_csv) else data_root / args.nodes_csv
    edge_path = Path(args.edge_catalog_csv) if os.path.isabs(args.edge_catalog_csv) else data_root / args.edge_catalog_csv

    if not nodes_path.is_file():
        raise FileNotFoundError(nodes_path)
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nodes:", nodes_path)
    x, y, node_order, sample_ids, old_to_new = _load_and_stack_nodes(nodes_path, NODE_FEAT_COLS)
    if args.max_samples is not None:
        x = x[: args.max_samples]
        y = y[: args.max_samples]
        sample_ids = sample_ids[: args.max_samples]

    print("Stacked:", x.shape, y.shape, "samples:", len(sample_ids))

    edge_index, edge_attr = _load_line_edges_supervised(edge_path, old_to_new)
    print("Line edges (bidir):", edge_index.shape[1], "edge_attr:", edge_attr.shape)

    n = x.shape[0]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = max(1, int(args.train_frac * n))
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:].tolist())
    if not val_idx:
        val_idx = [train_idx[-1]]
        train_idx = train_idx[:-1]

    if not args.no_normalize:
        train_idx_arr = np.array(train_idx, dtype=np.int64)
        x, feat_mean, feat_std = _zscore_features_train(x, train_idx_arr)
        torch.save({"mean": feat_mean, "std": feat_std, "feat_cols": list(NODE_FEAT_COLS)}, out_dir / "feature_norm.pt")

    dataset = HomoGINEDataset(x, y.unsqueeze(-1), edge_index, edge_attr)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    in_dim = x.shape[-1]
    if args.model == "gine":
        model = HomoGINE(
            in_dim=in_dim,
            edge_dim=2,
            hidden=args.hidden,
            n_layers=args.layers,
            dropout=args.dropout,
        )
    else:
        model = HomoGCNRes(in_dim=in_dim, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout)

    model = model.to(device)
    ckpt = out_dir / f"homo_{args.model}_h{args.hidden}_L{args.layers}_best.pt"

    best_mae = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=ckpt,
        use_compile=args.compile,
    )

    meta = {
        "best_val_mae_pu": float(best_mae),
        "nodes_csv": str(nodes_path),
        "edge_catalog_csv": str(edge_path),
        "n_samples": len(sample_ids),
        "n_nodes": int(x.shape[1]),
        "n_features": int(in_dim),
        "model": args.model,
        "hidden": args.hidden,
        "layers": args.layers,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "extra_csvs_used": [],
    }
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Best val MAE (p.u.):", best_mae)
    print("Saved:", ckpt)
    print("Wrote:", out_dir / "train_metrics.json")


if __name__ == "__main__":
    main()
