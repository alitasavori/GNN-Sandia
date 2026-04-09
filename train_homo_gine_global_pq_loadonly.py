"""
Train homogeneous GNN (GINE/GCN) on load-only MV nodes with global readout.

Design:
  - Node set: load-type nodes from hetero_mv_nodes_load_transformer_reg_tap_only.csv
  - Node features: P/Q only (p_load_kw, q_load_kvar)
  - Edges: compacted load-only line edges (R_full, X_full) from
      hetero_mv_line_edges_load_only_compacted.csv
  - Trunk: same family as train_homo_gine_csv.py (input proj + residual conv blocks)
  - Head: global MLP on concatenated per-node embeddings -> predicts vmag_pu for ALL nodes.

Output:
  - best checkpoint: <out_dir>/homo_<model>_global_pq_h*_L*_..._best.pt
  - train metrics:   <out_dir>/train_metrics_global.json
  - feature norm:    <out_dir>/feature_norm_pq.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

NODE_FEAT_COLS: tuple[str, str] = ("p_load_kw", "q_load_kvar")


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HomoGlobalDataset(Dataset):
    """One graph per sample; same edge_index/edge_attr; x and y vary."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self.x = x  # [S, N, 2]
        self.y = y  # [S, N]
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Data:
        return Data(
            x=self.x[idx],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y[idx],
        )


def _load_nodes_pq_target(
    nodes_csv: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], dict[str, int]]:
    usecols = ["sample_id", "node", "node_idx", "vmag_pu", *NODE_FEAT_COLS]
    print(f"Loading node CSV: {nodes_csv}", flush=True)
    df = pd.read_csv(nodes_csv, usecols=usecols)
    for c in usecols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in {nodes_csv}")

    sample_ids = sorted(df["sample_id"].unique().tolist())
    first_sid = sample_ids[0]
    first = df[df["sample_id"] == first_sid].sort_values("node_idx")
    node_order = first["node"].astype(str).str.strip().tolist()
    node_count = len(node_order)
    node_to_local = {n: i for i, n in enumerate(node_order)}

    s_ct = len(sample_ids)
    x_np = np.zeros((s_ct, node_count, 2), dtype=np.float32)
    y_np = np.zeros((s_ct, node_count), dtype=np.float32)

    for si, sid in enumerate(sample_ids):
        if si > 0 and si % 1000 == 0:
            print(f"  stacked {si}/{s_ct} samples...", flush=True)
        sub = df[df["sample_id"] == sid].sort_values("node_idx")
        if len(sub) != node_count:
            raise RuntimeError(f"sample_id={sid}: expected {node_count} rows, got {len(sub)}")
        if sub["node"].astype(str).str.strip().tolist() != node_order:
            raise RuntimeError(f"sample_id={sid}: node order differs from first sample")
        x_np[si, :, 0] = sub["p_load_kw"].to_numpy(dtype=np.float32)
        x_np[si, :, 1] = sub["q_load_kvar"].to_numpy(dtype=np.float32)
        y_np[si, :] = sub["vmag_pu"].to_numpy(dtype=np.float32)

    return torch.from_numpy(x_np), torch.from_numpy(y_np), sample_ids, node_order, node_to_local


def _load_compacted_edges(edge_csv: Path, node_to_local: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Loading compacted edge CSV: {edge_csv}", flush=True)
    df = pd.read_csv(edge_csv)
    need = ("from_node", "to_node", "R_full", "X_full")
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{edge_csv} missing required column {c!r}")

    src: list[int] = []
    dst: list[int] = []
    rs: list[float] = []
    xs: list[float] = []
    dropped = 0

    for _, r in df.iterrows():
        u = str(r["from_node"]).strip()
        v = str(r["to_node"]).strip()
        if u not in node_to_local or v not in node_to_local:
            dropped += 1
            continue
        iu = node_to_local[u]
        iv = node_to_local[v]
        rf = float(r.get("R_full", 0.0) or 0.0)
        xf = float(r.get("X_full", 0.0) or 0.0)
        # Make bidirectional graph.
        src.extend([iu, iv])
        dst.extend([iv, iu])
        rs.extend([rf, rf])
        xs.extend([xf, xf])

    if not src:
        raise RuntimeError("No valid edges remained after matching edge CSV to load-node set.")
    if dropped > 0:
        print(f"  dropped {dropped} unmatched edges", flush=True)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.column_stack([rs, xs]), dtype=torch.float32)
    print(f"  final edges (directed): {edge_index.shape[1]}", flush=True)
    return edge_index, edge_attr


def _zscore_features_train(
    x: torch.Tensor, train_idx: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xt = x[train_idx].reshape(-1, x.shape[-1])
    mean = xt.mean(dim=0, keepdim=True)
    std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


class HomoGINEGlobal(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        edge_dim: int,
        n_nodes: int,
        hidden: int,
        n_layers: int,
        node_out_dim: int,
        dropout: float,
        num_edges: int,
        node_emb_dim: int = 0,
        edge_emb_dim: int = 0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.node_emb: nn.Embedding | None
        if self.node_emb_dim > 0:
            self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim)
        else:
            self.node_emb = None

        self.edge_emb: nn.Embedding | None
        if self.edge_emb_dim > 0:
            self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim)
        else:
            self.edge_emb = None

        eff_in = in_dim + self.node_emb_dim
        eff_edge_dim = edge_dim + self.edge_emb_dim

        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=eff_edge_dim))
            self.norms.append(nn.LayerNorm(hidden))

        self.node_proj = nn.Linear(hidden, node_out_dim)
        gdim = self.n_nodes * node_out_dim
        # 3-layer global head
        self.global_head = nn.Sequential(
            nn.Linear(gdim, gdim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gdim, gdim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gdim // 2, self.n_nodes),
        )

    def _edge_ids(self, n_edges_total: int, device: torch.device) -> torch.Tensor:
        if n_edges_total % self.num_edges != 0:
            raise RuntimeError(f"Expected edge count multiple of {self.num_edges}, got {n_edges_total}")
        b = n_edges_total // self.num_edges
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(b)

    def _node_ids(self, n_nodes_total: int, device: torch.device) -> torch.Tensor:
        if n_nodes_total % self.n_nodes != 0:
            raise RuntimeError(f"Expected node count multiple of {self.n_nodes}, got {n_nodes_total}")
        b = n_nodes_total // self.n_nodes
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(b)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        ea = batch.edge_attr
        bvec = batch.batch

        if self.node_emb is not None:
            z = self.node_emb(self._node_ids(x.size(0), x.device))
            x = torch.cat([x, z], dim=-1)
        if self.edge_emb is not None:
            ze = self.edge_emb(self._edge_ids(ea.size(0), ea.device))
            ea = torch.cat([ea, ze], dim=-1)

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, ea))
            h = h + self.dropout(norm(h_msg))

        z_node = self.node_proj(h)
        if bvec is None:
            z_node = z_node.view(1, self.n_nodes, -1)
        else:
            b = int(bvec.max().item()) + 1
            z_node = z_node.view(b, self.n_nodes, -1)
        g = z_node.reshape(z_node.size(0), -1)
        out = self.global_head(g)  # [B, N]
        return out


class HomoGCNGlobal(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_nodes: int,
        hidden: int,
        n_layers: int,
        node_out_dim: int,
        dropout: float,
        node_emb_dim: int = 0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.node_emb: nn.Embedding | None
        if self.node_emb_dim > 0:
            self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim)
        else:
            self.node_emb = None

        eff_in = in_dim + self.node_emb_dim
        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.node_proj = nn.Linear(hidden, node_out_dim)
        gdim = self.n_nodes * node_out_dim
        self.global_head = nn.Sequential(
            nn.Linear(gdim, gdim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gdim, gdim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gdim // 2, self.n_nodes),
        )

    @staticmethod
    def edge_attr_to_weight(edge_attr: torch.Tensor) -> torch.Tensor:
        z = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2).clamp(min=1e-6)
        return 1.0 / z

    def _node_ids(self, n_nodes_total: int, device: torch.device) -> torch.Tensor:
        if n_nodes_total % self.n_nodes != 0:
            raise RuntimeError(f"Expected node count multiple of {self.n_nodes}, got {n_nodes_total}")
        b = n_nodes_total // self.n_nodes
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(b)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = self.edge_attr_to_weight(batch.edge_attr)
        bvec = batch.batch

        if self.node_emb is not None:
            z = self.node_emb(self._node_ids(x.size(0), x.device))
            x = torch.cat([x, z], dim=-1)

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, edge_weight=edge_weight))
            h = h + self.dropout(norm(h_msg))

        z_node = self.node_proj(h)
        if bvec is None:
            z_node = z_node.view(1, self.n_nodes, -1)
        else:
            b = int(bvec.max().item()) + 1
            z_node = z_node.view(b, self.n_nodes, -1)
        g = z_node.reshape(z_node.size(0), -1)
        out = self.global_head(g)
        return out


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
    log_every: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_mae = 0.0
        tr_mse = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)      # [B, N]
            tgt = batch.y            # [B, N]
            loss = criterion(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_mse += float(loss.item()) * batch.num_graphs
            tr_mae += float((pred - tgt).abs().mean(dim=1).sum().item())
            n_train += int(batch.num_graphs)
        tr_mse /= max(n_train, 1)
        tr_mae /= max(n_train, 1)

        model.eval()
        va_mae = 0.0
        va_mse = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                tgt = batch.y
                loss = criterion(pred, tgt)
                va_mse += float(loss.item()) * batch.num_graphs
                va_mae += float((pred - tgt).abs().mean(dim=1).sum().item())
                n_val += int(batch.num_graphs)
        va_mse /= max(n_val, 1)
        va_mae /= max(n_val, 1)

        scheduler.step(va_mse)
        if va_mse < best:
            best = va_mse
            bad = 0
            to_save = getattr(model, "_orig_mod", model)
            torch.save(to_save.state_dict(), checkpoint_path)
        else:
            bad += 1

        le = max(1, log_every)
        if epoch == 1 or epoch % le == 0:
            print(
                f"Epoch {epoch:4d} | train_mae={tr_mae:.6f} train_mse={tr_mse:.6f} | "
                f"val_mae={va_mae:.6f} val_mse={va_mse:.6f} | "
                f"best_val_mse={best:.6f} | patience {bad}/{patience}",
                flush=True,
            )

        if bad >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train homo GINE/GCN with global readout on load-only P/Q features.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--nodes_csv",
        type=str,
        default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
        help="Load-type node CSV with sample_id,node,node_idx,p_load_kw,q_load_kvar,vmag_pu.",
    )
    p.add_argument(
        "--edge_catalog_csv",
        type=str,
        default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
        help="Compacted load-only edge CSV with from_node,to_node,R_full,X_full.",
    )
    p.add_argument("--out_dir", type=str, default="checkpoints_homo_gine_global_pq")
    p.add_argument("--model", type=str, choices=("gine", "gcn"), default="gine")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--node_out_dim", type=int, default=2, help="Per-node embedding size before global head.")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--disable_dropout", action="store_true", help="Force dropout=0.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of samples to use (0,1].")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--node_emb_dim", type=int, default=0, metavar="D")
    p.add_argument("--edge_emb_dim", type=int, default=0, metavar="D", help="Ignored for model=gcn.")
    p.add_argument(
        "--cache_tensor",
        type=str,
        default="",
        help="Optional path to save/load preloaded tensors (x,y,edge_index,edge_attr,node_order,sample_ids).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_stdout()
    _set_seeds(args.seed)

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else repo / "datasets_gnn2" / "loadtype_8500_dailyagg"

    nodes_path = Path(args.nodes_csv) if os.path.isabs(args.nodes_csv) else data_root / args.nodes_csv
    edge_path = Path(args.edge_catalog_csv) if os.path.isabs(args.edge_catalog_csv) else data_root / args.edge_catalog_csv
    for p in (nodes_path, edge_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    drop = 0.0 if args.disable_dropout else float(args.dropout)
    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None

    if cache_path and cache_path.is_file():
        print(f"Loading preloaded tensor cache: {cache_path}", flush=True)
        try:
            pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            pack = torch.load(cache_path, map_location="cpu")
        x = pack["x"]
        y = pack["y"]
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        node_order = pack["node_order"]
        sample_ids = pack["sample_ids"]
    else:
        x, y, sample_ids, node_order, node_to_local = _load_nodes_pq_target(nodes_path)
        edge_index, edge_attr = _load_compacted_edges(edge_path, node_to_local)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "x": x,
                    "y": y,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "node_order": node_order,
                    "sample_ids": sample_ids,
                    "nodes_path": str(nodes_path.resolve()),
                    "edges_path": str(edge_path.resolve()),
                    "nodes_mtime": nodes_path.stat().st_mtime,
                    "edges_mtime": edge_path.stat().st_mtime,
                },
                cache_path,
            )
            print(f"Wrote tensor cache: {cache_path}", flush=True)

    n_all = x.shape[0]
    if args.max_samples is not None:
        k = min(int(args.max_samples), n_all)
        x, y = x[:k], y[:k]
        sample_ids = sample_ids[:k]
        print(f"Using --max_samples={k} (of {n_all}).", flush=True)
    elif args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0, 1].")
        k = max(1, int(round(n_all * args.sample_frac)))
        x, y = x[:k], y[:k]
        sample_ids = sample_ids[:k]
        print(f"Using --sample_frac={args.sample_frac} -> {k} samples (of {n_all}).", flush=True)
    else:
        print(f"Using all {n_all} samples (--sample_frac=1.0).", flush=True)

    n = x.shape[0]
    n_nodes = int(x.shape[1])
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = max(1, int(args.train_frac * n))
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:].tolist())
    if not val_idx:
        val_idx = [train_idx[-1]]
        train_idx = train_idx[:-1]

    x, feat_mean, feat_std = _zscore_features_train(x, np.array(train_idx, dtype=np.int64))
    torch.save(
        {"mean": feat_mean, "std": feat_std, "feat_cols": list(NODE_FEAT_COLS)},
        out_dir / "feature_norm_pq.pt",
    )

    dataset = HomoGlobalDataset(x, y, edge_index, edge_attr)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    nw = int(args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )

    in_dim = x.shape[-1]
    n_edges = int(edge_index.shape[1])
    ned = max(0, int(args.node_emb_dim))
    eed = max(0, int(args.edge_emb_dim))
    if args.model == "gcn" and eed > 0:
        print("[train_homo_gine_global_pq_loadonly] model=gcn ignores --edge_emb_dim; forcing to 0.", flush=True)
        eed = 0

    if args.model == "gine":
        model: nn.Module = HomoGINEGlobal(
            in_dim=in_dim,
            edge_dim=2,
            n_nodes=n_nodes,
            hidden=args.hidden,
            n_layers=args.layers,
            node_out_dim=args.node_out_dim,
            dropout=drop,
            num_edges=n_edges,
            node_emb_dim=ned,
            edge_emb_dim=eed,
        )
    else:
        model = HomoGCNGlobal(
            in_dim=in_dim,
            n_nodes=n_nodes,
            hidden=args.hidden,
            n_layers=args.layers,
            node_out_dim=args.node_out_dim,
            dropout=drop,
            node_emb_dim=ned,
        )

    model = model.to(device)
    emb_tag = f"_ne{ned}_ee{eed}"
    do_tag = "_do0" if drop == 0.0 else f"_do{drop:g}"
    ckpt = out_dir / (
        f"homo_{args.model}_global_pq_h{args.hidden}_L{args.layers}_nout{args.node_out_dim}"
        f"{emb_tag}{do_tag}_best.pt"
    )

    print(f"Device: {device}", flush=True)
    print(
        f"Train/val: {len(train_idx)}/{len(val_idx)} | N_nodes={n_nodes} "
        f"| model={args.model} hidden={args.hidden} layers={args.layers} node_out_dim={args.node_out_dim}",
        flush=True,
    )
    print(
        f"Embeddings: node_emb_dim={ned} edge_emb_dim={eed} | dropout={drop} | "
        f"batch_size={args.batch_size} epochs={args.epochs}",
        flush=True,
    )
    print("Starting training loop...", flush=True)

    best_val_mse = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=ckpt,
        log_every=args.log_every,
    )

    meta = {
        "best_val_mse_pu2": float(best_val_mse),
        "nodes_csv": str(nodes_path),
        "edge_catalog_csv": str(edge_path),
        "n_samples": int(len(sample_ids)),
        "sample_frac": float(args.sample_frac),
        "max_samples": args.max_samples,
        "n_nodes": int(n_nodes),
        "n_edges_directed": int(edge_index.shape[1]),
        "n_features": int(in_dim),
        "model": args.model,
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "node_out_dim": int(args.node_out_dim),
        "node_emb_dim": int(ned),
        "edge_emb_dim": int(eed),
        "dropout": float(drop),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
    }
    with open(out_dir / "train_metrics_global.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Best val MSE (p.u.^2):", best_val_mse, flush=True)
    print("Saved:", ckpt, flush=True)


if __name__ == "__main__":
    main()
