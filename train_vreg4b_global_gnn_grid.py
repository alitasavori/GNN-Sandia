"""
Grid training: VREG4_B tap (multiclass) with global node-embedding readout.

36 runs = 2 backbones (GINE, GCN) × 2 trunk depths (2, 3) × 3 node readout dims (2, 4, 8)
× 3 FCL head depths (1, 2, 3 hidden layers).

Inputs: P/Q only at each MV node (same CSV as homo MV); edges from hetero_mv_edge_catalog (lines).
Labels: reg_vreg4_b_tap_pu from gnn_sample_meta.csv (classes = rounded unique tap values).

Readout: project each node to d_node, then concatenate all N node embeddings → vector of size N * d_node → MLP → logits.

Colab: use ``python -u train_vreg4b_global_gnn_grid.py ...`` or ``PYTHONUNBUFFERED=1`` so each epoch prints live.

Required files (under --data_root, default …/loadtype_8500_dailyagg):
  - Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv  (or --nodes_csv)
  - Heterogenous GNN dataset/edges/hetero_mv_edge_catalog.csv  (or --edge_catalog_csv)
  - gnn_sample_meta.csv  (or --meta_csv)
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

TARGET_META_COL = "reg_vreg4_b_tap_pu"
FEAT_COLS: tuple[str, ...] = ("p_load_kw", "q_load_kvar")


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


def _norm_sid(s: object) -> int:
    if s is None or s == "":
        raise ValueError("empty sample_id")
    try:
        x = float(s)
        return int(x) if x == int(x) else int(x)
    except (TypeError, ValueError):
        return int(str(s).strip())


def _build_old_to_new(node_indices: np.ndarray) -> dict[int, int]:
    uniq = np.unique(node_indices)
    return {int(o): i for i, o in enumerate(uniq)}


def _load_line_edges_supervised(
    edge_catalog_path: Path,
    old_to_new: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _load_pq_and_ids(nodes_csv: Path) -> tuple[torch.Tensor, np.ndarray, list[int], dict[int, int]]:
    usecols = ["sample_id", "node_idx"] + list(FEAT_COLS)
    df = pd.read_csv(nodes_csv, usecols=usecols)
    for c in usecols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in {nodes_csv}")

    sample_ids = sorted(df["sample_id"].unique().tolist())
    first = df[df["sample_id"] == sample_ids[0]].sort_values("node_idx")
    node_order = first["node_idx"].to_numpy()
    old_to_new = _build_old_to_new(node_order)
    n_nodes = len(node_order)

    s_ct = len(sample_ids)
    f_ct = len(FEAT_COLS)
    x_np = np.zeros((s_ct, n_nodes, f_ct), dtype=np.float32)

    for si, sid in enumerate(sample_ids):
        sub = df[df["sample_id"] == sid].sort_values("node_idx")
        if len(sub) != n_nodes:
            raise RuntimeError(f"sample_id={sid}: expected {n_nodes} rows, got {len(sub)}")
        if not np.array_equal(sub["node_idx"].to_numpy(), node_order):
            raise RuntimeError(f"sample_id={sid}: node_idx order differs from first sample")
        x_np[si] = sub[list(FEAT_COLS)].to_numpy(dtype=np.float32)

    return torch.from_numpy(x_np), node_order, sample_ids, old_to_new


def _load_meta_taps(meta_csv: Path, sample_ids: list[int], col: str) -> np.ndarray:
    df = pd.read_csv(meta_csv, usecols=["sample_id", col])
    if col not in df.columns:
        raise ValueError(f"{meta_csv} missing column {col!r}")
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    lookup: dict[int, float] = {}
    for _, row in df.iterrows():
        lookup[_norm_sid(row["sample_id"])] = float(row[col])
    out: list[float] = []
    for sid in sample_ids:
        k = _norm_sid(sid)
        if k not in lookup:
            raise KeyError(f"sample_id {k} not in {meta_csv}")
        out.append(float(lookup[k]))
    return np.asarray(out, dtype=np.float64)


def _build_classes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_q = np.round(y.astype(np.float64), 6)
    classes = np.unique(y_q)
    idx = np.searchsorted(classes, y_q)
    return classes.astype(np.float64), idx.astype(np.int64)


def _zscore_train(x: torch.Tensor, train_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xt = x[train_idx].reshape(-1, x.shape[-1])
    mean = xt.mean(dim=0, keepdim=True)
    std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


def _class_weights(y_idx_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    c = torch.bincount(y_idx_train, minlength=n_classes).float().clamp_min(1.0)
    w = c.sum() / c
    return w / w.mean().clamp_min(1e-12)


class TapDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_idx: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        self.x = x
        self.y_idx = y_idx
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Data:
        return Data(
            x=self.x[i],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=torch.tensor(int(self.y_idx[i]), dtype=torch.long),
        )


class GineTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
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
            self.convs.append(GINEConv(mlp, edge_dim=2))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, edge_attr))
            h = h + self.dropout(norm(h_msg))
        return h


class GcnTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _edge_weight(edge_attr: torch.Tensor) -> torch.Tensor:
        ea = edge_attr[:, :2]
        z = torch.sqrt(ea[:, 0] ** 2 + ea[:, 1] ** 2).clamp(min=1e-6)
        return 1.0 / z

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        ew = self._edge_weight(edge_attr)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, edge_index, edge_weight=ew))
            h = h + self.dropout(norm(h_msg))
        return h


def build_tap_head(global_dim: int, n_classes: int, depth: int, dropout: float) -> nn.Module:
    """depth = number of hidden layers before logits (1, 2, or 3). Widths 512→256→128."""
    hid = [512, 256, 128]
    layers: list[nn.Module] = []
    d_in = global_dim
    for i in range(depth):
        out_d = hid[min(i, len(hid) - 1)]
        layers.append(nn.Linear(d_in, out_d))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d_in = out_d
    layers.append(nn.Linear(d_in, n_classes))
    return nn.Sequential(*layers)


class GlobalVreg4BTapModel(nn.Module):
    """Trunk → per-node linear to d_node → flatten all nodes → MLP → logits."""

    def __init__(
        self,
        backbone: str,
        n_nodes: int,
        hidden: int,
        n_layers: int,
        trunk_dropout: float,
        d_node: int,
        n_classes: int,
        head_depth: int,
        head_dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.d_node = int(d_node)
        self.backbone = backbone
        in_dim = len(FEAT_COLS)
        if backbone == "gine":
            self.trunk = GineTrunk(in_dim, hidden, n_layers, trunk_dropout)
        elif backbone == "gcn":
            self.trunk = GcnTrunk(in_dim, hidden, n_layers, trunk_dropout)
        else:
            raise ValueError(backbone)
        self.node_readout = nn.Linear(hidden, d_node)
        gdim = n_nodes * d_node
        self.head = build_tap_head(gdim, n_classes, head_depth, head_dropout)

    def forward(self, batch) -> torch.Tensor:
        h = self.trunk(batch.x, batch.edge_index, batch.edge_attr)
        z = self.node_readout(h)
        bvec = batch.batch
        if bvec is None:
            g = z.reshape(1, -1)
        else:
            b = int(bvec.max().item()) + 1
            rows = []
            for bi in range(b):
                m = bvec == bi
                rows.append(z[m].reshape(-1))
            g = torch.stack(rows, dim=0)
        return self.head(g)


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    run_name: str,
    log_every: int,
) -> tuple[float, float, int]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_ce = float("inf")
    best_acc = 0.0
    bad = 0
    best_ep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_ok = 0
        tr_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += float(loss.item()) * batch.num_graphs
            pred = logits.argmax(dim=1)
            tr_ok += int((pred == y).sum().item())
            tr_n += batch.num_graphs
        tr_loss /= max(tr_n, 1)
        tr_acc = tr_ok / max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_ok = 0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                y = batch.y
                loss = crit(logits, y)
                va_loss += float(loss.item()) * batch.num_graphs
                pred = logits.argmax(dim=1)
                va_ok += int((pred == y).sum().item())
                va_n += batch.num_graphs
        va_loss /= max(va_n, 1)
        va_acc = va_ok / max(va_n, 1)

        sched.step(va_loss)

        if va_loss < best_ce - 1e-12 or (abs(va_loss - best_ce) < 1e-12 and va_acc > best_acc):
            best_ce = va_loss
            best_acc = va_acc
            best_ep = epoch
            bad = 0
        else:
            bad += 1

        le = max(1, log_every)
        if epoch == 1 or epoch % le == 0 or epoch == epochs:
            lr_cur = opt.param_groups[0]["lr"]
            print(
                f"  [{run_name}] ep {epoch:4d}/{epochs}  train_ce={tr_loss:.6f} acc={tr_acc:.4f}  "
                f"val_ce={va_loss:.6f} acc={va_acc:.4f}  best_val_ce={best_ce:.6f} best_acc={best_acc:.4f}  "
                f"lr={lr_cur:.2e}  es {bad}/{patience}",
                flush=True,
            )

        if bad >= patience:
            print(f"  [{run_name}] early stop at epoch {epoch}", flush=True)
            break

    return best_ce, best_acc, best_ep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="36-run grid: VREG4_B tap, global concat readout.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--nodes_csv",
        type=str,
        default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    )
    p.add_argument(
        "--edge_catalog_csv",
        type=str,
        default="Heterogenous GNN dataset/edges/hetero_mv_edge_catalog.csv",
    )
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument("--out_dir", type=str, default="checkpoints_vreg4b_global_grid")
    p.add_argument("--hidden", type=int, default=64, help="GNN trunk hidden width.")
    p.add_argument("--trunk_dropout", type=float, default=0.15)
    p.add_argument("--head_dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sample_frac", type=float, default=1.0, help="Use first fraction of samples (after sort).")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1, help="Print every N epochs (1 = every epoch).")
    p.add_argument(
        "--cache_tensor",
        type=str,
        default="",
        help="Optional path to save/load preprocessed [x, y_idx, edge_index, edge_attr, ...] as .pt",
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
    meta_path = Path(args.meta_csv) if os.path.isabs(args.meta_csv) else data_root / args.meta_csv

    for pth in (nodes_path, edge_path, meta_path):
        if not pth.is_file():
            raise FileNotFoundError(pth)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None

    if cache_path and cache_path.is_file():
        print(f"Loading cache {cache_path}", flush=True)
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = pack["x"]
        y_idx = pack["y_idx"]
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        sample_ids = pack["sample_ids"]
        class_values = pack["class_values"]
        old_to_new = pack["old_to_new"]
    else:
        print(f"Loading nodes: {nodes_path}", flush=True)
        x, _node_order, sample_ids, old_to_new = _load_pq_and_ids(nodes_path)
        print(f"Loading meta taps: {meta_path}", flush=True)
        y_raw = _load_meta_taps(meta_path, sample_ids, TARGET_META_COL)
        class_values, y_idx_np = _build_classes(y_raw)
        y_idx = torch.from_numpy(y_idx_np)
        print(f"  samples={x.shape[0]}  n_nodes={x.shape[1]}  n_classes={len(class_values)}", flush=True)

        edge_index, edge_attr = _load_line_edges_supervised(edge_path, old_to_new)
        print(f"  line edges (directed): {edge_index.shape[1]}  edge_attr={tuple(edge_attr.shape)}", flush=True)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "x": x,
                    "y_idx": y_idx,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "sample_ids": sample_ids,
                    "class_values": class_values,
                    "old_to_new": old_to_new,
                },
                cache_path,
            )
            print(f"  wrote cache -> {cache_path}", flush=True)

    n_all = x.shape[0]
    if args.max_samples is not None:
        k = min(int(args.max_samples), n_all)
        x, y_idx = x[:k], y_idx[:k]
        print(f"Using --max_samples={k}", flush=True)
    elif args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0, 1].")
        k = max(1, int(round(n_all * args.sample_frac)))
        x, y_idx = x[:k], y_idx[:k]
        print(f"Using --sample_frac={args.sample_frac} -> {k} samples", flush=True)

    n = x.shape[0]
    n_nodes = int(x.shape[1])
    n_classes = int(y_idx.max().item()) + 1

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = max(1, int(args.train_frac * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.size == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    x, mean, std = _zscore_train(x, train_idx)
    torch.save({"mean": mean, "std": std, "feat_cols": list(FEAT_COLS)}, out_dir / "pq_norm.pt")

    dataset = TapDataset(x, y_idx, edge_index, edge_attr)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    y_train_t = y_idx[train_idx]
    cw = _class_weights(y_train_t, n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    nw = int(args.num_workers)

    grid = []
    for backbone in ("gine", "gcn"):
        for n_layers in (2, 3):
            for d_node in (2, 4, 8):
                for head_depth in (1, 2, 3):
                    grid.append(
                        {
                            "backbone": backbone,
                            "n_layers": n_layers,
                            "d_node": d_node,
                            "head_depth": head_depth,
                            "name": f"{backbone}_L{n_layers}_dn{d_node}_hd{head_depth}",
                        }
                    )

    assert len(grid) == 36

    results: list[dict] = []
    print(
        f"Device={device}  train={len(train_ds)} val={len(val_ds)}  N={n_nodes}  "
        f"global_dim = {n_nodes} * d_node  |  {len(grid)} runs\n",
        flush=True,
    )

    for gi, cfg in enumerate(grid):
        name = cfg["name"]
        print(f"\n======== Run {gi + 1}/36: {name} ========", flush=True)
        model = GlobalVreg4BTapModel(
            backbone=cfg["backbone"],
            n_nodes=n_nodes,
            hidden=args.hidden,
            n_layers=cfg["n_layers"],
            trunk_dropout=args.trunk_dropout,
            d_node=cfg["d_node"],
            n_classes=n_classes,
            head_depth=cfg["head_depth"],
            head_dropout=args.head_dropout,
        ).to(device)

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

        run_dir = out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_ce, best_acc, best_ep = train_one_run(
            model,
            train_loader,
            val_loader,
            device,
            cw.to(device),
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            run_name=name,
            log_every=args.log_every,
        )

        ckpt_path = run_dir / "best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "n_nodes": n_nodes,
                "n_classes": n_classes,
                "class_values": class_values.tolist(),
                "target_col": TARGET_META_COL,
                "best_val_ce": best_ce,
                "best_val_acc": best_acc,
                "best_epoch": best_ep,
                "feat_cols": list(FEAT_COLS),
                "hidden": args.hidden,
            },
            ckpt_path,
        )

        results.append(
            {
                **cfg,
                "best_val_ce": best_ce,
                "best_val_acc": best_acc,
                "best_epoch": best_ep,
                "ckpt": str(ckpt_path),
            }
        )

    summary = {
        "data_root": str(data_root),
        "nodes_csv": str(nodes_path),
        "edge_catalog_csv": str(edge_path),
        "meta_csv": str(meta_path),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_nodes": n_nodes,
        "n_classes": n_classes,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "args": vars(args),
        "runs": sorted(results, key=lambda r: (r["best_val_ce"], -r["best_val_acc"])),
    }
    (out_dir / "grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_overall = summary["runs"][0]
    print("\nDone. Best run by val_ce:", best_overall["name"], "ce=", best_overall["best_val_ce"], "acc=", best_overall["best_val_acc"])
    print("Wrote", out_dir / "grid_summary.json", flush=True)


if __name__ == "__main__":
    main()
