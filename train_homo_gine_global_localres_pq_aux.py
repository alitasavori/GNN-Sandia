"""
Train load-only homo GNN with local+global voltage head + auxiliary device heads.

Purpose:
  - Deep supervision / privileged-information training.
  - During training: optimize voltage loss + auxiliary regulator/capacitor losses.
  - During inference: use voltage head only (aux heads are ignored).

Base model: train_homo_gine_global_localres_pq_loadonly.py style
  - Node inputs: p_load_kw, q_load_kvar
  - Edge inputs: R_full, X_full from compacted load-only edge CSV
  - Optional node/edge embeddings
  - Final voltage prediction = V_local + DeltaV_global
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

from train_homo_gine_global_localres_pq_loadonly import (
    _load_compacted_edges,
    _load_nodes_pq_target,
    _zscore_features_train,
)

TARGET_REG_COLS: tuple[str, ...] = (
    "reg_feeder_rega_tap_pu",
    "reg_feeder_regb_tap_pu",
    "reg_feeder_regc_tap_pu",
    "reg_vreg2_a_tap_pu",
    "reg_vreg2_b_tap_pu",
    "reg_vreg2_c_tap_pu",
    "reg_vreg3_a_tap_pu",
    "reg_vreg3_b_tap_pu",
    "reg_vreg3_c_tap_pu",
    "reg_vreg4_a_tap_pu",
    "reg_vreg4_b_tap_pu",
    "reg_vreg4_c_tap_pu",
)

TARGET_CAP_COLS: tuple[str, ...] = (
    "cap_capbank0a_n_steps_on",
    "cap_capbank0b_n_steps_on",
    "cap_capbank0c_n_steps_on",
    "cap_capbank1a_n_steps_on",
    "cap_capbank1b_n_steps_on",
    "cap_capbank1c_n_steps_on",
    "cap_capbank2a_n_steps_on",
    "cap_capbank2b_n_steps_on",
    "cap_capbank2c_n_steps_on",
    "cap_capbank3_n_steps_on",
)


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
    try:
        x = float(s)
        return int(x)
    except Exception:
        return int(str(s).strip())


def _build_classes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build class ids from all available labels for stable CE indexing.
    (Using train-only class sets can crash when val has unseen labels.)
    """
    yq = np.round(y.astype(np.float64), 6)
    classes = np.unique(yq)
    cls_to_i = {float(c): i for i, c in enumerate(classes.tolist())}
    idx = np.array([cls_to_i[float(v)] for v in yq.tolist()], dtype=np.int64)
    return classes.astype(np.float32), idx


def _load_aux_targets(meta_csv: Path, sample_ids: list[int]) -> dict:
    usecols = ["sample_id", *TARGET_REG_COLS, *TARGET_CAP_COLS]
    df = pd.read_csv(meta_csv, usecols=usecols)
    lk = {_norm_sid(k): i for i, k in enumerate(df["sample_id"].tolist())}

    miss = [sid for sid in sample_ids if _norm_sid(sid) not in lk]
    if miss:
        raise KeyError(f"{len(miss)} sample_id values from nodes CSV not found in {meta_csv}")

    order = [lk[_norm_sid(sid)] for sid in sample_ids]
    out: dict = {
        "reg": [],
        "cap": [],
    }
    for c in TARGET_REG_COLS:
        y = df[c].to_numpy(dtype=np.float64)[order]
        cls, yi = _build_classes(y)
        out["reg"].append({"name": c, "classes": cls, "y_idx": torch.from_numpy(yi)})
    for c in TARGET_CAP_COLS:
        y = df[c].to_numpy(dtype=np.float64)[order]
        cls, yi = _build_classes(y)
        out["cap"].append({"name": c, "classes": cls, "y_idx": torch.from_numpy(yi)})
    return out


class AuxDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_v: torch.Tensor,
        y_reg: list[torch.Tensor],
        y_cap: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        self.x = x
        self.y_v = y_v
        self.y_reg = y_reg
        self.y_cap = y_cap
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Data:
        d = Data(
            x=self.x[i],
            y=self.y_v[i],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )
        d.y_reg = torch.stack([yr[i] for yr in self.y_reg], dim=0).long()  # [12]
        d.y_cap = torch.stack([yc[i] for yc in self.y_cap], dim=0).long()  # [10]
        return d


class GlobalLocalAuxBase(nn.Module):
    def __init__(self, n_nodes: int, hidden: int, node_out_dim: int, dropout: float):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.node_proj = nn.Linear(hidden, node_out_dim)
        self.local_head = nn.Linear(hidden, 1)
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
        self.aux_proj = nn.Linear(gdim, hidden)
        self.aux_reg_heads = nn.ModuleList()  # 12 heads, set by caller
        self.aux_cap_heads = nn.ModuleList()  # 10 heads, set by caller

    def _readout(self, h: torch.Tensor, bvec: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local_head(h).squeeze(-1)
        z = self.node_proj(h)
        if bvec is None:
            local = local.view(1, self.n_nodes)
            z = z.view(1, self.n_nodes, -1)
        else:
            b = int(bvec.max().item()) + 1
            local = local.view(b, self.n_nodes)
            z = z.view(b, self.n_nodes, -1)
        g = z.reshape(z.size(0), -1)  # [B, gdim]
        delta = self.global_head(g)
        v_pred = local + delta
        return v_pred, g

    def _aux_logits(self, g: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h = F.relu(self.aux_proj(g))
        reg_logits = [head(h) for head in self.aux_reg_heads]  # each [B, Cj]
        cap_logits = [head(h) for head in self.aux_cap_heads]  # each [B, Ck]
        return reg_logits, cap_logits


class HomoGINEGlobalLocalAux(GlobalLocalAuxBase):
    def __init__(
        self,
        *,
        in_dim: int,
        n_nodes: int,
        num_edges: int,
        hidden: int,
        n_layers: int,
        node_out_dim: int,
        dropout: float,
        node_emb_dim: int,
        edge_emb_dim: int,
        reg_nclasses: list[int],
        cap_nclasses: list[int],
    ):
        super().__init__(n_nodes=n_nodes, hidden=hidden, node_out_dim=node_out_dim, dropout=dropout)
        self.n_nodes = n_nodes
        self.num_edges = num_edges
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.node_emb = nn.Embedding(n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(num_edges, self.edge_emb_dim) if self.edge_emb_dim > 0 else None

        eff_in = in_dim + self.node_emb_dim
        eff_edge = 2 + self.edge_emb_dim
        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=eff_edge))
            self.norms.append(nn.LayerNorm(hidden))

        self.aux_reg_heads = nn.ModuleList([nn.Linear(hidden, c) for c in reg_nclasses])
        self.aux_cap_heads = nn.ModuleList([nn.Linear(hidden, c) for c in cap_nclasses])

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def forward_train(self, batch: Data) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        x = batch.x
        ea = batch.edge_attr
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        if self.edge_emb is not None:
            ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, batch.edge_index, ea))
            h = h + self.dropout(norm(h_msg))
        v_pred, g = self._readout(h, batch.batch)
        reg_logits, cap_logits = self._aux_logits(g)
        return v_pred, reg_logits, cap_logits

    def forward(self, batch: Data) -> torch.Tensor:
        v_pred, _, _ = self.forward_train(batch)
        return v_pred


class HomoGCNGlobalLocalAux(GlobalLocalAuxBase):
    def __init__(
        self,
        *,
        in_dim: int,
        n_nodes: int,
        hidden: int,
        n_layers: int,
        node_out_dim: int,
        dropout: float,
        node_emb_dim: int,
        reg_nclasses: list[int],
        cap_nclasses: list[int],
    ):
        super().__init__(n_nodes=n_nodes, hidden=hidden, node_out_dim=node_out_dim, dropout=dropout)
        self.n_nodes = n_nodes
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.dropout = nn.Dropout(dropout)
        self.node_emb = nn.Embedding(n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        eff_in = in_dim + self.node_emb_dim
        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.aux_reg_heads = nn.ModuleList([nn.Linear(hidden, c) for c in reg_nclasses])
        self.aux_cap_heads = nn.ModuleList([nn.Linear(hidden, c) for c in cap_nclasses])

    @staticmethod
    def _edge_weight(edge_attr: torch.Tensor) -> torch.Tensor:
        z = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2).clamp(min=1e-6)
        return 1.0 / z

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def forward_train(self, batch: Data) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        x = batch.x
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        h = self.input_proj(x)
        ew = self._edge_weight(batch.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, batch.edge_index, edge_weight=ew))
            h = h + self.dropout(norm(h_msg))
        v_pred, g = self._readout(h, batch.batch)
        reg_logits, cap_logits = self._aux_logits(g)
        return v_pred, reg_logits, cap_logits

    def forward(self, batch: Data) -> torch.Tensor:
        v_pred, _, _ = self.forward_train(batch)
        return v_pred


def _aux_loss(
    reg_logits: list[torch.Tensor],
    cap_logits: list[torch.Tensor],
    y_reg: torch.Tensor,
    y_cap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # y_reg [B,12], y_cap [B,10]
    reg_losses = []
    for j, lg in enumerate(reg_logits):
        reg_losses.append(F.cross_entropy(lg, y_reg[:, j]))
    cap_losses = []
    for j, lg in enumerate(cap_logits):
        cap_losses.append(F.cross_entropy(lg, y_cap[:, j]))
    lreg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=y_reg.device)
    lcap = torch.stack(cap_losses).mean() if cap_losses else torch.tensor(0.0, device=y_reg.device)
    return lreg, lcap


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
    lambda_reg: float,
    lambda_cap: float,
    checkpoint_path: Path,
    log_every: int,
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    mse = nn.MSELoss()
    best = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_mae = tr_mse = tr_auxr = tr_auxc = 0.0
        ntr = 0
        for batch in train_loader:
            batch = batch.to(device)
            v_pred, reg_logits, cap_logits = model.forward_train(batch)
            yv = batch.y.view(batch.num_graphs, -1)
            yr = batch.y_reg.view(batch.num_graphs, -1).long()  # [B, 12]
            yc = batch.y_cap.view(batch.num_graphs, -1).long()  # [B, 10]
            lv = mse(v_pred, yv)
            lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
            loss = lv + float(lambda_reg) * lr_aux + float(lambda_cap) * lc_aux
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_mse += float(lv.item()) * batch.num_graphs
            tr_mae += float((v_pred - yv).abs().mean(dim=1).sum().item())
            tr_auxr += float(lr_aux.item()) * batch.num_graphs
            tr_auxc += float(lc_aux.item()) * batch.num_graphs
            ntr += int(batch.num_graphs)
        tr_mse /= max(ntr, 1)
        tr_mae /= max(ntr, 1)
        tr_auxr /= max(ntr, 1)
        tr_auxc /= max(ntr, 1)

        model.eval()
        va_mae = va_mse = va_auxr = va_auxc = 0.0
        nva = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                v_pred, reg_logits, cap_logits = model.forward_train(batch)
                yv = batch.y.view(batch.num_graphs, -1)
                yr = batch.y_reg.view(batch.num_graphs, -1).long()
                yc = batch.y_cap.view(batch.num_graphs, -1).long()
                lv = mse(v_pred, yv)
                lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
                va_mse += float(lv.item()) * batch.num_graphs
                va_mae += float((v_pred - yv).abs().mean(dim=1).sum().item())
                va_auxr += float(lr_aux.item()) * batch.num_graphs
                va_auxc += float(lc_aux.item()) * batch.num_graphs
                nva += int(batch.num_graphs)
        va_mse /= max(nva, 1)
        va_mae /= max(nva, 1)
        va_auxr /= max(nva, 1)
        va_auxc /= max(nva, 1)
        sch.step(va_mse)

        if va_mse < best:
            best = va_mse
            bad = 0
            torch.save(getattr(model, "_orig_mod", model).state_dict(), checkpoint_path)
        else:
            bad += 1

        if ep == 1 or ep % max(1, log_every) == 0:
            print(
                f"Epoch {ep:4d} | train_mae={tr_mae:.6f} train_mse={tr_mse:.6f} "
                f"aux_reg={tr_auxr:.4f} aux_cap={tr_auxc:.4f} | "
                f"val_mae={va_mae:.6f} val_mse={va_mse:.6f} aux_reg={va_auxr:.4f} aux_cap={va_auxc:.4f} | "
                f"best_val_mse={best:.6f} | patience {bad}/{patience}",
                flush=True,
            )
        if bad >= patience:
            print(f"Early stopping at epoch {ep}", flush=True)
            break
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aux-head deep supervision training; drop aux heads at inference.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument("--out_dir", type=str, default="checkpoints_homo_gine_global_localres_pq_aux")
    p.add_argument("--model", type=str, choices=("gine", "gcn"), default="gine")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--node_out_dim", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--disable_dropout", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--node_emb_dim", type=int, default=0)
    p.add_argument("--edge_emb_dim", type=int, default=0)
    p.add_argument("--lambda_reg", type=float, default=0.2)
    p.add_argument("--lambda_cap", type=float, default=0.1)
    p.add_argument("--cache_tensor", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_stdout()
    _set_seeds(args.seed)
    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else repo / "datasets_gnn2" / "loadtype_8500_dailyagg"
    nodes_path = Path(args.nodes_csv) if os.path.isabs(args.nodes_csv) else data_root / args.nodes_csv
    edges_path = Path(args.edge_catalog_csv) if os.path.isabs(args.edge_catalog_csv) else data_root / args.edge_catalog_csv
    meta_path = Path(args.meta_csv) if os.path.isabs(args.meta_csv) else data_root / args.meta_csv
    for p in (nodes_path, edges_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = 0.0 if args.disable_dropout else float(args.dropout)
    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None

    if cache_path and cache_path.is_file():
        print(f"Loading tensor cache: {cache_path}", flush=True)
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = pack["x"]
        yv = pack["y"]
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        sample_ids = pack["sample_ids"]
    else:
        x, yv, sample_ids, _node_order, node_to_local = _load_nodes_pq_target(nodes_path)
        edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"x": x, "y": yv, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids},
                cache_path,
            )
            print(f"Wrote tensor cache: {cache_path}", flush=True)

    n_all = x.shape[0]
    if args.max_samples is not None:
        k = min(int(args.max_samples), n_all)
        x, yv, sample_ids = x[:k], yv[:k], sample_ids[:k]
        print(f"Using --max_samples={k}", flush=True)
    elif args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0,1].")
        k = max(1, int(round(n_all * args.sample_frac)))
        x, yv, sample_ids = x[:k], yv[:k], sample_ids[:k]
        print(f"Using --sample_frac={args.sample_frac} -> {k} samples", flush=True)
    else:
        print(f"Using all {n_all} samples", flush=True)

    n = x.shape[0]
    n_nodes = int(x.shape[1])
    n_edges = int(edge_index.shape[1])
    perm = np.random.RandomState(args.seed).permutation(n)
    n_train = max(1, int(args.train_frac * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.size == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    x, mean, std = _zscore_features_train(x, train_idx)
    torch.save({"mean": mean, "std": std, "feat_cols": ["p_load_kw", "q_load_kvar"]}, out_dir / "feature_norm_pq.pt")

    aux = _load_aux_targets(meta_path, sample_ids)
    y_reg = [d["y_idx"] for d in aux["reg"]]
    y_cap = [d["y_idx"] for d in aux["cap"]]
    reg_nclasses = [len(d["classes"]) for d in aux["reg"]]
    cap_nclasses = [len(d["classes"]) for d in aux["cap"]]

    ds = AuxDataset(x, yv, y_reg, y_cap, edge_index, edge_attr)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    nw = int(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    if args.model == "gine":
        model: nn.Module = HomoGINEGlobalLocalAux(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=n_edges,
            hidden=args.hidden,
            n_layers=args.layers,
            node_out_dim=args.node_out_dim,
            dropout=drop,
            node_emb_dim=max(0, int(args.node_emb_dim)),
            edge_emb_dim=max(0, int(args.edge_emb_dim)),
            reg_nclasses=reg_nclasses,
            cap_nclasses=cap_nclasses,
        )
    else:
        model = HomoGCNGlobalLocalAux(
            in_dim=2,
            n_nodes=n_nodes,
            hidden=args.hidden,
            n_layers=args.layers,
            node_out_dim=args.node_out_dim,
            dropout=drop,
            node_emb_dim=max(0, int(args.node_emb_dim)),
            reg_nclasses=reg_nclasses,
            cap_nclasses=cap_nclasses,
        )
    model = model.to(device)
    if args.model == "gcn" and int(args.edge_emb_dim) > 0:
        print("model=gcn ignores edge_emb_dim", flush=True)

    emb_tag = f"_ne{int(args.node_emb_dim)}_ee{int(args.edge_emb_dim)}"
    do_tag = "_do0" if drop == 0.0 else f"_do{drop:g}"
    ckpt = out_dir / (
        f"homo_{args.model}_global_localres_pq_aux_h{args.hidden}_L{args.layers}_nout{args.node_out_dim}"
        f"{emb_tag}{do_tag}_best.pt"
    )

    print(
        f"Device={device} train/val={len(train_idx)}/{len(val_idx)} N={n_nodes} E={n_edges} "
        f"model={args.model} hidden={args.hidden} layers={args.layers} nout={args.node_out_dim}",
        flush=True,
    )
    print(
        f"aux λ: reg={args.lambda_reg} cap={args.lambda_cap} | dropout={drop} "
        f"| node_emb={args.node_emb_dim} edge_emb={args.edge_emb_dim}",
        flush=True,
    )
    print("Starting training...", flush=True)

    best_val_mse = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_reg=float(args.lambda_reg),
        lambda_cap=float(args.lambda_cap),
        checkpoint_path=ckpt,
        log_every=args.log_every,
    )

    meta_out = {
        "best_val_mse_pu2": float(best_val_mse),
        "nodes_csv": str(nodes_path),
        "edge_catalog_csv": str(edges_path),
        "meta_csv": str(meta_path),
        "n_samples": int(len(sample_ids)),
        "sample_frac": float(args.sample_frac),
        "max_samples": args.max_samples,
        "n_nodes": int(n_nodes),
        "n_edges_directed": int(n_edges),
        "n_features": 2,
        "model": args.model,
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "node_out_dim": int(args.node_out_dim),
        "node_emb_dim": int(args.node_emb_dim),
        "edge_emb_dim": int(args.edge_emb_dim),
        "dropout": float(drop),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
        "lambda_reg": float(args.lambda_reg),
        "lambda_cap": float(args.lambda_cap),
        "aux_targets": {
            "reg": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["reg"]],
            "cap": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["cap"]],
        },
    }
    with open(out_dir / "train_metrics_global_localres_aux.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print("Best val MSE:", best_val_mse, flush=True)
    print("Saved:", ckpt, flush=True)


if __name__ == "__main__":
    main()

