"""
Compare GNN-only models (GINE, GraphSAGE, GCN) for complex voltage prediction.

Task:
  Inputs per node:  [p_load_kw, q_load_kvar]
  Outputs per node: [V_re, V_im]

Unlike GINE+MLP experiments, this script uses no global MLP head.
Each model predicts per-node complex voltage directly from node embeddings.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, SAGEConv

from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges, _load_nodes_pq_target


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_complex_targets(nodes_csv: Path, sample_ids: list[int], node_to_local: dict[str, int]) -> torch.Tensor:
    import pandas as pd

    sid_to_i = {int(s): i for i, s in enumerate(sample_ids)}
    y_ri = np.zeros((len(sample_ids), len(node_to_local), 2), dtype=np.float32)
    usecols = ["sample_id", "node", "vmag_pu", "vang_deg"]
    for chunk in pd.read_csv(nodes_csv, usecols=usecols, chunksize=500_000):
        row_s = chunk["sample_id"].map(lambda v: sid_to_i.get(int(float(v)), -1)).to_numpy(dtype=np.int64)
        row_n = chunk["node"].map(lambda v: node_to_local.get(str(v).strip(), -1)).to_numpy(dtype=np.int64)
        valid = (row_s >= 0) & (row_n >= 0)
        if not np.any(valid):
            continue
        s = row_s[valid]
        n = row_n[valid]
        vmag = chunk.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
        vang_rad = np.deg2rad(chunk.loc[valid, "vang_deg"].to_numpy(dtype=np.float32))
        y_ri[s, n, 0] = vmag * np.cos(vang_rad)
        y_ri[s, n, 1] = vmag * np.sin(vang_rad)
    return torch.from_numpy(y_ri)


class GraphVoltageDataset(Dataset):
    def __init__(self, x: torch.Tensor, y_ri: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self.x = x
        self.y_ri = y_ri
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Data:
        return Data(x=self.x[i], y=self.y_ri[i], edge_index=self.edge_index, edge_attr=self.edge_attr)


class GNNOnlyVoltageModel(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        in_dim: int,
        hidden: int,
        layers: int,
        n_nodes: int,
        num_edges: int,
        node_emb_dim: int,
        edge_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        mt = str(model_type).strip().lower()
        if mt not in {"gine", "sage", "gcn"}:
            raise ValueError(f"Unsupported model_type={model_type}")
        self.model_type = mt
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(float(dropout))

        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim) if (self.edge_emb_dim > 0 and mt == "gine") else None

        eff_in = in_dim + self.node_emb_dim
        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if mt == "gine":
            eff_edge = 2 + self.edge_emb_dim
            for _ in range(int(layers)):
                mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
                self.convs.append(GINEConv(mlp, edge_dim=eff_edge))
                self.norms.append(nn.LayerNorm(hidden))
        elif mt == "sage":
            for _ in range(int(layers)):
                self.convs.append(SAGEConv(hidden, hidden))
                self.norms.append(nn.LayerNorm(hidden))
        else:  # gcn
            for _ in range(int(layers)):
                self.convs.append(GCNConv(hidden, hidden))
                self.norms.append(nn.LayerNorm(hidden))

        self.out_head = nn.Linear(hidden, 2)  # [V_re, V_im]

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        h = self.input_proj(x)

        if self.model_type == "gine":
            ea = batch.edge_attr
            if self.edge_emb is not None:
                ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
            for conv, norm in zip(self.convs, self.norms):
                h_msg = F.relu(conv(h, batch.edge_index, ea))
                h = h + self.dropout(norm(h_msg))
        elif self.model_type == "sage":
            for conv, norm in zip(self.convs, self.norms):
                h_msg = F.relu(conv(h, batch.edge_index))
                h = h + self.dropout(norm(h_msg))
        else:  # gcn
            z = torch.sqrt(batch.edge_attr[:, 0] ** 2 + batch.edge_attr[:, 1] ** 2).clamp(min=1e-6)
            ew = 1.0 / z
            for conv, norm in zip(self.convs, self.norms):
                h_msg = F.relu(conv(h, batch.edge_index, edge_weight=ew))
                h = h + self.dropout(norm(h_msg))

        out = self.out_head(h)  # [B*N,2]
        return out.view(batch.num_graphs, self.n_nodes, 2).reshape(batch.num_graphs, 2 * self.n_nodes)


@dataclass
class RunResult:
    best_val_mse: float
    test_mae_vmag: float
    test_rmse_vmag: float
    test_mae_angle_deg: float
    test_rmse_angle_deg: float
    train_seconds: float
    checkpoint: str


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _metrics_from_ri_flat(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> dict[str, float]:
    bsz, two_n = pred_ri.shape
    n_nodes = two_n // 2
    pred = pred_ri.view(bsz, n_nodes, 2)
    true = true_ri.view(bsz, n_nodes, 2)
    pred_re, pred_im = pred[..., 0], pred[..., 1]
    true_re, true_im = true[..., 0], true[..., 1]
    pred_mag = torch.sqrt(pred_re * pred_re + pred_im * pred_im + 1e-12)
    true_mag = torch.sqrt(true_re * true_re + true_im * true_im + 1e-12)
    pred_ang = torch.atan2(pred_im, pred_re)
    true_ang = torch.atan2(true_im, true_re)
    ang_err_deg = _angle_diff_deg(pred_ang, true_ang)
    vmag_err = pred_mag - true_mag
    return {
        "mae_vmag_pu": float(vmag_err.abs().mean().item()),
        "rmse_vmag_pu": float(torch.sqrt((vmag_err * vmag_err).mean()).item()),
        "mae_angle_deg": float(ang_err_deg.abs().mean().item()),
        "rmse_angle_deg": float(torch.sqrt((ang_err_deg * ang_err_deg).mean()).item()),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, dl: DataLoader, device: torch.device, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    for batch in dl:
        batch = batch.to(device)
        yp_n = model(batch)
        yp = yp_n * y_std.to(device) + y_mean.to(device)
        preds.append(yp.cpu())
        tgts.append(batch.y.view(batch.num_graphs, -1).cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    return _metrics_from_ri_flat(pred, tgt)


def _train_one(
    *,
    model_type: str,
    args: argparse.Namespace,
    n_nodes: int,
    n_edges: int,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dl_te: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    out_dir: Path,
) -> RunResult:
    model = GNNOnlyVoltageModel(
        model_type=model_type,
        in_dim=2,
        hidden=int(args.hidden),
        layers=int(args.layers),
        n_nodes=n_nodes,
        num_edges=n_edges,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        dropout=0.0 if args.disable_dropout else float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            yb = batch.y.view(batch.num_graphs, -1)
            yb_n = (yb - y_mean.to(device)) / y_std.to(device)
            pred_n = model(batch)
            loss = mse(pred_n, yb_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                yb = batch.y.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean.to(device)) / y_std.to(device)
                pred_n = model(batch)
                lv = mse(pred_n, yb_n)
                val_loss += float(lv.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
        val_loss /= max(nv, 1)
        sch.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0:
            print(f"[{model_type}] epoch {ep:4d}/{args.epochs} val_mse_norm={val_loss:.6f} best={best_val:.6f}", flush=True)
        if bad >= args.patience:
            print(f"[{model_type}] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    met = _evaluate(model, dl_te, device, y_mean, y_std)
    ckpt_path = out_dir / f"{model_type}_gnn_only_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "n_nodes": int(n_nodes),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "x_norm_paths": {
                "x_mean": str((out_dir / "x_mean.pt").resolve()),
                "x_std": str((out_dir / "x_std.pt").resolve()),
            },
            "y_norm_paths": {
                "y_mean": str((out_dir / "y_mean.pt").resolve()),
                "y_std": str((out_dir / "y_std.pt").resolve()),
            },
        },
        ckpt_path,
    )
    return RunResult(
        best_val_mse=float(best_val),
        test_mae_vmag=met["mae_vmag_pu"],
        test_rmse_vmag=met["rmse_vmag_pu"],
        test_mae_angle_deg=met["mae_angle_deg"],
        test_rmse_angle_deg=met["rmse_angle_deg"],
        train_seconds=float(train_seconds),
        checkpoint=str(ckpt_path.resolve()),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare GNN-only models (gine,sage,gcn) for complex voltage prediction.")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg")
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv")
    p.add_argument("--out_dir", type=str, default="gnn_only_compare_complex_8500")
    p.add_argument("--models", type=str, default="gine,sage,gcn", help="Comma-separated list from {gine,sage,gcn}.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--edge_emb_dim", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--disable_dropout", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_tensor", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    allowed = {"gine", "sage", "gcn"}
    if not models or any(m not in allowed for m in models):
        raise ValueError(f"--models must be comma-separated subset of {sorted(allowed)}")

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv)
    if not nodes_path.is_absolute():
        nodes_path = (data_root / nodes_path).resolve()
    edges_path = Path(args.edge_catalog_csv)
    if not edges_path.is_absolute():
        edges_path = (data_root / edges_path).resolve()
    for p in (nodes_path, edges_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None
    node_to_local = None
    if cache_path and cache_path.is_file():
        print(f"Loading cache: {cache_path}", flush=True)
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = pack["x"]
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        sample_ids = pack["sample_ids"]
    else:
        x, _y_unused, sample_ids, _node_order, node_to_local = _load_nodes_pq_target(nodes_path)
        edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids},
                cache_path,
            )
            print(f"Wrote cache: {cache_path}", flush=True)

    if node_to_local is None:
        _x_tmp, _y_tmp, _sid_tmp, _node_order_tmp, node_to_local = _load_nodes_pq_target(nodes_path)
        del _x_tmp, _y_tmp, _sid_tmp, _node_order_tmp
    y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)

    if args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0,1].")
        k = max(1, int(round(len(sample_ids) * args.sample_frac)))
        x = x[:k]
        y_ri = y_ri[:k]
        sample_ids = sample_ids[:k]
        print(f"Using sample_frac={args.sample_frac} => {k} samples", flush=True)

    n = int(x.shape[0])
    n_nodes = int(x.shape[1])
    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("Invalid split; require at least one sample in train/val/test.")
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]
    print(f"Split train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}", flush=True)

    xt = x[idx_train].reshape(-1, 2)
    x_mean = xt.mean(dim=0, keepdim=True)
    x_std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_n = (x - x_mean) / x_std

    y_train = y_ri[idx_train].reshape(len(idx_train), -1)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    torch.save(y_mean, out_dir / "y_mean.pt")
    torch.save(y_std, out_dir / "y_std.pt")

    ds = GraphVoltageDataset(x_n, y_ri, edge_index, edge_attr)
    dl_tr = DataLoader(Subset(ds, idx_train.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(Subset(ds, idx_val.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(Subset(ds, idx_test.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} n_nodes={n_nodes} n_edges={int(edge_index.shape[1])} models={models}", flush=True)

    results: dict[str, dict] = {}
    for m in models:
        print(f"\n=== Training {m.upper()} (GNN-only) ===", flush=True)
        res = _train_one(
            model_type=m,
            args=args,
            n_nodes=n_nodes,
            n_edges=int(edge_index.shape[1]),
            dl_tr=dl_tr,
            dl_va=dl_va,
            dl_te=dl_te,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            out_dir=out_dir,
        )
        results[m] = res.__dict__
        print(
            f"[{m}] test |V| MAE={res.test_mae_vmag:.6f} pu, angle MAE={res.test_mae_angle_deg:.6f} deg, time={res.train_seconds:.1f}s",
            flush=True,
        )

    report = {
        "task": "GNN-only PQ -> complex voltage [V_re,V_im]",
        "dataset_nodes_csv": str(nodes_path),
        "dataset_edges_csv": str(edges_path),
        "n_samples": int(n),
        "n_nodes": int(n_nodes),
        "split": {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
        "models_requested": models,
        "hyperparameters": {
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "dropout": 0.0 if args.disable_dropout else float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        "results": results,
        "normalization": {
            "x_mean_path": str((out_dir / "x_mean.pt").resolve()),
            "x_std_path": str((out_dir / "x_std.pt").resolve()),
            "y_mean_path": str((out_dir / "y_mean.pt").resolve()),
            "y_std_path": str((out_dir / "y_std.pt").resolve()),
        },
    }
    report_path = out_dir / "gnn_only_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
