"""
High-capacity trainer: global pre-MLP -> deep GINE -> global post-MLP.

Goal:
  maximize in-distribution fit capacity (no aux, no per-node heads).
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
from torch_geometric.nn import GINEConv

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


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_proj(h)


class GINEUltraPrePost(nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        num_edges: int,
        pre_hidden: int,
        pre_depth: int,
        pre_out_dim: int,
        hidden_gnn: int,
        layers: int,
        state_dim: int,
        post_hidden: int,
        post_depth: int,
        node_emb_dim: int,
        edge_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.state_dim = int(state_dim)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(float(dropout))

        self.pre_global = ResidualMLP(
            in_dim=2 * self.n_nodes,
            out_dim=self.n_nodes * int(pre_out_dim),
            hidden=int(pre_hidden),
            depth=int(pre_depth),
            dropout=float(dropout),
        )

        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim) if self.edge_emb_dim > 0 else None

        eff_in = int(pre_out_dim) + self.node_emb_dim
        eff_edge = 2 + self.edge_emb_dim
        self.input_proj = nn.Linear(eff_in, int(hidden_gnn))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(layers)):
            mlp = nn.Sequential(
                nn.Linear(int(hidden_gnn), int(hidden_gnn) * 2),
                nn.GELU(),
                nn.Linear(int(hidden_gnn) * 2, int(hidden_gnn)),
            )
            self.convs.append(GINEConv(mlp, edge_dim=eff_edge))
            self.norms.append(nn.LayerNorm(int(hidden_gnn)))
        self.state_head = nn.Linear(int(hidden_gnn), self.state_dim)

        self.post_global = ResidualMLP(
            in_dim=self.n_nodes * self.state_dim,
            out_dim=2 * self.n_nodes,
            hidden=int(post_hidden),
            depth=int(post_depth),
            dropout=float(dropout),
        )

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        ea = batch.edge_attr
        x_flat = x.view(batch.num_graphs, self.n_nodes * 2)
        x = self.pre_global(x_flat).view(batch.num_graphs * self.n_nodes, -1)
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        if self.edge_emb is not None:
            ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(norm(F.gelu(conv(h, batch.edge_index, ea))))
        s = self.state_head(h).view(batch.num_graphs, self.n_nodes, self.state_dim)
        return self.post_global(s.reshape(batch.num_graphs, self.n_nodes * self.state_dim))


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
def _evaluate_voltage(model: nn.Module, dl: DataLoader, device: torch.device, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    for batch in dl:
        batch = batch.to(device)
        yp_n = model(batch)
        yp = yp_n * y_std.to(device) + y_mean.to(device)
        preds.append(yp.cpu())
        tgts.append(batch.y.view(batch.num_graphs, -1).cpu())
    return _metrics_from_ri_flat(torch.cat(preds, dim=0), torch.cat(tgts, dim=0))


@dataclass
class RunResult:
    best_val_mse: float
    test_mae_vmag: float
    test_rmse_vmag: float
    test_mae_angle_deg: float
    test_rmse_angle_deg: float
    train_seconds: float
    checkpoint: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultra-capacity global pre/post MLP + GINE trainer.")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg")
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv")
    p.add_argument("--out_dir", type=str, default="gine_ultra_pre_post_complex_8500")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_gnn", type=int, default=128)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--state_dim", type=int, default=8)
    p.add_argument("--pre_hidden", type=int, default=2048)
    p.add_argument("--pre_depth", type=int, default=2)
    p.add_argument("--pre_out_dim", type=int, default=32)
    p.add_argument("--post_hidden", type=int, default=2048)
    p.add_argument("--post_depth", type=int, default=2)
    p.add_argument("--node_emb_dim", type=int, default=8)
    p.add_argument("--edge_emb_dim", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--disable_dropout", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_sched_factor", type=float, default=0.8)
    p.add_argument("--lr_sched_patience", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--cache_tensor", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv) if Path(args.nodes_csv).is_absolute() else (data_root / args.nodes_csv).resolve()
    edges_path = Path(args.edge_catalog_csv) if Path(args.edge_catalog_csv).is_absolute() else (data_root / args.edge_catalog_csv).resolve()
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
            torch.save({"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids}, cache_path)
            print(f"Wrote cache: {cache_path}", flush=True)

    if node_to_local is None:
        _x_tmp, _y_tmp, _sid_tmp, _n_tmp, node_to_local = _load_nodes_pq_target(nodes_path)
        del _x_tmp, _y_tmp, _sid_tmp, _n_tmp

    y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)
    if args.sample_frac < 1.0:
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

    ds = GraphVoltageDataset(x_n, y_ri, edge_index, edge_attr)
    dl_tr = DataLoader(Subset(ds, idx_train.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(Subset(ds, idx_val.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(Subset(ds, idx_test.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINEUltraPrePost(
        n_nodes=n_nodes,
        num_edges=int(edge_index.shape[1]),
        pre_hidden=int(args.pre_hidden),
        pre_depth=int(args.pre_depth),
        pre_out_dim=int(args.pre_out_dim),
        hidden_gnn=int(args.hidden_gnn),
        layers=int(args.layers),
        state_dim=int(args.state_dim),
        post_hidden=int(args.post_hidden),
        post_depth=int(args.post_depth),
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        dropout=0.0 if args.disable_dropout else float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(args.lr_sched_factor), patience=int(args.lr_sched_patience)
    )
    mse = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    bad = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            yv = batch.y.view(batch.num_graphs, -1)
            yv_n = (yv - y_mean.to(device)) / y_std.to(device)
            pred_n = model(batch)
            loss = mse(pred_n, yv_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_sum = 0.0
        nv = 0
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                yv = batch.y.view(batch.num_graphs, -1)
                yv_n = (yv - y_mean.to(device)) / y_std.to(device)
                lv = mse(model(batch), yv_n)
                val_sum += float(lv.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
        val_v_mse = val_sum / max(nv, 1)
        sch.step(val_v_mse)
        live = _evaluate_voltage(model, dl_va, device, y_mean, y_std)
        lr = float(opt.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(ep),
                "lr": lr,
                "val_mse_norm": float(val_v_mse),
                "val_mae_vmag_pu": float(live["mae_vmag_pu"]),
                "val_rmse_vmag_pu": float(live["rmse_vmag_pu"]),
            }
        )
        if val_v_mse < best_val_mse:
            best_val_mse = val_v_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % int(args.log_every) == 0:
            print(
                f"[gine-ultra-pre-post] epoch {ep:4d}/{args.epochs} lr={lr:.6g} "
                f"val_mse_norm={val_v_mse:.6f} val_mae_vmag={live['mae_vmag_pu']:.6f} "
                f"best_val_mse={best_val_mse:.6f}",
                flush=True,
            )
        if bad >= args.patience:
            print(f"[gine-ultra-pre-post] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    met = _evaluate_voltage(model, dl_te, device, y_mean, y_std)

    ckpt_path = out_dir / "gine_ultra_pre_post_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_nodes": int(n_nodes),
            "hidden_gnn": int(args.hidden_gnn),
            "layers": int(args.layers),
            "state_dim": int(args.state_dim),
            "pre_hidden": int(args.pre_hidden),
            "pre_depth": int(args.pre_depth),
            "pre_out_dim": int(args.pre_out_dim),
            "post_hidden": int(args.post_hidden),
            "post_depth": int(args.post_depth),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        ckpt_path,
    )
    result = RunResult(
        best_val_mse=float(best_val_mse),
        test_mae_vmag=met["mae_vmag_pu"],
        test_rmse_vmag=met["rmse_vmag_pu"],
        test_mae_angle_deg=met["mae_angle_deg"],
        test_rmse_angle_deg=met["rmse_angle_deg"],
        train_seconds=float(train_seconds),
        checkpoint=str(ckpt_path.resolve()),
    )
    report = {
        "task": "Ultra global pre/post MLP + deep GINE (no aux)",
        "result": result.__dict__,
        "history": history,
    }
    (out_dir / "gine_ultra_pre_post_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

