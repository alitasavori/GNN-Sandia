"""
Compare GNN+MLP models (GINE, GraphSAGE, GCN) for complex voltage prediction.

Task:
  Inputs per node:  [p_load_kw, q_load_kvar]
  Outputs per node: [V_re, V_im]

Design:
  - GNN encoder produces 2D state per node.
  - Flattened node states [2N] feed the same RealMLP head used in MLP baseline.
  - No local/global decomposition path in this script.
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


def _append_shared_pe_features(
    x: torch.Tensor,
    *,
    node_to_local: dict[str, int],
    node_pe_csv: Path,
    node_pe_cols: str,
) -> torch.Tensor:
    import pandas as pd

    if not node_pe_csv.is_file():
        raise FileNotFoundError(node_pe_csv)
    pe_df = pd.read_csv(node_pe_csv)
    if "node" not in pe_df.columns:
        raise ValueError(f"{node_pe_csv} must contain 'node' column.")

    if str(node_pe_cols).strip().lower() == "auto":
        pe_cols = [c for c in pe_df.columns if str(c).lower().startswith("pe_")]
    else:
        pe_cols = [c.strip() for c in str(node_pe_cols).split(",") if c.strip()]
    if not pe_cols:
        raise ValueError("No PE columns selected/found. Use --node_pe_cols auto or provide explicit column names.")

    pe_map_exact: dict[str, np.ndarray] = {}
    pe_map_lower: dict[str, np.ndarray] = {}
    for _, r in pe_df[["node", *pe_cols]].iterrows():
        n = str(r["node"]).strip()
        v = np.asarray([float(r[c]) for c in pe_cols], dtype=np.float32)
        pe_map_exact[n] = v
        pe_map_lower[n.lower()] = v

    pe_np = np.zeros((len(node_to_local), len(pe_cols)), dtype=np.float32)
    missing = 0
    for n, i in node_to_local.items():
        v = pe_map_exact.get(n, pe_map_lower.get(n.lower()))
        if v is None:
            missing += 1
            continue
        pe_np[i, :] = v

    pe = torch.from_numpy(pe_np).unsqueeze(0).expand(int(x.shape[0]), -1, -1)
    print(f"Using PE from {node_pe_csv} with columns: {pe_cols}", flush=True)
    if missing > 0:
        print(f"[warn] PE missing for {missing}/{len(node_to_local)} nodes; zero-filled.", flush=True)
    return torch.cat([x, pe], dim=-1)


def _load_chunked_dataset(
    *,
    chunk_parent: Path,
    chunk_subdir_glob: str,
    nodes_csv_name: str,
    edges_csv_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], dict[str, int]]:
    chunk_dirs = sorted([p for p in chunk_parent.glob(chunk_subdir_glob) if p.is_dir()])
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk dirs matched {chunk_subdir_glob!r} under {chunk_parent}")

    print(f"[chunk_parent] {len(chunk_dirs)} chunks under {chunk_parent}", flush=True)
    for ch in chunk_dirs:
        print(f"  - {ch.name}", flush=True)

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    sample_ids_all: list[int] = []
    node_to_local_ref: dict[str, int] | None = None
    edge_index_ref: torch.Tensor | None = None
    edge_attr_ref: torch.Tensor | None = None

    for i, ch in enumerate(chunk_dirs):
        nodes_path = (ch / nodes_csv_name).resolve()
        edges_path = (ch / edges_csv_name).resolve()
        for p in (nodes_path, edges_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        x_i, _y_unused, sids_i, _node_order_i, node_to_local_i = _load_nodes_pq_target(nodes_path)
        if node_to_local_ref is None:
            node_to_local_ref = node_to_local_i
            edge_index_ref, edge_attr_ref = _load_compacted_edges(edges_path, node_to_local_ref)
        else:
            if node_to_local_i != node_to_local_ref:
                raise RuntimeError(f"{ch.name}: node order/mapping mismatch vs first chunk")
        y_i = _build_complex_targets(nodes_path, sids_i, node_to_local_ref)

        xs.append(x_i)
        ys.append(y_i)
        sample_ids_all.extend(sids_i)
        print(f"[chunk {i+1}/{len(chunk_dirs)}] samples={len(sids_i)}", flush=True)

    assert node_to_local_ref is not None
    assert edge_index_ref is not None and edge_attr_ref is not None
    x = torch.cat(xs, dim=0)
    y_ri = torch.cat(ys, dim=0)
    return x, y_ri, edge_index_ref, edge_attr_ref, sample_ids_all, node_to_local_ref


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


class RealMLP(nn.Module):
    # Same architecture as MLP baseline.
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNEncoder2D(nn.Module):
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

        self.state_head = nn.Linear(hidden, 2)

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

        state = self.state_head(h)  # [B*N,2]
        return state.view(batch.num_graphs, self.n_nodes, 2).reshape(batch.num_graphs, 2 * self.n_nodes)


class GNNPlusMLP(nn.Module):
    def __init__(self, encoder: GNNEncoder2D, mlp_hidden: int, n_nodes: int):
        super().__init__()
        self.encoder = encoder
        self.mlp = RealMLP(in_dim=2 * n_nodes, out_dim=2 * n_nodes, hidden=mlp_hidden)

    def forward(self, batch: Data) -> torch.Tensor:
        state_flat = self.encoder(batch)
        return self.mlp(state_flat)


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


def _val_r2_and_worst_mae_from_ri(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> tuple[float, float, float]:
    """
    Compute DA-GPS-style validation diagnostics on |V|:
      - mean/min R^2 across nodes
      - worst-node MAE across nodes
    """
    bsz, two_n = pred_ri.shape
    n_nodes = two_n // 2
    pred = pred_ri.view(bsz, n_nodes, 2)
    true = true_ri.view(bsz, n_nodes, 2)
    pred_mag = torch.sqrt(pred[..., 0] * pred[..., 0] + pred[..., 1] * pred[..., 1] + 1e-12)
    true_mag = torch.sqrt(true[..., 0] * true[..., 0] + true[..., 1] * true[..., 1] + 1e-12)

    mae_per_node = (pred_mag - true_mag).abs().mean(dim=0)  # [N]
    worst_mae = float(mae_per_node.max().item())

    # R^2 per node over batch dimension
    y_true_mean = true_mag.mean(dim=0, keepdim=True)
    ss_res = ((true_mag - pred_mag) ** 2).sum(dim=0)
    ss_tot = ((true_mag - y_true_mean) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    r2_mean = float(r2.mean().item())
    r2_min = float(r2.min().item())
    return r2_mean, r2_min, worst_mae


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
    in_dim: int,
    n_edges: int,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dl_te: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    run_label: str,
) -> RunResult:
    encoder = GNNEncoder2D(
        model_type=model_type,
        in_dim=in_dim,
        hidden=int(args.hidden_gnn),
        layers=int(args.layers),
        n_nodes=n_nodes,
        num_edges=n_edges,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        dropout=0.0 if args.disable_dropout else float(args.dropout),
    )
    model = GNNPlusMLP(encoder=encoder, mlp_hidden=int(args.hidden_mlp), n_nodes=n_nodes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_seen = 0
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
            train_loss_sum += float(loss.item()) * int(batch.num_graphs)
            n_train_seen += int(batch.num_graphs)

        train_loss = train_loss_sum / max(n_train_seen, 1)

        model.eval()
        val_loss = 0.0
        nv = 0
        val_preds = []
        val_tgts = []
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                yb = batch.y.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean.to(device)) / y_std.to(device)
                pred_n = model(batch)
                lv = mse(pred_n, yb_n)
                val_loss += float(lv.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
                yp = pred_n * y_std.to(device) + y_mean.to(device)
                val_preds.append(yp.detach().cpu())
                val_tgts.append(yb.detach().cpu())
        val_loss /= max(nv, 1)
        if val_preds:
            val_pred = torch.cat(val_preds, dim=0)
            val_tgt = torch.cat(val_tgts, dim=0)
            val_r2_mean, val_r2_min, val_worst_mae = _val_r2_and_worst_mae_from_ri(val_pred, val_tgt)
        else:
            val_r2_mean, val_r2_min, val_worst_mae = float("nan"), float("nan"), float("nan")
        sch.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % max(1, int(args.log_every)) == 0:
            print(
                f"[{model_type} {run_label}] epoch {ep:4d}/{args.epochs} | "
                f"train_tot={train_loss:.4f} train_volt={train_loss:.4f} | "
                f"val_tot={val_loss:.4f} val_volt={val_loss:.4f} | "
                f"val_r2_mean={val_r2_mean:.4f} val_r2_min={val_r2_min:.4f} "
                f"val_worst_mae={val_worst_mae:.4f} | best={best_val:.4f}",
                flush=True,
            )
        if bad >= args.patience:
            print(f"[{model_type}] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    met = _evaluate(model, dl_te, device, y_mean, y_std)
    ckpt_path = out_dir / f"{model_type}_gnn_plus_mlp_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "n_nodes": int(n_nodes),
            "hidden_gnn": int(args.hidden_gnn),
            "layers": int(args.layers),
            "hidden_mlp": int(args.hidden_mlp),
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
    p = argparse.ArgumentParser(description="Compare GNN+MLP models (gine,sage,gcn) for complex voltage prediction.")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg")
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv")
    p.add_argument("--chunk_parent", type=str, default="", help="If set, reads all chunk folders and concatenates samples.")
    p.add_argument("--chunk_subdir_glob", type=str, default="run_*")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv", help="Accepted for CLI parity; not used by this trainer.")
    p.add_argument("--out_dir", type=str, default="gnn_plus_mlp_compare_complex_8500")
    p.add_argument("--models", type=str, default="gine,sage,gcn", help="Comma-separated list from {gine,sage,gcn}.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_gnn", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden_mlp", type=int, default=1024)
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
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--cache_tensor", type=str, default="")
    p.add_argument("--node_pe_csv", type=str, default="")
    p.add_argument("--node_pe_cols", type=str, default="auto")
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

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_parent = Path(args.chunk_parent).resolve() if str(args.chunk_parent).strip() else None
    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None
    node_to_local = None
    if chunk_parent is not None:
        x, y_ri, edge_index, edge_attr, sample_ids, node_to_local = _load_chunked_dataset(
            chunk_parent=chunk_parent,
            chunk_subdir_glob=str(args.chunk_subdir_glob),
            nodes_csv_name=str(args.nodes_csv),
            edges_csv_name=str(args.edge_catalog_csv),
        )
    else:
        nodes_path = Path(args.nodes_csv)
        if not nodes_path.is_absolute():
            nodes_path = (data_root / nodes_path).resolve()
        edges_path = Path(args.edge_catalog_csv)
        if not edges_path.is_absolute():
            edges_path = (data_root / edges_path).resolve()
        for p in (nodes_path, edges_path):
            if not p.is_file():
                raise FileNotFoundError(p)

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

    if args.node_pe_csv:
        node_pe_path = Path(args.node_pe_csv)
        if not node_pe_path.is_absolute():
            node_pe_path = (repo / node_pe_path).resolve()
        x = _append_shared_pe_features(
            x,
            node_to_local=node_to_local,
            node_pe_csv=node_pe_path,
            node_pe_cols=args.node_pe_cols,
        )

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

    in_dim = int(x.shape[-1])
    xt = x[idx_train].reshape(-1, in_dim)
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
    run_label = "chunk_parent" if chunk_parent is not None else "single_csv"
    for m in models:
        print(f"\n=== Training {m.upper()} (GNN+MLP no local/global split) ===", flush=True)
        res = _train_one(
            model_type=m,
            args=args,
            n_nodes=n_nodes,
            in_dim=in_dim,
            n_edges=int(edge_index.shape[1]),
            dl_tr=dl_tr,
            dl_va=dl_va,
            dl_te=dl_te,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            out_dir=out_dir,
            run_label=run_label,
        )
        results[m] = res.__dict__
        print(
            f"[{m}] test |V| MAE={res.test_mae_vmag:.6f} pu, angle MAE={res.test_mae_angle_deg:.6f} deg, time={res.train_seconds:.1f}s",
            flush=True,
        )

    report = {
        "task": "GNN+MLP (no local/global split) PQ -> complex voltage [V_re,V_im]",
        "dataset_nodes_csv": str(args.nodes_csv),
        "dataset_edges_csv": str(args.edge_catalog_csv),
        "chunk_parent": str(chunk_parent) if chunk_parent is not None else "",
        "chunk_subdir_glob": str(args.chunk_subdir_glob),
        "n_samples": int(n),
        "n_nodes": int(n_nodes),
        "node_input_dim": int(in_dim),
        "split": {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
        "models_requested": models,
        "hyperparameters": {
            "hidden_gnn": int(args.hidden_gnn),
            "layers": int(args.layers),
            "hidden_mlp": int(args.hidden_mlp),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "dropout": 0.0 if args.disable_dropout else float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "log_every": int(args.log_every),
            "node_pe_csv": str(args.node_pe_csv),
            "node_pe_cols": str(args.node_pe_cols),
        },
        "results": results,
        "normalization": {
            "x_mean_path": str((out_dir / "x_mean.pt").resolve()),
            "x_std_path": str((out_dir / "x_std.pt").resolve()),
            "y_mean_path": str((out_dir / "y_mean.pt").resolve()),
            "y_std_path": str((out_dir / "y_std.pt").resolve()),
        },
    }
    report_path = out_dir / "gnn_plus_mlp_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
