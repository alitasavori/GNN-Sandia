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
from torch_geometric.utils import to_dense_batch

from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_node_feature_cols(spec: str) -> list[str]:
    cols = [c.strip() for c in str(spec).split(",") if c.strip()]
    if not cols:
        raise ValueError("--node_feature_cols must provide at least one column.")
    return cols


def _load_nodes_features_target(
    nodes_csv: Path,
    node_feature_cols: list[str],
    selected_sample_ids: list[int] | None = None,
) -> tuple[torch.Tensor, list[int], list[str], dict[str, int]]:
    import pandas as pd

    usecols = ["sample_id", "node", "node_idx", *node_feature_cols]
    print(f"Loading nodes: {nodes_csv}", flush=True)
    df = pd.read_csv(nodes_csv, usecols=usecols)
    sample_ids = sorted(int(x) for x in df["sample_id"].unique().tolist())
    if selected_sample_ids is not None:
        keep = {int(x) for x in selected_sample_ids}
        sample_ids = [sid for sid in sample_ids if sid in keep]
        if not sample_ids:
            raise RuntimeError(f"No selected sample IDs found in {nodes_csv}")
        df = df[df["sample_id"].astype(int).isin(sample_ids)].copy()
    first = df[df["sample_id"] == sample_ids[0]].sort_values("node_idx")
    node_order = first["node"].astype(str).str.strip().tolist()
    node_to_local = {n: i for i, n in enumerate(node_order)}
    n_nodes = len(node_order)
    n_feat = len(node_feature_cols)

    x_np = np.zeros((len(sample_ids), n_nodes, n_feat), dtype=np.float32)
    for si, sid in enumerate(sample_ids):
        if si > 0 and si % 1000 == 0:
            print(f"  stacked {si}/{len(sample_ids)} samples...", flush=True)
        sub = df[df["sample_id"] == sid].sort_values("node_idx")
        if len(sub) != n_nodes:
            raise RuntimeError(f"sample_id={sid}: expected {n_nodes}, got {len(sub)}")
        if sub["node"].astype(str).str.strip().tolist() != node_order:
            raise RuntimeError(f"sample_id={sid}: node order mismatch")
        x_np[si, :, :] = sub[node_feature_cols].to_numpy(np.float32)

    return torch.from_numpy(x_np), sample_ids, node_order, node_to_local


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


def _select_chunk_sample_ids(sids: list[int], sample_frac: float, seed: int, chunk_idx: int) -> list[int]:
    if float(sample_frac) >= 1.0:
        return [int(x) for x in sids]
    if not (0.0 < float(sample_frac) <= 1.0):
        raise ValueError("--sample_frac must be in (0,1].")
    if len(sids) == 0:
        return []
    rng = np.random.default_rng(int(seed) + int(chunk_idx) * 100_003)
    k = max(1, int(round(len(sids) * float(sample_frac))))
    pick = rng.choice(len(sids), size=k, replace=False)
    pick_sorted = np.sort(pick)
    return [int(sids[i]) for i in pick_sorted]


def _chunk_cache_path(cache_dir: Path, chunk_name: str, sample_frac: float, seed: int, chunk_idx: int) -> Path:
    if float(sample_frac) >= 1.0:
        tag = "full"
    else:
        tag = f"sf{float(sample_frac):.6f}_s{int(seed)}_c{int(chunk_idx)}"
    return cache_dir / f"{chunk_name}__{tag}.pt"


def _ensure_chunk_tensor_cache_gnn(
    chunk_dir: Path,
    *,
    nodes_name: str,
    node_feature_cols: list[str],
    selected_sample_ids: list[int] | None,
    cache_pt: Path,
    ref_ntl: dict[str, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], dict[str, int]]:
    np_ = chunk_dir / nodes_name
    if cache_pt.is_file():
        z = torch.load(cache_pt, map_location="cpu", weights_only=False)
        if not all(k in z for k in ("x", "y_ri", "sample_ids", "node_to_local")):
            raise RuntimeError(f"Cache missing required keys: {cache_pt}")
        ntl = z["node_to_local"]
        if ref_ntl is not None and ntl != ref_ntl:
            raise RuntimeError(f"node_to_local mismatch vs first chunk: {chunk_dir}")
        sids = z["sample_ids"]
        if isinstance(sids, torch.Tensor):
            sids = [int(x) for x in sids.tolist()]
        else:
            sids = [int(x) for x in list(sids)]
        x = z["x"].to(dtype=torch.float32)
        y_ri = z["y_ri"].to(dtype=torch.float32)
        return x, y_ri, sids, ntl

    if not np_.is_file():
        raise FileNotFoundError(np_)
    x, sample_ids, _node_order, node_to_local = _load_nodes_features_target(
        np_,
        node_feature_cols=node_feature_cols,
        selected_sample_ids=selected_sample_ids,
    )
    y_ri = _build_complex_targets(np_, sample_ids, node_to_local).to(dtype=torch.float32)
    x = x.to(dtype=torch.float32)
    if ref_ntl is not None and node_to_local != ref_ntl:
        raise RuntimeError(f"node_to_local mismatch vs first chunk: {chunk_dir}")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": x,
            "y_ri": y_ri,
            "sample_ids": sample_ids,
            "node_to_local": node_to_local,
        },
        cache_pt,
    )
    print(f"Wrote chunk tensor cache: {cache_pt}", flush=True)
    return x, y_ri, sample_ids, node_to_local


def _load_chunked_dataset(
    *,
    chunk_parent: Path,
    chunk_subdir_glob: str,
    nodes_csv_name: str,
    edges_csv_name: str,
    node_feature_cols: list[str],
    edge_shared_csv: Path | None,
    cache_dir: Path | None,
    sample_frac: float,
    seed: int,
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
        edge_path_to_check = edge_shared_csv if edge_shared_csv is not None else edges_path
        for p in (nodes_path, edge_path_to_check):
            if not p.is_file():
                raise FileNotFoundError(p)

        if cache_dir is not None:
            full_sids = sorted(int(x) for x in __import__("pandas").read_csv(nodes_path, usecols=["sample_id"])["sample_id"].unique().tolist())
            selected_sids = _select_chunk_sample_ids(full_sids, float(sample_frac), int(seed), int(i))
            cache_pt = _chunk_cache_path(cache_dir, ch.name, float(sample_frac), int(seed), int(i))
            x_i, y_i, sids_i, node_to_local_i = _ensure_chunk_tensor_cache_gnn(
                ch,
                nodes_name=nodes_csv_name,
                node_feature_cols=node_feature_cols,
                selected_sample_ids=selected_sids,
                cache_pt=cache_pt,
                ref_ntl=node_to_local_ref,
            )
        else:
            x_i, sids_i, _node_order_i, node_to_local_i = _load_nodes_features_target(
                nodes_path, node_feature_cols
            )
            y_i = _build_complex_targets(nodes_path, sids_i, node_to_local_i)

        if node_to_local_ref is None:
            node_to_local_ref = node_to_local_i
            edge_path = edge_shared_csv if edge_shared_csv is not None else edges_path
            if edge_shared_csv is not None:
                print(f"Using shared edges from {edge_path}", flush=True)
            edge_index_ref, edge_attr_ref = _load_compacted_edges(edge_path, node_to_local_ref)
        else:
            if node_to_local_i != node_to_local_ref:
                raise RuntimeError(f"{ch.name}: node order/mapping mismatch vs first chunk")

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


class GPSBlock(nn.Module):
    def __init__(self, *, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by gps_heads ({heads}).")
        self.local = GCNConv(hidden, hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        h_local = self.drop(F.gelu(self.local(x, edge_index)))
        if batch is None:
            x_seq = x.unsqueeze(0)
            h_global, _ = self.attn(x_seq, x_seq, x_seq)
            h_global = h_global.squeeze(0)
        else:
            x_dense, node_mask = to_dense_batch(x, batch)
            key_padding_mask = ~node_mask
            hg, _ = self.attn(x_dense, x_dense, x_dense, key_padding_mask=key_padding_mask)
            h_global = hg[node_mask]
        x = self.norm1(x + h_local + h_global)
        x = self.norm2(x + self.ffn(x))
        return x


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
        gps_heads: int,
        gps_mlp_state_dim: int,
        gps_mlp_post_hidden: int,
        dropout: float,
    ):
        super().__init__()
        mt = str(model_type).strip().lower()
        if mt not in {"gine", "sage", "gcn", "gps", "gps_mlp"}:
            raise ValueError(f"Unsupported model_type={model_type}")
        self.model_type = mt
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.gps_heads = int(gps_heads)
        self.gps_mlp_state_dim = int(gps_mlp_state_dim)
        self.dropout = nn.Dropout(float(dropout))

        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim) if (self.edge_emb_dim > 0 and mt == "gine") else None

        self.gps_global_feats = 3 if mt in {"gps", "gps_mlp"} else 0
        eff_in = in_dim + self.node_emb_dim + self.gps_global_feats
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
        elif mt in {"gps", "gps_mlp"}:
            for _ in range(int(layers)):
                self.convs.append(GPSBlock(hidden=hidden, heads=self.gps_heads, dropout=dropout))
        else:  # gcn
            for _ in range(int(layers)):
                self.convs.append(GCNConv(hidden, hidden))
                self.norms.append(nn.LayerNorm(hidden))

        if mt == "gps_mlp":
            if self.gps_mlp_state_dim <= 0:
                raise ValueError("--gps_mlp_state_dim must be >= 1 when using model=gps_mlp.")
            self.state_head = nn.Linear(hidden, self.gps_mlp_state_dim)
            self.post_mlp = nn.Sequential(
                nn.Linear(self.n_nodes * self.gps_mlp_state_dim, int(gps_mlp_post_hidden)),
                nn.ReLU(),
                nn.Linear(int(gps_mlp_post_hidden), int(gps_mlp_post_hidden)),
                nn.ReLU(),
                nn.Linear(int(gps_mlp_post_hidden), 2 * self.n_nodes),
            )
            self.out_head = None
        else:
            self.out_head = nn.Linear(hidden, 2)  # [V_re, V_im]

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def _append_gps_global_feats(self, x: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if self.gps_global_feats <= 0:
            return x
        if batch is None:
            total_p = x[:, 0].sum().view(1, 1).expand(x.size(0), 1)
            total_q = x[:, 1].sum().view(1, 1).expand(x.size(0), 1)
            mean_f = x.mean().view(1, 1).expand(x.size(0), 1)
            gf = torch.cat([total_p, total_q, mean_f], dim=1)
            return torch.cat([x, gf], dim=1)
        out = torch.zeros(x.size(0), self.gps_global_feats, device=x.device, dtype=x.dtype)
        for bi in batch.unique(sorted=True):
            m = batch == bi
            xb = x[m]
            total_p = xb[:, 0].sum()
            total_q = xb[:, 1].sum()
            mean_f = xb.mean()
            out[m, 0] = total_p
            out[m, 1] = total_q
            out[m, 2] = mean_f
        return torch.cat([x, out], dim=1)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        x = self._append_gps_global_feats(x, batch.batch if hasattr(batch, "batch") else None)
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
        elif self.model_type in {"gps", "gps_mlp"}:
            for blk in self.convs:
                h = blk(h, batch.edge_index, batch.batch if hasattr(batch, "batch") else None)
        else:  # gcn
            z = torch.sqrt(batch.edge_attr[:, 0] ** 2 + batch.edge_attr[:, 1] ** 2).clamp(min=1e-6)
            ew = 1.0 / z
            for conv, norm in zip(self.convs, self.norms):
                h_msg = F.relu(conv(h, batch.edge_index, edge_weight=ew))
                h = h + self.dropout(norm(h_msg))

        if self.model_type == "gps_mlp":
            s = self.state_head(h).view(batch.num_graphs, self.n_nodes, self.gps_mlp_state_dim)
            return self.post_mlp(s.reshape(batch.num_graphs, self.n_nodes * self.gps_mlp_state_dim))

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


def _val_r2_and_worst_mae_from_ri(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> tuple[float, float, float]:
    bsz, two_n = pred_ri.shape
    n_nodes = two_n // 2
    pred = pred_ri.view(bsz, n_nodes, 2)
    true = true_ri.view(bsz, n_nodes, 2)
    pred_mag = torch.sqrt(pred[..., 0] * pred[..., 0] + pred[..., 1] * pred[..., 1] + 1e-12)
    true_mag = torch.sqrt(true[..., 0] * true[..., 0] + true[..., 1] * true[..., 1] + 1e-12)

    mae_per_node = (pred_mag - true_mag).abs().mean(dim=0)
    worst_mae = float(mae_per_node.max().item())

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
    in_dim: int,
    n_nodes: int,
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
    model = GNNOnlyVoltageModel(
        model_type=model_type,
        in_dim=in_dim,
        hidden=int(args.hidden),
        layers=int(args.layers),
        n_nodes=n_nodes,
        num_edges=n_edges,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        gps_heads=int(args.gps_heads),
        gps_mlp_state_dim=int(args.gps_mlp_state_dim),
        gps_mlp_post_hidden=int(args.gps_mlp_post_hidden),
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
    p.add_argument("--node_feature_cols", type=str, default="p_load_kw,q_load_kvar")
    p.add_argument("--chunk_parent", type=str, default="", help="If set, reads all chunk folders and concatenates samples.")
    p.add_argument("--chunk_subdir_glob", type=str, default="run_*")
    p.add_argument("--edge_shared_csv", type=str, default="", help="Optional single shared edge CSV used for all chunks.")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv", help="Accepted for CLI parity; not used by this trainer.")
    p.add_argument("--out_dir", type=str, default="gnn_only_compare_complex_8500")
    p.add_argument("--models", type=str, default="gine,sage,gcn", help="Comma-separated list from {gine,sage,gcn}.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--edge_emb_dim", type=int, default=8)
    p.add_argument("--gps_heads", type=int, default=4, help="Only used when model includes gps.")
    p.add_argument("--gps_mlp_state_dim", type=int, default=4, help="Only used when model includes gps_mlp.")
    p.add_argument("--gps_mlp_post_hidden", type=int, default=1024, help="Only used when model includes gps_mlp.")
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
    p.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Chunk mode only: directory for per-chunk tensor caches (GPS-compatible naming).",
    )
    p.add_argument("--node_pe_csv", type=str, default="")
    p.add_argument("--node_pe_cols", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    node_feature_cols = _parse_node_feature_cols(args.node_feature_cols)

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    allowed = {"gine", "sage", "gcn"}
    if not models or any(m not in allowed for m in models):
        raise ValueError(f"--models must be comma-separated subset of {sorted(allowed)}")

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    chunk_parent = Path(args.chunk_parent).resolve() if str(args.chunk_parent).strip() else None

    # Single-CSV mode only: nodes_csv / edge_catalog_csv live under data_root.
    # Chunk mode uses those names as filenames inside each run_* folder — do not resolve here.
    if chunk_parent is None:
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
    cache_dir = Path(args.cache_dir).resolve() if str(args.cache_dir).strip() else None
    node_to_local: dict[str, int] | None = None
    edge_shared_csv = Path(args.edge_shared_csv).resolve() if str(args.edge_shared_csv).strip() else None
    sample_frac_applied_in_chunk = False
    if chunk_parent is not None:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            x, y_ri, edge_index, edge_attr, sample_ids, node_to_local = _load_chunked_dataset(
                chunk_parent=chunk_parent,
                chunk_subdir_glob=str(args.chunk_subdir_glob),
                nodes_csv_name=str(args.nodes_csv),
                edges_csv_name=str(args.edge_catalog_csv),
                node_feature_cols=node_feature_cols,
                edge_shared_csv=edge_shared_csv,
                cache_dir=cache_dir,
                sample_frac=float(args.sample_frac),
                seed=int(args.seed),
            )
            sample_frac_applied_in_chunk = float(args.sample_frac) < 1.0
        elif cache_path and cache_path.is_file():
            print(f"Loading cache: {cache_path}", flush=True)
            pack = torch.load(cache_path, map_location="cpu", weights_only=False)
            x = pack["x"]
            y_ri = pack["y_ri"]
            edge_index = pack["edge_index"]
            edge_attr = pack["edge_attr"]
            sample_ids = pack["sample_ids"]
            node_to_local = pack["node_to_local"]
        else:
            x, y_ri, edge_index, edge_attr, sample_ids, node_to_local = _load_chunked_dataset(
                chunk_parent=chunk_parent,
                chunk_subdir_glob=str(args.chunk_subdir_glob),
                nodes_csv_name=str(args.nodes_csv),
                edges_csv_name=str(args.edge_catalog_csv),
                node_feature_cols=node_feature_cols,
                edge_shared_csv=edge_shared_csv,
                cache_dir=None,
                sample_frac=1.0,
                seed=int(args.seed),
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "x": x,
                        "y_ri": y_ri,
                        "edge_index": edge_index,
                        "edge_attr": edge_attr,
                        "sample_ids": sample_ids,
                        "node_to_local": node_to_local,
                    },
                    cache_path,
                )
                print(f"Wrote cache: {cache_path}", flush=True)
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
            x, sample_ids, _node_order, node_to_local = _load_nodes_features_target(nodes_path, node_feature_cols)
            edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids},
                    cache_path,
                )
                print(f"Wrote cache: {cache_path}", flush=True)

        if node_to_local is None:
            _x_tmp, _sid_tmp, _node_order_tmp, node_to_local = _load_nodes_features_target(nodes_path, node_feature_cols)
            del _x_tmp, _sid_tmp, _node_order_tmp
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

    if args.sample_frac < 1.0 and not sample_frac_applied_in_chunk:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GraphVoltageDataset(x_n, y_ri, edge_index, edge_attr)
    pin = device.type == "cuda"
    nw = int(args.num_workers)
    dl_tr = DataLoader(
        Subset(ds, idx_train.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    dl_va = DataLoader(
        Subset(ds, idx_val.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    dl_te = DataLoader(
        Subset(ds, idx_test.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    print(f"Device={device} n_nodes={n_nodes} n_edges={int(edge_index.shape[1])} models={models}", flush=True)

    results: dict[str, dict] = {}
    run_label = "chunk_parent" if chunk_parent is not None else "single_csv"
    for m in models:
        print(f"\n=== Training {m.upper()} (GNN-only) ===", flush=True)
        res = _train_one(
            model_type=m,
            args=args,
            in_dim=in_dim,
            n_nodes=n_nodes,
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
        "task": "GNN-only PQ -> complex voltage [V_re,V_im]",
        "dataset_nodes_csv": str(args.nodes_csv),
        "dataset_edges_csv": str(args.edge_catalog_csv),
        "edge_shared_csv": str(args.edge_shared_csv),
        "node_feature_cols": node_feature_cols,
        "chunk_parent": str(chunk_parent) if chunk_parent is not None else "",
        "chunk_subdir_glob": str(args.chunk_subdir_glob),
        "n_samples": int(n),
        "n_nodes": int(n_nodes),
        "node_input_dim": int(in_dim),
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
    report_path = out_dir / "gnn_only_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
