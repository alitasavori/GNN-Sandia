"""
Device-Aware GPS (DA-GPS): Perceiver-style latent tokens + cross-attention + local MPNN,
multitask voltage + cap (BCE) + regulator (MSE on z-scored taps).

Global features per graph (broadcast to every node before the node MLP):
  g = [ sum(P), sum(Q), g3 ] with g3 = z-scored slack |V| (pu) when --substation_node_idx is set,
  else g3 = mean P over the graph (not mean of P and Q together).

Expects full-MV static graph (same topology for all samples) and aux labels in meta_csv.
Device→bus mapping: JSON (see da_gps_device_config.example.json).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch.utils.checkpoint import checkpoint

try:
    from torch_scatter import scatter as _scatter_op
except Exception:  # pragma: no cover
    _scatter_op = None
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from train_gnn_only_compare_complex_voltage import _build_complex_targets
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges, _load_nodes_pq_target


def _scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    reduce: str,
) -> torch.Tensor:
    """Per-graph reduction: index is [0..B-1] per node. Fallback if torch_scatter missing."""
    if _scatter_op is not None:
        return _scatter_op(src, index, dim=0, dim_size=dim_size, reduce=reduce)
    if reduce == "sum":
        out = src.new_zeros(dim_size, *src.shape[1:])
        out.index_add_(0, index, src)
        return out
    if reduce == "mean":
        s = _scatter_reduce(src, index, dim_size, "sum")
        ones = torch.ones(index.size(0), device=src.device, dtype=src.dtype)
        c = _scatter_reduce(ones, index, dim_size, "sum")
        if s.dim() > 1:
            c = c.unsqueeze(-1)
        return s / c.clamp_min(1e-8)
    raise ValueError(f"reduce={reduce} not supported in fallback scatter")


def _batched_global_features(
    x: torch.Tensor,
    batch: torch.Tensor,
    *,
    v_sub_per_graph: torch.Tensor | None,
) -> torch.Tensor:
    """
    Per-node broadcast: [sum P, sum Q, third] for each node's graph.
    Third = v_sub_per_graph (slack |V| pu) when provided; else mean P over that graph.
    """
    B = int(batch.max().item()) + 1
    p = x[:, 0]
    q = x[:, 1]
    sum_p = _scatter_reduce(p, batch, B, "sum")[batch]
    sum_q = _scatter_reduce(q, batch, B, "sum")[batch]
    if v_sub_per_graph is not None:
        g3 = v_sub_per_graph[batch]
    else:
        g3 = _scatter_reduce(p, batch, B, "mean")[batch]
    return torch.stack([sum_p, sum_q, g3], dim=-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm_sid(s: object) -> int:
    try:
        return int(float(s))
    except Exception:
        return int(str(s).strip())


def load_device_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in ("cap_host_nodes", "reg_host_nodes", "cap_target_cols", "reg_target_cols"):
        if k not in data:
            raise ValueError(f"device config {path} missing required key {k!r}")
    caps = data["cap_host_nodes"]
    regs = data["reg_host_nodes"]
    ccols = data["cap_target_cols"]
    rcols = data["reg_target_cols"]
    if len(caps) != len(ccols):
        raise ValueError(f"cap_host_nodes ({len(caps)}) vs cap_target_cols ({len(ccols)})")
    if len(regs) != len(rcols):
        raise ValueError(f"reg_host_nodes ({len(regs)}) vs reg_target_cols ({len(rcols)})")
    return data


def _resolve_node_idx(node_to_local: dict[str, int], name: str) -> int:
    s = str(name).strip()
    if s in node_to_local:
        return int(node_to_local[s])
    low = {k.lower(): v for k, v in node_to_local.items()}
    if s.lower() in low:
        return int(low[s.lower()])
    raise KeyError(f"Node {name!r} not in graph node list (check device JSON).")


def pe_from_pinv(Lp: np.ndarray) -> torch.Tensor:
    """PE_i from L+: row stats of Omega_ij = L+_ii + L+_jj - 2 L+_ij."""
    n = int(Lp.shape[0])
    diag = np.diag(Lp)
    Omega = diag.reshape(-1, 1) + diag.reshape(1, -1) - 2.0 * Lp
    pe = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        row = Omega[i, :].astype(np.float64)
        pe[i, 0] = float(row.min())
        pe[i, 1] = float(row.max())
        pe[i, 2] = float(row.std())
        pe[i, 3] = float(np.median(row))
        pe[i, 4] = float(row.mean())
    return torch.from_numpy(pe)


def build_omega_device_columns(Lp: np.ndarray, host_indices: list[int]) -> np.ndarray:
    """Omega[:, host_j] for each device j. Lp is N×N Moore–Penrose inverse of L."""
    n = Lp.shape[0]
    out = np.zeros((n, len(host_indices)), dtype=np.float32)
    diag = np.diag(Lp)
    for j, h in enumerate(host_indices):
        out[:, j] = diag + Lp[h, h] - 2.0 * Lp[:, h]
    return out


def laplacian_pinv(edge_index: torch.Tensor, edge_attr: torch.Tensor, n_nodes: int, min_x: float = 1e-4) -> np.ndarray:
    ei = edge_index.cpu().numpy()
    ea = edge_attr.cpu().numpy()
    n = int(n_nodes)
    L = np.zeros((n, n), dtype=np.float64)
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        x = float(abs(ea[k, 1]))
        w = 1.0 / max(x, min_x)
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return np.linalg.pinv(L, rcond=1e-9)


def _multihead_cross_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    n_heads: int,
    dropout_p: float,
    training: bool,
    key_padding_mask: torch.Tensor | None,
    attn_bias: torch.Tensor | None,
    query_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """query (B,L,d), key/value (B,S,d). key_padding_mask: True where KEY is pad (ignore)."""
    B, L, d = query.shape
    _, S, _ = key.shape
    dh = d // n_heads
    qh = query.view(B, L, n_heads, dh).transpose(1, 2)
    kh = key.view(B, S, n_heads, dh).transpose(1, 2)
    vh = value.view(B, S, n_heads, dh).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dh)
    if attn_bias is not None:
        scores = scores + attn_bias.unsqueeze(1)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    if query_padding_mask is not None:
        scores = scores.masked_fill(~query_padding_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    attn = F.dropout(attn, dropout_p, training)
    out = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, L, d)
    if query_padding_mask is not None:
        out = out * query_padding_mask.unsqueeze(-1).to(out.dtype)
    return out


class EdgeAttnMPNN(nn.Module):
    """One hop: gated messages with edge (R, X), residual to linear(self)."""

    def __init__(self, d: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * d + edge_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.gate = nn.Linear(d, 1)
        self.self_lin = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        m = self.msg(torch.cat([h[row], h[col], edge_attr], dim=-1))
        alpha = torch.sigmoid(self.gate(m))
        acc = torch.zeros_like(h)
        acc.index_add_(0, col, alpha * m)
        return self.drop(self.self_lin(h) + acc)


class DAGPSBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        edge_dim: int,
        n_dev: int,
        dropout: float,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must divide heads")
        self.hidden = hidden
        self.heads = heads
        self.n_dev = int(n_dev)
        self.dropout_p = float(dropout)
        self.wq_nt = nn.Linear(hidden, hidden)
        self.wk_nt = nn.Linear(hidden, hidden)
        self.wv_nt = nn.Linear(hidden, hidden)
        self.wo_nt = nn.Linear(hidden, hidden)
        self.wq_tn = nn.Linear(hidden, hidden)
        self.wk_tn = nn.Linear(hidden, hidden)
        self.wv_tn = nn.Linear(hidden, hidden)
        self.wo_tn = nn.Linear(hidden, hidden)
        self.norm_t1 = nn.LayerNorm(hidden)
        self.norm_t2 = nn.LayerNorm(hidden)
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.norm_h_mid = nn.LayerNorm(hidden)
        self.mpnn = EdgeAttnMPNN(hidden, edge_dim, dropout)
        self.norm_out = nn.LayerNorm(hidden)
        self.ffn_h = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h_in: torch.Tensor,
        T_in: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        omega_dev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_dense, node_mask = to_dense_batch(h_in, batch)
        B, N, _ = h_dense.shape
        key_pad = ~node_mask

        q = self.wq_nt(T_in)
        k = self.wk_nt(h_dense)
        v = self.wv_nt(h_dense)
        zt = _multihead_cross_attn(
            q, k, v, n_heads=self.heads, dropout_p=self.dropout_p, training=self.training,
            key_padding_mask=key_pad, attn_bias=None, query_padding_mask=None,
        )
        zt = self.wo_nt(zt)
        T_mid = self.norm_t1(T_in + F.dropout(zt, self.dropout_p, self.training))
        T_mid = self.norm_t2(T_mid + self.ffn_t(T_mid))

        attn_bias = None
        if omega_dev is not None and self.n_dev > 0:
            o = omega_dev.to(h_dense.device, dtype=h_dense.dtype)
            bias_ng = -self.beta * o.unsqueeze(0).expand(B, -1, -1)
            Gtok = T_mid.size(1)
            attn_bias = h_dense.new_zeros(B, N, Gtok)
            attn_bias[:, :, : self.n_dev] = bias_ng
            attn_bias = attn_bias.masked_fill(~node_mask.unsqueeze(-1), 0.0)

        q2 = self.wq_tn(h_dense)
        k2 = self.wk_tn(T_mid)
        v2 = self.wv_tn(T_mid)
        zh = _multihead_cross_attn(
            q2, k2, v2, n_heads=self.heads, dropout_p=self.dropout_p, training=self.training,
            key_padding_mask=None, attn_bias=attn_bias, query_padding_mask=node_mask,
        )
        zh = self.wo_tn(zh)
        z_flat = zh[node_mask]

        h_loc = self.mpnn(h_in, edge_index, edge_attr)
        h_mid = self.norm_h_mid(h_in + z_flat + h_loc)
        h_out = self.norm_out(h_mid + self.ffn_h(h_mid))
        return h_out, T_mid


class DAGPSModel(nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        hidden: int,
        heads: int,
        n_layers: int,
        n_cap: int,
        n_reg: int,
        n_system: int,
        pe_static: torch.Tensor,
        cap_host_idx: torch.Tensor,
        reg_host_idx: torch.Tensor,
        omega_dev: torch.Tensor | None,
        edge_dim: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.n_cap = int(n_cap)
        self.n_reg = int(n_reg)
        self.n_system = int(n_system)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.g_tokens = int(n_cap + n_reg + n_system)
        pe_dim = int(pe_static.size(1))
        self.register_buffer("pe_static", pe_static)
        self.node_in = nn.Sequential(
            nn.Linear(2 + pe_dim + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.token_latent = nn.Parameter(torch.randn(self.g_tokens, hidden) * 0.02)
        self.token_pe_proj = nn.Linear(pe_dim, hidden)
        self.register_buffer("cap_host_idx", cap_host_idx.long().clamp(0, pe_static.size(0) - 1))
        self.register_buffer("reg_host_idx", reg_host_idx.long().clamp(0, pe_static.size(0) - 1))
        if omega_dev is not None:
            self.register_buffer("omega_dev", omega_dev)
        else:
            self.omega_dev = None
        self.blocks = nn.ModuleList(
            [
                DAGPSBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=edge_dim,
                    n_dev=n_cap + n_reg,
                    dropout=dropout,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.volt_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.cap_head = nn.Linear(hidden, 1, bias=False)
        self.reg_head = nn.Linear(hidden, 1, bias=False)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = data.x
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        pe0 = self.pe_static.to(x.device, dtype=x.dtype)
        if batch is None:
            pe = pe0
            B = 1
        else:
            B = int(batch.max().item()) + 1
            pe = pe0.unsqueeze(0).expand(B, -1, -1).reshape(-1, pe0.size(-1))
        v_sub_graph: torch.Tensor | None = None
        if hasattr(data, "v_sub") and data.v_sub is not None:
            v_sub_graph = data.v_sub.view(-1)
        if batch is None:
            b_one = x.new_zeros(x.size(0), dtype=torch.long)
            g = _batched_global_features(x, b_one, v_sub_per_graph=v_sub_graph)
        else:
            g = _batched_global_features(x, batch, v_sub_per_graph=v_sub_graph)
        h = self.node_in(torch.cat([x, pe, g], dim=-1))
        T = self.token_latent.unsqueeze(0).expand(B, -1, -1).clone()
        for k in range(self.n_cap):
            T[:, k] = T[:, k] + self.token_pe_proj(pe0[self.cap_host_idx[k]])
        for k in range(self.n_reg):
            T[:, self.n_cap + k] = T[:, self.n_cap + k] + self.token_pe_proj(pe0[self.reg_host_idx[k]])
        bptr = batch if batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        od = self.omega_dev if hasattr(self, "omega_dev") and self.omega_dev is not None else None
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                h, T = checkpoint(
                    blk,
                    h,
                    T,
                    data.edge_index,
                    data.edge_attr,
                    bptr,
                    od,
                    use_reentrant=False,
                )
            else:
                h, T = blk(h, T, data.edge_index, data.edge_attr, bptr, od)

        volt = self.volt_head(h)
        cap_logits = self.cap_head(T[:, : self.n_cap, :]).squeeze(-1)
        reg_pred = self.reg_head(T[:, self.n_cap : self.n_cap + self.n_reg, :]).squeeze(-1)
        return volt, cap_logits, reg_pred


class DAGPSDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_ri: torch.Tensor,
        y_cap: torch.Tensor,
        y_reg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        v_sub: torch.Tensor | None = None,
    ):
        self.x = x
        self.y_ri = y_ri
        self.y_cap = y_cap
        self.y_reg = y_reg
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.v_sub = v_sub

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Data:
        d = Data(
            x=self.x[i],
            y=self.y_ri[i],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y_cap=self.y_cap[i],
            y_reg=self.y_reg[i],
        )
        if self.v_sub is not None:
            d.v_sub = self.v_sub[i].view(1)
        return d


def _load_substation_vmag(
    nodes_csv: Path,
    sample_ids: list[int],
    substation_node_idx: int,
) -> torch.Tensor:
    """|V| pu at the configured node index (e.g. slack); one value per sample_id."""
    import pandas as pd

    sid_to_i = {int(s): j for j, s in enumerate(sample_ids)}
    out = np.full(len(sample_ids), np.nan, dtype=np.float32)
    usecols = ["sample_id", "node_idx", "vmag_pu"]
    for chunk in pd.read_csv(nodes_csv, usecols=usecols, chunksize=500_000):
        row_s = chunk["sample_id"].map(lambda v: sid_to_i.get(int(float(v)), -1)).to_numpy(dtype=np.int64)
        ni = chunk["node_idx"].to_numpy(dtype=np.int64)
        valid = (row_s >= 0) & (ni == int(substation_node_idx))
        if not np.any(valid):
            continue
        out[row_s[valid]] = chunk.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
    if np.isnan(out).any():
        miss = int(np.isnan(out).sum())
        raise ValueError(
            f"substation node_idx={substation_node_idx}: missing vmag for {miss}/{len(sample_ids)} samples; check nodes CSV / index"
        )
    return torch.from_numpy(out)


def _load_meta_aux(
    meta_csv: Path,
    sample_ids: list[int],
    cap_cols: list[str],
    reg_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd

    usecols = ["sample_id", *cap_cols, *reg_cols]
    df = pd.read_csv(meta_csv, usecols=usecols)
    lk = {_norm_sid(k): j for j, k in enumerate(df["sample_id"].tolist())}
    miss = [sid for sid in sample_ids if _norm_sid(sid) not in lk]
    if miss:
        raise KeyError(f"{len(miss)} sample_id values missing from {meta_csv} (showing up to 5): {miss[:5]}")
    order = [lk[_norm_sid(sid)] for sid in sample_ids]
    cap_raw = df[list(cap_cols)].to_numpy(dtype=np.float64)[order]
    reg_raw = df[list(reg_cols)].to_numpy(dtype=np.float64)[order]
    y_cap = (cap_raw > 0.5).astype(np.float32)
    return torch.from_numpy(y_cap), torch.from_numpy(reg_raw.astype(np.float32))


def _metrics_voltage(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> dict[str, float]:
    pred = pred_ri.view(pred_ri.size(0), -1, 2)
    true = true_ri.view(true_ri.size(0), -1, 2)
    pred_re, pred_im = pred[..., 0], pred[..., 1]
    true_re, true_im = true[..., 0], true[..., 1]
    pred_mag = torch.sqrt(pred_re * pred_re + pred_im * pred_im + 1e-12)
    true_mag = torch.sqrt(true_re * true_re + true_im * true_im + 1e-12)
    pred_ang = torch.atan2(pred_im, pred_re)
    true_ang = torch.atan2(true_im, true_re)
    d_ang = pred_ang - true_ang
    d_ang = (d_ang + math.pi) % (2.0 * math.pi) - math.pi
    ang_err_deg = torch.rad2deg(d_ang)
    vmag_err = pred_mag - true_mag
    return {
        "mae_vmag_pu": float(vmag_err.abs().mean().item()),
        "rmse_vmag_pu": float(torch.sqrt((vmag_err * vmag_err).mean()).item()),
        "mae_angle_deg": float(ang_err_deg.abs().mean().item()),
        "rmse_angle_deg": float(torch.sqrt((ang_err_deg * ang_err_deg).mean()).item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    reg_mean: torch.Tensor,
    reg_std: torch.Tensor,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    preds, tgts = [], []
    cap_logits_all, cap_tgt_all = [], []
    reg_pred_all, reg_tgt_all = [], []
    for batch in dl:
        batch = batch.to(device)
        yb = batch.y.view(batch.num_graphs, -1)
        with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
            v_n, c_log, r_p = model(batch)
        preds.append((v_n * y_std.to(device) + y_mean.to(device)).view(batch.num_graphs, -1).cpu())
        tgts.append(yb.cpu())
        cap_logits_all.append(c_log.cpu())
        cap_tgt_all.append(batch.y_cap.cpu())
        reg_pred_all.append(r_p.cpu())
        reg_tgt_all.append(batch.y_reg.cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    met = _metrics_voltage(pred, tgt)
    cap_log = torch.cat(cap_logits_all, dim=0)
    cap_t = torch.cat(cap_tgt_all, dim=0)
    met["cap_bce"] = float(F.binary_cross_entropy_with_logits(cap_log, cap_t).item())
    rp = torch.cat(reg_pred_all, dim=0)
    rt = torch.cat(reg_tgt_all, dim=0)
    met["reg_mse_normalized"] = float(F.mse_loss(rp, rt.to(rp.dtype)).item())
    rp_denorm = rp * reg_std.to(rp.device) + reg_mean.to(rp.device)
    rt_denorm = rt * reg_std.to(rt.device) + reg_mean.to(rt.device)
    met["reg_mse_tap_pu"] = float(F.mse_loss(rp_denorm, rt_denorm.to(rp_denorm.dtype)).item())
    return met


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA-GPS multitask: voltage + cap + reg (full MV).")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg_full_mv")
    p.add_argument("--nodes_csv", type=str, default="gnn_node_features_and_targets_full_mv.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="gnn_edges_phase_static_full_mv.csv")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument("--device_config_json", type=str, required=True, help="Path to device/host/column mapping JSON.")
    p.add_argument("--n_system_tokens", type=int, default=10, help="Unsupervised latent tokens after cap+reg tokens.")
    p.add_argument("--out_dir", type=str, default="da_gps_multitask_full_mv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64, help="Per-step graphs; A100 can usually fit 32–64+ for N~3.8k, d=256.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--disable_dropout", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_cap", type=float, default=0.1)
    p.add_argument("--lambda_reg", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers; 0 only for tiny debug runs.")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--cache_tensor", type=str, default="")
    p.add_argument("--pe_cache", type=str, default="", help="Optional .pt path for cached PE + omega_dev.")
    p.add_argument(
        "--early_stop_on",
        type=str,
        default="total",
        choices=("total", "voltage"),
        help="Validation metric for best checkpoint / patience.",
    )
    p.add_argument(
        "--substation_node_idx",
        type=int,
        default=None,
        help="If set, third global = vmag_pu for this node_idx from nodes CSV (slack / substation |V|). "
        "Else third global = mean P over the graph (not mixed P&Q).",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (default: AMP on when cuda).",
    )
    p.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile on CUDA (default: try compile on PyTorch 2+).",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Checkpoint each GPS block in training to save activation memory (~30%% slower).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    dropout = 0.0 if args.disable_dropout else float(args.dropout)

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv) if Path(args.nodes_csv).is_absolute() else (data_root / args.nodes_csv).resolve()
    edges_path = Path(args.edge_catalog_csv) if Path(args.edge_catalog_csv).is_absolute() else (data_root / args.edge_catalog_csv).resolve()
    meta_path = Path(args.meta_csv) if Path(args.meta_csv).is_absolute() else (data_root / args.meta_csv).resolve()
    dev_cfg_path = Path(args.device_config_json) if Path(args.device_config_json).is_absolute() else (repo / args.device_config_json).resolve()
    if not dev_cfg_path.is_file():
        dev_cfg_path = Path(args.device_config_json).resolve()
    if not dev_cfg_path.is_file():
        raise FileNotFoundError(dev_cfg_path)

    for pth in (nodes_path, edges_path, meta_path):
        if not pth.is_file():
            raise FileNotFoundError(pth)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = load_device_config(dev_cfg_path)
    cap_cols = list(dev["cap_target_cols"])
    reg_cols = list(dev["reg_target_cols"])
    n_cap = len(cap_cols)
    n_reg = len(reg_cols)
    n_sys = int(args.n_system_tokens)
    g_tot = n_cap + n_reg + n_sys

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

    cap_hosts = [_resolve_node_idx(node_to_local, n) for n in dev["cap_host_nodes"]]
    reg_hosts = [_resolve_node_idx(node_to_local, n) for n in dev["reg_host_nodes"]]
    cap_idx_t = torch.tensor(cap_hosts, dtype=torch.long)
    reg_idx_t = torch.tensor(reg_hosts, dtype=torch.long)

    n_nodes = int(x.shape[1])
    pe_path = Path(args.pe_cache).resolve() if args.pe_cache else None
    if pe_path and pe_path.is_file():
        print(f"Loading PE cache: {pe_path}", flush=True)
        pe_pack = torch.load(pe_path, map_location="cpu", weights_only=False)
        pe_static = pe_pack["pe_static"]
        omega_dev = pe_pack["omega_dev"]
        if int(omega_dev.shape[1]) != n_cap + n_reg:
            raise ValueError(
                f"PE cache omega_dev columns {omega_dev.shape[1]} != n_cap+n_reg {n_cap + n_reg}; delete cache or fix JSON"
            )
    else:
        print("Computing effective-resistance PE + omega device columns (dense pinv; one-time)...", flush=True)
        t0 = time.perf_counter()
        Lp = laplacian_pinv(edge_index, edge_attr, n_nodes)
        pe_static = pe_from_pinv(Lp)
        host_all = cap_hosts + reg_hosts
        omega_np = build_omega_device_columns(Lp, host_all)
        omega_dev = torch.from_numpy(omega_np)
        if int(omega_dev.shape[1]) != n_cap + n_reg:
            raise ValueError(
                f"omega_dev columns {omega_dev.shape[1]} != n_cap+n_reg {n_cap + n_reg} (check device JSON hosts)"
            )
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)
        if pe_path:
            pe_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"pe_static": pe_static, "omega_dev": omega_dev}, pe_path)
            print(f"Wrote PE cache: {pe_path}", flush=True)

    y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)
    y_cap, y_reg = _load_meta_aux(meta_path, sample_ids, cap_cols, reg_cols)

    if args.sample_frac < 1.0:
        k = max(1, int(round(len(sample_ids) * args.sample_frac)))
        x = x[:k]
        y_ri = y_ri[:k]
        y_cap = y_cap[:k]
        y_reg = y_reg[:k]
        sample_ids = sample_ids[:k]
        print(f"sample_frac={args.sample_frac} => {k} samples", flush=True)

    n = int(x.shape[0])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Invalid train/val/test split.")
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    xt = x[idx_train].reshape(-1, 2)
    x_mean = xt.mean(dim=0, keepdim=True)
    x_std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_n = (x - x_mean) / x_std

    y_train = y_ri[idx_train].reshape(len(idx_train), -1)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    reg_mean = y_reg[idx_train].mean(dim=0, keepdim=True)
    reg_std = y_reg[idx_train].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    y_reg_n = (y_reg - reg_mean) / reg_std

    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    torch.save(y_mean, out_dir / "y_mean.pt")
    torch.save(y_std, out_dir / "y_std.pt")
    torch.save(reg_mean, out_dir / "reg_mean.pt")
    torch.save(reg_std, out_dir / "reg_std.pt")

    v_sub_n: torch.Tensor | None = None
    if args.substation_node_idx is not None:
        v_raw = _load_substation_vmag(nodes_path, sample_ids, int(args.substation_node_idx))
        v_m = v_raw[idx_train].mean()
        v_s = v_raw[idx_train].std(unbiased=False).clamp_min(1e-6)
        v_sub_n = (v_raw - v_m) / v_s
        torch.save(v_m, out_dir / "v_sub_mean.pt")
        torch.save(v_s, out_dir / "v_sub_std.pt")
        print(
            f"Third global: z-scored vmag_pu at node_idx={int(args.substation_node_idx)} (train mean/std).",
            flush=True,
        )
    else:
        print("Third global: mean P per graph (not V_sub; pass --substation_node_idx to use slack |V|).", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DAGPSDataset(x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, v_sub=v_sub_n)
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

    base_model = DAGPSModel(
        n_nodes=n_nodes,
        hidden=int(args.hidden),
        heads=int(args.heads),
        n_layers=int(args.layers),
        n_cap=n_cap,
        n_reg=n_reg,
        n_system=n_sys,
        pe_static=pe_static,
        cap_host_idx=cap_idx_t,
        reg_host_idx=reg_idx_t,
        omega_dev=omega_dev,
        edge_dim=int(edge_attr.size(1)),
        dropout=dropout,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    ).to(device)
    model = base_model
    if device.type == "cuda" and not args.no_compile:
        try:
            model = torch.compile(base_model)  # type: ignore[assignment]
            print("torch.compile: enabled", flush=True)
        except Exception as ex:  # pragma: no cover
            print(f"torch.compile: skipped ({ex})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    y_mean_d = y_mean.to(device)
    y_std_d = y_std.to(device)
    reg_mean_d = reg_mean.to(device)
    reg_std_d = reg_std.to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        from torch.cuda.amp import GradScaler as _GradScaler

        scaler = _GradScaler()
        print("AMP (autocast + GradScaler): enabled", flush=True)
    else:
        scaler = None
    if args.gradient_checkpointing:
        print("gradient_checkpointing: per-block recompute (training only)", flush=True)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            yb = batch.y.view(batch.num_graphs, -1)
            yb_n = (yb - y_mean_d) / y_std_d
            opt.zero_grad(set_to_none=True)
            with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                v_n, c_log, r_p = model(batch)
                loss_v = mse(v_n.view_as(yb_n), yb_n)
                loss_c = bce(c_log, batch.y_cap.to(device))
                loss_r = mse(r_p, batch.y_reg.to(device))
                loss = loss_v + float(args.lambda_cap) * loss_c + float(args.lambda_reg) * loss_r
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        val_tot = val_v = 0.0
        nv = 0
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                yb = batch.y.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean_d) / y_std_d
                with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                    v_n, c_log, r_p = model(batch)
                    lv = mse(v_n.view_as(yb_n), yb_n)
                    lc = bce(c_log, batch.y_cap.to(device))
                    lr_ = mse(r_p, batch.y_reg.to(device))
                    lt = lv + float(args.lambda_cap) * lc + float(args.lambda_reg) * lr_
                val_tot += float(lt.item()) * batch.num_graphs
                val_v += float(lv.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
        val_tot /= max(nv, 1)
        val_v /= max(nv, 1)
        sch.step(val_tot)
        crit = val_tot if args.early_stop_on == "total" else val_v
        if crit < best_val:
            best_val = crit
            best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % max(1, int(args.log_every)) == 0:
            print(
                f"[da_gps] epoch {ep:4d}/{args.epochs} val_tot={val_tot:.6f} val_volt={val_v:.6f} best={best_val:.6f}",
                flush=True,
            )
        if bad >= args.patience:
            print(f"[da_gps] early stop at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        base_model.load_state_dict(best_state)

    met = evaluate(model, dl_te, device, y_mean_d, y_std_d, reg_mean_d, reg_std_d, use_amp=use_amp)
    ckpt = out_dir / "da_gps_multitask_best.pt"
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "n_nodes": n_nodes,
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "heads": int(args.heads),
            "n_cap": n_cap,
            "n_reg": n_reg,
            "n_system_tokens": n_sys,
            "device_config": str(dev_cfg_path.resolve()),
            "cap_target_cols": cap_cols,
            "reg_target_cols": reg_cols,
        },
        ckpt,
    )
    report = {
        "task": "DA-GPS multitask full MV",
        "nodes_csv": str(nodes_path),
        "edges_csv": str(edges_path),
        "meta_csv": str(meta_path),
        "device_config": str(dev_cfg_path.resolve()),
        "n_samples": n,
        "n_nodes": n_nodes,
        "g_tokens": g_tot,
        "split": {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
        "hyperparameters": vars(args),
        "test_metrics": met,
        "train_seconds": train_seconds,
        "checkpoint": str(ckpt.resolve()),
    }
    (out_dir / "da_gps_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Test |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"cap_BCE={met['cap_bce']:.6f}  reg_MSE(pu)={met['reg_mse_tap_pu']:.6f}  time={train_seconds:.1f}s",
        flush=True,
    )
    print(f"Saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
