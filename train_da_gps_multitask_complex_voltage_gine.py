"""
DA-GPS v2 (GINE local message passing): Perceiver-style latent tokens + cross-attention
+ standard GINEConv on the graph (replaces EdgeAttnMPNN),
multitask voltage + cap (BCE) + regulator (MSE on z-scored taps).

v2 alignment:
- Node inputs: dynamic columns from nodes CSV (default load P/Q) plus optional shared PE columns from a single master CSV.
- Tokens are pure learnable parameters (no host-node warm start).
- No effective-resistance attention bias.
- Aux targets are hardcoded in-script (old aux-trainer style).
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import gc
import hashlib
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

from train_gnn_only_compare_complex_voltage import _build_complex_targets


def _to_dense_batch_mv(
    x: torch.Tensor,
    batch: torch.Tensor,
    *,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack batched nodes into ``(B, N, *)`` without ``batch.max()`` (PyG ``to_dense_batch``).

    DA-GPS MV batches use a fixed ``n_nodes`` per graph in standard PyG order; this avoids
    Dynamo graph breaks from ``Tensor.item()`` inside ``to_dense_batch``.
    Falls back to ``to_dense_batch`` if the total node count is not ``B * n_nodes``.
    """
    n = int(n_nodes)
    ntot = int(x.size(0))
    if n > 0 and ntot % n == 0:
        bsz = ntot // n
        dense = x.view(bsz, n, -1)
        mask = torch.ones(bsz, n, dtype=torch.bool, device=x.device)
        return dense, mask
    return to_dense_batch(x, batch)
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges

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


def _parse_csv_cols(spec: str) -> list[str]:
    cols = [c.strip() for c in str(spec).split(",") if c.strip()]
    if not cols:
        raise ValueError("node feature column list is empty.")
    return cols


def _resolve_pe_cols(pe_df_cols: list[str], pe_spec: str) -> list[str]:
    spec = str(pe_spec).strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "auto":
        cols = [c for c in pe_df_cols if str(c).lower().startswith("pe_")]
        return sorted(cols)
    return _parse_csv_cols(pe_spec)


def _load_nodes_features_complex_targets(
    nodes_csv: Path,
    *,
    node_feature_cols: list[str],
    node_pe_csv: Path | None,
    node_pe_cols: str,
    selected_sample_ids: list[int] | None = None,
    csv_chunksize: int = 500_000,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], dict[str, int]]:
    import pandas as pd

    req = ["sample_id", "node", "node_idx", "vmag_pu", "vang_deg", *node_feature_cols]
    print(f"Loading nodes: {nodes_csv}", flush=True)
    if selected_sample_ids is not None:
        sample_ids = [int(_norm_sid(s)) for s in selected_sample_ids]
    else:
        sid_set_all: set[int] = set()
        for ch in pd.read_csv(nodes_csv, usecols=["sample_id"], chunksize=int(csv_chunksize)):
            sid_set_all.update(int(_norm_sid(s)) for s in ch["sample_id"].tolist())
        sample_ids = sorted(sid_set_all)
    if not sample_ids:
        raise RuntimeError(f"No sample IDs found for {nodes_csv}")
    selected_set = set(sample_ids)

    sid0 = int(sample_ids[0])
    first_rows = []
    for ch in pd.read_csv(nodes_csv, usecols=["sample_id", "node", "node_idx"], chunksize=int(csv_chunksize)):
        sid_col = ch["sample_id"].map(_norm_sid)
        sub = ch.loc[sid_col == sid0, ["node", "node_idx"]]
        if len(sub):
            first_rows.append(sub)
    if not first_rows:
        raise RuntimeError(f"sample_id={sid0} not found in {nodes_csv}")
    first = pd.concat(first_rows, ignore_index=True).sort_values("node_idx")
    node_order = first["node"].astype(str).str.strip().str.lower().tolist()
    node_to_local = {n: i for i, n in enumerate(node_order)}
    n_nodes = len(node_order)

    pe_cols: list[str] = []
    pe_mat = None
    if node_pe_csv is not None:
        if not node_pe_csv.is_file():
            raise FileNotFoundError(node_pe_csv)
        pe_df = pd.read_csv(node_pe_csv)
        if "node" not in pe_df.columns:
            raise ValueError(f"{node_pe_csv} must contain a 'node' column.")
        pe_df["node"] = pe_df["node"].astype(str).str.strip().str.lower()
        pe_cols = _resolve_pe_cols(list(pe_df.columns), node_pe_cols)
        if pe_cols:
            miss = [c for c in pe_cols if c not in pe_df.columns]
            if miss:
                raise ValueError(f"{node_pe_csv} missing PE columns: {miss}")
            pe_map = pe_df.set_index("node")[pe_cols]
            # In this dataset, PE is often computed for a filtered graph-node set,
            # while node samples may include extra source/substation nodes.
            # Keep PE where available; zero-fill missing nodes.
            pe_aligned = pe_map.reindex(node_order)
            miss_nodes = pe_aligned.index[pe_aligned.isna().any(axis=1)].tolist()
            if miss_nodes:
                print(
                    f"WARNING: {node_pe_csv} missing PE for {len(miss_nodes)} nodes "
                    f"(showing up to 5): {miss_nodes[:5]} -- filling zeros.",
                    flush=True,
                )
                pe_aligned = pe_aligned.fillna(0.0)
            pe_mat = pe_aligned.to_numpy(dtype=np.float32)
            print(f"Using PE from {node_pe_csv} with columns: {pe_cols}", flush=True)

    d_dyn = len(node_feature_cols)
    d_pe = 0 if pe_mat is None else int(pe_mat.shape[1])
    x_np = np.zeros((len(sample_ids), n_nodes, d_dyn + d_pe), dtype=np.float32)
    y_ri_np = np.zeros((len(sample_ids), n_nodes, 2), dtype=np.float32)
    if pe_mat is not None:
        x_np[:, :, d_dyn:] = pe_mat[None, :, :]

    sid_to_i = {int(s): i for i, s in enumerate(sample_ids)}
    fill_counts = np.zeros((len(sample_ids),), dtype=np.int64)
    for ch in pd.read_csv(nodes_csv, usecols=req, chunksize=int(csv_chunksize)):
        sid_arr = ch["sample_id"].map(_norm_sid).to_numpy(dtype=np.int64)
        node_arr = ch["node"].astype(str).str.strip().str.lower().map(node_to_local).fillna(-1).to_numpy(dtype=np.int64)
        valid = np.array([(int(s) in selected_set) for s in sid_arr], dtype=bool) & (node_arr >= 0)
        if not np.any(valid):
            continue
        s_local = np.array([sid_to_i[int(s)] for s in sid_arr[valid]], dtype=np.int64)
        n_local = node_arr[valid]
        for j, c in enumerate(node_feature_cols):
            x_np[s_local, n_local, j] = ch.loc[valid, c].to_numpy(dtype=np.float32)
        vmag = ch.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
        vang_rad = np.deg2rad(ch.loc[valid, "vang_deg"].to_numpy(dtype=np.float32))
        y_ri_np[s_local, n_local, 0] = vmag * np.cos(vang_rad)
        y_ri_np[s_local, n_local, 1] = vmag * np.sin(vang_rad)
        np.add.at(fill_counts, s_local, 1)
    bad = np.where(fill_counts != n_nodes)[0]
    if len(bad):
        sid_bad = [sample_ids[int(i)] for i in bad[:5]]
        raise RuntimeError(f"Incomplete node rows for {len(bad)} samples in {nodes_csv}; sample_ids like {sid_bad}")

    return torch.from_numpy(x_np), torch.from_numpy(y_ri_np), sample_ids, node_order, node_to_local


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
    attn_mask = None
    if attn_bias is not None:
        # Accept (B, L, S) bias and broadcast across heads.
        attn_mask = attn_bias.unsqueeze(1)
    if key_padding_mask is not None:
        # Build additive mask where padded keys are -inf.
        kp_mask = torch.zeros((B, 1, L, S), device=query.device, dtype=query.dtype)
        kp_mask = kp_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_mask = kp_mask if attn_mask is None else (attn_mask + kp_mask)
    if query_padding_mask is not None:
        # Fully mask invalid query positions.
        q_mask = torch.zeros((B, 1, L, S), device=query.device, dtype=query.dtype)
        q_mask = q_mask.masked_fill(~query_padding_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
        attn_mask = q_mask if attn_mask is None else (attn_mask + q_mask)

    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
    )
    out = out.transpose(1, 2).contiguous().view(B, L, d)
    if query_padding_mask is not None:
        out = out * query_padding_mask.unsqueeze(-1).to(out.dtype)
    return out


def _attn_probs_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    n_heads: int,
    key_padding_mask: torch.Tensor | None,
    query_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Softmax attention weights (no dropout). Same masking as ``_multihead_cross_attn``.
    Returns (B, n_heads, L, S) where L = query length, S = key length."""
    B, L, d = query.shape
    _, S, _ = key.shape
    dh = d // n_heads
    qh = query.view(B, L, n_heads, dh).transpose(1, 2)
    kh = key.view(B, S, n_heads, dh).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(float(dh))
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    if query_padding_mask is not None:
        scores = scores.masked_fill(~query_padding_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
    return torch.softmax(scores, dim=-1)


class GINELayer(nn.Module):
    """Standard GINE message passing layer.

    GINEConv computes internally:
        out_i = MLP( (1+eps)*h_i  +  sum_{j in N(i)} ReLU(h_j + W_e * e_ij) )

    We subtract h back before returning so the output is a pure message term —
    the residual h_in is added once externally in DAGPSBlock.forward (step 1.4),
    keeping the same pattern as the original EdgeAttnMPNN.
    """

    def __init__(self, d: int, edge_dim: int, dropout: float):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(h, edge_index, edge_attr)  # includes (1+eps)*h internally
        return self.drop(out - h)  # return messages only, strip self-loop


class DAGPSBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        edge_dim: int,
        dropout: float,
        n_nodes: int,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must divide heads")
        self.n_nodes = int(n_nodes)
        self.hidden = hidden
        self.heads = heads
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
        self.mpnn = GINELayer(hidden, edge_dim, dropout)
        self.norm_out = nn.LayerNorm(hidden)
        self.ffn_h = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_in: torch.Tensor,
        T_in: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        h_dense: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_dense is None or node_mask is None:
            h_dense, node_mask = _to_dense_batch_mv(h_in, batch, n_nodes=self.n_nodes)

        has_padding = int(h_in.size(0)) != int(h_dense.size(0) * h_dense.size(1))
        key_pad = (~node_mask) if has_padding else None

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

        q2 = self.wq_tn(h_dense)
        k2 = self.wk_tn(T_mid)
        v2 = self.wv_tn(T_mid)
        zh = _multihead_cross_attn(
            q2, k2, v2, n_heads=self.heads, dropout_p=self.dropout_p, training=self.training,
            key_padding_mask=None, attn_bias=attn_bias, query_padding_mask=node_mask if has_padding else None,
        )
        zh = self.wo_tn(zh)
        z_flat = zh[node_mask] if has_padding else zh.reshape(-1, zh.size(-1))

        h_loc = self.mpnn(h_in, edge_index, edge_attr)
        h_mid = self.norm_h_mid(h_in + z_flat + h_loc)
        h_out = self.norm_out(h_mid + self.ffn_h(h_mid))
        return h_out, T_mid

    def token_to_node_attention_probs(
        self,
        T_in: torch.Tensor,
        h_dense: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """First cross-attn: queries = tokens, keys = nodes.
        Returns (B, heads, n_tokens, n_nodes) — distribution over nodes per token."""
        q = self.wq_nt(T_in)
        k = self.wk_nt(h_dense)
        return _attn_probs_qk(
            q, k, n_heads=self.heads, key_padding_mask=key_padding_mask, query_padding_mask=None
        )

    def node_to_token_attention_probs(
        self,
        h_dense: torch.Tensor,
        T_mid: torch.Tensor,
        node_mask: torch.Tensor | None,
        has_padding: bool,
    ) -> torch.Tensor:
        """Second cross-attn in the block: queries = nodes, keys = tokens.
        Returns (B, heads, n_nodes, n_tokens) — distribution over tokens per node."""
        q2 = self.wq_tn(h_dense)
        k2 = self.wk_tn(T_mid)
        qpm = node_mask if has_padding else None
        return _attn_probs_qk(
            q2, k2, n_heads=self.heads, key_padding_mask=None, query_padding_mask=qpm
        )


class DAGPSModel(nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        num_edges: int,
        hidden: int,
        heads: int,
        n_layers: int,
        n_cap: int,
        n_reg: int,
        n_system: int,
        node_in_dim: int,
        node_emb_dim: int,
        edge_emb_dim: int,
        edge_dim: int,
        dropout: float,
        gradient_checkpointing: bool = False,
        per_node_heads: bool = False,
        per_device_cap_head: bool = False,
        per_device_reg_head: bool = False,
        n_pv_aux: int = 0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.n_cap = int(n_cap)
        self.n_reg = int(n_reg)
        self.n_system = int(n_system)
        self.n_pv_aux = int(n_pv_aux)
        if self.n_pv_aux > 0 and self.n_pv_aux > self.n_system:
            raise ValueError(f"n_pv_aux={self.n_pv_aux} exceeds n_system={self.n_system}")
        self.node_in_dim = int(node_in_dim)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.per_node_heads = bool(per_node_heads)
        self.per_device_cap_head = bool(per_device_cap_head)
        self.per_device_reg_head = bool(per_device_reg_head)
        self.g_tokens = int(n_cap + n_reg + n_system)
        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim) if self.edge_emb_dim > 0 else None
        self.node_in = nn.Sequential(
            nn.Linear(self.node_in_dim + self.node_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        eff_edge_dim = int(edge_dim + self.edge_emb_dim)
        self.token_latent = nn.Parameter(torch.randn(self.g_tokens, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [
                DAGPSBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=eff_edge_dim,
                    dropout=dropout,
                    n_nodes=int(n_nodes),
                )
                for _ in range(int(n_layers))
            ]
        )
        if self.per_node_heads:
            self.volt_W = nn.Parameter(torch.randn(self.n_nodes, self.hidden, 2) * 0.02)
            self.volt_b = nn.Parameter(torch.zeros(self.n_nodes, 2))
            self.volt_head = None
        else:
            self.volt_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
            self.volt_W = None
            self.volt_b = None
        if self.per_device_cap_head:
            self.cap_W = nn.Parameter(torch.randn(self.n_cap, self.hidden) * 0.02)
            self.cap_b = nn.Parameter(torch.zeros(self.n_cap))
            self.cap_head = None
        else:
            self.cap_head = nn.Linear(hidden, 1, bias=False)
            self.cap_W = None
            self.cap_b = None

        if self.per_device_reg_head:
            self.reg_W = nn.Parameter(torch.randn(self.n_reg, self.hidden) * 0.02)
            self.reg_b = nn.Parameter(torch.zeros(self.n_reg))
            self.reg_head = None
        else:
            self.reg_head = nn.Linear(hidden, 1, bias=False)
            self.reg_W = None
            self.reg_b = None

        if self.n_pv_aux > 0:
            self.pv_W = nn.Parameter(torch.randn(self.n_pv_aux, self.hidden) * 0.02)
            self.pv_b = nn.Parameter(torch.zeros(self.n_pv_aux))
        else:
            self.pv_W = None
            self.pv_b = None

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = data.x
        ea = data.edge_attr
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        if self.edge_emb is not None:
            ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        B = int(data.num_graphs) if hasattr(data, "num_graphs") and data.num_graphs is not None else 1
        h = self.node_in(x)
        T = self.token_latent.unsqueeze(0).repeat(B, 1, 1)
        bptr = batch if batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_dense, node_mask = _to_dense_batch_mv(h, bptr, n_nodes=self.n_nodes)
        can_view_dense = int(h.size(0)) == int(h_dense.size(0) * h_dense.size(1))
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                h, T = checkpoint(
                    blk,
                    h,
                    T,
                    data.edge_index,
                    ea,
                    bptr,
                    h_dense,
                    node_mask,
                    use_reentrant=False,
                )
            else:
                h, T = blk(h, T, data.edge_index, ea, bptr, h_dense, node_mask)
            if can_view_dense:
                h_dense = h.view(h_dense.size(0), h_dense.size(1), h.size(-1))
            else:
                h_dense = torch.zeros_like(h_dense)
                h_dense[node_mask] = h

        if self.per_node_heads:
            h_per = h.view(B, self.n_nodes, self.hidden)
            volt = torch.einsum("bnd,ndo->bno", h_per, self.volt_W) + self.volt_b
            volt = volt.reshape(B * self.n_nodes, 2)
        else:
            volt = self.volt_head(h)
        T_cap = T[:, : self.n_cap, :]
        if self.per_device_cap_head:
            cap_logits = (T_cap * self.cap_W.unsqueeze(0)).sum(-1) + self.cap_b.unsqueeze(0)
        else:
            cap_logits = self.cap_head(T_cap).squeeze(-1)

        T_reg = T[:, self.n_cap : self.n_cap + self.n_reg, :]
        if self.per_device_reg_head:
            reg_pred = (T_reg * self.reg_W.unsqueeze(0)).sum(-1) + self.reg_b.unsqueeze(0)
        else:
            reg_pred = self.reg_head(T_reg).squeeze(-1)
        if self.n_pv_aux > 0 and self.pv_W is not None:
            T_pv = T[:, self.n_cap + self.n_reg : self.n_cap + self.n_reg + self.n_pv_aux, :]
            pv_pred = (T_pv * self.pv_W.unsqueeze(0)).sum(-1) + self.pv_b.unsqueeze(0)
        else:
            pv_pred = reg_pred.new_zeros((reg_pred.size(0), 0))
        return volt, cap_logits, reg_pred, pv_pred

    @torch.no_grad()
    def forward_node_to_token_attention(
        self, data: Data
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward, collecting both cross-attention softmax weights per GPS block.

        **First cross-attn (token → node):** each global token attends over nodes.
        For each layer ``l``: ``(B, heads, n_tokens, n_nodes)``.

        **Second cross-attn (node → token):** each node attends over tokens.
        For each layer ``l``: ``(B, heads, n_nodes, n_tokens)``; token index
        ``n_cap + j`` is regulator ``j`` (``reg_target_cols[j]``).

        Returns:
            layer_probs_nt: node→token, list length ``n_layers``
            layer_probs_tn: token→node, list length ``n_layers``
            volt, cap_logits, reg_pred, pv_pred: same as ``forward`` (after all blocks).
        """
        self.eval()
        x = data.x
        ea = data.edge_attr
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        if self.edge_emb is not None:
            ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        B = int(data.num_graphs) if hasattr(data, "num_graphs") and data.num_graphs is not None else 1
        h = self.node_in(x)
        T = self.token_latent.unsqueeze(0).repeat(B, 1, 1)
        bptr = batch if batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_dense, node_mask = _to_dense_batch_mv(h, bptr, n_nodes=self.n_nodes)
        can_view_dense = int(h.size(0)) == int(h_dense.size(0) * h_dense.size(1))
        has_padding = int(h.size(0)) != int(h_dense.size(0) * h_dense.size(1))
        key_pad = (~node_mask) if has_padding else None
        layer_probs_nt: list[torch.Tensor] = []
        layer_probs_tn: list[torch.Tensor] = []

        for blk in self.blocks:
            probs_tn = blk.token_to_node_attention_probs(T, h_dense, key_pad)
            layer_probs_tn.append(probs_tn.cpu())

            q = blk.wq_nt(T)
            k = blk.wk_nt(h_dense)
            v = blk.wv_nt(h_dense)
            zt = _multihead_cross_attn(
                q,
                k,
                v,
                n_heads=blk.heads,
                dropout_p=0.0,
                training=False,
                key_padding_mask=key_pad,
                attn_bias=None,
                query_padding_mask=None,
            )
            zt = blk.wo_nt(zt)
            T_mid = blk.norm_t1(T + zt)
            T_mid = blk.norm_t2(T_mid + blk.ffn_t(T_mid))

            probs_nt = blk.node_to_token_attention_probs(h_dense, T_mid, node_mask, has_padding)
            layer_probs_nt.append(probs_nt.cpu())

            q2 = blk.wq_tn(h_dense)
            k2 = blk.wk_tn(T_mid)
            v2 = blk.wv_tn(T_mid)
            zh = _multihead_cross_attn(
                q2,
                k2,
                v2,
                n_heads=blk.heads,
                dropout_p=0.0,
                training=False,
                key_padding_mask=None,
                attn_bias=None,
                query_padding_mask=node_mask if has_padding else None,
            )
            zh = blk.wo_tn(zh)
            z_flat = zh[node_mask] if has_padding else zh.reshape(-1, zh.size(-1))

            h_loc = blk.mpnn(h, data.edge_index, ea)
            h_mid = blk.norm_h_mid(h + z_flat + h_loc)
            h = blk.norm_out(h_mid + blk.ffn_h(h_mid))
            T = T_mid

            if can_view_dense:
                h_dense = h.view(h_dense.size(0), h_dense.size(1), h.size(-1))
            else:
                h_dense = torch.zeros_like(h_dense)
                h_dense[node_mask] = h

        if self.per_node_heads:
            h_per = h.view(B, self.n_nodes, self.hidden)
            volt = torch.einsum("bnd,ndo->bno", h_per, self.volt_W) + self.volt_b
            volt = volt.reshape(B * self.n_nodes, 2)
        else:
            volt = self.volt_head(h)
        T_cap = T[:, : self.n_cap, :]
        if self.per_device_cap_head:
            cap_logits = (T_cap * self.cap_W.unsqueeze(0)).sum(-1) + self.cap_b.unsqueeze(0)
        else:
            cap_logits = self.cap_head(T_cap).squeeze(-1)
        T_reg = T[:, self.n_cap : self.n_cap + self.n_reg, :]
        if self.per_device_reg_head:
            reg_pred = (T_reg * self.reg_W.unsqueeze(0)).sum(-1) + self.reg_b.unsqueeze(0)
        else:
            reg_pred = self.reg_head(T_reg).squeeze(-1)
        if self.n_pv_aux > 0 and self.pv_W is not None:
            T_pv = T[:, self.n_cap + self.n_reg : self.n_cap + self.n_reg + self.n_pv_aux, :]
            pv_pred = (T_pv * self.pv_W.unsqueeze(0)).sum(-1) + self.pv_b.unsqueeze(0)
        else:
            pv_pred = reg_pred.new_zeros((reg_pred.size(0), 0))
        return layer_probs_nt, layer_probs_tn, volt, cap_logits, reg_pred, pv_pred


class DAGPSDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_ri: torch.Tensor,
        y_cap: torch.Tensor,
        y_reg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y_pv: torch.Tensor | None = None,
    ):
        self.x = x
        self.y_ri = y_ri
        self.y_cap = y_cap
        self.y_reg = y_reg
        self.y_pv = y_pv
        self.edge_index = edge_index
        self.edge_attr = edge_attr

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
        if self.y_pv is not None:
            d.y_pv = self.y_pv[i]
        return d


def _load_meta_aux(
    meta_csv: Path,
    sample_ids: list[int],
    cap_cols: list[str],
    reg_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd

    usecols = ["sample_id", *cap_cols, *reg_cols]
    df = pd.read_csv(meta_csv)
    ren = {}
    for c in df.columns:
        cs = str(c)
        if cs.startswith("cap_") or cs.startswith("reg_"):
            cl = cs.lower()
            if cl != cs:
                ren[c] = cl
    if ren:
        df = df.rename(columns=ren)
    df = df[usecols]
    lk = {_norm_sid(k): j for j, k in enumerate(df["sample_id"].tolist())}
    miss = [sid for sid in sample_ids if _norm_sid(sid) not in lk]
    if miss:
        raise KeyError(f"{len(miss)} sample_id values missing from {meta_csv} (showing up to 5): {miss[:5]}")
    order = [lk[_norm_sid(sid)] for sid in sample_ids]
    cap_raw = df[list(cap_cols)].to_numpy(dtype=np.float64)[order]
    reg_raw = df[list(reg_cols)].to_numpy(dtype=np.float64)[order]
    y_cap = (cap_raw > 0.5).astype(np.float32)
    return torch.from_numpy(y_cap), torch.from_numpy(reg_raw.astype(np.float32))


def _load_meta_pv(meta_csv: Path, sample_ids: list[int], pv_cols: list[str]) -> torch.Tensor:
    """Numeric columns from ``gnn_sample_meta`` (float targets), rows aligned to ``sample_ids`` order."""
    import pandas as pd

    if not pv_cols:
        raise ValueError("pv_cols must be non-empty")
    df = pd.read_csv(meta_csv)
    lower_to_orig = {str(c).lower(): c for c in df.columns}
    if "sample_id" not in lower_to_orig:
        raise KeyError(f"sample_id missing in {meta_csv}")
    sid_col = lower_to_orig["sample_id"]
    use_orig: list[str] = []
    for c in pv_cols:
        cl = str(c).lower()
        if cl not in lower_to_orig:
            raise KeyError(f"Column {c!r} not in {meta_csv} (available include: {sorted(lower_to_orig.keys())[:30]}...)")
        use_orig.append(lower_to_orig[cl])
    df = df[[sid_col, *use_orig]]
    lk = {_norm_sid(k): j for j, k in enumerate(df[sid_col].tolist())}
    miss = [sid for sid in sample_ids if _norm_sid(sid) not in lk]
    if miss:
        raise KeyError(f"{len(miss)} sample_id values missing from {meta_csv} for PV aux (showing up to 5): {miss[:5]}")
    order = [lk[_norm_sid(sid)] for sid in sample_ids]
    raw = df[use_orig].to_numpy(dtype=np.float64)[order]
    return torch.from_numpy(raw.astype(np.float32))


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
    var_true = ((true_mag - true_mag.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
    mse_node = ((pred_mag - true_mag) ** 2).mean(dim=0)
    r2_per_node = 1.0 - mse_node / var_true.clamp_min(1e-8)
    worst_node_mae = (pred_mag - true_mag).abs().max(dim=1).values.mean()
    return {
        "mae_vmag_pu": float(vmag_err.abs().mean().item()),
        "rmse_vmag_pu": float(torch.sqrt((vmag_err * vmag_err).mean()).item()),
        "mae_angle_deg": float(ang_err_deg.abs().mean().item()),
        "rmse_angle_deg": float(torch.sqrt((ang_err_deg * ang_err_deg).mean()).item()),
        "r2_vmag_mean": float(r2_per_node.mean().item()),
        "r2_vmag_min": float(r2_per_node.min().item()),
        "mae_vmag_worst_node": float(worst_node_mae.item()),
    }


def _cast_batch_float_tensors(batch: Data) -> Data:
    # Defensive cast: keep graph index tensors as-is, force numeric tensors to float32.
    if hasattr(batch, "x") and batch.x is not None:
        batch.x = batch.x.float()
    if hasattr(batch, "y") and batch.y is not None:
        batch.y = batch.y.float()
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr.float()
    if hasattr(batch, "y_cap") and batch.y_cap is not None:
        batch.y_cap = batch.y_cap.float()
    if hasattr(batch, "y_reg") and batch.y_reg is not None:
        batch.y_reg = batch.y_reg.float()
    if hasattr(batch, "y_pv") and batch.y_pv is not None:
        batch.y_pv = batch.y_pv.float()
    return batch


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
    *,
    pv_mean: torch.Tensor | None = None,
    pv_std: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    preds, tgts = [], []
    cap_logits_all, cap_tgt_all = [], []
    reg_pred_all, reg_tgt_all = [], []
    pv_pred_all, pv_tgt_all = [], []
    use_pv = pv_mean is not None and pv_std is not None
    for batch in dl:
        batch = batch.to(device)
        batch = _cast_batch_float_tensors(batch)
        yb = batch.y.view(batch.num_graphs, -1)
        y_cap = batch.y_cap.view(batch.num_graphs, -1)
        y_reg = batch.y_reg.view(batch.num_graphs, -1)
        with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
            v_n, c_log, r_p, pv_p = model(batch)
        v_n_flat = v_n.view(batch.num_graphs, -1)
        preds.append((v_n_flat * y_std.to(device) + y_mean.to(device)).cpu())
        tgts.append(yb.cpu())
        cap_logits_all.append(c_log.cpu())
        cap_tgt_all.append(y_cap.cpu())
        reg_pred_all.append(r_p.cpu())
        reg_tgt_all.append(y_reg.cpu())
        if use_pv and hasattr(batch, "y_pv") and batch.y_pv is not None and pv_p.size(-1) > 0:
            y_pv_b = batch.y_pv.view(batch.num_graphs, -1)
            pv_pred_all.append(pv_p.cpu())
            pv_tgt_all.append(y_pv_b.cpu())
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
    if use_pv and pv_pred_all:
        pp = torch.cat(pv_pred_all, dim=0)
        pt = torch.cat(pv_tgt_all, dim=0)
        met["pv_mse_normalized"] = float(F.mse_loss(pp, pt.to(pp.dtype)).item())
        pp_den = pp * pv_std.to(pp.device) + pv_mean.to(pp.device)
        pt_den = pt * pv_std.to(pt.device) + pv_mean.to(pt.device)
        met["pv_mse_raw"] = float(F.mse_loss(pp_den, pt_den.to(pp_den.dtype)).item())
    else:
        met["pv_mse_normalized"] = float("nan")
        met["pv_mse_raw"] = float("nan")
    return met


def _file_digest(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _select_sample_ids_from_meta(meta_csv: Path, sample_frac: float, seed: int, chunk_idx: int) -> list[int] | None:
    if float(sample_frac) >= 1.0:
        return None
    import pandas as pd

    df = pd.read_csv(meta_csv, usecols=["sample_id"])
    sids = [int(_norm_sid(s)) for s in df["sample_id"].tolist()]
    if not sids:
        return []
    rng = np.random.default_rng(int(seed) + int(chunk_idx) * 100_003)
    k = max(1, int(round(len(sids) * float(sample_frac))))
    pick = rng.choice(len(sids), size=k, replace=False)
    pick_sorted = np.sort(pick)
    return [int(sids[i]) for i in pick_sorted]


def _chunk_cache_path(
    cache_dir: Path,
    chunk_name: str,
    sample_frac: float,
    seed: int,
    chunk_idx: int,
    *,
    feat_slug: str = "",
    meta_aux_slug: str = "",
) -> Path:
    if float(sample_frac) >= 1.0:
        tag = "full"
    else:
        tag = f"sf{float(sample_frac):.6f}_s{int(seed)}_c{int(chunk_idx)}"
    base = f"{chunk_name}__{tag}"
    if str(feat_slug).strip():
        base = f"{base}__{str(feat_slug).strip()}"
    if str(meta_aux_slug).strip():
        base = f"{base}__maux{str(meta_aux_slug).strip()}"
    return cache_dir / f"{base}.pt"


def _meta_aux_cols_from_args(args: argparse.Namespace) -> list[str]:
    """Prefer --aux_meta_cols; fall back to deprecated --aux_pv_meta_cols."""
    raw = str(getattr(args, "aux_meta_cols", "") or "").strip()
    if raw:
        return [c.strip().lower() for c in raw.split(",") if c.strip()]
    raw = str(getattr(args, "aux_pv_meta_cols", "") or "").strip()
    return [c.strip().lower() for c in raw.split(",") if c.strip()]


def _meta_aux_cache_slug(meta_aux_cols: list[str]) -> str:
    if not meta_aux_cols:
        return ""
    return hashlib.md5(",".join(meta_aux_cols).encode("utf-8")).hexdigest()[:8]


def _ensure_chunk_tensor_cache(
    chunk_dir: Path,
    *,
    nodes_name: str,
    meta_name: str,
    node_feature_cols: list[str],
    node_pe_csv: Path | None,
    node_pe_cols: str,
    selected_sample_ids: list[int] | None,
    cap_cols: list[str],
    reg_cols: list[str],
    cache_pt: Path,
    bootstrap_gnn_cache_pt: Path | None,
    ref_ntl: dict[str, int] | None,
    pv_aux_cols: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], dict[str, int]]:
    np_ = chunk_dir / nodes_name
    mp_ = chunk_dir / meta_name
    pv_cols = [str(c).strip().lower() for c in (pv_aux_cols or []) if str(c).strip()]

    def _maybe_attach_y_pv(z: dict, sids: list[int]) -> torch.Tensor | None:
        if not pv_cols:
            return None
        k = len(pv_cols)
        stored = z.get("meta_aux_cols")
        stored_l = [str(x).lower() for x in stored] if stored is not None else None
        existing = z.get("y_pv")
        if existing is not None:
            ex = existing.to(dtype=torch.float32)
            if stored_l == pv_cols and ex.dim() == 2 and ex.shape[1] == k:
                return ex
            print(
                f"chunk cache y_pv out of date (cols or shape); recomputing meta aux: {cache_pt}",
                flush=True,
            )
            z.pop("y_pv", None)
            z.pop("meta_aux_cols", None)
        y_pv = _load_meta_pv(mp_, sids, pv_cols)
        z["y_pv"] = y_pv
        z["meta_aux_cols"] = list(pv_cols)
        torch.save(z, cache_pt)
        print(f"Added meta-aux columns to chunk cache: {cache_pt}", flush=True)
        return y_pv.to(dtype=torch.float32)

    if cache_pt.is_file():
        z = torch.load(cache_pt, map_location="cpu", weights_only=False)
        ntl = z["node_to_local"]
        if ref_ntl is not None and ntl != ref_ntl:
            raise RuntimeError(f"node_to_local mismatch vs first chunk: {chunk_dir}")
        sids = z["sample_ids"]
        if isinstance(sids, torch.Tensor):
            sids = [int(x) for x in sids.tolist()]
        else:
            sids = list(sids)
        x = z["x"].to(dtype=torch.float32)
        y_ri = z["y_ri"].to(dtype=torch.float32)
        y_cap = z["y_cap"].to(dtype=torch.float32)
        y_reg = z["y_reg"].to(dtype=torch.float32)
        y_pv = None
        if pv_cols:
            y_pv = _maybe_attach_y_pv(z, sids)
            if y_pv is not None:
                y_pv = y_pv.to(dtype=torch.float32)
        return x, y_ri, y_cap, y_reg, y_pv, sids, ntl

    if bootstrap_gnn_cache_pt is not None and bootstrap_gnn_cache_pt.is_file():
        z = torch.load(bootstrap_gnn_cache_pt, map_location="cpu", weights_only=False)
        need = {"x", "y_ri", "sample_ids", "node_to_local"}
        if need.issubset(set(z.keys())):
            node_to_local = z["node_to_local"]
            if ref_ntl is not None and node_to_local != ref_ntl:
                raise RuntimeError(f"node_to_local mismatch vs first chunk: {chunk_dir}")
            sample_ids = z["sample_ids"]
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = [int(x) for x in sample_ids.tolist()]
            else:
                sample_ids = [int(x) for x in list(sample_ids)]
            x = z["x"].to(dtype=torch.float32)
            y_ri = z["y_ri"].to(dtype=torch.float32)
            if selected_sample_ids is not None:
                want = set(int(s) for s in selected_sample_ids)
                keep_idx = [i for i, sid in enumerate(sample_ids) if int(sid) in want]
                if not keep_idx:
                    raise RuntimeError(f"No selected sample IDs found in bootstrap GNN cache: {bootstrap_gnn_cache_pt}")
                idx_t = torch.tensor(keep_idx, dtype=torch.long)
                x = x.index_select(0, idx_t)
                y_ri = y_ri.index_select(0, idx_t)
                sample_ids = [sample_ids[i] for i in keep_idx]
            if not mp_.is_file():
                raise FileNotFoundError(mp_)
            y_cap, y_reg = _load_meta_aux(mp_, sample_ids, cap_cols, reg_cols)
            y_cap = y_cap.to(dtype=torch.float32)
            y_reg = y_reg.to(dtype=torch.float32)
            y_pv = _load_meta_pv(mp_, sample_ids, pv_cols) if pv_cols else None
            cache_pt.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "x": x,
                "y_ri": y_ri,
                "y_cap": y_cap,
                "y_reg": y_reg,
                "sample_ids": sample_ids,
                "node_to_local": node_to_local,
            }
            if y_pv is not None:
                row["y_pv"] = y_pv
                row["meta_aux_cols"] = list(pv_cols)
            torch.save(row, cache_pt)
            print(f"Bootstrapped DA cache from GNN cache: {bootstrap_gnn_cache_pt} -> {cache_pt}", flush=True)
            return x, y_ri, y_cap, y_reg, y_pv, sample_ids, node_to_local

    if not np_.is_file() or not mp_.is_file():
        raise FileNotFoundError(f"{np_} / {mp_}")
    x, y_ri, sample_ids, _, node_to_local = _load_nodes_features_complex_targets(
        np_,
        node_feature_cols=node_feature_cols,
        node_pe_csv=node_pe_csv,
        node_pe_cols=node_pe_cols,
        selected_sample_ids=selected_sample_ids,
    )
    x = x.to(dtype=torch.float32)
    y_ri = y_ri.to(dtype=torch.float32)
    if ref_ntl is not None and node_to_local != ref_ntl:
        raise RuntimeError(f"node_to_local mismatch vs first chunk: {chunk_dir}")
    y_cap, y_reg = _load_meta_aux(mp_, sample_ids, cap_cols, reg_cols)
    y_cap = y_cap.to(dtype=torch.float32)
    y_reg = y_reg.to(dtype=torch.float32)
    y_pv = _load_meta_pv(mp_, sample_ids, pv_cols) if pv_cols else None
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "x": x,
        "y_ri": y_ri,
        "y_cap": y_cap,
        "y_reg": y_reg,
        "sample_ids": sample_ids,
        "node_to_local": node_to_local,
    }
    if y_pv is not None:
        row["y_pv"] = y_pv
        row["meta_aux_cols"] = list(pv_cols)
    torch.save(row, cache_pt)
    print(f"Wrote chunk tensor cache: {cache_pt}", flush=True)
    return x, y_ri, y_cap, y_reg, y_pv, sample_ids, node_to_local


def _evaluate_multi_chunks(
    model: nn.Module,
    chunk_dirs: list[Path],
    idx_lists: list[np.ndarray],
    cache_pts: list[Path],
    bootstrap_cache_pts: list[Path | None],
    selected_ids_list: list[list[int] | None],
    *,
    nodes_name: str,
    meta_name: str,
    node_feature_cols: list[str],
    node_pe_csv: Path | None,
    node_pe_cols: str,
    cap_cols: list[str],
    reg_cols: list[str],
    cache_dir: Path,
    ref_ntl: dict[str, int],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    reg_mean: torch.Tensor,
    reg_std: torch.Tensor,
    pv_mean: torch.Tensor | None,
    pv_std: torch.Tensor | None,
    pv_aux_cols: list[str] | None,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    met_acc: dict[str, float] | None = None
    wtot = 0
    for ch, idx_te, cpt, boot_pt, sel_ids in zip(
        chunk_dirs, idx_lists, cache_pts, bootstrap_cache_pts, selected_ids_list
    ):
        if len(idx_te) == 0:
            continue
        x, y_ri, y_cap, y_reg, y_pv, _sids, _ntl = _ensure_chunk_tensor_cache(
            ch,
            nodes_name=nodes_name,
            meta_name=meta_name,
            node_feature_cols=node_feature_cols,
            node_pe_csv=node_pe_csv,
            node_pe_cols=node_pe_cols,
            selected_sample_ids=sel_ids,
            cap_cols=cap_cols,
            reg_cols=reg_cols,
            cache_pt=cpt,
            bootstrap_gnn_cache_pt=boot_pt,
            ref_ntl=ref_ntl,
            pv_aux_cols=pv_aux_cols,
        )
        y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)
        x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
        y_pv_n = None
        if y_pv is not None and pv_mean is not None and pv_std is not None:
            y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
        ds = DAGPSDataset(x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n)
        dl = DataLoader(
            Subset(ds, idx_te.tolist()),
            batch_size=min(64, max(1, len(idx_te))),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        met = evaluate(
            model,
            dl,
            device,
            y_mean,
            y_std,
            reg_mean,
            reg_std,
            use_amp=use_amp,
            pv_mean=pv_mean,
            pv_std=pv_std,
        )
        w = int(len(idx_te))
        if met_acc is None:
            met_acc = {k: met[k] * w for k in met}
        else:
            for k in met:
                met_acc[k] += met[k] * w
        wtot += w
        del x, y_ri, y_cap, y_reg, y_pv, y_reg_n, y_pv_n, x_n, ds, dl
        gc.collect()
    if met_acc is None or wtot == 0:
        return {
            "mae_vmag_pu": float("nan"),
            "rmse_vmag_pu": float("nan"),
            "mae_angle_deg": float("nan"),
            "rmse_angle_deg": float("nan"),
            "r2_vmag_mean": float("nan"),
            "r2_vmag_min": float("nan"),
            "mae_vmag_worst_node": float("nan"),
            "cap_bce": float("nan"),
            "reg_mse_normalized": float("nan"),
            "reg_mse_tap_pu": float("nan"),
            "pv_mse_normalized": float("nan"),
            "pv_mse_raw": float("nan"),
        }
    return {k: met_acc[k] / float(wtot) for k in met_acc}


def _save_periodic_training_checkpoint(
    path: Path,
    base_model: nn.Module,
    opt: torch.optim.Optimizer,
    sch: object,
    scaler: object | None,
    *,
    epoch: int,
    bad: int,
    best_val: float,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    """Atomic write of resumable training state (current + best-so-far weights)."""
    payload: dict[str, object] = {
        "epoch": int(epoch),
        "bad": int(bad),
        "best_val": float(best_val),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sch.state_dict(),
        "best_model_state_dict": (
            {k: v.detach().cpu().clone() for k, v in best_state.items()} if best_state is not None else None
        ),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()  # type: ignore[union-attr]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main_multi_chunk(args: argparse.Namespace, repo: Path) -> None:
    """Train on many chunk folders (no merged mega-CSV). One chunk loaded at a time."""
    _set_seed(args.seed)
    dropout = 0.0 if args.disable_dropout else float(args.dropout)

    chunk_parent = Path(args.chunk_parent).resolve()
    if not chunk_parent.is_dir():
        raise FileNotFoundError(chunk_parent)

    glob_pat = str(args.chunk_subdir_glob)
    chunk_dirs = sorted(
        [p for p in chunk_parent.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob_pat)],
        key=lambda p: p.name,
    )
    if not chunk_dirs:
        raise FileNotFoundError(f"No subdirs matching {glob_pat!r} under {chunk_parent}")

    nodes_name = Path(args.nodes_csv).name
    edge_name = Path(args.edge_catalog_csv).name
    meta_name = Path(args.meta_csv).name
    node_feature_cols = _parse_csv_cols(args.node_feature_cols)
    if bool(args.exclude_bess_features):
        _bess = ("p_bess_kw", "q_bess_kvar")
        node_feature_cols = [c for c in node_feature_cols if c not in _bess]
        print("exclude_bess_features: using node_feature_cols=", node_feature_cols, flush=True)
    feat_slug = "nobess" if bool(args.exclude_bess_features) else ""
    _raw_meta = str(getattr(args, "aux_meta_cols", "") or "").strip()
    _raw_pv = str(getattr(args, "aux_pv_meta_cols", "") or "").strip()
    if _raw_meta and _raw_pv:
        print(
            "NOTE: both --aux_meta_cols and --aux_pv_meta_cols are set; using --aux_meta_cols only.",
            flush=True,
        )
    pv_aux_cols = _meta_aux_cols_from_args(args)
    _bad = {"sample_id"} & set(pv_aux_cols)
    if _bad:
        raise ValueError(f"--aux_meta_cols must not include reserved column name(s): {_bad}")
    n_pv_aux = len(pv_aux_cols)
    maux_slug = _meta_aux_cache_slug(pv_aux_cols)
    if n_pv_aux > int(args.n_system_tokens):
        raise ValueError(
            f"--n_system_tokens ({args.n_system_tokens}) must be >= number of meta-aux columns ({n_pv_aux}). "
            "Each listed column supervises one system token in order (after cap and reg tokens)."
        )
    node_pe_csv = Path(args.node_pe_csv).resolve() if str(args.node_pe_csv).strip() else None
    node_pe_cols = str(args.node_pe_cols)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.cache_dir).strip():
        cache_dir = Path(args.cache_dir).resolve()
        print(f"chunk_parent cache override via --cache_dir: {cache_dir}", flush=True)
    elif args.cache_tensor:
        cache_override = Path(args.cache_tensor).resolve()
        if cache_override.suffix.lower() == ".pt":
            cache_dir = cache_override.parent / f"{cache_override.stem}_chunk_tensor_cache"
            print(
                f"chunk_parent cache override from --cache_tensor file path -> using directory: {cache_dir}",
                flush=True,
            )
        else:
            cache_dir = cache_override
            print(f"chunk_parent cache override: {cache_dir}", flush=True)
    else:
        cache_dir = out_dir / "chunk_tensor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_gnn_cache_dir = Path(args.bootstrap_gnn_cache_dir).resolve() if str(args.bootstrap_gnn_cache_dir).strip() else None
    if bootstrap_gnn_cache_dir is not None:
        print(f"bootstrap GNN cache dir: {bootstrap_gnn_cache_dir}", flush=True)

    cap_cols = list(TARGET_CAP_COLS)
    reg_cols = list(TARGET_REG_COLS)
    n_cap = len(cap_cols)
    n_reg = len(reg_cols)
    n_sys = int(args.n_system_tokens)
    g_tot = n_cap + n_reg + n_sys
    if n_pv_aux > 0:
        print(
            f"Meta aux (sample_meta): {n_pv_aux} column(s); chunk DA caches use suffix __maux{maux_slug} (per chunk name).",
            flush=True,
        )
        for j, cname in enumerate(pv_aux_cols):
            tok_i = n_cap + n_reg + j
            print(f"  global token index {tok_i} (system slot {j}): column {cname!r}", flush=True)

    ref_digest = _file_digest(chunk_dirs[0] / edge_name)
    for ch in chunk_dirs[1:]:
        ep = ch / edge_name
        if not ep.is_file():
            raise FileNotFoundError(ep)
        if _file_digest(ep) != ref_digest:
            raise RuntimeError(f"Edge catalog differs from first chunk (must be identical topology): {ep}")

    print(f"[chunk_parent] {len(chunk_dirs)} chunks under {chunk_parent}", flush=True)
    for d in chunk_dirs:
        print(f"  - {d.name}", flush=True)

    idx_train_list: list[np.ndarray] = []
    idx_val_list: list[np.ndarray] = []
    idx_test_list: list[np.ndarray] = []
    selected_ids_list: list[list[int] | None] = []
    cache_pts: list[Path] = []
    bootstrap_cache_pts: list[Path | None] = []

    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None
    cnt_x = 0
    sum_y: torch.Tensor | None = None
    sum_y2: torch.Tensor | None = None
    cnt_y = 0
    sum_reg: torch.Tensor | None = None
    sum_reg2: torch.Tensor | None = None
    cnt_reg = 0
    sum_pv: torch.Tensor | None = None
    sum_pv2: torch.Tensor | None = None
    cnt_pv = 0

    ref_ntl: dict[str, int] | None = None
    edge_index: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    n_nodes = 0
    n_node_features = 0

    for ci, ch in enumerate(chunk_dirs):
        meta_path = ch / meta_name
        sel_ids = _select_sample_ids_from_meta(meta_path, float(args.sample_frac), int(args.seed), ci)
        selected_ids_list.append(sel_ids)
        da_pt = _chunk_cache_path(
            cache_dir, ch.name, float(args.sample_frac), int(args.seed), ci, feat_slug=feat_slug, meta_aux_slug=maux_slug
        )
        cache_pts.append(da_pt)
        if bootstrap_gnn_cache_dir is not None:
            boot_name = _chunk_cache_path(
                cache_dir, ch.name, float(args.sample_frac), int(args.seed), ci, feat_slug=feat_slug, meta_aux_slug=""
            ).name
            bootstrap_cache_pts.append(bootstrap_gnn_cache_dir / boot_name)
        else:
            bootstrap_cache_pts.append(None)
        boot_pt = bootstrap_cache_pts[-1]
        x, y_ri, y_cap, y_reg, y_pv, _sids, ntl = _ensure_chunk_tensor_cache(
            ch,
            nodes_name=nodes_name,
            meta_name=meta_name,
            node_feature_cols=node_feature_cols,
            node_pe_csv=node_pe_csv,
            node_pe_cols=node_pe_cols,
            selected_sample_ids=sel_ids,
            cap_cols=cap_cols,
            reg_cols=reg_cols,
            cache_pt=da_pt,
            bootstrap_gnn_cache_pt=boot_pt,
            ref_ntl=ref_ntl,
            pv_aux_cols=pv_aux_cols if n_pv_aux > 0 else None,
        )
        if ci == 0:
            ref_ntl = ntl
            n_nodes = int(x.shape[1])
            n_node_features = int(x.shape[2])
            ep = ch / edge_name
            edge_index, edge_attr = _load_compacted_edges(ep, ref_ntl)
            sum_x = torch.zeros(n_node_features, dtype=torch.float64)
            sum_x2 = torch.zeros(n_node_features, dtype=torch.float64)
            sum_y = torch.zeros(n_nodes * 2, dtype=torch.float64)
            sum_y2 = torch.zeros(n_nodes * 2, dtype=torch.float64)
            sum_reg = torch.zeros(n_reg, dtype=torch.float64)
            sum_reg2 = torch.zeros(n_reg, dtype=torch.float64)
            if n_pv_aux > 0:
                sum_pv = torch.zeros(n_pv_aux, dtype=torch.float64)
                sum_pv2 = torch.zeros(n_pv_aux, dtype=torch.float64)
        assert sum_x is not None and sum_x2 is not None and sum_y is not None and sum_reg is not None and edge_index is not None

        n = int(x.shape[0])
        rng = np.random.default_rng(int(args.seed) + ci * 100_003)
        perm = rng.permutation(n)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f"Invalid train/val/test split for chunk {ch.name}.")
        idx_train_list.append(perm[:n_train])
        idx_val_list.append(perm[n_train : n_train + n_val])
        idx_test_list.append(perm[n_train + n_val :])

        itr = idx_train_list[-1]
        xt = x[itr].reshape(-1, n_node_features).to(dtype=torch.float64)
        sum_x += xt.sum(dim=0)
        sum_x2 += (xt * xt).sum(dim=0)
        cnt_x += int(xt.shape[0])

        yt = y_ri[itr].reshape(len(itr), -1).to(dtype=torch.float64)
        sum_y += yt.sum(dim=0)
        sum_y2 += (yt * yt).sum(dim=0)
        cnt_y += len(itr)

        rt = y_reg[itr].to(dtype=torch.float64)
        sum_reg += rt.sum(dim=0)
        sum_reg2 += (rt * rt).sum(dim=0)
        cnt_reg += len(itr)

        if n_pv_aux > 0 and y_pv is not None and sum_pv is not None and sum_pv2 is not None:
            ypv = y_pv[itr].to(dtype=torch.float64)
            sum_pv += ypv.sum(dim=0)
            sum_pv2 += (ypv * ypv).sum(dim=0)
            cnt_pv += len(itr)

        del x, y_ri, y_cap, y_reg, y_pv
        gc.collect()

    assert ref_ntl is not None and edge_index is not None and edge_attr is not None
    assert sum_y is not None and cnt_x > 0

    assert sum_x is not None and sum_x2 is not None
    x_mean = (sum_x / float(cnt_x)).view(1, n_node_features).float()
    x_var = sum_x2 / float(cnt_x) - (sum_x / float(cnt_x)) ** 2
    x_std = torch.sqrt(x_var.clamp_min(1e-24)).view(1, n_node_features).clamp_min(1e-8).float()

    y_mean = (sum_y / float(cnt_y)).view(1, -1).float()
    y_var = sum_y2 / float(cnt_y) - (sum_y / float(cnt_y)) ** 2
    y_std = torch.sqrt(y_var.clamp_min(1e-24)).view(1, -1).clamp_min(1e-6).float()

    reg_mean = (sum_reg / float(cnt_reg)).view(1, -1).float()
    reg_var = sum_reg2 / float(cnt_reg) - (sum_reg / float(cnt_reg)) ** 2
    reg_std = torch.sqrt(reg_var.clamp_min(1e-24)).view(1, -1).clamp_min(1e-6).float()

    if n_pv_aux > 0:
        if sum_pv is None or cnt_pv < 1:
            raise RuntimeError(
                "Meta aux enabled but no train statistics accumulated for meta columns (missing y_pv in caches?)."
            )
        pv_mean = (sum_pv / float(cnt_pv)).view(1, -1).float()
        pv_var = sum_pv2 / float(cnt_pv) - (sum_pv / float(cnt_pv)) ** 2
        pv_std = torch.sqrt(pv_var.clamp_min(1e-24)).view(1, -1).clamp_min(1e-6).float()
        torch.save(pv_mean, out_dir / "pv_mean.pt")
        torch.save(pv_std, out_dir / "pv_std.pt")
    else:
        pv_mean = None
        pv_std = None

    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    torch.save(y_mean, out_dir / "y_mean.pt")
    torch.save(y_std, out_dir / "y_std.pt")
    torch.save(reg_mean, out_dir / "reg_mean.pt")
    torch.save(reg_std, out_dir / "reg_std.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    nw = int(args.num_workers)

    base_model = DAGPSModel(
        n_nodes=n_nodes,
        num_edges=int(edge_index.shape[1]),
        hidden=int(args.hidden),
        heads=int(args.heads),
        n_layers=int(args.layers),
        n_cap=n_cap,
        n_reg=n_reg,
        n_system=n_sys,
        node_in_dim=n_node_features,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        edge_dim=int(edge_attr.size(1)),
        dropout=dropout,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        per_node_heads=bool(args.per_node_heads),
        per_device_cap_head=bool(args.per_device_cap_head),
        per_device_reg_head=bool(args.per_device_reg_head),
        n_pv_aux=int(n_pv_aux),
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

    y_mean_d = y_mean.to(device).float()
    y_std_d = y_std.to(device).float()
    reg_mean_d = reg_mean.to(device).float()
    reg_std_d = reg_std.to(device).float()
    pv_mean_d = pv_mean.to(device).float() if pv_mean is not None else None
    pv_std_d = pv_std.to(device).float() if pv_std is not None else None
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
        train_loss_sum = train_v_sum = train_c_sum = train_r_sum = train_pv_sum = 0.0
        train_n = 0
        train_order = np.random.default_rng(args.seed + ep * 17).permutation(len(chunk_dirs))
        for ci in train_order:
            ci_i = int(ci)
            ch = chunk_dirs[ci_i]
            cpt = cache_pts[ci_i]
            boot_pt = bootstrap_cache_pts[ci_i]
            x, y_ri, y_cap, y_reg, y_pv, _sids, _ntl = _ensure_chunk_tensor_cache(
                ch,
                nodes_name=nodes_name,
                meta_name=meta_name,
                node_feature_cols=node_feature_cols,
                node_pe_csv=node_pe_csv,
                node_pe_cols=node_pe_cols,
                selected_sample_ids=selected_ids_list[ci_i],
                cap_cols=cap_cols,
                reg_cols=reg_cols,
                cache_pt=cpt,
                bootstrap_gnn_cache_pt=boot_pt,
                ref_ntl=ref_ntl,
                pv_aux_cols=pv_aux_cols if n_pv_aux > 0 else None,
            )
            y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)
            x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
            y_pv_n = None
            if n_pv_aux > 0 and y_pv is not None and pv_mean is not None and pv_std is not None:
                y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
            ds = DAGPSDataset(x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n)
            dl_tr = DataLoader(
                Subset(ds, idx_train_list[ci_i].tolist()),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=pin,
                persistent_workers=nw > 0,
            )
            for batch in dl_tr:
                batch = batch.to(device)
                batch = _cast_batch_float_tensors(batch)
                yb = batch.y.view(batch.num_graphs, -1)
                y_cap_b = batch.y_cap.view(batch.num_graphs, -1)
                y_reg_b = batch.y_reg.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean_d) / y_std_d
                opt.zero_grad(set_to_none=True)
                with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                    v_n, c_log, r_p, pv_p = model(batch)
                    loss_v = mse(v_n.view_as(yb_n), yb_n)
                    loss_c = bce(c_log, y_cap_b)
                    loss_r = mse(r_p, y_reg_b)
                    loss = loss_v + float(args.lambda_cap) * loss_c + float(args.lambda_reg) * loss_r
                    if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                        y_pv_b = batch.y_pv.view(batch.num_graphs, -1)
                        loss_pv = mse(pv_p, y_pv_b)
                        loss = loss + float(args.lambda_pv) * loss_pv
                        train_pv_sum += float(loss_pv.item()) * batch.num_graphs
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
                with torch.no_grad():
                    train_loss_sum += float(loss.item()) * batch.num_graphs
                    train_v_sum += float(loss_v.item()) * batch.num_graphs
                    train_c_sum += float(loss_c.item()) * batch.num_graphs
                    train_r_sum += float(loss_r.item()) * batch.num_graphs
                    train_n += int(batch.num_graphs)
            del x, y_ri, y_cap, y_reg, y_pv, y_reg_n, y_pv_n, x_n, ds, dl_tr
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model.eval()
        val_tot = val_v = 0.0
        val_c_sum = val_r_sum = val_pv_sum = 0.0
        nv = 0
        val_sum_true = torch.zeros(n_nodes, device=device)
        val_sum_true2 = torch.zeros(n_nodes, device=device)
        val_sum_se = torch.zeros(n_nodes, device=device)
        val_sum_worst = 0.0
        with torch.no_grad():
            for ci, ch in enumerate(chunk_dirs):
                cpt = cache_pts[ci]
                boot_pt = bootstrap_cache_pts[ci]
                x, y_ri, y_cap, y_reg, y_pv, _sids, _ntl = _ensure_chunk_tensor_cache(
                    ch,
                    nodes_name=nodes_name,
                    meta_name=meta_name,
                    node_feature_cols=node_feature_cols,
                    node_pe_csv=node_pe_csv,
                    node_pe_cols=node_pe_cols,
                    selected_sample_ids=selected_ids_list[ci],
                    cap_cols=cap_cols,
                    reg_cols=reg_cols,
                    cache_pt=cpt,
                    bootstrap_gnn_cache_pt=boot_pt,
                    ref_ntl=ref_ntl,
                    pv_aux_cols=pv_aux_cols if n_pv_aux > 0 else None,
                )
                y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)
                x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
                y_pv_n = None
                if n_pv_aux > 0 and y_pv is not None and pv_mean is not None and pv_std is not None:
                    y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
                ds = DAGPSDataset(x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n)
                dl_va = DataLoader(
                    Subset(ds, idx_val_list[ci].tolist()),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=nw,
                    pin_memory=pin,
                    persistent_workers=nw > 0,
                )
                for batch in dl_va:
                    batch = batch.to(device)
                    batch = _cast_batch_float_tensors(batch)
                    yb = batch.y.view(batch.num_graphs, -1)
                    y_cap_b = batch.y_cap.view(batch.num_graphs, -1)
                    y_reg_b = batch.y_reg.view(batch.num_graphs, -1)
                    yb_n = (yb - y_mean_d) / y_std_d
                    with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                        v_n, c_log, r_p, pv_p = model(batch)
                        lv = mse(v_n.view_as(yb_n), yb_n)
                        lc = bce(c_log, y_cap_b)
                        lr_ = mse(r_p, y_reg_b)
                        lt = lv + float(args.lambda_cap) * lc + float(args.lambda_reg) * lr_
                        if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                            lpv = mse(pv_p, batch.y_pv.view(batch.num_graphs, -1))
                            lt = lt + float(args.lambda_pv) * lpv
                            val_pv_sum += float(lpv.item()) * batch.num_graphs
                    val_tot += float(lt.item()) * batch.num_graphs
                    val_v += float(lv.item()) * batch.num_graphs
                    val_c_sum += float(lc.item()) * batch.num_graphs
                    val_r_sum += float(lr_.item()) * batch.num_graphs
                    nv += int(batch.num_graphs)
                    v_flat = v_n.view(batch.num_graphs, -1)
                    pred_ri = (v_flat * y_std_d + y_mean_d).view(batch.num_graphs, n_nodes, 2)
                    true_ri = yb.view(batch.num_graphs, n_nodes, 2)
                    pred_mag = torch.sqrt(pred_ri[..., 0] * pred_ri[..., 0] + pred_ri[..., 1] * pred_ri[..., 1] + 1e-12)
                    true_mag = torch.sqrt(true_ri[..., 0] * true_ri[..., 0] + true_ri[..., 1] * true_ri[..., 1] + 1e-12)
                    err = pred_mag - true_mag
                    val_sum_true += true_mag.sum(dim=0)
                    val_sum_true2 += (true_mag * true_mag).sum(dim=0)
                    val_sum_se += (err * err).sum(dim=0)
                    val_sum_worst += float(err.abs().max(dim=1).values.sum().item())
                del x, y_ri, y_cap, y_reg, y_pv, y_reg_n, y_pv_n, x_n, ds, dl_va
                gc.collect()

        val_tot /= max(nv, 1)
        val_v /= max(nv, 1)
        val_c = val_c_sum / max(nv, 1)
        val_r = val_r_sum / max(nv, 1)
        val_pv = val_pv_sum / max(nv, 1) if n_pv_aux > 0 else float("nan")
        true_mean = val_sum_true / max(nv, 1)
        var_true = val_sum_true2 / max(nv, 1) - true_mean * true_mean
        mse_node = val_sum_se / max(nv, 1)
        r2_node = 1.0 - mse_node / var_true.clamp_min(1e-8)
        val_r2_mean = float(r2_node.mean().item())
        val_r2_min = float(r2_node.min().item())
        val_worst_node_mae = val_sum_worst / max(nv, 1)
        train_v = train_v_sum / max(train_n, 1)
        train_c = train_c_sum / max(train_n, 1)
        train_r = train_r_sum / max(train_n, 1)
        train_pv = train_pv_sum / max(train_n, 1) if n_pv_aux > 0 else float("nan")
        train_tot = train_loss_sum / max(train_n, 1)
        sch.step(val_tot)
        crit = val_tot if args.early_stop_on == "total" else val_v
        if crit < best_val:
            best_val = crit
            best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % max(1, int(args.log_every)) == 0:
            _log = (
                f"[da_gps chunk_parent] epoch {ep:4d}/{args.epochs} "
                f"| train_tot={train_tot:.4f} train_volt={train_v:.4f} train_cap={train_c:.4f} train_reg={train_r:.4f}"
            )
            if n_pv_aux > 0:
                _log += f" train_meta_aux={train_pv:.4f} val_meta_aux={val_pv:.4f}"
            _log += (
                f" | val_tot={val_tot:.4f} val_volt={val_v:.4f} val_cap={val_c:.4f} val_reg={val_r:.4f} "
                f"| val_r2_mean={val_r2_mean:.4f} val_r2_min={val_r2_min:.4f} val_worst_mae={val_worst_node_mae:.4f} "
                f"| best={best_val:.4f}"
            )
            print(_log, flush=True)
        _ce = int(args.checkpoint_every)
        if _ce > 0 and ep % _ce == 0:
            _ck = out_dir / "training_last.pt"
            _save_periodic_training_checkpoint(
                _ck, base_model, opt, sch, scaler, epoch=ep, bad=bad, best_val=best_val, best_state=best_state
            )
            print(f"  periodic checkpoint -> {_ck}", flush=True)
        if bad >= args.patience:
            print(f"[da_gps chunk_parent] early stop at epoch {ep}", flush=True)
            if int(args.checkpoint_every) > 0:
                _ck = out_dir / "training_last.pt"
                _save_periodic_training_checkpoint(
                    _ck, base_model, opt, sch, scaler, epoch=ep, bad=bad, best_val=best_val, best_state=best_state
                )
                print(f"  periodic checkpoint (early stop) -> {_ck}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        base_model.load_state_dict(best_state)

    met = _evaluate_multi_chunks(
        model,
        chunk_dirs,
        idx_test_list,
        cache_pts,
        bootstrap_cache_pts,
        selected_ids_list,
        nodes_name=nodes_name,
        meta_name=meta_name,
        node_feature_cols=node_feature_cols,
        node_pe_csv=node_pe_csv,
        node_pe_cols=node_pe_cols,
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        cache_dir=cache_dir,
        ref_ntl=ref_ntl,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        reg_mean=reg_mean,
        reg_std=reg_std,
        pv_mean=pv_mean,
        pv_std=pv_std,
        pv_aux_cols=pv_aux_cols if n_pv_aux > 0 else None,
        device=device,
        use_amp=use_amp,
    )

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
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "per_node_heads": bool(args.per_node_heads),
            "per_device_cap_head": bool(args.per_device_cap_head),
            "per_device_reg_head": bool(args.per_device_reg_head),
            "n_pv_aux": int(n_pv_aux),
            "pv_target_cols": list(pv_aux_cols) if n_pv_aux > 0 else [],
            "meta_aux_target_cols": list(pv_aux_cols) if n_pv_aux > 0 else [],
            "cap_target_cols": cap_cols,
            "reg_target_cols": reg_cols,
            "chunk_parent": str(chunk_parent),
            "chunk_folders": [str(p) for p in chunk_dirs],
        },
        ckpt,
    )
    report = {
        "task": "DA-GPS multitask chunk_parent",
        "chunk_parent": str(chunk_parent),
        "chunks": [str(p) for p in chunk_dirs],
        "normalization": "aggregated train statistics across all chunks",
        "chunk_tensor_cache_dir": str(cache_dir),
        "n_chunks": len(chunk_dirs),
        "hyperparameters": vars(args),
        "test_metrics": met,
        "train_seconds": train_seconds,
        "checkpoint": str(ckpt.resolve()),
    }
    (out_dir / "da_gps_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _pv_n = met.get("pv_mse_normalized", float("nan"))
    _pv_raw = met.get("pv_mse_raw", float("nan"))
    _pv_tail = f"  meta_aux_MSE(nrm)={_pv_n:.6f}  meta_aux_MSE(raw)={_pv_raw:.6f}" if n_pv_aux > 0 else ""
    print(
        f"Test |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"cap_BCE={met['cap_bce']:.6f}  reg_MSE(pu)={met['reg_mse_tap_pu']:.6f}{_pv_tail}  time={train_seconds:.1f}s",
        flush=True,
    )
    print(f"Saved {ckpt}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA-GPS v2 multitask: voltage + cap + reg (full MV, hardcoded aux cols).")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg_full_mv")
    p.add_argument("--nodes_csv", type=str, default="gnn_node_features_and_targets_full_mv.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="gnn_edges_phase_static_full_mv.csv")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument(
        "--node_feature_cols",
        type=str,
        default="p_load_kw,q_load_kvar",
        help="Comma-separated dynamic node feature columns from nodes_csv.",
    )
    p.add_argument(
        "--node_pe_csv",
        type=str,
        default="",
        help="Optional single PE CSV shared by all chunks/runs (e.g., gnn_node_index_master.csv).",
    )
    p.add_argument(
        "--node_pe_cols",
        type=str,
        default="auto",
        help="'auto' to use all pe_* columns from node_pe_csv, 'none' to disable, or comma list (e.g. pe_1,pe_2).",
    )
    p.add_argument("--n_system_tokens", type=int, default=10, help="Unsupervised latent tokens after cap+reg tokens.")
    p.add_argument("--out_dir", type=str, default="da_gps_multitask_full_mv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64, help="Per-step graphs; A100 can usually fit 32–64+ for N~3.8k, d=256.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--node_emb_dim", type=int, default=0, help="Optional learned node-id embedding dim.")
    p.add_argument("--edge_emb_dim", type=int, default=0, help="Optional learned edge-id embedding dim.")
    p.add_argument("--dropout", type=float, default=0.1)
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
    p.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Chunk mode only: directory for per-chunk tensor caches. Lets you reuse cache across runs while keeping out_dir timestamped.",
    )
    p.add_argument(
        "--bootstrap_gnn_cache_dir",
        type=str,
        default="",
        help="Chunk mode only: optional directory of GNN chunk caches (run_*__*.pt with x,y_ri,sample_ids,node_to_local). "
        "If DA cache is missing, bootstrap from GNN cache and compute only y_cap/y_reg from meta.",
    )
    p.add_argument(
        "--early_stop_on",
        type=str,
        default="total",
        choices=("total", "voltage"),
        help="Validation metric for best checkpoint / patience.",
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
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help="Every N epochs save out_dir/training_last.pt (model+optimizer+scheduler+epoch); 0 disables.",
    )
    p.add_argument(
        "--per_node_heads",
        action="store_true",
        help="Use independent per-node voltage decoder instead of shared MLP head.",
    )
    p.add_argument(
        "--per_device_cap_head",
        action="store_true",
        help="Use independent decoder per cap bank token instead of shared linear.",
    )
    p.add_argument(
        "--per_device_reg_head",
        action="store_true",
        help="Use independent decoder per regulator token instead of shared linear.",
    )
    p.add_argument(
        "--chunk_parent",
        type=str,
        default="",
        help="If set, train sequentially on each matching subfolder (see --chunk_subdir_glob) without merging CSVs. "
        "Filenames are --nodes_csv / --edge_catalog_csv / --meta_csv inside each folder. "
        "Normalization aggregates train-split statistics across all chunks. Tensor caches go to out_dir/chunk_tensor_cache/.",
    )
    p.add_argument(
        "--chunk_subdir_glob",
        type=str,
        default="run_*",
        help="Only used with --chunk_parent: fnmatch pattern for subdirectory names (e.g. run_*).",
    )
    p.add_argument(
        "--exclude_bess_features",
        action="store_true",
        help="Remove p_bess_kw and q_bess_kvar from --node_feature_cols if present (chunk mode uses cache filename __nobess).",
    )
    p.add_argument(
        "--aux_meta_cols",
        type=str,
        default="",
        help="Comma-separated numeric columns from gnn_sample_meta: column i supervises global system token "
        "index (n_cap+n_reg+i) with normalized MSE. Use any meta names you want (not only PV). "
        "Overrides --aux_pv_meta_cols when non-empty. Empty disables.",
    )
    p.add_argument(
        "--aux_pv_meta_cols",
        type=str,
        default="",
        help="Deprecated alias for --aux_meta_cols (used only when --aux_meta_cols is empty).",
    )
    p.add_argument(
        "--lambda_pv",
        type=float,
        default=0.1,
        help="Loss weight for meta-aux MSE (all --aux_meta_cols targets; normalized like regulators).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent
    if str(args.chunk_parent).strip():
        main_multi_chunk(args, repo)
        return

    _set_seed(args.seed)
    dropout = 0.0 if args.disable_dropout else float(args.dropout)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv) if Path(args.nodes_csv).is_absolute() else (data_root / args.nodes_csv).resolve()
    edges_path = Path(args.edge_catalog_csv) if Path(args.edge_catalog_csv).is_absolute() else (data_root / args.edge_catalog_csv).resolve()
    meta_path = Path(args.meta_csv) if Path(args.meta_csv).is_absolute() else (data_root / args.meta_csv).resolve()
    node_feature_cols = _parse_csv_cols(args.node_feature_cols)
    if bool(args.exclude_bess_features):
        node_feature_cols = [c for c in node_feature_cols if c not in ("p_bess_kw", "q_bess_kvar")]
        print("exclude_bess_features: using node_feature_cols=", node_feature_cols, flush=True)
    _raw_meta = str(getattr(args, "aux_meta_cols", "") or "").strip()
    _raw_pv = str(getattr(args, "aux_pv_meta_cols", "") or "").strip()
    if _raw_meta and _raw_pv:
        print(
            "NOTE: both --aux_meta_cols and --aux_pv_meta_cols are set; using --aux_meta_cols only.",
            flush=True,
        )
    pv_aux_cols = _meta_aux_cols_from_args(args)
    _bad = {"sample_id"} & set(pv_aux_cols)
    if _bad:
        raise ValueError(f"--aux_meta_cols must not include reserved column name(s): {_bad}")
    n_pv_aux = len(pv_aux_cols)
    if n_pv_aux > int(args.n_system_tokens):
        raise ValueError(
            f"--n_system_tokens ({args.n_system_tokens}) must be >= number of meta-aux columns ({n_pv_aux})."
        )
    node_pe_csv = Path(args.node_pe_csv).resolve() if str(args.node_pe_csv).strip() else None
    node_pe_cols = str(args.node_pe_cols)

    for pth in (nodes_path, edges_path, meta_path):
        if not pth.is_file():
            raise FileNotFoundError(pth)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_cols = list(TARGET_CAP_COLS)
    reg_cols = list(TARGET_REG_COLS)
    n_cap = len(cap_cols)
    n_reg = len(reg_cols)
    n_sys = int(args.n_system_tokens)
    g_tot = n_cap + n_reg + n_sys
    if n_pv_aux > 0:
        print(f"Meta aux (sample_meta): {n_pv_aux} column(s): {pv_aux_cols}", flush=True)
        for j, cname in enumerate(pv_aux_cols):
            tok_i = n_cap + n_reg + j
            print(f"  global token index {tok_i} (system slot {j}): column {cname!r}", flush=True)

    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None
    node_to_local = None
    if cache_path and cache_path.is_file():
        print(f"Loading cache: {cache_path}", flush=True)
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = pack["x"].to(dtype=torch.float32)
        y_ri = pack.get("y_ri")
        if y_ri is not None:
            y_ri = y_ri.to(dtype=torch.float32)
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        sample_ids = pack["sample_ids"]
    else:
        x, y_ri, sample_ids, _node_order, node_to_local = _load_nodes_features_complex_targets(
            nodes_path,
            node_feature_cols=node_feature_cols,
            node_pe_csv=node_pe_csv,
            node_pe_cols=node_pe_cols,
        )
        x = x.to(dtype=torch.float32)
        y_ri = y_ri.to(dtype=torch.float32)
        edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"x": x, "y_ri": y_ri, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids},
                cache_path,
            )
            print(f"Wrote cache: {cache_path}", flush=True)

    if node_to_local is None:
        _x_tmp, _y_tmp, _sid_tmp, _node_order_tmp, node_to_local = _load_nodes_features_complex_targets(
            nodes_path,
            node_feature_cols=node_feature_cols,
            node_pe_csv=node_pe_csv,
            node_pe_cols=node_pe_cols,
        )
        del _x_tmp, _y_tmp, _sid_tmp, _node_order_tmp

    n_nodes = int(x.shape[1])
    n_node_features = int(x.shape[2])

    if y_ri is None:
        y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)
    sid_list = (
        [int(x) for x in sample_ids.tolist()]
        if isinstance(sample_ids, torch.Tensor)
        else [int(_norm_sid(s)) for s in sample_ids]
    )
    y_cap, y_reg = _load_meta_aux(meta_path, sid_list, cap_cols, reg_cols)
    y_cap = y_cap.to(dtype=torch.float32)
    y_reg = y_reg.to(dtype=torch.float32)
    y_pv = _load_meta_pv(meta_path, sid_list, pv_aux_cols) if n_pv_aux > 0 else None

    if args.sample_frac < 1.0:
        k = max(1, int(round(len(sample_ids) * args.sample_frac)))
        x = x[:k]
        y_ri = y_ri[:k]
        y_cap = y_cap[:k]
        y_reg = y_reg[:k]
        if y_pv is not None:
            y_pv = y_pv[:k]
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

    xt = x[idx_train].reshape(-1, n_node_features)
    x_mean = xt.mean(dim=0, keepdim=True)
    x_std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8).float()
    x_n = (x - x_mean) / x_std

    y_train = y_ri[idx_train].reshape(len(idx_train), -1)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6).float()

    reg_mean = y_reg[idx_train].mean(dim=0, keepdim=True)
    reg_std = y_reg[idx_train].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6).float()
    y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)

    if n_pv_aux > 0 and y_pv is not None:
        pv_mean = y_pv[idx_train].mean(dim=0, keepdim=True)
        pv_std = y_pv[idx_train].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6).float()
        y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
        torch.save(pv_mean, out_dir / "pv_mean.pt")
        torch.save(pv_std, out_dir / "pv_std.pt")
    else:
        pv_mean = None
        pv_std = None
        y_pv_n = None

    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    torch.save(y_mean, out_dir / "y_mean.pt")
    torch.save(y_std, out_dir / "y_std.pt")
    torch.save(reg_mean, out_dir / "reg_mean.pt")
    torch.save(reg_std, out_dir / "reg_std.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DAGPSDataset(x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n)
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
        num_edges=int(edge_index.shape[1]),
        hidden=int(args.hidden),
        heads=int(args.heads),
        n_layers=int(args.layers),
        n_cap=n_cap,
        n_reg=n_reg,
        n_system=n_sys,
        node_in_dim=n_node_features,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        edge_dim=int(edge_attr.size(1)),
        dropout=dropout,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        per_node_heads=bool(args.per_node_heads),
        per_device_cap_head=bool(args.per_device_cap_head),
        per_device_reg_head=bool(args.per_device_reg_head),
        n_pv_aux=int(n_pv_aux),
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

    y_mean_d = y_mean.to(device).float()
    y_std_d = y_std.to(device).float()
    reg_mean_d = reg_mean.to(device).float()
    reg_std_d = reg_std.to(device).float()
    pv_mean_d = pv_mean.to(device).float() if pv_mean is not None else None
    pv_std_d = pv_std.to(device).float() if pv_std is not None else None
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
        train_loss_sum = train_v_sum = train_c_sum = train_r_sum = train_pv_sum = 0.0
        train_n = 0
        for batch in dl_tr:
            batch = batch.to(device)
            batch = _cast_batch_float_tensors(batch)
            yb = batch.y.view(batch.num_graphs, -1)
            y_cap = batch.y_cap.view(batch.num_graphs, -1)
            y_reg = batch.y_reg.view(batch.num_graphs, -1)
            yb_n = (yb - y_mean_d) / y_std_d
            opt.zero_grad(set_to_none=True)
            with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                v_n, c_log, r_p, pv_p = model(batch)
                loss_v = mse(v_n.view_as(yb_n), yb_n)
                loss_c = bce(c_log, y_cap)
                loss_r = mse(r_p, y_reg)
                loss = loss_v + float(args.lambda_cap) * loss_c + float(args.lambda_reg) * loss_r
                if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                    loss_pv = mse(pv_p, batch.y_pv.view(batch.num_graphs, -1))
                    loss = loss + float(args.lambda_pv) * loss_pv
                    train_pv_sum += float(loss_pv.item()) * batch.num_graphs
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
            with torch.no_grad():
                train_loss_sum += float(loss.item()) * batch.num_graphs
                train_v_sum += float(loss_v.item()) * batch.num_graphs
                train_c_sum += float(loss_c.item()) * batch.num_graphs
                train_r_sum += float(loss_r.item()) * batch.num_graphs
                train_n += int(batch.num_graphs)

        model.eval()
        val_tot = val_v = 0.0
        val_c_sum = val_r_sum = val_pv_sum = 0.0
        nv = 0
        val_sum_true = torch.zeros(n_nodes, device=device)
        val_sum_true2 = torch.zeros(n_nodes, device=device)
        val_sum_se = torch.zeros(n_nodes, device=device)
        val_sum_worst = 0.0
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                batch = _cast_batch_float_tensors(batch)
                yb = batch.y.view(batch.num_graphs, -1)
                y_cap = batch.y_cap.view(batch.num_graphs, -1)
                y_reg = batch.y_reg.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean_d) / y_std_d
                with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                    v_n, c_log, r_p, pv_p = model(batch)
                    lv = mse(v_n.view_as(yb_n), yb_n)
                    lc = bce(c_log, y_cap)
                    lr_ = mse(r_p, y_reg)
                    lt = lv + float(args.lambda_cap) * lc + float(args.lambda_reg) * lr_
                    if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                        lpv = mse(pv_p, batch.y_pv.view(batch.num_graphs, -1))
                        lt = lt + float(args.lambda_pv) * lpv
                        val_pv_sum += float(lpv.item()) * batch.num_graphs
                val_tot += float(lt.item()) * batch.num_graphs
                val_v += float(lv.item()) * batch.num_graphs
                val_c_sum += float(lc.item()) * batch.num_graphs
                val_r_sum += float(lr_.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
                v_flat = v_n.view(batch.num_graphs, -1)
                pred_ri = (v_flat * y_std_d + y_mean_d).view(batch.num_graphs, n_nodes, 2)
                true_ri = yb.view(batch.num_graphs, n_nodes, 2)
                pred_mag = torch.sqrt(pred_ri[..., 0] * pred_ri[..., 0] + pred_ri[..., 1] * pred_ri[..., 1] + 1e-12)
                true_mag = torch.sqrt(true_ri[..., 0] * true_ri[..., 0] + true_ri[..., 1] * true_ri[..., 1] + 1e-12)
                err = pred_mag - true_mag
                val_sum_true += true_mag.sum(dim=0)
                val_sum_true2 += (true_mag * true_mag).sum(dim=0)
                val_sum_se += (err * err).sum(dim=0)
                val_sum_worst += float(err.abs().max(dim=1).values.sum().item())
        val_tot /= max(nv, 1)
        val_v /= max(nv, 1)
        val_c = val_c_sum / max(nv, 1)
        val_r = val_r_sum / max(nv, 1)
        val_pv = val_pv_sum / max(nv, 1) if n_pv_aux > 0 else float("nan")
        true_mean = val_sum_true / max(nv, 1)
        var_true = val_sum_true2 / max(nv, 1) - true_mean * true_mean
        mse_node = val_sum_se / max(nv, 1)
        r2_node = 1.0 - mse_node / var_true.clamp_min(1e-8)
        val_r2_mean = float(r2_node.mean().item())
        val_r2_min = float(r2_node.min().item())
        val_worst_node_mae = val_sum_worst / max(nv, 1)
        train_v = train_v_sum / max(train_n, 1)
        train_c = train_c_sum / max(train_n, 1)
        train_r = train_r_sum / max(train_n, 1)
        train_pv = train_pv_sum / max(train_n, 1) if n_pv_aux > 0 else float("nan")
        train_tot = train_loss_sum / max(train_n, 1)
        sch.step(val_tot)
        crit = val_tot if args.early_stop_on == "total" else val_v
        if crit < best_val:
            best_val = crit
            best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % max(1, int(args.log_every)) == 0:
            _log = (
                f"[da_gps] epoch {ep:4d}/{args.epochs} "
                f"| train_tot={train_tot:.4f} train_volt={train_v:.4f} train_cap={train_c:.4f} train_reg={train_r:.4f}"
            )
            if n_pv_aux > 0:
                _log += f" train_meta_aux={train_pv:.4f} val_meta_aux={val_pv:.4f}"
            _log += (
                f" | val_tot={val_tot:.4f} val_volt={val_v:.4f} val_cap={val_c:.4f} val_reg={val_r:.4f} "
                f"| val_r2_mean={val_r2_mean:.4f} val_r2_min={val_r2_min:.4f} val_worst_mae={val_worst_node_mae:.4f} "
                f"| best={best_val:.4f}"
            )
            print(_log, flush=True)
        _ce = int(args.checkpoint_every)
        if _ce > 0 and ep % _ce == 0:
            _ck = out_dir / "training_last.pt"
            _save_periodic_training_checkpoint(
                _ck, base_model, opt, sch, scaler, epoch=ep, bad=bad, best_val=best_val, best_state=best_state
            )
            print(f"  periodic checkpoint -> {_ck}", flush=True)
        if bad >= args.patience:
            print(f"[da_gps] early stop at epoch {ep}", flush=True)
            if int(args.checkpoint_every) > 0:
                _ck = out_dir / "training_last.pt"
                _save_periodic_training_checkpoint(
                    _ck, base_model, opt, sch, scaler, epoch=ep, bad=bad, best_val=best_val, best_state=best_state
                )
                print(f"  periodic checkpoint (early stop) -> {_ck}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        base_model.load_state_dict(best_state)

    met = evaluate(
        model,
        dl_te,
        device,
        y_mean_d,
        y_std_d,
        reg_mean_d,
        reg_std_d,
        use_amp=use_amp,
        pv_mean=pv_mean_d,
        pv_std=pv_std_d,
    )
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
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "per_node_heads": bool(args.per_node_heads),
            "per_device_cap_head": bool(args.per_device_cap_head),
            "per_device_reg_head": bool(args.per_device_reg_head),
            "n_pv_aux": int(n_pv_aux),
            "pv_target_cols": list(pv_aux_cols) if n_pv_aux > 0 else [],
            "meta_aux_target_cols": list(pv_aux_cols) if n_pv_aux > 0 else [],
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
    _pv_n = met.get("pv_mse_normalized", float("nan"))
    _pv_raw = met.get("pv_mse_raw", float("nan"))
    _pv_tail = f"  meta_aux_MSE(nrm)={_pv_n:.6f}  meta_aux_MSE(raw)={_pv_raw:.6f}" if n_pv_aux > 0 else ""
    print(
        f"Test |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"cap_BCE={met['cap_bce']:.6f}  reg_MSE(pu)={met['reg_mse_tap_pu']:.6f}{_pv_tail}  time={train_seconds:.1f}s",
        flush=True,
    )
    print(f"Saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
