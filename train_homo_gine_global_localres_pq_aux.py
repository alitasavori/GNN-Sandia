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
import math
import os
import random
import sys
import time
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


def _load_voltage_target_complex_ri(
    nodes_csv: Path,
    sample_ids: list[int],
    node_to_local: dict[str, int],
) -> torch.Tensor:
    """
    Build voltage targets as real/imag from vmag_pu + vang_deg.
    Returns tensor [S, N, 2] where last dim is [V_re, V_im].
    """
    sid_to_i = {_norm_sid(s): i for i, s in enumerate(sample_ids)}
    S = len(sample_ids)
    N = len(node_to_local)
    y_ri = np.zeros((S, N, 2), dtype=np.float32)
    usecols = ["sample_id", "node", "vmag_pu", "vang_deg"]
    for chunk in pd.read_csv(nodes_csv, usecols=usecols, chunksize=500_000):
        row_s = chunk["sample_id"].map(lambda v: sid_to_i.get(_norm_sid(v), -1)).to_numpy(dtype=np.int64)
        row_n = chunk["node"].map(lambda v: node_to_local.get(str(v).strip(), -1)).to_numpy(dtype=np.int64)
        valid = (row_s >= 0) & (row_n >= 0)
        if not np.any(valid):
            continue
        s = row_s[valid]
        n = row_n[valid]
        vmag = chunk.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
        vang_deg = chunk.loc[valid, "vang_deg"].to_numpy(dtype=np.float32)
        vang_rad = np.deg2rad(vang_deg)
        y_ri[s, n, 0] = vmag * np.cos(vang_rad)  # V_re
        y_ri[s, n, 1] = vmag * np.sin(vang_rad)  # V_im
    n_nonzero = int(np.count_nonzero(np.abs(y_ri) > 0.0))
    if n_nonzero == 0:
        raise RuntimeError(
            "complex_ri target tensor is all zeros; node mapping likely failed. "
            "Check that nodes CSV 'node' names match _load_nodes_pq_target ordering."
        )
    return torch.from_numpy(y_ri)


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
    def __init__(
        self,
        n_nodes: int,
        hidden: int,
        node_out_dim: int,
        voltage_out_components: int,
        dropout_global: float,
        dropout_aux: float,
        global_proj_dim: int,
        global_hidden_dim: int,
        global_gate_learnable: bool,
        global_gate_init_scale: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.voltage_out_components = int(voltage_out_components)
        self.global_residual_scale = 1.0
        self.global_gate_learnable_enabled = bool(global_gate_learnable)
        dg = float(dropout_global)
        da = float(dropout_aux)
        self.node_proj = nn.Linear(hidden, node_out_dim)
        self.local_head = nn.Linear(hidden, self.voltage_out_components)
        gdim = self.n_nodes * node_out_dim
        out_dim = self.n_nodes * self.voltage_out_components
        proj_dim = max(0, int(global_proj_dim))
        self.global_proj = nn.Linear(gdim, proj_dim) if proj_dim > 0 else None
        if self.global_proj is None:
            hidden_x = max(1, out_dim // 2)
            self.global_head = nn.Sequential(
                # Backward-compatible default global residual head.
                nn.Linear(gdim, out_dim),
                nn.ReLU(),
                nn.Dropout(dg),
                nn.Linear(out_dim, hidden_x),
                nn.ReLU(),
                nn.Dropout(dg),
                nn.Linear(hidden_x, hidden_x),
                nn.ReLU(),
                nn.Dropout(dg),
                nn.Linear(hidden_x, out_dim),
            )
        else:
            hidden_g = max(1, int(global_hidden_dim))
            self.global_head = nn.Sequential(
                nn.Linear(proj_dim, hidden_g),
                nn.ReLU(),
                nn.Dropout(dg),
                nn.Linear(hidden_g, hidden_g),
                nn.ReLU(),
                nn.Dropout(dg),
                nn.Linear(hidden_g, out_dim),
            )
        self.aux_proj = nn.Linear(gdim, hidden)
        self.aux_dropout = nn.Dropout(da)
        self.aux_reg_heads = nn.ModuleList()  # 12 heads, set by caller
        self.aux_cap_heads = nn.ModuleList()  # 10 heads, set by caller
        if self.global_gate_learnable_enabled:
            p0 = float(global_gate_init_scale)
            if p0 < 0.0 or p0 > 1.0:
                raise ValueError("--global_gate_init_scale must be in [0,1].")
            p0 = min(max(p0, 1e-4), 1.0 - 1e-4)
            logit0 = math.log(p0 / (1.0 - p0))
            self.global_gate_logit = nn.Parameter(torch.tensor(logit0, dtype=torch.float32))
        else:
            self.register_parameter("global_gate_logit", None)

    def set_global_residual_scale(self, scale: float) -> None:
        self.global_residual_scale = float(scale)

    def get_learned_global_gate(self) -> float | None:
        if self.global_gate_logit is None:
            return None
        return float(torch.sigmoid(self.global_gate_logit.detach()).item())

    def _readout(self, h: torch.Tensor, bvec: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local_head(h)
        z = self.node_proj(h)
        if bvec is None:
            local = local.view(1, self.n_nodes, self.voltage_out_components)
            z = z.view(1, self.n_nodes, -1)
        else:
            b = int(bvec.max().item()) + 1
            local = local.view(b, self.n_nodes, self.voltage_out_components)
            z = z.view(b, self.n_nodes, -1)
        g = z.reshape(z.size(0), -1)  # [B, gdim]
        g_in = self.global_proj(g) if self.global_proj is not None else g
        delta = self.global_head(g_in).view(local.size(0), self.n_nodes, self.voltage_out_components)
        gate = float(self.global_residual_scale)
        if self.global_gate_logit is not None:
            gate = gate * torch.sigmoid(self.global_gate_logit).to(dtype=delta.dtype, device=delta.device)
        delta = delta * gate
        v_pred = local + delta
        if self.voltage_out_components == 1:
            v_pred = v_pred.squeeze(-1)
        return v_pred, g

    def _aux_logits(self, g: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h = self.aux_dropout(F.relu(self.aux_proj(g)))
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
        voltage_out_components: int,
        dropout_trunk: float,
        dropout_global: float,
        dropout_aux: float,
        node_emb_dim: int,
        edge_emb_dim: int,
        global_proj_dim: int,
        global_hidden_dim: int,
        global_gate_learnable: bool,
        global_gate_init_scale: float,
        reg_nclasses: list[int],
        cap_nclasses: list[int],
    ):
        super().__init__(
            n_nodes=n_nodes,
            hidden=hidden,
            node_out_dim=node_out_dim,
            voltage_out_components=voltage_out_components,
            dropout_global=dropout_global,
            dropout_aux=dropout_aux,
            global_proj_dim=global_proj_dim,
            global_hidden_dim=global_hidden_dim,
            global_gate_learnable=global_gate_learnable,
            global_gate_init_scale=global_gate_init_scale,
        )
        self.n_nodes = n_nodes
        self.num_edges = num_edges
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(float(dropout_trunk))

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
        voltage_out_components: int,
        dropout_trunk: float,
        dropout_global: float,
        dropout_aux: float,
        node_emb_dim: int,
        global_proj_dim: int,
        global_hidden_dim: int,
        global_gate_learnable: bool,
        global_gate_init_scale: float,
        reg_nclasses: list[int],
        cap_nclasses: list[int],
    ):
        super().__init__(
            n_nodes=n_nodes,
            hidden=hidden,
            node_out_dim=node_out_dim,
            voltage_out_components=voltage_out_components,
            dropout_global=dropout_global,
            dropout_aux=dropout_aux,
            global_proj_dim=global_proj_dim,
            global_hidden_dim=global_hidden_dim,
            global_gate_learnable=global_gate_learnable,
            global_gate_init_scale=global_gate_init_scale,
        )
        self.n_nodes = n_nodes
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.dropout = nn.Dropout(float(dropout_trunk))
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


def _aux_lambda_scale(epoch_1based: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """
    Multiplier in [0, 1] for auxiliary loss weights (voltage-only when 0).

    - Epochs 1..warmup: 0 (no aux gradient).
    - Next ramp_epochs epochs: linear ramp 1/ramp, 2/ramp, ..., 1.
    - After that: 1.

    If warmup_epochs and ramp_epochs are both 0, returns 1 (full aux from epoch 1).
    """
    if warmup_epochs <= 0 and ramp_epochs <= 0:
        return 1.0
    if epoch_1based <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    t = epoch_1based - warmup_epochs
    if t > ramp_epochs:
        return 1.0
    return float(t) / float(ramp_epochs)


def _global_residual_scale(epoch_1based: int, warmup_epochs: int, ramp_epochs: int, start_scale: float) -> float:
    """
    Global residual scale in [start_scale, 1].
    """
    s0 = float(start_scale)
    if s0 < 0.0 or s0 > 1.0:
        raise ValueError("--global_gate_start_scale must be in [0,1].")
    if warmup_epochs <= 0 and ramp_epochs <= 0:
        return 1.0
    if epoch_1based <= warmup_epochs:
        return s0
    if ramp_epochs <= 0:
        return 1.0
    t = epoch_1based - warmup_epochs
    if t > ramp_epochs:
        return 1.0
    alpha = float(t) / float(ramp_epochs)
    return s0 + alpha * (1.0 - s0)


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
    aux_warmup_epochs: int,
    aux_ramp_epochs: int,
    global_gate_warmup_epochs: int,
    global_gate_ramp_epochs: int,
    global_gate_start_scale: float,
    y_mean_flat: torch.Tensor,
    y_std_flat: torch.Tensor,
    checkpoint_path: Path,
    log_every: int,
    lr_plateau_factor: float,
    lr_plateau_patience: int,
    lr_min: float,
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(lr_plateau_factor),
        patience=int(lr_plateau_patience),
        min_lr=float(lr_min),
    )
    mse = nn.MSELoss()
    best = float("inf")
    bad = 0
    y_mean_flat = y_mean_flat.to(device)
    y_std_flat = y_std_flat.to(device)

    def _flatten_voltage_tensor(t: torch.Tensor, n_graphs: int) -> torch.Tensor:
        # vmag mode: [B, N] -> [B, N]
        # complex_ri mode: [B, N, 2] -> [B, 2N]
        if t.dim() <= 2:
            return t.view(n_graphs, -1)
        return t.reshape(n_graphs, -1)

    for ep in range(1, epochs + 1):
        aux_scale = _aux_lambda_scale(ep, aux_warmup_epochs, aux_ramp_epochs)
        global_scale = _global_residual_scale(
            ep,
            global_gate_warmup_epochs,
            global_gate_ramp_epochs,
            global_gate_start_scale,
        )
        model_ref = getattr(model, "_orig_mod", model)
        if hasattr(model_ref, "set_global_residual_scale"):
            model_ref.set_global_residual_scale(global_scale)
        learned_global_gate = None
        if hasattr(model_ref, "get_learned_global_gate"):
            learned_global_gate = model_ref.get_learned_global_gate()
        lr_reg = float(lambda_reg) * aux_scale
        lr_cap = float(lambda_cap) * aux_scale
        model.train()
        tr_mae = tr_mse = tr_auxr = tr_auxc = 0.0
        ntr = 0
        for batch in train_loader:
            batch = batch.to(device)
            v_pred, reg_logits, cap_logits = model.forward_train(batch)
            yv = _flatten_voltage_tensor(batch.y, batch.num_graphs)
            v_pred_f = _flatten_voltage_tensor(v_pred, batch.num_graphs)
            yr = batch.y_reg.view(batch.num_graphs, -1).long()  # [B, 12]
            yc = batch.y_cap.view(batch.num_graphs, -1).long()  # [B, 10]
            yv_n = (yv - y_mean_flat) / y_std_flat
            v_pred_n = (v_pred_f - y_mean_flat) / y_std_flat
            lv = mse(v_pred_n, yv_n)
            lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
            loss = lv + lr_reg * lr_aux + lr_cap * lc_aux
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_mse += float(lv.item()) * batch.num_graphs
            tr_mae += float((v_pred_f - yv).abs().mean(dim=1).sum().item())
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
                yv = _flatten_voltage_tensor(batch.y, batch.num_graphs)
                v_pred_f = _flatten_voltage_tensor(v_pred, batch.num_graphs)
                yr = batch.y_reg.view(batch.num_graphs, -1).long()
                yc = batch.y_cap.view(batch.num_graphs, -1).long()
                yv_n = (yv - y_mean_flat) / y_std_flat
                v_pred_n = (v_pred_f - y_mean_flat) / y_std_flat
                lv = mse(v_pred_n, yv_n)
                lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
                va_mse += float(lv.item()) * batch.num_graphs
                va_mae += float((v_pred_f - yv).abs().mean(dim=1).sum().item())
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
            gate_dbg = (
                f"learned_gate={learned_global_gate:.4f} eff_global={(learned_global_gate * global_scale):.4f} "
                if learned_global_gate is not None
                else ""
            )
            print(
                f"Epoch {ep:4d} | aux_scale={aux_scale:.4f} global_scale={global_scale:.4f} {gate_dbg}"
                f"eff_λ_reg={lr_reg:.6f} eff_λ_cap={lr_cap:.6f} | "
                f"train_mae={tr_mae:.6f} train_mse={tr_mse:.6f} "
                f"aux_reg={tr_auxr:.4f} aux_cap={tr_auxc:.4f} | "
                f"val_mae={va_mae:.6f} val_mse={va_mse:.6f} aux_reg={va_auxr:.4f} aux_cap={va_auxc:.4f} | "
                f"best_val_mse={best:.6f} | patience {bad}/{patience}",
                flush=True,
            )
        if bad >= patience:
            print(f"Early stopping at epoch {ep}", flush=True)
            break
    return best


def _angle_diff_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
    d = pred_deg - true_deg
    return (d + 180.0) % 360.0 - 180.0


def _compute_voltage_target_norm(
    y: torch.Tensor,
    train_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-output normalization stats from train split only.
    Returns tensors with shape [1, N] for vmag, or [1, N, 2] for complex_ri.
    """
    y_train = y[train_idx]
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return y_mean, y_std


@torch.no_grad()
def eval_voltage_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    voltage_target_mode: str,
) -> dict[str, float]:
    model.eval()
    sum_abs_v = 0.0
    sum_sq_v = 0.0
    n_v = 0
    sum_abs_a = 0.0
    sum_sq_a = 0.0
    n_a = 0
    for batch in loader:
        batch = batch.to(device)
        # Model predicts in raw voltage space; normalization is used only inside the loss.
        pred = model(batch)
        y_true = batch.y
        if voltage_target_mode == "complex_ri":
            # pred: [B, N, 2]. PyG stacks targets as batch.y: [B*N, 2] (same node order as batch.x).
            b = int(batch.num_graphs)
            if pred.dim() != 3:
                raise RuntimeError(f"complex_ri eval expected pred [B,N,2], got shape {tuple(pred.shape)}")
            nloc = int(pred.size(1))
            y_true = y_true.view(b, nloc, 2)
            # pred, y_true: [B, N, 2] with [V_re, V_im]
            pr = pred[..., 0]
            pi = pred[..., 1]
            tr = y_true[..., 0]
            ti = y_true[..., 1]

            pm = torch.sqrt(pr * pr + pi * pi + 1e-12)
            tm = torch.sqrt(tr * tr + ti * ti + 1e-12)
            dv = pm - tm
            sum_abs_v += float(dv.abs().sum().item())
            sum_sq_v += float((dv * dv).sum().item())
            n_v += int(dv.numel())

            pa = torch.rad2deg(torch.atan2(pi, pr))
            ta = torch.rad2deg(torch.atan2(ti, tr))
            da = _angle_diff_deg(pa, ta)
            sum_abs_a += float(da.abs().sum().item())
            sum_sq_a += float((da * da).sum().item())
            n_a += int(da.numel())
        else:
            # vmag mode: pred, y_true are [B, N]
            dv = pred.view(batch.num_graphs, -1) - y_true.view(batch.num_graphs, -1)
            sum_abs_v += float(dv.abs().sum().item())
            sum_sq_v += float((dv * dv).sum().item())
            n_v += int(dv.numel())

    out = {
        "mae_vmag_pu": float(sum_abs_v / max(n_v, 1)),
        "rmse_vmag_pu": float((sum_sq_v / max(n_v, 1)) ** 0.5),
    }
    if n_a > 0:
        out["mae_angle_deg"] = float(sum_abs_a / n_a)
        out["rmse_angle_deg"] = float((sum_sq_a / n_a) ** 0.5)
    else:
        out["mae_angle_deg"] = float("nan")
        out["rmse_angle_deg"] = float("nan")
    return out


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
    p.add_argument(
        "--dropout_trunk",
        type=float,
        default=0.0,
        help="Dropout after each GNN layer (GINE/GCN). 0 = off.",
    )
    p.add_argument(
        "--dropout_global",
        type=float,
        default=0.0,
        help="Dropout inside the global ΔV MLP on concatenated node embeddings. 0 = off.",
    )
    p.add_argument(
        "--dropout_aux",
        type=float,
        default=0.0,
        help="Dropout on aux path (after aux_proj, before regulator/cap heads). 0 = off.",
    )
    p.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Force dropout_trunk, dropout_global, and dropout_aux to 0.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau: multiply LR by this factor when val plateaus (default 0.5, legacy).",
    )
    p.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=10,
        help="ReduceLROnPlateau: epochs with no val improvement before reducing LR.",
    )
    p.add_argument(
        "--lr_min",
        type=float,
        default=0.0,
        help="ReduceLROnPlateau: minimum LR (0 = PyTorch default, no floor).",
    )
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction; test gets remaining samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--node_emb_dim", type=int, default=0)
    p.add_argument("--edge_emb_dim", type=int, default=0)
    p.add_argument(
        "--global_proj_dim",
        type=int,
        default=0,
        help="If >0, add global bottleneck Linear(gdim->proj_dim) before global head. 0 keeps legacy head.",
    )
    p.add_argument(
        "--global_hidden_dim",
        type=int,
        default=1024,
        help="Hidden width used by bottlenecked global head (proj->hidden->hidden->out).",
    )
    p.add_argument(
        "--enforce_small_node_out_without_proj",
        action="store_true",
        help="If set, require node_out_dim <= --max_node_out_without_proj when global_proj_dim=0.",
    )
    p.add_argument(
        "--max_node_out_without_proj",
        type=int,
        default=4,
        help="Upper bound enforced only when --enforce_small_node_out_without_proj is set and no bottleneck is used.",
    )
    p.add_argument(
        "--warn_node_out_threshold",
        type=int,
        default=8,
        help="Emit warning when global_proj_dim=0 and node_out_dim >= this value.",
    )
    p.add_argument("--lambda_reg", type=float, default=0.2)
    p.add_argument("--lambda_cap", type=float, default=0.1)
    p.add_argument(
        "--aux_warmup_epochs",
        type=int,
        default=0,
        help="Train voltage only (aux loss weight 0) for this many epochs before applying aux.",
    )
    p.add_argument(
        "--aux_ramp_epochs",
        type=int,
        default=0,
        help="After warmup, linearly ramp aux weight multiplier from 0 to 1 over this many epochs. 0 = jump to full λ.",
    )
    p.add_argument(
        "--global_gate_warmup_epochs",
        type=int,
        default=0,
        help="Keep global residual scale at --global_gate_start_scale for this many epochs.",
    )
    p.add_argument(
        "--global_gate_ramp_epochs",
        type=int,
        default=0,
        help="Linearly ramp global residual scale to 1 after warmup. 0 = jump to 1.",
    )
    p.add_argument(
        "--global_gate_start_scale",
        type=float,
        default=1.0,
        help="Starting global residual scale in [0,1]. Default 1 keeps legacy behavior.",
    )
    p.add_argument(
        "--global_gate_learnable",
        action="store_true",
        help="Enable learnable global gate g=Sigmoid(logit), applied as effective_scale=schedule_scale*g.",
    )
    p.add_argument(
        "--global_gate_init_scale",
        type=float,
        default=1.0,
        help="Initial value for learnable global gate in [0,1] (used only with --global_gate_learnable).",
    )
    p.add_argument("--cache_tensor", type=str, default="")
    p.add_argument(
        "--voltage-target",
        type=str,
        choices=("vmag", "complex_ri"),
        default="vmag",
        help="Voltage training target: vmag (default) or complex_ri (MSE on V_re,V_im).",
    )
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Load existing best checkpoint from --out_dir (same naming as training) and only run val/test voltage metrics + write JSON. Requires matching data args and seed.",
    )
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
    if args.disable_dropout:
        dropout_trunk = dropout_global = dropout_aux = 0.0
    else:
        dropout_trunk = float(args.dropout_trunk)
        dropout_global = float(args.dropout_global)
        dropout_aux = float(args.dropout_aux)
    global_proj_dim = max(0, int(args.global_proj_dim))
    if global_proj_dim == 0 and int(args.node_out_dim) >= int(args.warn_node_out_threshold):
        print(
            f"Warning: node_out_dim={args.node_out_dim} without bottleneck can greatly increase global-head params. "
            "Consider --global_proj_dim 256 (or 384).",
            flush=True,
        )
    if args.enforce_small_node_out_without_proj and global_proj_dim == 0:
        if int(args.node_out_dim) > int(args.max_node_out_without_proj):
            raise ValueError(
                "node_out_dim exceeds --max_node_out_without_proj without bottleneck. "
                "Reduce --node_out_dim or set --global_proj_dim > 0."
            )
    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None

    node_to_local = None
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
    n_val = max(1, int(args.val_frac * n))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    if test_idx.size == 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    x, mean, std = _zscore_features_train(x, train_idx)
    torch.save({"mean": mean, "std": std, "feat_cols": ["p_load_kw", "q_load_kvar"]}, out_dir / "feature_norm_pq.pt")

    voltage_target_mode = str(args.voltage_target).strip().lower()
    if voltage_target_mode == "complex_ri":
        if node_to_local is None:
            _x_tmp, _y_tmp, _sample_ids_tmp, _node_order_tmp, node_to_local = _load_nodes_pq_target(nodes_path)
            del _x_tmp, _y_tmp, _sample_ids_tmp, _node_order_tmp
        yv = _load_voltage_target_complex_ri(nodes_path, sample_ids, node_to_local)
        print("Voltage target mode: complex_ri (training on V_re,V_im)", flush=True)
    else:
        print("Voltage target mode: vmag (training on |V|)", flush=True)

    aux = _load_aux_targets(meta_path, sample_ids)
    y_reg = [d["y_idx"] for d in aux["reg"]]
    y_cap = [d["y_idx"] for d in aux["cap"]]
    reg_nclasses = [len(d["classes"]) for d in aux["reg"]]
    cap_nclasses = [len(d["classes"]) for d in aux["cap"]]

    ds = AuxDataset(x, yv, y_reg, y_cap, edge_index, edge_attr)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())
    y_mean, y_std = _compute_voltage_target_norm(yv, train_idx)
    y_mean_flat = y_mean.reshape(1, -1)
    y_std_flat = y_std.reshape(1, -1)
    torch.save(
        {"y_mean": y_mean.cpu(), "y_std": y_std.cpu(), "voltage_target_mode": voltage_target_mode},
        out_dir / "voltage_target_norm.pt",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    nw = int(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    if args.model == "gine":
        model: nn.Module = HomoGINEGlobalLocalAux(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=n_edges,
            hidden=args.hidden,
            n_layers=args.layers,
            node_out_dim=args.node_out_dim,
            voltage_out_components=(2 if voltage_target_mode == "complex_ri" else 1),
            dropout_trunk=dropout_trunk,
            dropout_global=dropout_global,
            dropout_aux=dropout_aux,
            node_emb_dim=max(0, int(args.node_emb_dim)),
            edge_emb_dim=max(0, int(args.edge_emb_dim)),
            global_proj_dim=global_proj_dim,
            global_hidden_dim=max(1, int(args.global_hidden_dim)),
            global_gate_learnable=bool(args.global_gate_learnable),
            global_gate_init_scale=float(args.global_gate_init_scale),
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
            voltage_out_components=(2 if voltage_target_mode == "complex_ri" else 1),
            dropout_trunk=dropout_trunk,
            dropout_global=dropout_global,
            dropout_aux=dropout_aux,
            node_emb_dim=max(0, int(args.node_emb_dim)),
            global_proj_dim=global_proj_dim,
            global_hidden_dim=max(1, int(args.global_hidden_dim)),
            global_gate_learnable=bool(args.global_gate_learnable),
            global_gate_init_scale=float(args.global_gate_init_scale),
            reg_nclasses=reg_nclasses,
            cap_nclasses=cap_nclasses,
        )
    model = model.to(device)
    if args.model == "gcn" and int(args.edge_emb_dim) > 0:
        print("model=gcn ignores edge_emb_dim", flush=True)

    emb_tag = f"_ne{int(args.node_emb_dim)}_ee{int(args.edge_emb_dim)}"

    def _fmt_do(x: float) -> str:
        return f"{x:g}" if x > 0 else "0"

    do_tag = f"_dt{_fmt_do(dropout_trunk)}_dg{_fmt_do(dropout_global)}_da{_fmt_do(dropout_aux)}"
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
        f"aux λ (targets): reg={args.lambda_reg} cap={args.lambda_cap} | "
        f"warmup_epochs={args.aux_warmup_epochs} ramp_epochs={args.aux_ramp_epochs} | "
        f"global_gate: start={args.global_gate_start_scale} warmup={args.global_gate_warmup_epochs} "
        f"ramp={args.global_gate_ramp_epochs} learnable={bool(args.global_gate_learnable)} "
        f"init={args.global_gate_init_scale} | "
        f"global_proj_dim={global_proj_dim} global_hidden_dim={args.global_hidden_dim} | "
        f"DO trunk={dropout_trunk} global={dropout_global} aux={dropout_aux} | "
        f"node_emb={args.node_emb_dim} edge_emb={args.edge_emb_dim} | "
        f"lr_plateau: factor={args.lr_plateau_factor} patience={args.lr_plateau_patience} min={args.lr_min}",
        flush=True,
    )
    if args.skip_train:
        if not ckpt.is_file():
            raise FileNotFoundError(f"--skip_train requires checkpoint at {ckpt}")
        print(f"--skip_train: loading {ckpt}", flush=True)
        best_val_mse = float("nan")
        train_seconds = 0.0
    else:
        print("Starting training...", flush=True)
        t_train_start = time.perf_counter()

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
            aux_warmup_epochs=int(args.aux_warmup_epochs),
            aux_ramp_epochs=int(args.aux_ramp_epochs),
            global_gate_warmup_epochs=int(args.global_gate_warmup_epochs),
            global_gate_ramp_epochs=int(args.global_gate_ramp_epochs),
            global_gate_start_scale=float(args.global_gate_start_scale),
            y_mean_flat=y_mean_flat,
            y_std_flat=y_std_flat,
            checkpoint_path=ckpt,
            log_every=args.log_every,
            lr_plateau_factor=float(args.lr_plateau_factor),
            lr_plateau_patience=int(args.lr_plateau_patience),
            lr_min=float(args.lr_min),
        )
        train_seconds = float(time.perf_counter() - t_train_start)

    # Evaluate the best checkpoint on val/test with physical metrics.
    best_state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_state, strict=False)
    val_voltage_metrics = eval_voltage_metrics(model, val_loader, device, voltage_target_mode)
    test_voltage_metrics = eval_voltage_metrics(model, test_loader, device, voltage_target_mode)

    meta_out = {
        # This is the checkpointing objective (normalized target space), not physical pu^2.
        "best_val_mse_normalized": (None if args.skip_train else float(best_val_mse)),
        # Backward-compat alias for old consumers; same numeric value as normalized MSE.
        "best_val_mse_pu2": (None if args.skip_train else float(best_val_mse)),
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
        "global_proj_dim": int(global_proj_dim),
        "global_hidden_dim": int(args.global_hidden_dim),
        "global_gate_start_scale": float(args.global_gate_start_scale),
        "global_gate_warmup_epochs": int(args.global_gate_warmup_epochs),
        "global_gate_ramp_epochs": int(args.global_gate_ramp_epochs),
        "global_gate_learnable": bool(args.global_gate_learnable),
        "global_gate_init_scale": float(args.global_gate_init_scale),
        "dropout_trunk": float(dropout_trunk),
        "dropout_global": float(dropout_global),
        "dropout_aux": float(dropout_aux),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "lr_plateau_factor": float(args.lr_plateau_factor),
        "lr_plateau_patience": int(args.lr_plateau_patience),
        "lr_min": float(args.lr_min),
        "seed": int(args.seed),
        "lambda_reg": float(args.lambda_reg),
        "lambda_cap": float(args.lambda_cap),
        "aux_warmup_epochs": int(args.aux_warmup_epochs),
        "aux_ramp_epochs": int(args.aux_ramp_epochs),
        "aux_targets": {
            "reg": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["reg"]],
            "cap": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["cap"]],
        },
        "voltage_target_mode": voltage_target_mode,
        "voltage_target_normalization": {
            "enabled": True,
            "stats_path": str((out_dir / "voltage_target_norm.pt").resolve()),
        },
        "skip_train": bool(args.skip_train),
        "train_seconds": train_seconds,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "val_voltage_metrics": val_voltage_metrics,
        "test_voltage_metrics": test_voltage_metrics,
    }
    metrics_path = out_dir / "train_metrics_global_localres_aux.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print("Best val MSE (normalized target space):", "(skipped --skip_train)" if args.skip_train else best_val_mse, flush=True)
    print("Val voltage metrics:", val_voltage_metrics, flush=True)
    print("Test voltage metrics:", test_voltage_metrics, flush=True)
    print("Saved checkpoint:", ckpt, flush=True)
    print("Saved metrics JSON:", metrics_path, flush=True)


if __name__ == "__main__":
    main()

