"""
Search 5 heterogeneous-GNN architectures for the MV anchor graph (8500 dailyagg).

Dataset layout (default):
  datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset/
    nodes/  — 4 CSVs (regulator up/down, cap, load transformer)
    edges/  — catalog + line attrs + regulator tap features

Graph:
  - Four PyG node types: upstream, downstream, capacitor, load (disjoint assignment:
    load > capacitor > downstream > upstream when a bus appears in multiple CSVs).
  - Buses that appear in more than one of the four node CSVs are left in the graph but
    excluded from **training** supervision (train_mask False); val supervision unchanged.
  - Two edge relations: line (R_full, X_full), reg (reg_tap_pu) with per-(src,rel,dst) message
    modules (HeteroConv) and per-node-type update MLPs after each layer.
  - Node raw inputs are type-specific: cap — q_capacitor_bank; load — p_load_kw, q_load_kvar;
    upstream/downstream — electrical_distance_ohm only (see TYPE_FEAT_COLS).
  - Topology is static from hetero_mv_edge_catalog.csv; per-sample node features / reg taps vary.

Target: vmag_pu only. Use --target-node-types (e.g. `load`) to supervise all nodes in those storages,
or --target-nodes / --target-nodes-file for named buses. Empty = all labeled nodes on all types.
Training mask excludes chosen nodes from train loss (--exclude-train-nodes).

Requires: torch, torch_geometric, pandas, numpy.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GINEConv, HeteroConv, SAGEConv
except ImportError as e:
    raise SystemExit("Install torch_geometric: pip install torch_geometric") from e

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

DEFAULT_DATASET = (
    REPO / "datasets_gnn2" / "loadtype_8500_dailyagg" / "Heterogenous GNN dataset"
)

# Filenames only: joined with dataset_dir / "nodes" in run_search (not nodes/nodes/...).
NODE_FILES = {
    "upstream": "hetero_mv_nodes_regulator_upstream.csv",
    "downstream": "hetero_mv_nodes_regulator_downstream.csv",
    "capacitor": "hetero_mv_nodes_capacitor_related.csv",
    "load": "hetero_mv_nodes_load_transformer.csv",
}

# Canonical order for ModuleDict / iteration
NODE_TYPES: tuple[str, ...] = ("upstream", "downstream", "capacitor", "load")

# Disjoint role assignment: first match wins (matches typical cap/load overlap handling).
TYPE_PRIORITY: tuple[str, ...] = ("load", "capacitor", "downstream", "upstream")

FEAT_COLS = [
    "electrical_distance_ohm",
    "p_load_kw",
    "q_load_kvar",
    "q_capacitor_bank",
]
FEAT_COL_INDEX: dict[str, int] = {c: i for i, c in enumerate(FEAT_COLS)}

# Raw model inputs per PyG node type (subset of FEAT_COLS; CSV accumulation still loads all four).
TYPE_FEAT_COLS: dict[str, tuple[str, ...]] = {
    "upstream": ("electrical_distance_ohm",),
    "downstream": ("electrical_distance_ohm",),
    "capacitor": ("q_capacitor_bank",),
    "load": ("p_load_kw", "q_load_kvar"),
}
IN_DIMS: dict[str, int] = {t: len(TYPE_FEAT_COLS[t]) for t in NODE_TYPES}

SEARCH_ROOT = REPO / "gnn2_architecture_search"
DATA_FRAC_DEFAULT = 1.0 / 3.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_node_idx_master(path: Path) -> dict[str, int]:
    df = pd.read_csv(path, usecols=["node", "node_idx"])
    out: dict[str, int] = {}
    for _, r in df.iterrows():
        k = str(r["node"]).strip().lower()
        out[k] = int(r["node_idx"])
    return out


def _collect_global_node_indices(
    catalog: pd.DataFrame,
    name_to_gidx: dict[str, int],
    extra_node_names: set[str],
) -> list[int]:
    s: set[int] = set()
    for _, row in catalog.iterrows():
        ui, vi = row["u_idx"], row["v_idx"]
        if pd.isna(ui) or pd.isna(vi):
            continue
        s.add(int(ui))
        s.add(int(vi))
    for nm in extra_node_names:
        k = nm.strip().lower()
        if k in name_to_gidx:
            s.add(name_to_gidx[k])
    return sorted(s)


def _membership_by_csv(nodes_dir: Path) -> dict[str, set[int]]:
    """Global node_idx sets appearing in each CSV (all samples)."""
    use = ["node_idx", "node"]
    out: dict[str, set[int]] = {t: set() for t in NODE_TYPES}
    for kind, rel in NODE_FILES.items():
        path = nodes_dir / rel
        if kind == "load":
            for chunk in pd.read_csv(path, usecols=lambda c: c in use, chunksize=400_000):
                for _, r in chunk.iterrows():
                    g = int(float(r["node_idx"])) if pd.notna(r["node_idx"]) else None
                    if g is None:
                        continue
                    out[kind].add(int(g))
        else:
            df = pd.read_csv(path, usecols=lambda c: c in use)
            for _, r in df.iterrows():
                g = int(float(r["node_idx"])) if pd.notna(r["node_idx"]) else None
                if g is None:
                    continue
                out[kind].add(int(g))
    return out


def _assign_disjoint_type(gidx: int, membership: dict[str, set[int]]) -> str | None:
    for t in TYPE_PRIORITY:
        if gidx in membership[t]:
            return t
    return None


def _globals_in_multiple_node_csvs(membership: dict[str, set[int]]) -> set[int]:
    """Global node_idx values listed in two or more of the four node CSVs (union over rows)."""
    cnt: dict[int, int] = defaultdict(int)
    for t in NODE_TYPES:
        for g in membership[t]:
            cnt[g] += 1
    return {g for g, c in cnt.items() if c > 1}


def _build_typed_topology(
    catalog: pd.DataFrame,
    line_attr: pd.DataFrame,
    g_list: list[int],
    membership: dict[str, set[int]],
) -> tuple[
    dict[str, dict[int, int]],
    dict[int, str],
    dict[str, int],
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor],
]:
    """
    disjoint g -> type, local indices per type, edge_index_dict, edge_attr_dict (line 2D, reg 1D).
    """
    gset = set(g_list)
    g2l: dict[str, dict[int, int]] = {t: {} for t in NODE_TYPES}
    global_type: dict[int, str] = {}
    orphans: list[int] = []
    for g in g_list:
        tt = _assign_disjoint_type(g, membership)
        if tt is None:
            orphans.append(g)
            continue
        global_type[g] = tt
        loc = len(g2l[tt])
        g2l[tt][g] = loc

    # attach orphans as load (MV buses on edges but missing from all four CSVs)
    for g in orphans:
        tt = "load"
        global_type[g] = tt
        g2l[tt][g] = len(g2l[tt])

    counts = {t: len(g2l[t]) for t in NODE_TYPES}

    leid = line_attr.set_index("edge_id")
    ei: dict[tuple[str, str, str], list[list[int]]] = defaultdict(lambda: [[], []])
    ea: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)

    for _, row in catalog.iterrows():
        eid = int(row["edge_id"])
        u = int(row["u_idx"])
        v = int(row["v_idx"])
        if u not in gset or v not in gset:
            continue
        tu = global_type[u]
        tv = global_type[v]
        lu, lv = g2l[tu][u], g2l[tv][v]
        et = str(row["edge_type"]).strip().lower()
        if et == "line":
            rrow = leid.loc[eid]
            rx = float(rrow["R_full"])
            xx = float(rrow["X_full"])
            kf = (tu, "line", tv)
            ei[kf][0].append(lu)
            ei[kf][1].append(lv)
            ea[kf].append([rx, xx])
            kr = (tv, "line", tu)
            ei[kr][0].append(lv)
            ei[kr][1].append(lu)
            ea[kr].append([rx, xx])
        elif et == "regulator":
            kf = (tu, "reg", tv)
            ei[kf][0].append(lu)
            ei[kf][1].append(lv)
            ea[kf].append([0.0])  # placeholder; filled per-sample below pattern
            kr = (tv, "reg", tu)
            ei[kr][0].append(lv)
            ei[kr][1].append(lu)
            ea[kr].append([0.0])

    # Regulator taps are per-sample: keep edge_index fixed, edge_attr placeholder zeros here;
    # forward pass overwrites reg edge_attr from reg_attr tensors per key.
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    for key, ij in ei.items():
        if not ij[0]:
            continue
        edge_index_dict[key] = torch.tensor(ij, dtype=torch.long)
        arr = np.array(ea[key], dtype=np.float32)
        edge_attr_dict[key] = torch.from_numpy(arr)

    return g2l, global_type, counts, edge_index_dict, edge_attr_dict


def _reg_attr_dict_per_keys(
    sid: int,
    catalog: pd.DataFrame,
    gset: set[int],
    global_type: dict[int, str],
    g2l: dict[str, dict[int, int]],
    reg_tap_map: dict[tuple[int, int], float],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    device: torch.device,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Build [E,1] reg edge_attr per (s,reg,t) key matching edge_index_dict row order."""
    out: dict[tuple[str, str, str], torch.Tensor] = {}
    # count edges per key in same order as catalog scan
    buffers: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for _, row in catalog.iterrows():
        if str(row["edge_type"]).strip().lower() != "regulator":
            continue
        eid = int(row["edge_id"])
        u = int(row["u_idx"])
        v = int(row["v_idx"])
        if u not in gset or v not in gset:
            continue
        tu, tv = global_type[u], global_type[v]
        tap = float(reg_tap_map.get((sid, eid), 0.0))
        kf = (tu, "reg", tv)
        if kf in edge_index_dict:
            buffers[kf].append(tap)
        kr = (tv, "reg", tu)
        if kr in edge_index_dict:
            buffers[kr].append(tap)

    for key, tbuf in buffers.items():
        if key not in edge_index_dict:
            continue
        e = edge_index_dict[key].shape[1]
        if len(tbuf) != e:
            # align by recomputing from catalog in lockstep if mismatch (should not happen)
            tbuf = (tbuf + [0.0] * e)[:e]
        out[key] = torch.tensor(tbuf, dtype=torch.float32, device=device).view(-1, 1)
    return out


def _load_reg_taps(reg_csv: Path) -> dict[tuple[int, int], float]:
    df = pd.read_csv(reg_csv, usecols=["sample_id", "edge_id", "reg_tap_pu"])
    out: dict[tuple[int, int], float] = {}
    for r in df.itertuples(index=False):
        sid = int(float(r.sample_id))
        eid = int(r.edge_id)
        out[(sid, eid)] = float(r.reg_tap_pu)
    return out


def _target_selection_typed_masks(
    name_to_gidx: dict[str, int],
    g2l: dict[str, dict[int, int]],
    counts: dict[str, int],
    target_names: frozenset[str] | None,
    restrict_types: frozenset[str] | None,
) -> dict[str, np.ndarray]:
    """Per-type boolean mask: True where vmag loss allowed (before exclude-train)."""
    masks = {t: np.zeros(counts[t], dtype=bool) for t in NODE_TYPES}
    if restrict_types is not None:
        for t in NODE_TYPES:
            if t in restrict_types:
                masks[t][:] = True
    else:
        for t in NODE_TYPES:
            masks[t][:] = True

    if not target_names:
        return masks

    name_hit = {t: np.zeros(counts[t], dtype=bool) for t in NODE_TYPES}
    missing: list[str] = []
    for raw in target_names:
        k = raw.strip().lower()
        if not k:
            continue
        g = name_to_gidx.get(k)
        if g is None:
            missing.append(raw.strip())
            continue
        placed = False
        for t in NODE_TYPES:
            if g in g2l[t]:
                name_hit[t][g2l[t][g]] = True
                placed = True
                break
        if not placed:
            missing.append(raw.strip())
    if missing:
        head = missing[:12]
        more = "" if len(missing) <= 12 else f" (+{len(missing) - 12} more)"
        print(f"[hetero_mv_search] warning: target nodes not in graph/index ({len(missing)}): {head}{more}")

    for t in NODE_TYPES:
        masks[t] &= name_hit[t]
    return masks


def _norm_sid(v: Any) -> int:
    try:
        x = float(v)
        return int(x) if x == int(x) else int(x)
    except (TypeError, ValueError):
        return int(v)


def _accumulate_node_rows(
    df: pd.DataFrame,
    kind: str,
    per_sample: dict[int, dict[int, dict[str, Any]]],
    name_to_gidx: dict[str, int],
) -> None:
    for r in df.itertuples(index=False):
        sid = _norm_sid(r.sample_id)
        name = str(r.node).strip()
        gidx = int(float(r.node_idx)) if not pd.isna(r.node_idx) else name_to_gidx.get(name.lower())
        if gidx is None or (isinstance(gidx, float) and np.isnan(gidx)):
            gidx = name_to_gidx.get(name.lower())
        if gidx is None:
            continue
        gidx = int(gidx)
        bucket = per_sample[sid].setdefault(
            gidx,
            {
                "by_kind": {},
                "vmag": np.nan,
            },
        )
        feats = np.zeros(len(FEAT_COLS), dtype=np.float64)
        for i, c in enumerate(FEAT_COLS):
            val = getattr(r, c)
            if pd.notna(val):
                feats[i] = float(val)
        bucket["by_kind"][kind] = feats
        vm = getattr(r, "vmag_pu")
        if pd.notna(vm):
            bucket["vmag"] = float(vm)


def _accumulate_node_csv(
    path: Path,
    kind: str,
    per_sample: dict[int, dict[int, dict[str, Any]]],
    name_to_gidx: dict[str, int],
) -> None:
    use = ["sample_id", "node", "node_idx"] + FEAT_COLS + ["vmag_pu"]
    if kind == "load":
        for chunk in pd.read_csv(path, usecols=lambda c: c in use, chunksize=400_000):
            _accumulate_node_rows(chunk, kind, per_sample, name_to_gidx)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in use)
        _accumulate_node_rows(df, kind, per_sample, name_to_gidx)


def _typed_feat_vector(blob: dict[str, Any], node_type: str) -> np.ndarray:
    """Slice TYPE_FEAT_COLS[node_type] from the best available CSV row for this bus."""
    cols = TYPE_FEAT_COLS[node_type]
    out = np.zeros(len(cols), dtype=np.float32)
    bk = blob.get("by_kind", {})
    src_kind: str | None = None
    if node_type in bk:
        src_kind = node_type
    else:
        for k in TYPE_PRIORITY:
            if k in bk:
                src_kind = k
                break
    if src_kind is None:
        return out
    full = np.asarray(bk[src_kind], dtype=np.float64)
    for i, c in enumerate(cols):
        out[i] = float(full[FEAT_COL_INDEX[c]])
    return out


def _make_hetero_conv_sage(
    edge_index_dict: dict,
    d: int,
    hidden: int,
) -> HeteroConv:
    mods = {key: SAGEConv((d, d), hidden, aggr="mean") for key in edge_index_dict}
    return HeteroConv(mods, aggr="sum")


def _make_hetero_conv_gat(
    edge_index_dict: dict,
    d: int,
    h_per: int,
    heads: int,
    dropout: float,
) -> HeteroConv:
    mods = {}
    for key in edge_index_dict:
        mods[key] = GATConv(d, h_per, heads=heads, concat=True, dropout=dropout)
    return HeteroConv(mods, aggr="sum")


def _make_hetero_conv_gine(
    edge_index_dict: dict,
    d: int,
    hidden: int,
) -> HeteroConv:
    mods = {}
    for key in edge_index_dict:
        _s, rel, _t = key
        mlp = nn.Sequential(
            nn.Linear(d, 2 * hidden),
            nn.ReLU(),
            nn.Linear(2 * hidden, hidden),
        )
        edim = 2 if rel == "line" else 1
        mods[key] = GINEConv(mlp, edge_dim=edim, train_eps=True)
    return HeteroConv(mods, aggr="sum")


class PerTypeUpdate(nn.Module):
    def __init__(self, node_types: tuple[str, ...], dim: int, use_norm: bool, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.node_types = node_types
        self.linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for t in node_types:
            self.linears[t] = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
            self.norms[t] = nn.LayerNorm(dim) if use_norm else nn.Identity()

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for t in self.node_types:
            if t not in x_dict:
                continue
            h = self.norms[t](x_dict[t])
            h = self.linears[t](h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out[t] = h
        return out


class HeteroTypedSAGE(nn.Module):
    def __init__(
        self,
        node_types: tuple[str, ...],
        in_dims: dict[str, int],
        hidden: int,
        num_layers: int,
        dropout: float,
        use_norm: bool,
        edge_index_dict: dict,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout
        self.embed = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden) for t in node_types})
        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        d = hidden
        for _ in range(num_layers):
            self.convs.append(_make_hetero_conv_sage(edge_index_dict, d, d))
            self.updates.append(PerTypeUpdate(node_types, d, use_norm, dropout))
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in node_types})

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        h = {t: self.embed[t](x_dict[t]) for t in self.node_types}
        for conv, upd in zip(self.convs, self.updates, strict=True):
            h = conv(h, edge_index_dict)
            h = {t: F.relu(h[t]) for t in self.node_types}
            h = upd(h)
        return {t: self.heads[t](h[t]).squeeze(-1) for t in self.node_types}


class HeteroTypedGAT(nn.Module):
    def __init__(
        self,
        node_types: tuple[str, ...],
        in_dims: dict[str, int],
        hidden_total: int,
        heads: int,
        num_layers: int,
        dropout: float,
        edge_index_dict: dict,
    ) -> None:
        super().__init__()
        assert hidden_total % heads == 0
        h_per = hidden_total // heads
        self.node_types = node_types
        self.dropout = dropout
        self.embed = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden_total) for t in node_types})
        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        d = hidden_total
        for lay in range(num_layers):
            self.convs.append(_make_hetero_conv_gat(edge_index_dict, d, h_per, heads, dropout))
            self.updates.append(PerTypeUpdate(node_types, d, False, dropout))
            d = h_per * heads
        self.heads = nn.ModuleDict({t: nn.Linear(d, 1) for t in node_types})

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        h = {t: self.embed[t](x_dict[t]) for t in self.node_types}
        for conv, upd in zip(self.convs, self.updates, strict=True):
            h = conv(h, edge_index_dict)
            h = {t: F.elu(h[t]) for t in self.node_types}
            h = upd(h)
        return {t: self.heads[t](h[t]).squeeze(-1) for t in self.node_types}


class HeteroTypedGINE(nn.Module):
    def __init__(
        self,
        node_types: tuple[str, ...],
        in_dims: dict[str, int],
        hidden: int,
        num_layers: int,
        dropout: float,
        edge_index_dict: dict,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout
        self.embed = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden) for t in node_types})
        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        d = hidden
        for _ in range(num_layers):
            self.convs.append(_make_hetero_conv_gine(edge_index_dict, d, d))
            self.updates.append(PerTypeUpdate(node_types, d, False, dropout))
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in node_types})

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h = {t: self.embed[t](x_dict[t]) for t in self.node_types}
        for conv, upd in zip(self.convs, self.updates, strict=True):
            h = conv(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
            h = {t: F.relu(h[t]) for t in self.node_types}
            h = upd(h)
        return {t: self.heads[t](h[t]).squeeze(-1) for t in self.node_types}


def _typed_supervised_loss(
    pred: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    train_mask: dict[str, torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for t in NODE_TYPES:
        m = train_mask[t]
        if m.sum() == 0:
            continue
        parts.append(F.mse_loss(pred[t][m], y[t][m]))
    if not parts:
        return torch.tensor(0.0, device=next(iter(pred.values())).device)
    return torch.stack(parts).mean()


def run_search(
    dataset_dir: Path,
    node_index_master: Path,
    data_frac: float,
    seed: int,
    epochs: int,
    patience: int,
    exclude_nodes: frozenset[str],
    max_samples: int | None,
    target_nodes: frozenset[str] | None,
    target_node_types: frozenset[str] | None,
    log_every: int,
) -> Path:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges_dir = dataset_dir / "edges"
    nodes_dir = dataset_dir / "nodes"

    print("[hetero_mv_search] reading edge catalog + line attrs...", flush=True)
    catalog = pd.read_csv(edges_dir / "hetero_mv_edge_catalog.csv")
    line_attr = pd.read_csv(edges_dir / "hetero_mv_line_edge_attr.csv")
    reg_csv = edges_dir / "hetero_mv_regulator_edge_features.csv"

    name_to_gidx = _read_node_idx_master(node_index_master)
    extra_names: set[str] = set()
    print("[hetero_mv_search] scanning node CSVs for bus names (load file can take several minutes)...", flush=True)
    for fn in NODE_FILES.values():
        df = pd.read_csv(nodes_dir / fn, usecols=["node"])
        for n in df["node"].astype(str):
            extra_names.add(n.strip())

    g_list = _collect_global_node_indices(catalog, name_to_gidx, extra_names)
    gset = set(g_list)
    print("[hetero_mv_search] membership scan (full pass over load CSV in chunks)...", flush=True)
    membership = _membership_by_csv(nodes_dir)
    print("[hetero_mv_search] building typed topology + tensors...", flush=True)
    g2l, global_type, counts, edge_index_dict_cpu, line_edge_attr_dict_cpu = _build_typed_topology(
        catalog, line_attr, g_list, membership
    )

    multi_csv_gidx = _globals_in_multiple_node_csvs(membership)
    multi_csv_in_graph = sorted(g for g in multi_csv_gidx if g in global_type)

    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict_cpu.items()}
    base_line_attr = {k: v.to(device) for k, v in line_edge_attr_dict_cpu.items()}

    print("[hetero_mv_search] loading regulator tap table...", flush=True)
    reg_tap_map = _load_reg_taps(reg_csv)

    exclude_typed: tuple[str, int] | None = None
    for ex in exclude_nodes:
        k = ex.strip().lower()
        if k not in name_to_gidx:
            continue
        gi = name_to_gidx[k]
        for t in NODE_TYPES:
            if gi in g2l[t]:
                exclude_typed = (t, g2l[t][gi])
                break

    all_sids = sorted(
        pd.read_csv(nodes_dir / NODE_FILES["upstream"], usecols=["sample_id"])["sample_id"].map(_norm_sid).unique().tolist()
    )
    if max_samples is not None:
        all_sids = all_sids[:max_samples]
    n_pool = max(1, int(len(all_sids) * data_frac))
    pool = all_sids[:n_pool]
    random.shuffle(pool)
    n_val = max(1, int(0.2 * len(pool)))
    val_ids = sorted(pool[:n_val])
    train_ids = sorted(pool[n_val:])

    n_line_edges = sum(edge_index_dict[k].shape[1] for k in edge_index_dict if k[1] == "line") // 2
    n_reg_edges = sum(edge_index_dict[k].shape[1] for k in edge_index_dict if k[1] == "reg") // 2
    print(f"[hetero_mv_search] device={device} typed_nodes={counts} relations={len(edge_index_dict)}")
    print(f"  E_line(undir pairs)~{n_line_edges} E_reg~{n_reg_edges}")
    print(f"  samples total={len(all_sids)} pool(frac)={len(pool)} train={len(train_ids)} val={len(val_ids)}")
    if exclude_typed is not None:
        print(f"  exclude from train loss: type={exclude_typed[0]} local={exclude_typed[1]} ({exclude_nodes})")
    if multi_csv_in_graph:
        print(
            f"  exclude from train loss: {len(multi_csv_in_graph)} bus(es) in 2+ node CSVs "
            f"(global_idx sample: {multi_csv_in_graph[:8]}{'...' if len(multi_csv_in_graph) > 8 else ''})"
        )

    sel_typed = _target_selection_typed_masks(
        name_to_gidx, g2l, counts, target_nodes, target_node_types
    )
    if not any(sel_typed[t].any() for t in NODE_TYPES):
        raise ValueError(
            "No supervised targets. Check --target-node-types (storages may be empty), "
            "and --target-nodes / --target-nodes-file against gnn_node_index_master.csv."
        )
    if target_nodes:
        print(f"  vmag_pu loss restricted to {sum(int(sel_typed[t].sum()) for t in NODE_TYPES)} named bus slot(s)")
    elif target_node_types:
        nslots = sum(int(sel_typed[t].sum()) for t in NODE_TYPES)
        print(
            f"  vmag_pu loss restricted to storages {sorted(target_node_types)} "
            f"({nslots} node positions; loss only where labels exist)"
        )

    per_sample: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    print("[hetero_mv_search] accumulating per-sample node features (another full load-CSV pass)...", flush=True)
    for kind, rel in NODE_FILES.items():
        print(f"  ... reading {kind} CSV", flush=True)
        _accumulate_node_csv(nodes_dir / rel, kind, per_sample, name_to_gidx)

    print("[hetero_mv_search] building GPU cache for train/val samples...", flush=True)
    cache: dict[int, dict[str, Any]] = {}
    for sid in pool:
        x_d: dict[str, np.ndarray] = {}
        y_d: dict[str, np.ndarray] = {}
        for t in NODE_TYPES:
            nt = counts[t]
            x_d[t] = np.zeros((nt, IN_DIMS[t]), dtype=np.float32)
            y_d[t] = np.full(nt, np.nan, dtype=np.float32)

        raw = per_sample.get(sid, {})
        for gidx, blob in raw.items():
            if gidx not in global_type:
                continue
            tt = global_type[gidx]
            li = g2l[tt][gidx]
            x_d[tt][li] = _typed_feat_vector(blob, tt)
            vm = blob.get("vmag", np.nan)
            if not np.isnan(vm):
                y_d[tt][li] = float(vm)

        train_m: dict[str, np.ndarray] = {}
        val_m: dict[str, np.ndarray] = {}
        for t in NODE_TYPES:
            labeled = np.isfinite(y_d[t])
            train_m[t] = labeled & sel_typed[t]
            val_m[t] = labeled & sel_typed[t]

        if exclude_typed is not None:
            et, el = exclude_typed
            train_m[et][el] = False

        for g in multi_csv_gidx:
            if g not in global_type:
                continue
            tt = global_type[g]
            train_m[tt][g2l[tt][g]] = False

        reg_attr_d = _reg_attr_dict_per_keys(
            sid, catalog, gset, global_type, g2l, reg_tap_map, edge_index_dict, device
        )

        cache[sid] = {
            "x_dict": {t: torch.from_numpy(x_d[t]).to(device) for t in NODE_TYPES},
            "y_dict": {t: torch.from_numpy(y_d[t]).to(device) for t in NODE_TYPES},
            "train_mask": {t: torch.from_numpy(train_m[t]).to(device) for t in NODE_TYPES},
            "val_mask": {t: torch.from_numpy(val_m[t]).to(device) for t in NODE_TYPES},
            "reg_attr_dict": reg_attr_d,
        }

    results: list[dict[str, Any]] = []

    class GINEWrapper(nn.Module):
        def __init__(self, core: HeteroTypedGINE, base_line: dict) -> None:
            super().__init__()
            self.core = core
            self.line_keys: list[tuple[str, str, str]] = list(base_line.keys())
            for i, k in enumerate(self.line_keys):
                self.register_buffer(f"line_ea_{i}", base_line[k])

        def _line_edge_attr_dict(self) -> dict[tuple[str, str, str], torch.Tensor]:
            return {k: getattr(self, f"line_ea_{i}") for i, k in enumerate(self.line_keys)}

        def forward(self, b: dict[str, Any]) -> dict[str, torch.Tensor]:
            ead = {**self._line_edge_attr_dict(), **b["reg_attr_dict"]}
            return self.core(b["x_dict"], edge_index_dict, ead)

    candidates: list[tuple[str, nn.Module, bool]] = [
        ("hetero_sage_2x96", HeteroTypedSAGE(NODE_TYPES, IN_DIMS, 96, 2, 0.0, False, edge_index_dict).to(device), False),
        ("hetero_sage_4x64_ln_drop", HeteroTypedSAGE(NODE_TYPES, IN_DIMS, 64, 4, 0.15, True, edge_index_dict).to(device), False),
        ("hetero_gat_2L4h128", HeteroTypedGAT(NODE_TYPES, IN_DIMS, 128, 4, 2, 0.1, edge_index_dict).to(device), False),
        ("hetero_sage_3x112_wide", HeteroTypedSAGE(NODE_TYPES, IN_DIMS, 112, 3, 0.05, False, edge_index_dict).to(device), False),
        ("hetero_gine_3x80", HeteroTypedGINE(NODE_TYPES, IN_DIMS, 80, 3, 0.1, edge_index_dict).to(device), True),
    ]

    for name, model, use_gine in candidates:
        print(f"\n>>> Training {name} (gine_edge_attr={use_gine})")
        if use_gine:
            wrapped = GINEWrapper(model, base_line_attr).to(device)
        else:
            wrapped = model

        opt = torch.optim.Adam(wrapped.parameters(), lr=1e-3)
        best_val = float("inf")
        best_state = None
        bad = 0
        t0 = time.time()
        ep = 0
        for ep in range(epochs):
            wrapped.train()
            for sid in train_ids:
                b = cache[sid]
                opt.zero_grad()
                pred = wrapped(b) if use_gine else wrapped(b["x_dict"], edge_index_dict)
                loss = _typed_supervised_loss(pred, b["y_dict"], b["train_mask"])
                if loss.isnan().any():
                    continue
                if sum(int(b["train_mask"][t].sum().item()) for t in NODE_TYPES) == 0:
                    continue
                loss.backward()
                opt.step()
            wrapped.eval()
            val_se = 0.0
            val_n = 0
            with torch.no_grad():
                for sid in val_ids:
                    b = cache[sid]
                    pred = wrapped(b) if use_gine else wrapped(b["x_dict"], edge_index_dict)
                    for t in NODE_TYPES:
                        m = b["val_mask"][t]
                        if m.sum() == 0:
                            continue
                        err = pred[t][m] - b["y_dict"][t][m]
                        val_se += float((err ** 2).sum().item())
                        val_n += int(m.sum().item())
            val_rmse = (val_se / max(val_n, 1)) ** 0.5
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k, v in wrapped.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if log_every > 0 and ((ep + 1) % log_every == 0 or ep == 0):
                print(
                    f"    epoch {ep + 1:4d}/{epochs}  val_rmse={val_rmse:.6f}  "
                    f"best_val_rmse={best_val:.6f}  no_improve_streak={bad}/{patience}"
                )
            if bad >= patience:
                break
        if best_state:
            wrapped.load_state_dict(best_state)
        mae_sum = 0.0
        cnt = 0
        wrapped.eval()
        with torch.no_grad():
            for sid in val_ids:
                b = cache[sid]
                pred = wrapped(b) if use_gine else wrapped(b["x_dict"], edge_index_dict)
                for t in NODE_TYPES:
                    m = b["val_mask"][t]
                    if m.sum() == 0:
                        continue
                    mae_sum += float((pred[t][m] - b["y_dict"][t][m]).abs().sum().item())
                    cnt += int(m.sum().item())
        mae = mae_sum / max(cnt, 1)
        results.append(
            {
                "cfg_name": name,
                "best_val_rmse": best_val,
                "val_mae": mae,
                "epochs_ran": ep + 1,
                "sec": time.time() - t0,
                "state": best_state,
                "use_gine": use_gine,
            }
        )

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("state", "use_gine")} for r in results])
    df = df.sort_values(["best_val_rmse", "val_mae", "cfg_name"]).reset_index(drop=True)

    out_dir = SEARCH_ROOT / "hetero_mv_8500"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "search_results.csv"
    df.to_csv(csv_path, index=False)

    best_name = str(df.iloc[0]["cfg_name"]) if len(df) else None
    best = next((r for r in results if r["cfg_name"] == best_name), None) if best_name else None
    if best and best.get("state"):
        ckpt = out_dir / f"{best['cfg_name']}_best.pt"
        meta = {
            "schema": "hetero_typed_4node_2rel",
            "node_types": list(NODE_TYPES),
            "type_priority": list(TYPE_PRIORITY),
            "node_input_dims": {t: IN_DIMS[t] for t in NODE_TYPES},
            "type_feat_cols": {t: list(TYPE_FEAT_COLS[t]) for t in NODE_TYPES},
            "line_edge_attr_cols": ["R_full", "X_full"],
            "reg_edge_attr_cols": ["reg_tap_pu"],
            "counts": counts,
            "edge_index_keys": [list(k) for k in edge_index_dict.keys()],
            "exclude_train": {"type": exclude_typed[0], "local": exclude_typed[1]} if exclude_typed else None,
            "exclude_train_multi_csv_global_idx": multi_csv_in_graph,
            "target_nodes": sorted(target_nodes) if target_nodes else None,
            "target_node_types": sorted(target_node_types) if target_node_types else None,
        }
        torch.save(
            {
                "cfg_name": best["cfg_name"],
                "best_val_rmse": best["best_val_rmse"],
                "val_mae": best["val_mae"],
                "state_dict": best["state"],
                "node_input_dims": {t: IN_DIMS[t] for t in NODE_TYPES},
                "meta": meta,
            },
            ckpt,
        )

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "data_frac": data_frac,
        "target_nodes": sorted(target_nodes) if target_nodes else None,
        "target_node_types": sorted(target_node_types) if target_node_types else None,
        "results_csv": str(csv_path),
        "best_cfg": str(df.iloc[0]["cfg_name"]) if len(df) else None,
        "best_val_rmse": float(df.iloc[0]["best_val_rmse"]) if len(df) else None,
    }
    with open(out_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description="Search 5 hetero GNN architectures (MV 8500 bundle, typed nodes).")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--node-index",
        type=Path,
        default=REPO / "datasets_gnn2" / "loadtype_8500_dailyagg" / "gnn_node_index_master.csv",
    )
    p.add_argument("--data-frac", type=float, default=DATA_FRAC_DEFAULT, help="Fraction of samples to use (train+val pool)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print val_rmse every N epochs during each architecture run (0 to disable). Also prints after epoch 1.",
    )
    p.add_argument(
        "--exclude-train-nodes",
        type=str,
        default="l2823592.1",
        help="Comma-separated node names to drop from train mask only (empty to disable)",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of samples (debug)")
    p.add_argument(
        "--target-nodes",
        type=str,
        default="",
        help="Comma-separated node names. Intersected with masks (and with --target-node-types if set).",
    )
    p.add_argument(
        "--target-nodes-file",
        type=Path,
        default=None,
        help="Optional file: one node name per line (# comments ok). Merged with --target-nodes.",
    )
    p.add_argument(
        "--target-node-types",
        type=str,
        default="",
        help="Comma-separated storages: upstream,downstream,capacitor,load. "
        "Supervise vmag only on those node types (all buses in that storage). "
        "Empty = all types. Example: --target-node-types load for all ~load transformer MV buses.",
    )
    args = p.parse_args()
    ex: set[str] = set()
    if args.exclude_train_nodes.strip():
        ex = frozenset(x.strip() for x in args.exclude_train_nodes.split(",") if x.strip())
    else:
        ex = frozenset()

    tset: set[str] = set()
    if args.target_nodes.strip():
        tset.update(x.strip() for x in args.target_nodes.split(",") if x.strip())
    if args.target_nodes_file is not None:
        tf = args.target_nodes_file.resolve()
        if not tf.is_file():
            raise SystemExit(f"Missing --target-nodes-file: {tf}")
        for line in tf.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tset.add(s)
    target_frozen: frozenset[str] | None = frozenset(tset) if tset else None

    tnt: frozenset[str] | None = None
    if args.target_node_types.strip():
        tnt = frozenset(x.strip().lower() for x in args.target_node_types.split(",") if x.strip())
        bad = tnt - set(NODE_TYPES)
        if bad:
            raise SystemExit(f"Invalid --target-node-types {bad!r}. Allowed: {list(NODE_TYPES)}")

    log_every = max(0, int(args.log_every))

    run_search(
        args.dataset_dir.resolve(),
        args.node_index.resolve(),
        args.data_frac,
        args.seed,
        args.epochs,
        args.patience,
        frozenset(ex),
        args.max_samples,
        target_frozen,
        tnt,
        log_every,
    )


if __name__ == "__main__":
    main()
