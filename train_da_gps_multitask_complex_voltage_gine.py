"""
DA-GPS v2 (GINE local message passing): Perceiver-style latent tokens + cross-attention
+ standard GINEConv on the graph (replaces EdgeAttnMPNN),
multitask voltage + cap (BCE) + regulator (MSE or MAE on z-scored taps via ``--reg_loss``).

v2 alignment:
- Node inputs: dynamic columns from nodes CSV (default load P/Q) plus optional shared PE columns from a single master CSV.
- Tokens are pure learnable parameters (no host-node warm start).
- No effective-resistance attention bias.
- Aux targets are hardcoded in-script (old aux-trainer style).

Optional physics loss (``--loss_power_balance_weight`` > 0, strict — no silent physics fallbacks):
- After denormalizing predicted complex voltage (Re/Im head), penalize nodal P/Q balance residuals
  in **kW/kvar** with Huber aggregation (``--pf_huber_delta_kw``, default 10 kW) on selected nodes
  (slack/source excluded).
- Base ``Y`` from **line-only** ``R_full``/``X_full`` in ``--edge_catalog_csv`` (regulator transformer branches and
  cap shunts are excluded from the base stamp). Regulator branches from ``--pf_reg_edge_catalog`` are stamped with
  **predicted** tap ratios; capacitor banks add shunt ``j*B`` at mapped cap buses from **predicted** cap states
  (``--pf_detach_controls`` optionally stops gradients through controls). Caps are **not** added to ``Q_inj``.
- ``P_inj``/``Q_inj`` use **known** denormalized node features only (OpenDSS convention:
  ``P_inj = P_pv - P_load``, ``Q_inj = -Q_pv - Q_load``). Meta-aux PV predictions are supervised separately and
  are **not** used in the physics residual. Missing catalog/mapping/normalization → error.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import gc
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
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


def _device_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _cap_col_stem(col: str) -> str | None:
    m = re.match(r"^cap_(.+)_n_steps_on$", str(col).strip(), flags=re.IGNORECASE)
    return _device_stem(m.group(1)) if m else None


def _reg_col_stem(col: str) -> str | None:
    m = re.match(r"^reg_(.+)_tap_pu$", str(col).strip(), flags=re.IGNORECASE)
    return _device_stem(m.group(1)) if m else None


def _parse_capacitors_dss(dss_path: Path) -> dict[str, tuple[str, float]]:
    """``stem -> (bus1_node, kvar)`` from ``New Capacitor.NAME Bus1=... kvar=...`` lines."""
    out: dict[str, tuple[str, float]] = {}
    if not dss_path.is_file():
        return out
    pat = re.compile(
        r"New\s+Capacitor\.(\S+)\s+.*?Bus1=(\S+)\s+.*?kvar=([0-9.eE+-]+)",
        flags=re.IGNORECASE,
    )
    for line in dss_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line.replace("~", " "))
        if not m:
            continue
        nm, bus, kvar = m.group(1), m.group(2), float(m.group(3))
        out[_device_stem(nm)] = (str(bus).strip().lower(), float(kvar))
    return out


def _cap_bus_local_indices(bus: str, node_to_local: dict[str, int]) -> list[int]:
    """Map a capacitor bus name (phased or bare) to one or more local node indices."""
    b = str(bus).strip().lower()
    if b in node_to_local:
        return [int(node_to_local[b])]
    if re.search(r"\.\d+$", b):
        raise ValueError(f"Capacitor bus {bus!r} not in node index map")
    phased = [int(node_to_local[f"{b}{ph}"]) for ph in (".1", ".2", ".3") if f"{b}{ph}" in node_to_local]
    if phased:
        return phased
    raise ValueError(f"Capacitor bus {bus!r} not in node index map")


def _resolve_cap_bus_nodes(
    cap_cols: list[str],
    node_to_local: dict[str, int],
    *,
    cap_nodes_csv: Path | None,
    meta_csv: Path | None,
    capacitors_dss: Path | None,
) -> list[tuple[int, float, int]]:
    """``(local_node_idx, q_nominal_kvar_per_stamped_node, cap_col_index)`` for shunt-Y only."""
    import pandas as pd

    q_nom_by_stem: dict[str, float] = {}
    if meta_csv is not None and meta_csv.is_file():
        hdr = [str(c).lower() for c in pd.read_csv(meta_csv, nrows=0).columns.tolist()]
        for c in hdr:
            m = re.match(r"^cap_(.+)_q_nominal_kvar$", c, flags=re.IGNORECASE)
            if m:
                st = _device_stem(m.group(1))
                row = pd.read_csv(meta_csv, usecols=[c], nrows=1)
                q_nom_by_stem[st] = float(row[c].iloc[0])

    bus_nodes_by_stem: dict[str, list[str]] = {}
    if cap_nodes_csv is not None and cap_nodes_csv.is_file():
        cdf = pd.read_csv(cap_nodes_csv)
        cap_name_col = next((c for c in cdf.columns if str(c).strip().upper() == "CAP"), None)
        bus_col = next(
            (
                c
                for c in cdf.columns
                if str(c).strip().lower() in ("cap bus node", "feeder-side node", "from node")
            ),
            None,
        )
        if cap_name_col and bus_col:
            for _, row in cdf.iterrows():
                st = _device_stem(row[cap_name_col])
                bus = str(row[bus_col]).strip().lower()
                if bus and bus != "nan":
                    bus_nodes_by_stem.setdefault(st, []).append(bus)

    if capacitors_dss is not None:
        for st, (bus, kvar) in _parse_capacitors_dss(capacitors_dss).items():
            q_nom_by_stem.setdefault(st, float(kvar))
            bus_nodes_by_stem.setdefault(st, [bus])

    banks: list[tuple[int, float, int]] = []
    for j, col in enumerate(cap_cols):
        st = _cap_col_stem(col)
        if not st:
            raise ValueError(f"Cannot parse capacitor stem from meta column {col!r}")
        buses = bus_nodes_by_stem.get(st)
        if not buses:
            raise ValueError(
                f"No bus mapping for capacitor {st!r} (column {col!r}). "
                "Set --pf_cap_nodes_csv or provide capacitor_involved_nodes.csv with CAP and bus columns."
            )
        if st not in q_nom_by_stem:
            raise ValueError(
                f"No q_nominal_kvar for capacitor {st!r}; need cap_*_q_nominal_kvar in meta CSV or Capacitors.dss"
            )
        q_nom = float(q_nom_by_stem[st])
        node_indices: list[int] = []
        for bus in buses:
            for ni in _cap_bus_local_indices(bus, node_to_local):
                if ni not in node_indices:
                    node_indices.append(ni)
        if not node_indices:
            raise ValueError(f"No local nodes resolved for capacitor {st!r} buses {buses!r}")
        q_each = q_nom / float(len(node_indices))
        for ni in node_indices:
            banks.append((int(ni), float(q_each), int(j)))
    return banks


def _load_regulator_edges_for_pf(
    reg_catalog_csv: Path,
    node_to_local: dict[str, int],
    reg_cols: list[str],
    z_base_ohm: float | None,
) -> list[tuple[int, int, float, float, int]]:
    """Regulator series branches: ``(iu, iv, g, b, reg_col_idx)`` in pu or Siemens.

    When ``z_base_ohm`` is ``None``, ``g=R/(R^2+X^2)`` and ``b=-X/(R^2+X^2)`` (ohms → Siemens).
    Otherwise values are per-unit on ``z_base_ohm``.

    Catalog row ``from_node`` is the regulated (downstream) bus; ``to_node`` is ``regxfmr_*``.
    OpenDSS places the off-nominal tap on winding 2 (downstream). Stamp uses tap ``a`` on node ``iu``.
    """
    import pandas as pd

    if not reg_catalog_csv.is_file():
        raise FileNotFoundError(f"Regulator edge catalog not found: {reg_catalog_csv}")
    df = pd.read_csv(reg_catalog_csv)
    if "edge_type" not in df.columns:
        raise ValueError(f"{reg_catalog_csv} missing 'edge_type' column")
    reg_rows = df[df["edge_type"].astype(str).str.strip().str.lower() == "regulator"]
    col_to_j = {_reg_col_stem(c): j for j, c in enumerate(reg_cols) if _reg_col_stem(c)}
    z_base = None if z_base_ohm is None else float(z_base_ohm)
    edges: list[tuple[int, int, float, float, int]] = []
    for _, row in reg_rows.iterrows():
        u = str(row["from_node"]).strip().lower()
        v = str(row["to_node"]).strip().lower()
        if u not in node_to_local or v not in node_to_local:
            raise ValueError(
                f"Regulator edge nodes {u!r}/{v!r} from {reg_catalog_csv} not in node index map"
            )
        tap_col = str(row.get("tap_column", "")).strip().lower()
        rj = col_to_j.get(_reg_col_stem(tap_col))
        rlab = str(row.get("Regulator", "")).strip()
        if rj is None:
            rj = col_to_j.get(_device_stem(rlab))
        if rj is None:
            raise ValueError(
                f"Cannot map regulator row (tap_column={tap_col!r}, Regulator={rlab!r}) "
                f"to reg_cols {reg_cols!r} in {reg_catalog_csv}"
            )
        rf = float(row.get("R_full", 0.0) or 0.0)
        xf = float(row.get("X_full", 0.0) or 0.0)
        z2 = rf * rf + xf * xf
        if z2 < 1e-24:
            continue
        g = rf / z2
        b = -xf / z2
        if z_base is not None:
            g *= z_base
            b *= z_base
        edges.append((int(node_to_local[u]), int(node_to_local[v]), float(g), float(b), int(rj)))
    if reg_cols:
        mapped = {e[4] for e in edges}
        missing = [reg_cols[j] for j in range(len(reg_cols)) if j not in mapped]
        if missing:
            raise ValueError(
                f"Regulator columns not mapped to catalog edges in {reg_catalog_csv}: {missing}"
            )
    return edges


@dataclass
class PfYbusCoo:
    """COO line-Y for O(E) ``Y @ V`` (row sums into nodal current)."""

    row: torch.Tensor
    col: torch.Tensor
    y_re: torch.Tensor
    y_im: torch.Tensor

    def to(self, device: torch.device) -> PfYbusCoo:
        return PfYbusCoo(
            row=self.row.to(device),
            col=self.col.to(device),
            y_re=self.y_re.to(device),
            y_im=self.y_im.to(device),
        )


@dataclass
class PfPhysicsState:
    weight: float = 0.0
    s_base_kva: float = 5000.0
    huber_delta_kw: float = 10.0
    v_scale_volts: torch.Tensor | None = None
    Y_re_base: torch.Tensor | None = None
    Y_im_base: torch.Tensor | None = None
    y_coo: PfYbusCoo | None = None
    use_sparse_y: bool = True
    mask: torch.Tensor | None = None
    reg_edges: list[tuple[int, int, float, float, int]] = field(default_factory=list)
    cap_banks: list[tuple[int, float, int]] = field(default_factory=list)
    detach_controls: bool = False
    node_feature_cols: list[str] = field(default_factory=list)
    pf_debug_nan: bool = False
    idx_to_node: dict[int, str] = field(default_factory=dict)


def _undirected_node_pair(iu: int, iv: int) -> tuple[int, int]:
    return (min(int(iu), int(iv)), max(int(iu), int(iv)))


def _is_pf_slack_source_node(node: str) -> bool:
    """Substation / slack buses where nodal balance is not enforced."""
    n = str(node).strip().lower()
    bus = n.split(".")[0]
    if bus in ("sourcebus", "800", "substation"):
        return True
    if bus.startswith("_hvmv_sub") or bus.startswith("hvmv_sub"):
        return True
    return False


_PF_INTERFACE_BUS_PREFIXES: tuple[str, ...] = ("regxfmr", "190-")
_PF_INTERFACE_BUS_FIRST_CHARS: frozenset[str] = frozenset("mpn")


def _is_pf_interface_node(node: str) -> bool:
    """MV interface / monitoring / cap-feeder buses omitted from the MV Y-bus subgraph."""
    bus = str(node).strip().lower().split(".")[0]
    if any(bus.startswith(p) for p in _PF_INTERFACE_BUS_PREFIXES):
        return True
    return len(bus) > 0 and bus[0] in _PF_INTERFACE_BUS_FIRST_CHARS


def _load_pf_hetero_node_indices(
    data_root: Path,
    node_to_local: dict[str, int] | None = None,
) -> set[int]:
    """Local indices for nodes in ``hetero_mv_nodes_load_transformer.csv``.

    When ``node_to_local`` is set, map catalog ``node`` names to the training graph
    (required for chunk/subgraph runs where global ``node_idx`` values differ).
    """
    import pandas as pd

    het_path = data_root / _PF_HETERO_MV_NODES_REL
    if not het_path.is_file():
        return set()
    if node_to_local is not None:
        try:
            df = pd.read_csv(het_path, usecols=["node", "node_idx"])
        except ValueError:
            df = pd.read_csv(het_path, usecols=["node_idx"])
        else:
            if "node" in df.columns:
                out: set[int] = set()
                for raw in df["node"].dropna().unique():
                    key = str(raw).strip().lower()
                    if key in node_to_local:
                        out.add(int(node_to_local[key]))
                if out:
                    return out
    else:
        df = pd.read_csv(het_path, usecols=["node_idx"])
    return {int(v) for v in df["node_idx"].dropna().unique()}


def _refine_pf_mv_balance_mask(
    mask: torch.Tensor,
    node_to_local: dict[str, int],
    hetero_nodes: set[int],
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    *,
    exclude_interface: bool = True,
    hetero_y_neighbors_only: bool = True,
) -> torch.Tensor:
    """Tighten MV mask: hetero load nodes only, drop interface buses and Y couplings to them."""
    if not (exclude_interface or hetero_y_neighbors_only):
        return mask
    idx_to_node = {int(li): str(node) for node, li in node_to_local.items()}
    out = mask.clone()
    y_re_np = y_re.detach().cpu().numpy()
    y_im_np = y_im.detach().cpu().numpy()
    n_nodes = int(out.numel())
    for li in range(n_nodes):
        if not bool(out[li].item()):
            continue
        node = idx_to_node.get(li, "")
        if exclude_interface and _is_pf_interface_node(node):
            out[li] = False
            continue
        if hetero_y_neighbors_only and int(li) not in hetero_nodes:
            out[li] = False
            continue
        if not hetero_y_neighbors_only:
            continue
        nbrs = np.where((np.abs(y_re_np[li, :]) + np.abs(y_im_np[li, :])) > 1e-9)[0]
        nbrs = [int(j) for j in nbrs if int(j) != int(li)]
        if any(
            int(j) not in hetero_nodes or (exclude_interface and _is_pf_interface_node(idx_to_node.get(j, "")))
            for j in nbrs
        ):
            out[li] = False
    return out


def _apply_pf_balance_mask_refinement(
    mask: torch.Tensor,
    node_to_local: dict[str, int],
    pf_root: Path,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    args: argparse.Namespace,
    *,
    label: str = "PF",
) -> torch.Tensor:
    """Drop interface / non-hetero balance nodes (shared by MV distance and explicit lists)."""
    exclude_iface = bool(getattr(args, "pf_exclude_interface_buses", True))
    het_y_nbrs = bool(getattr(args, "pf_hetero_y_neighbors_only", True))
    if not (exclude_iface or het_y_nbrs):
        return mask
    hetero_nodes = _load_pf_hetero_node_indices(pf_root, node_to_local)
    effective_het_y = het_y_nbrs and bool(hetero_nodes)
    if het_y_nbrs and not hetero_nodes:
        print(
            f"WARNING: {label} mask refinement (hetero_y_neighbors_only) skipped — "
            f"hetero_mv_nodes_load_transformer.csv not found under {pf_root}; "
            "applying interface exclusion only.",
            flush=True,
        )
    if not exclude_iface and not effective_het_y:
        return mask
    n_before = int(mask.sum().item())
    mask = _refine_pf_mv_balance_mask(
        mask,
        node_to_local,
        hetero_nodes,
        y_re,
        y_im,
        exclude_interface=exclude_iface,
        hetero_y_neighbors_only=effective_het_y,
    )
    n_after = int(mask.sum().item())
    if n_after < n_before:
        print(
            f"{label} mask refined: {n_before} -> {n_after} nodes "
            f"(exclude_interface={exclude_iface}, hetero_y_neighbors_only={effective_het_y})",
            flush=True,
        )
    if not bool(mask.any()):
        raise ValueError(
            f"{label} balance mask empty after refinement "
            f"(exclude_interface={exclude_iface}, hetero_y_neighbors_only={effective_het_y})."
        )
    return mask


def _build_ybus_from_edge_csv(
    edge_csv: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
    z_base_ohm: float | None,
    *,
    skip_undirected: set[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense Ybus (Re/Im) from undirected ``R_full``/``X_full`` **line** edges only.

    When ``z_base_ohm`` is set, admittance is per-unit on that impedance base; when ``None``,
    ``R_full``/``X_full`` are treated as ohms and stamped in Siemens.

    Skips regulator-transformer branches (``skip_undirected`` from hetero catalog, and rows whose
    ``line_name`` starts with ``Transformer.`` or ``linecode`` is ``xfmr``) so taps are not double-stamped.
    """
    import pandas as pd

    df = pd.read_csv(edge_csv)
    for c in ("from_node", "to_node", "R_full", "X_full"):
        if c not in df.columns:
            raise ValueError(f"{edge_csv} missing {c!r}")
    y_re = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    y_im = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    skip = skip_undirected or set()
    z_base = None if z_base_ohm is None else float(z_base_ohm)
    for _, row in df.iterrows():
        line_name = str(row.get("line_name", "") or "").strip()
        linecode = str(row.get("linecode", "") or "").strip().lower()
        if line_name.startswith("Transformer.") or linecode == "xfmr":
            continue
        u = str(row["from_node"]).strip().lower()
        v = str(row["to_node"]).strip().lower()
        if u not in node_to_local or v not in node_to_local:
            continue
        iu = int(node_to_local[u])
        iv = int(node_to_local[v])
        if iu == iv:
            continue
        key = _undirected_node_pair(iu, iv)
        if key in skip or key in seen:
            continue
        seen.add(key)
        rf = float(row.get("R_full", 0.0) or 0.0)
        xf = float(row.get("X_full", 0.0) or 0.0)
        z2 = rf * rf + xf * xf
        if z2 < 1e-24:
            continue
        g = rf / z2
        b = -xf / z2
        if z_base is not None:
            g *= z_base
            b *= z_base
        y_line_re = g
        y_line_im = b
        y_re[iu, iv] -= y_line_re
        y_re[iv, iu] -= y_line_re
        y_im[iu, iv] -= y_line_im
        y_im[iv, iu] -= y_line_im
        y_re[iu, iu] += y_line_re
        y_re[iv, iv] += y_line_re
        y_im[iu, iu] += y_line_im
        y_im[iv, iv] += y_line_im
    return torch.from_numpy(y_re).float(), torch.from_numpy(y_im).float()


def _build_ybus_siemens_from_edge_csv(
    edge_csv: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
    *,
    skip_undirected: set[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical Siemens Y-bus from ``R_full``/``X_full`` in ohms."""
    return _build_ybus_from_edge_csv(
        edge_csv, node_to_local, n_nodes, None, skip_undirected=skip_undirected
    )


def _dense_y_to_coo(
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    *,
    tol: float = 1e-12,
) -> PfYbusCoo:
    """Compress dense ``(N,N)`` Y into COO for edge-local ``Y @ V``."""
    yr = y_re.detach()
    yi = y_im.detach()
    mask = (yr.abs() > tol) | (yi.abs() > tol)
    rows, cols = torch.where(mask)
    flat = mask.flatten()
    return PfYbusCoo(
        row=rows.to(dtype=torch.long),
        col=cols.to(dtype=torch.long),
        y_re=yr.reshape(-1)[flat],
        y_im=yi.reshape(-1)[flat],
    )


def _yv_from_line_coo(
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    coo: PfYbusCoo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``I = Y @ V`` via COO gather-scatter; ``v_*`` shape ``(B, N)``."""
    row = coo.row
    col = coo.col
    yr = coo.y_re.to(device=v_re.device, dtype=v_re.dtype)
    yi = coo.y_im.to(device=v_re.device, dtype=v_re.dtype)
    v_re_j = v_re.index_select(1, col)
    v_im_j = v_im.index_select(1, col)
    term_re = yr.unsqueeze(0) * v_re_j - yi.unsqueeze(0) * v_im_j
    term_im = yr.unsqueeze(0) * v_im_j + yi.unsqueeze(0) * v_re_j
    batch_size, n_nodes = v_re.shape
    row_b = row.unsqueeze(0).expand(batch_size, -1)
    i_re = torch.zeros(batch_size, n_nodes, device=v_re.device, dtype=v_re.dtype)
    i_im = torch.zeros(batch_size, n_nodes, device=v_re.device, dtype=v_re.dtype)
    i_re.scatter_add_(1, row_b, term_re)
    i_im.scatter_add_(1, row_b, term_im)
    return i_re, i_im


def _yv_add_reg_branch_contrib(
    i_re: torch.Tensor,
    i_im: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    iu: int,
    iv: int,
    g: float,
    b: float,
    tap: torch.Tensor,
) -> None:
    """Add regulator branch stamp to nodal current (matches ``_stamp_reg_branch_ybus``)."""
    a = tap.clamp(0.9, 1.1)
    a2 = a * a
    g_t = torch.as_tensor(g, device=v_re.device, dtype=v_re.dtype)
    b_t = torch.as_tensor(b, device=v_re.device, dtype=v_re.dtype)
    i_re[:, iu] = i_re[:, iu] + g_t * v_re[:, iu] - b_t * v_im[:, iu]
    i_im[:, iu] = i_im[:, iu] + g_t * v_im[:, iu] + b_t * v_re[:, iu]
    i_re[:, iv] = i_re[:, iv] + (g_t / a2) * v_re[:, iv] - (b_t / a2) * v_im[:, iv]
    i_im[:, iv] = i_im[:, iv] + (g_t / a2) * v_im[:, iv] + (b_t / a2) * v_re[:, iv]
    i_re[:, iu] = i_re[:, iu] - (g_t / a) * v_re[:, iv] + (b_t / a) * v_im[:, iv]
    i_im[:, iu] = i_im[:, iu] - (g_t / a) * v_im[:, iv] - (b_t / a) * v_re[:, iv]
    i_re[:, iv] = i_re[:, iv] - (g_t / a) * v_re[:, iu] + (b_t / a) * v_im[:, iu]
    i_im[:, iv] = i_im[:, iv] - (g_t / a) * v_im[:, iu] - (b_t / a) * v_re[:, iu]


def _yv_add_cap_shunt_contrib(
    i_re: torch.Tensor,
    i_im: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    ni: int,
    b_shunt: torch.Tensor,
) -> None:
    """Add shunt ``j*B`` diagonal stamp to nodal current."""
    i_re[:, ni] = i_re[:, ni] - b_shunt * v_im[:, ni]
    i_im[:, ni] = i_im[:, ni] + b_shunt * v_re[:, ni]


def _pf_node_nominal_volts(v_scale_volts: torch.Tensor, ni: int) -> torch.Tensor:
    """Per-bus nominal LN volts; accepts ``(N,)`` or ``(B, N)``."""
    if v_scale_volts.dim() == 1:
        return v_scale_volts[int(ni)].clamp(min=1.0)
    if v_scale_volts.dim() == 2:
        return v_scale_volts[:, int(ni)].clamp(min=1.0)
    raise ValueError(f"v_scale_volts must be 1D or 2D, got shape {tuple(v_scale_volts.shape)}")


def _compute_yv_current(
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    *,
    Y_re: torch.Tensor | None = None,
    Y_im: torch.Tensor | None = None,
    y_coo: PfYbusCoo | None = None,
    reg_edges: list[tuple[int, int, float, float, int]] | None = None,
    cap_banks: list[tuple[int, float, int]] | None = None,
    tap_pu: torch.Tensor | None = None,
    cap_on: torch.Tensor | None = None,
    use_sparse_y: bool = True,
    s_base_kva: float = 5000.0,
    v_scale_volts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch current ``I = Y @ V`` as ``(i_re, i_im)`` each ``(B, N)``."""
    if use_sparse_y and y_coo is not None:
        i_re, i_im = _yv_from_line_coo(v_re, v_im, y_coo)
        if reg_edges and tap_pu is not None:
            for iu, iv, g, b, rj in reg_edges:
                _yv_add_reg_branch_contrib(i_re, i_im, v_re, v_im, iu, iv, g, b, tap_pu[:, rj])
        if cap_banks and cap_on is not None:
            if v_scale_volts is None:
                raise ValueError("cap shunt stamping requires v_scale_volts")
            v_scale = v_scale_volts.to(device=v_re.device, dtype=v_re.dtype)
            for ni, q_nom, cj in cap_banks:
                v_nom = _pf_node_nominal_volts(v_scale, ni)
                b_shunt = cap_on[:, cj] * (float(q_nom) * 1000.0) / (v_nom * v_nom)
                _yv_add_cap_shunt_contrib(i_re, i_im, v_re, v_im, ni, b_shunt)
        return i_re, i_im
    if Y_re is None or Y_im is None:
        raise ValueError("dense Y@V requires Y_re and Y_im when sparse coo is disabled")
    if Y_re.dim() == 2:
        i_re = torch.matmul(v_re, Y_re.T) - torch.matmul(v_im, Y_im.T)
        i_im = torch.matmul(v_re, Y_im.T) + torch.matmul(v_im, Y_re.T)
    else:
        i_re = torch.matmul(v_re, Y_re.transpose(-1, -2)) - torch.matmul(v_im, Y_im.transpose(-1, -2))
        i_im = torch.matmul(v_re, Y_im.transpose(-1, -2)) + torch.matmul(v_im, Y_re.transpose(-1, -2))
    return i_re, i_im


_PF_ELECTRICAL_DISTANCE_REL = Path("electrical_distance_from_substation.csv")
_PF_NODE_INDEX_REL = Path("gnn_node_index_master.csv")
_PF_HETERO_MV_NODES_REL = (
    Path("Heterogenous GNN dataset") / "nodes" / "hetero_mv_nodes_load_transformer.csv"
)


def _resolve_pf_data_root_arg(args: argparse.Namespace, repo: Path) -> Path | None:
    """Explicit --pf_data_root override for mvagg topology CSVs (not chunk run_* dirs)."""
    raw = str(getattr(args, "pf_data_root", "") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (repo / p).resolve()
    return p.resolve()


def _effective_pf_data_root(
    *,
    data_root: Path,
    args: argparse.Namespace,
    repo: Path,
    chunk_parent: Path | None = None,
) -> Path:
    """Root for PF catalog / distance / hetero-node lookups (may differ from chunk run_* dir)."""
    explicit = _resolve_pf_data_root_arg(args, repo)
    if explicit is not None:
        return explicit
    if not data_root.name.startswith("run_"):
        return data_root
    from gnn2_pf_data_paths import resolve_pf_catalog_paths

    _, _, root = resolve_pf_catalog_paths(
        repo=repo,
        preferred_root=None,
        chunk_parent=chunk_parent,
    )
    return root


def _csv_has_columns(csv_path: Path, *cols: str) -> bool:
    import pandas as pd

    if not csv_path.is_file():
        return False
    hdr = {str(c).strip().lower() for c in pd.read_csv(csv_path, nrows=0).columns.tolist()}
    return all(str(c).strip().lower() in hdr for c in cols)


def _pf_distance_csv_candidates(
    *,
    nodes_csv: Path,
    node_pe_csv: Path | None,
    data_root: Path | None,
    repo: Path,
    pf_preferred: Path | None = None,
) -> list[tuple[str, Path]]:
    """Ordered (label, path) candidates that may carry ``electrical_distance_ohm``."""
    out: list[tuple[str, Path]] = [("nodes_csv", nodes_csv)]
    if node_pe_csv is not None:
        out.append(("node_pe_csv", node_pe_csv))
    extra: list[Path] = []
    if data_root is not None:
        extra.extend(
            [
                data_root / _PF_ELECTRICAL_DISTANCE_REL,
                data_root / _PF_NODE_INDEX_REL,
                data_root / _PF_HETERO_MV_NODES_REL,
            ]
        )
    from gnn2_pf_data_paths import candidate_pf_data_roots

    chunk_parent = nodes_csv.parent.parent if nodes_csv.parent.name.startswith("run_") else None
    preferred = pf_preferred
    if preferred is None and data_root is not None and not data_root.name.startswith("run_"):
        preferred = data_root
    for root in candidate_pf_data_roots(
        repo=repo, preferred=preferred, chunk_parent=chunk_parent
    ):
        extra.extend(
            [
                root / _PF_ELECTRICAL_DISTANCE_REL,
                root / _PF_NODE_INDEX_REL,
                root / _PF_HETERO_MV_NODES_REL,
            ]
        )
    seen: set[str] = {str(nodes_csv.resolve()).lower()} if nodes_csv.is_file() else set()
    if node_pe_csv is not None and node_pe_csv.is_file():
        seen.add(str(node_pe_csv.resolve()).lower())
    for p in extra:
        try:
            key = str(p.resolve()).lower()
        except OSError:
            key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.name == _PF_ELECTRICAL_DISTANCE_REL.name:
            out.append(("pf_data_electrical_distance", p))
        elif p.name == _PF_NODE_INDEX_REL.name:
            out.append(("pf_data_node_index", p))
        else:
            out.append(("pf_data_hetero_mv_nodes", p))
    return out


def _resolve_pf_electrical_distance_csv(
    *,
    nodes_csv: Path,
    node_pe_csv: Path | None,
    data_root: Path | None,
    repo: Path,
    mode: str,
    pf_preferred: Path | None = None,
) -> tuple[Path | None, list[str]]:
    """Pick a CSV with ``electrical_distance_ohm`` for ``--pf_balance_nodes mv``."""
    m = str(mode).strip().lower()
    if m != "mv":
        return None, []
    tried: list[str] = []
    for label, path in _pf_distance_csv_candidates(
        nodes_csv=nodes_csv,
        node_pe_csv=node_pe_csv,
        data_root=data_root,
        repo=repo,
        pf_preferred=pf_preferred,
    ):
        tried.append(f"{label}={path}")
        if _csv_has_columns(path, "node", "electrical_distance_ohm"):
            if path != nodes_csv:
                print(
                    f"WARNING: {nodes_csv} lacks electrical_distance_ohm; "
                    f"using {path} ({label}) for --pf_balance_nodes mv mask.",
                    flush=True,
                )
            return path, tried
    return None, tried


def _load_pf_distance_by_node(
    distance_csv: Path,
    *,
    sample_id: int | None,
) -> dict[str, float]:
    import pandas as pd

    hdr = pd.read_csv(distance_csv, nrows=0).columns.tolist()
    usecols = ["node", "electrical_distance_ohm"]
    if "sample_id" in hdr:
        usecols.insert(0, "sample_id")
    df = pd.read_csv(distance_csv, usecols=usecols)
    if "sample_id" in df.columns and sample_id is not None:
        df = df[df["sample_id"].map(_norm_sid) == int(sample_id)]
    return {
        str(row["node"]).strip().lower(): float(row["electrical_distance_ohm"])
        for _, row in df.iterrows()
    }


def _load_pf_balance_mask(
    nodes_csv: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
    mode: str,
    *,
    distance_csv: Path | None = None,
    mv_fallback_all_non_slack: bool = False,
    distance_tried: list[str] | None = None,
) -> torch.Tensor:
    """Balance mask excluding slack/source buses.

    ``all`` = every non-slack node; ``mv`` = ``electrical_distance_ohm > 0`` and non-slack (downstream MV).
    When ``nodes_csv`` lacks ``electrical_distance_ohm``, pass ``distance_csv`` (e.g. ``--node_pe_csv``).
    """
    import pandas as pd

    m = str(mode).strip().lower()
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    dist_by_node: dict[str, float] | None = None
    dist_src = distance_csv if distance_csv is not None else nodes_csv

    if m == "mv":
        if not _csv_has_columns(dist_src, "node", "electrical_distance_ohm"):
            tried = distance_tried or [str(dist_src)]
            if mv_fallback_all_non_slack:
                print(
                    "WARNING: --pf_balance_nodes mv but electrical_distance_ohm was not found; "
                    f"falling back to all non-slack nodes. Tried:\n  "
                    + "\n  ".join(tried)
                    + "\nPass --node_pe_csv (e.g. gnn_node_index_master.csv) or use --pf_balance_nodes all.",
                    flush=True,
                )
            else:
                raise ValueError(
                    "--pf_balance_nodes mv requires electrical_distance_ohm but it is missing from "
                    f"{nodes_csv}. Tried:\n  "
                    + "\n  ".join(tried)
                    + "\nPass --node_pe_csv (e.g. gnn_node_index_master.csv) or use --pf_balance_nodes all."
                )
        elif dist_src != nodes_csv or not _csv_has_columns(nodes_csv, "electrical_distance_ohm"):
            sid0 = int(pd.read_csv(nodes_csv, usecols=["sample_id"], nrows=1)["sample_id"].iloc[0])
            dist_by_node = _load_pf_distance_by_node(dist_src, sample_id=sid0)

    if dist_by_node is not None:
        for node, li in node_to_local.items():
            if _is_pf_slack_source_node(node):
                continue
            dist = dist_by_node.get(node)
            if dist is None:
                continue
            if m == "mv" and float(dist) <= 1e-9:
                continue
            mask[int(li)] = True
    elif m == "all" or mv_fallback_all_non_slack:
        for node, li in node_to_local.items():
            if _is_pf_slack_source_node(node):
                continue
            mask[int(li)] = True
    else:
        usecols = ["sample_id", "node"]
        if m == "mv":
            usecols.append("electrical_distance_ohm")
        sid0 = int(pd.read_csv(nodes_csv, usecols=["sample_id"], nrows=1)["sample_id"].iloc[0])
        sub = pd.read_csv(nodes_csv, usecols=usecols)
        sub = sub[sub["sample_id"].map(_norm_sid) == sid0]
        for _, row in sub.iterrows():
            node = str(row["node"]).strip().lower()
            if node not in node_to_local:
                continue
            if _is_pf_slack_source_node(node):
                continue
            if m == "mv" and float(row["electrical_distance_ohm"]) <= 1e-9:
                continue
            mask[int(node_to_local[node])] = True

    if not bool(mask.any()):
        raise ValueError(
            f"--pf_balance_nodes {mode!r} produced empty mask from {nodes_csv}; "
            "check electrical_distance_ohm, slack exclusion, and node index mapping."
        )
    return mask


def _load_pf_balance_mask_from_explicit_list(
    list_csv: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
) -> torch.Tensor:
    """Build balance mask from an explicit node list CSV (``node_idx``, ``bus``, and/or ``node``)."""
    import pandas as pd

    path = Path(list_csv).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--pf_balance_node_list_csv not found: {path}")
    df = pd.read_csv(path)
    cols = {str(c).strip().lower() for c in df.columns}
    id_cols = {"node_idx", "bus", "node"} & cols
    if not id_cols:
        raise ValueError(
            f"--pf_balance_node_list_csv {path} must include at least one of: node_idx, bus, node "
            f"(found columns: {list(df.columns)!r})"
        )

    bus_to_nodes: dict[str, list[str]] = {}
    for node in node_to_local:
        bus = str(node).strip().lower().split(".")[0]
        bus_to_nodes.setdefault(bus, []).append(node)

    mask = torch.zeros(n_nodes, dtype=torch.bool)
    n_slack_dropped = 0
    n_unknown = 0
    idx_to_node = {int(li): str(node) for node, li in node_to_local.items()}

    def _stamp_local(li: int) -> None:
        nonlocal n_slack_dropped, n_unknown
        if li < 0 or li >= n_nodes:
            n_unknown += 1
            return
        node = idx_to_node.get(int(li), "")
        if node and _is_pf_slack_source_node(node):
            n_slack_dropped += 1
            return
        mask[int(li)] = True

    for _, row in df.iterrows():
        # Prefer exact phase node names (chunk-safe), then bus (all phases), then node_idx.
        if "node" in id_cols and pd.notna(row.get("node")):
            key = str(row["node"]).strip().lower()
            if key in node_to_local:
                _stamp_local(int(node_to_local[key]))
            else:
                n_unknown += 1
            continue
        if "bus" in id_cols and pd.notna(row.get("bus")):
            bus = str(row["bus"]).strip().lower()
            hits = bus_to_nodes.get(bus, [])
            if not hits:
                n_unknown += 1
                continue
            for node in hits:
                _stamp_local(int(node_to_local[node]))
            continue
        if "node_idx" in id_cols and pd.notna(row.get("node_idx")):
            _stamp_local(int(row["node_idx"]))

    if n_slack_dropped:
        print(
            f"WARNING: PF explicit balance list dropped {n_slack_dropped} slack/source node(s) from {path.name}",
            flush=True,
        )
    if n_unknown:
        print(
            f"WARNING: PF explicit balance list skipped {n_unknown} row(s) not mapped to training graph",
            flush=True,
        )
    if not bool(mask.any()):
        raise ValueError(
            f"--pf_balance_node_list_csv {path} produced empty mask after slack exclusion "
            "and node index mapping."
        )
    print(f"PF explicit balance nodes: {int(mask.sum().item())} from {path}", flush=True)
    return mask


def _pf_huber_mean_sq(r: torch.Tensor, *, delta: float) -> torch.Tensor:
    """Huber on residuals; ``delta`` is in the same units as ``r`` (kW/kvar)."""
    d = max(float(delta), 1e-12)
    abs_r = r.abs()
    quad = 0.5 * r.square()
    lin = d * (abs_r - 0.5 * d)
    return torch.where(abs_r <= d, quad, lin).mean()


def _pf_physics_fp32_ctx():
    """Disable AMP inside the physics block (Y@V and Huber need fp32)."""
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


_PF_BALANCE_NODE_WARNED: set[int] = set()


def _warn_pf_balance_node_issues(
    mask: torch.Tensor,
    idx_to_node: dict[int, str],
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    *,
    hetero_nodes: set[int] | None = None,
) -> None:
    """One-shot warnings for explicit balance nodes that may skew physics loss."""
    y_re_np = y_re.detach().cpu().numpy()
    y_im_np = y_im.detach().cpu().numpy()
    for li in range(int(mask.numel())):
        if not bool(mask[li].item()):
            continue
        node = idx_to_node.get(int(li), "")
        issues: list[str] = []
        if _is_pf_interface_node(node):
            issues.append("interface bus")
        row_norm = float(np.abs(y_re_np[li, :]).sum() + np.abs(y_im_np[li, :]).sum())
        if row_norm < 1e-9:
            issues.append("zero Y-bus row")
        if hetero_nodes is not None and int(li) not in hetero_nodes:
            issues.append("not in hetero_mv_nodes_load_transformer")
        if not issues or int(li) in _PF_BALANCE_NODE_WARNED:
            continue
        _PF_BALANCE_NODE_WARNED.add(int(li))
        print(
            f"WARNING: PF balance node_idx={li} ({node or '?'}) flagged: {', '.join(issues)}",
            flush=True,
        )


_PF_DEBUG_NAN_EMITTED = False


def _resolve_pf_debug_nan(args: argparse.Namespace) -> bool:
    if bool(int(getattr(args, "pf_debug_nan", 0) or 0)):
        return True
    env = str(os.environ.get("GNN2_PF_DEBUG_NAN", "")).strip().lower()
    return env in ("1", "true", "yes", "on")


def _pf_should_emit_debug(
    pf: PfPhysicsState,
    *,
    epoch: int,
    first_batch_of_epoch: bool,
    loss_pf: torch.Tensor | None,
) -> bool:
    global _PF_DEBUG_NAN_EMITTED
    if _PF_DEBUG_NAN_EMITTED or loss_pf is None:
        return False
    if not torch.isfinite(loss_pf).all():
        return True
    return bool(pf.pf_debug_nan and epoch == 1 and first_batch_of_epoch)


def _pf_masked_finite_line(name: str, t: torch.Tensor, mask: torch.Tensor) -> str:
    """One-line nan/inf/range summary for ``t`` shaped ``(B, N)`` on boolean ``mask`` ``(N,)``."""
    m = mask.to(device=t.device, dtype=torch.bool).view(1, -1).expand_as(t)
    sel = t.masked_select(m)
    n = int(sel.numel())
    if n == 0:
        return f"  {name}: (no masked elements)"
    nan_c = int(torch.isnan(sel).sum().item())
    inf_c = int(torch.isinf(sel).sum().item())
    fin = sel[torch.isfinite(sel)]
    if fin.numel() == 0:
        stats = "all non-finite"
    else:
        stats = (
            f"min={float(fin.min()):.4e} max={float(fin.max()):.4e} "
            f"mean={float(fin.mean()):.4e} std={float(fin.std(unbiased=False)):.4e}"
        )
    return f"  {name}: nan={nan_c} inf={inf_c} of {n} | {stats}"


def _pf_top_residual_nodes(
    r_p: torch.Tensor,
    r_q: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 5,
    idx_to_node: dict[int, str] | None = None,
) -> list[str]:
    m = mask.to(device=r_p.device, dtype=torch.bool).view(1, -1)
    score = (r_p.abs() + r_q.abs()).masked_fill(~m, -1.0)
    flat = score.reshape(-1)
    n_pick = min(int(k), int((flat >= 0).sum().item()))
    if n_pick <= 0:
        return ["  (no masked nodes)"]
    vals, idx = torch.topk(flat, k=n_pick)
    n_nodes = int(r_p.shape[-1])
    lines: list[str] = []
    for v, fi in zip(vals.tolist(), idx.tolist()):
        b = int(fi // n_nodes)
        ni = int(fi % n_nodes)
        rp = float(r_p[b, ni].item())
        rq = float(r_q[b, ni].item())
        rp_s = "nan" if math.isnan(rp) else f"{rp:.4e}"
        rq_s = "nan" if math.isnan(rq) else f"{rq:.4e}"
        bus = ""
        if idx_to_node:
            bus_name = idx_to_node.get(ni, "")
            if bus_name:
                bus = f"  bus={bus_name}"
        lines.append(
            f"  node_idx={ni}{bus} batch={b} |r_p|+|r_q|={v:.4e}  r_p={rp_s} kW  r_q={rq_s} kvar"
        )
    return lines


@torch.no_grad()
def _pf_debug_nan_report(
    *,
    loss_pf: torch.Tensor,
    v_n: torch.Tensor,
    batch: Data,
    n_nodes: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    pf: PfPhysicsState,
    cap_logits: torch.Tensor,
    reg_pred: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    reg_loss: str,
    reg_mean: torch.Tensor | None,
    reg_std: torch.Tensor | None,
    reg_logits: list[torch.Tensor] | None,
    reg_class_values: torch.Tensor | None,
    use_amp: bool,
    epoch: int,
    batch_idx: int,
    trigger: str,
) -> None:
    """Print one-shot diagnostics for physics loss NaN / first-batch smoke (Colab-friendly)."""
    global _PF_DEBUG_NAN_EMITTED
    _PF_DEBUG_NAN_EMITTED = True

    pf_loss = float(loss_pf.detach().float().item())
    pf_finite = math.isfinite(pf_loss)
    lines = [
        "",
        f"=== PF physics debug ({trigger}; epoch {epoch} batch {batch_idx}) ===",
        f"AMP (autocast) enabled for model: {use_amp}",
        f"loss_pf (fp32 physics path): {pf_loss:.6e}  finite={pf_finite}",
        f"use_sparse_y: {bool(pf.use_sparse_y and pf.y_coo is not None)}",
        f"huber_delta_kw: {pf.huber_delta_kw}",
        f"detach_controls: {pf.detach_controls}",
        f"masked balance nodes: {int(pf.mask.sum().item()) if pf.mask is not None else 0} / {n_nodes}",
        f"batch graphs: {int(batch.num_graphs)}",
        "--- Per-stage counts on masked nodes (current dtype) ---",
    ]

    mask = pf.mask if pf.mask is not None else torch.ones(n_nodes, dtype=torch.bool, device=v_n.device)
    hub_p_val = hub_q_val = hub_tot_val = float("nan")
    stage_lines: list[str] = []
    with _pf_physics_fp32_ctx():
        y_mean_ri = y_mean.view(1, n_nodes, 2).float()
        y_std_ri = y_std.view(1, n_nodes, 2).float()
        pred_ri = v_n.view(batch.num_graphs, n_nodes, 2).float() * y_std_ri + y_mean_ri
        cap_on = torch.sigmoid(cap_logits.float())
        tap_pu = _expected_reg_tap_pu(
            reg_pred.float(),
            reg_loss=reg_loss,
            reg_mean=reg_mean,
            reg_std=reg_std,
            reg_logits=reg_logits,
            reg_class_values=reg_class_values,
        )
        x_den = _denorm_node_features(batch.x.float(), batch, n_nodes, x_mean.float(), x_std.float())
        p_inj, q_inj = _assemble_pf_injections(
            x_den,
            pf.node_feature_cols,
            batch=batch,
            n_nodes=n_nodes,
        )

        v_re = pred_ri[..., 0]
        v_im = pred_ri[..., 1]
        vs = pf.v_scale_volts.to(device=v_re.device, dtype=torch.float32)
        if vs.dim() == 1:
            vs_b = vs.reshape(1, -1).expand(v_re.shape[0], -1)
        else:
            vs_b = vs.reshape(v_re.shape[0], -1)
        v_phys_re = v_re * vs_b
        v_phys_im = v_im * vs_b

        use_sparse = bool(pf.use_sparse_y and pf.y_coo is not None)
        if use_sparse:
            i_re, i_im = _compute_yv_current(
                v_phys_re,
                v_phys_im,
                y_coo=pf.y_coo,
                reg_edges=pf.reg_edges,
                cap_banks=pf.cap_banks,
                tap_pu=tap_pu,
                cap_on=cap_on,
                use_sparse_y=True,
                s_base_kva=pf.s_base_kva,
                v_scale_volts=pf.v_scale_volts,
            )
        else:
            assert pf.Y_re_base is not None and pf.Y_im_base is not None
            y_re, y_im = _ybus_with_predicted_controls(
                pf.Y_re_base,
                pf.Y_im_base,
                reg_edges=pf.reg_edges,
                cap_banks=pf.cap_banks,
                tap_pu=tap_pu,
                cap_on=cap_on,
                s_base_kva=pf.s_base_kva,
                batch_size=int(batch.num_graphs),
                v_scale_volts=pf.v_scale_volts,
            )
            i_re, i_im = _compute_yv_current(
                v_phys_re,
                v_phys_im,
                Y_re=y_re,
                Y_im=y_im,
                use_sparse_y=False,
            )

        s_re = v_phys_re * i_re + v_phys_im * i_im
        s_im = v_phys_im * i_re - v_phys_re * i_im
        p_yv_kw = s_re / 1000.0
        q_yv_kvar = s_im / 1000.0
        r_p = p_inj - p_yv_kw
        r_q = q_inj - q_yv_kvar

        m = mask.to(device=r_p.device, dtype=torch.bool).view(1, -1)
        r_p_m = r_p.masked_select(m)
        r_q_m = r_q.masked_select(m)
        hub_p = _pf_huber_mean_sq(r_p_m, delta=pf.huber_delta_kw)
        hub_q = _pf_huber_mean_sq(r_q_m, delta=pf.huber_delta_kw)
        hub_tot = hub_p + hub_q
        hub_p_val = float(hub_p.item())
        hub_q_val = float(hub_q.item())
        hub_tot_val = float(hub_tot.item())

        stage_lines = [
            _pf_masked_finite_line("pred_ri (V pu)", pred_ri.norm(dim=-1), mask),
            _pf_masked_finite_line("V_phys (volts, |V|)", torch.sqrt(v_phys_re * v_phys_re + v_phys_im * v_phys_im), mask),
            _pf_masked_finite_line("p_inj_kw", p_inj, mask),
            _pf_masked_finite_line("q_inj_kvar", q_inj, mask),
            _pf_masked_finite_line("i_re (Y@V, A)", i_re, mask),
            _pf_masked_finite_line("i_im (Y@V, A)", i_im, mask),
            _pf_masked_finite_line("p_yv_kw (S_re/1e3)", p_yv_kw, mask),
            _pf_masked_finite_line("q_yv_kvar (S_im/1e3)", q_yv_kvar, mask),
            _pf_masked_finite_line("r_p residual (kW)", r_p, mask),
            _pf_masked_finite_line("r_q residual (kvar)", r_q, mask),
            f"  huber_p: {hub_p_val:.6e}  finite={math.isfinite(hub_p_val)}",
            f"  huber_q: {hub_q_val:.6e}  finite={math.isfinite(hub_q_val)}",
            f"  huber_total: {hub_tot_val:.6e}  finite={math.isfinite(hub_tot_val)}",
            "--- Worst masked nodes by |r_p|+|r_q| ---",
            *_pf_top_residual_nodes(r_p, r_q, mask, k=5, idx_to_node=pf.idx_to_node or None),
        ]

    lines.extend(stage_lines)

    if use_amp:
        with _pf_physics_fp32_ctx():
            loss_fp32 = _power_balance_loss_from_batch(
                v_n,
                batch,
                n_nodes=n_nodes,
                y_mean=y_mean,
                y_std=y_std,
                pf=pf,
                cap_logits=cap_logits,
                reg_pred=reg_pred,
                x_mean=x_mean,
                x_std=x_std,
                reg_loss=reg_loss,
                reg_mean=reg_mean,
                reg_std=reg_std,
                reg_logits=reg_logits,
                reg_class_values=reg_class_values,
            )
        fp32_val = float(loss_fp32.item())
        lines.append("--- FP32 replay sanity check ---")
        lines.append(f"  loss_pf_fp32: {fp32_val:.6e}  finite={math.isfinite(fp32_val)}")
        if pf_finite and math.isfinite(fp32_val) and abs(fp32_val - pf_loss) < max(1e-3, 1e-6 * abs(fp32_val)):
            lines.append("  OK: physics loss runs in fp32 (matches replay).")
        elif (
            math.isfinite(hub_tot_val)
            and math.isfinite(fp32_val)
            and abs(fp32_val - hub_tot_val) < max(1e-3, 1e-6 * abs(fp32_val))
        ):
            lines.append("  OK: huber_total matches fp32 replay (debug aligned with loss).")
        elif pf_finite and not math.isfinite(fp32_val):
            lines.append("  NOTE: reported loss finite but fp32 replay non-finite (unexpected).")
        elif not pf_finite and math.isfinite(fp32_val):
            lines.append("  LIKELY CAUSE: non-fp32 physics path or stale loss tensor.")
        elif not pf_finite and not math.isfinite(fp32_val):
            lines.append("  LIKELY CAUSE: non-finite values in physics tensors.")
        if math.isfinite(fp32_val) and fp32_val > 1.0e5:
            lines.append(
                f"  NOTE: large fp32 loss ({fp32_val:.3e}); early-epoch V preds or outlier nodes "
                "can dominate Huber — check worst nodes above."
            )

    lines.append("=== end PF physics debug ===")
    print("\n".join(lines), flush=True)


def nodal_power_balance_residual(
    pred_ri: torch.Tensor,
    p_inj_kw: torch.Tensor,
    q_inj_kvar: torch.Tensor,
    Y_re: torch.Tensor | None,
    Y_im: torch.Tensor | None,
    node_mask: torch.Tensor | None,
    s_base_kva: float = 5000.0,
    *,
    v_scale_volts: torch.Tensor | None = None,
    huber_delta_kw: float = 10.0,
    y_coo: PfYbusCoo | None = None,
    reg_edges: list[tuple[int, int, float, float, int]] | None = None,
    cap_banks: list[tuple[int, float, int]] | None = None,
    tap_pu: torch.Tensor | None = None,
    cap_on: torch.Tensor | None = None,
    use_sparse_y: bool = True,
) -> torch.Tensor:
    """Huber-smoothed mean P/Q balance residual (physical units: V volts, Y Siemens, kW/kvar)."""
    if v_scale_volts is None:
        raise ValueError("nodal_power_balance_residual requires v_scale_volts")
    pred_ri = pred_ri.float()
    p_inj_kw = p_inj_kw.float()
    q_inj_kvar = q_inj_kvar.float()
    v_re = pred_ri[..., 0]
    v_im = pred_ri[..., 1]
    vs = v_scale_volts.to(device=v_re.device, dtype=torch.float32)
    if vs.dim() == 1:
        vs = vs.reshape(1, -1).expand(v_re.shape[0], -1)
    else:
        vs = vs.reshape(v_re.shape[0], -1)
    v_re = v_re * vs
    v_im = v_im * vs
    i_re, i_im = _compute_yv_current(
        v_re,
        v_im,
        Y_re=Y_re,
        Y_im=Y_im,
        y_coo=y_coo,
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        tap_pu=tap_pu,
        cap_on=cap_on,
        use_sparse_y=use_sparse_y,
        s_base_kva=s_base_kva,
        v_scale_volts=v_scale_volts,
    )
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    p_yv_kw = s_re / 1000.0
    q_yv_kvar = s_im / 1000.0
    r_p = p_inj_kw - p_yv_kw
    r_q = q_inj_kvar - q_yv_kvar
    delta = huber_delta_kw
    if node_mask is not None:
        m = node_mask.to(device=pred_ri.device, dtype=torch.bool).view(1, -1)
        r_p = r_p.masked_select(m)
        r_q = r_q.masked_select(m)
    return _pf_huber_mean_sq(r_p, delta=delta) + _pf_huber_mean_sq(r_q, delta=delta)


def _expected_reg_tap_pu(
    reg_pred: torch.Tensor,
    *,
    reg_loss: str,
    reg_mean: torch.Tensor | None,
    reg_std: torch.Tensor | None,
    reg_logits: list[torch.Tensor] | None,
    reg_class_values: torch.Tensor | None,
) -> torch.Tensor:
    """Differentiable tap (pu) per graph for physics Y stamping."""
    if reg_loss == "ce":
        if not reg_logits or reg_class_values is None:
            return reg_pred
        taps = []
        cv = reg_class_values.to(device=reg_pred.device, dtype=torch.float32)
        for j, lg in enumerate(reg_logits):
            probs = F.softmax(lg.float(), dim=-1)
            nc = int(probs.shape[-1])
            taps.append(torch.matmul(probs, cv[j, :nc]))
        return torch.stack(taps, dim=1)
    if reg_mean is not None and reg_std is not None:
        return reg_pred.float() * reg_std.view(1, -1) + reg_mean.view(1, -1)
    return reg_pred.float()


def _stamp_reg_branch_ybus(
    Y_re: torch.Tensor,
    Y_im: torch.Tensor,
    iu: int,
    iv: int,
    g: float,
    b: float,
    tap: torch.Tensor,
) -> None:
    """Off-nominal tap ``a`` on secondary (node ``iu``); primary at ``iv``. In-place on batched ``(B,N,N)``."""
    a = tap.clamp(0.9, 1.1)
    a2 = a * a
    g_t = torch.as_tensor(g, device=Y_re.device, dtype=Y_re.dtype)
    b_t = torch.as_tensor(b, device=Y_im.device, dtype=Y_im.dtype)
    Y_re[..., iu, iu] = Y_re[..., iu, iu] + g_t
    Y_im[..., iu, iu] = Y_im[..., iu, iu] + b_t
    Y_re[..., iv, iv] = Y_re[..., iv, iv] + g_t / a2
    Y_im[..., iv, iv] = Y_im[..., iv, iv] + b_t / a2
    Y_re[..., iu, iv] = Y_re[..., iu, iv] - g_t / a
    Y_re[..., iv, iu] = Y_re[..., iv, iu] - g_t / a
    Y_im[..., iu, iv] = Y_im[..., iu, iv] - b_t / a
    Y_im[..., iv, iu] = Y_im[..., iv, iu] - b_t / a


def _ybus_with_predicted_controls(
    Y_re_base: torch.Tensor,
    Y_im_base: torch.Tensor,
    *,
    reg_edges: list[tuple[int, int, float, float, int]],
    cap_banks: list[tuple[int, float, int]],
    tap_pu: torch.Tensor,
    cap_on: torch.Tensor,
    s_base_kva: float,
    batch_size: int,
    v_scale_volts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(B,N,N)`` Ybus: line base + regulator taps + cap shunt ``j*B`` (Siemens)."""
    dev, dt = Y_re_base.device, Y_re_base.dtype
    y_re = Y_re_base.unsqueeze(0).expand(batch_size, -1, -1).clone()
    y_im = Y_im_base.unsqueeze(0).expand(batch_size, -1, -1).clone()
    for iu, iv, g, b, rj in reg_edges:
        _stamp_reg_branch_ybus(y_re, y_im, iu, iv, g, b, tap_pu[:, rj])
    v_scale = v_scale_volts.to(device=dev, dtype=dt)
    for ni, q_nom, cj in cap_banks:
        v_nom = _pf_node_nominal_volts(v_scale, ni)
        b_siemens = cap_on[:, cj] * (float(q_nom) * 1000.0) / (v_nom * v_nom)
        y_im[:, ni, ni] = y_im[:, ni, ni] + b_siemens
    return y_re, y_im


def _denorm_node_features(
    x_n: torch.Tensor,
    batch: Data,
    n_nodes: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
) -> torch.Tensor:
    xm = x_mean.view(1, 1, -1).float()
    xs = x_std.view(1, 1, -1).float()
    return x_n.view(batch.num_graphs, n_nodes, -1).float() * xs + xm


def _assemble_pf_injections(
    x_denorm: torch.Tensor,
    node_feature_cols: list[str],
    *,
    batch: Data,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodal injections from known denormalized features (OpenDSS convention).

    ``P_inj = P_pv - P_load`` (kW), ``Q_inj = -Q_pv - Q_load`` (kvar).
    Capacitor banks are modeled as shunt ``j*B`` in ``Y_bus`` only — not added here.
    Meta-aux PV predictions are intentionally excluded from physics residuals.
    """
    _ = batch  # signature parity with call sites
    dev, dt = x_denorm.device, x_denorm.dtype
    cols = {str(c).lower(): i for i, c in enumerate(node_feature_cols)}
    if "p_load_kw" not in cols or "q_load_kvar" not in cols:
        raise ValueError(
            "Physics injection assembly requires p_load_kw and q_load_kvar in --node_feature_cols "
            f"(got {node_feature_cols!r})."
        )
    if "p_pv_kw" not in cols:
        raise ValueError(
            "Physics injection assembly requires p_pv_kw in --node_feature_cols "
            f"(got {node_feature_cols!r})."
        )

    p_load = x_denorm[..., cols["p_load_kw"]]
    q_load = x_denorm[..., cols["q_load_kvar"]]
    p_pv = x_denorm[..., cols["p_pv_kw"]]
    q_pv = x_denorm[..., cols["q_pv_kvar"]] if "q_pv_kvar" in cols else torch.zeros_like(p_load)

    p_inj = p_pv - p_load
    q_inj = -q_pv - q_load
    return p_inj.to(device=dev, dtype=dt), q_inj.to(device=dev, dtype=dt)


def _power_balance_loss_from_batch(
    v_n: torch.Tensor,
    batch: Data,
    *,
    n_nodes: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    pf: PfPhysicsState,
    cap_logits: torch.Tensor | None = None,
    reg_pred: torch.Tensor | None = None,
    x_mean: torch.Tensor | None = None,
    x_std: torch.Tensor | None = None,
    reg_loss: str = "mse",
    reg_mean: torch.Tensor | None = None,
    reg_std: torch.Tensor | None = None,
    reg_logits: list[torch.Tensor] | None = None,
    reg_class_values: torch.Tensor | None = None,
) -> torch.Tensor:
    y_mean_ri = y_mean.view(1, n_nodes, 2).float()
    y_std_ri = y_std.view(1, n_nodes, 2).float()
    pred_ri = v_n.view(batch.num_graphs, n_nodes, 2).float() * y_std_ri + y_mean_ri
    assert pf.Y_re_base is not None and pf.Y_im_base is not None

    if cap_logits is None or reg_pred is None:
        raise ValueError("Power-balance loss requires model cap_logits and reg_pred outputs.")
    if x_mean is None or x_std is None:
        raise ValueError("Power-balance loss requires x_mean and x_std for node-feature denormalization.")

    cap_on = torch.sigmoid(cap_logits.float())
    if pf.detach_controls:
        cap_on = cap_on.detach()
    tap_pu = _expected_reg_tap_pu(
        reg_pred,
        reg_loss=reg_loss,
        reg_mean=reg_mean,
        reg_std=reg_std,
        reg_logits=reg_logits,
        reg_class_values=reg_class_values,
    )
    if pf.detach_controls:
        tap_pu = tap_pu.detach()
    x_den = _denorm_node_features(batch.x, batch, n_nodes, x_mean.float(), x_std.float())
    p_inj, q_inj = _assemble_pf_injections(
        x_den,
        pf.node_feature_cols,
        batch=batch,
        n_nodes=n_nodes,
    )

    use_sparse = bool(pf.use_sparse_y and pf.y_coo is not None)
    if use_sparse:
        y_re, y_im = None, None
    else:
        y_re, y_im = _ybus_with_predicted_controls(
            pf.Y_re_base,
            pf.Y_im_base,
            reg_edges=pf.reg_edges,
            cap_banks=pf.cap_banks,
            tap_pu=tap_pu,
            cap_on=cap_on,
            s_base_kva=pf.s_base_kva,
            batch_size=int(batch.num_graphs),
            v_scale_volts=pf.v_scale_volts,
        )

    return nodal_power_balance_residual(
        pred_ri,
        p_inj,
        q_inj,
        y_re,
        y_im,
        pf.mask,
        pf.s_base_kva,
        v_scale_volts=pf.v_scale_volts,
        huber_delta_kw=pf.huber_delta_kw,
        y_coo=pf.y_coo if use_sparse else None,
        reg_edges=pf.reg_edges if use_sparse else None,
        cap_banks=pf.cap_banks if use_sparse else None,
        tap_pu=tap_pu if use_sparse else None,
        cap_on=cap_on if use_sparse else None,
        use_sparse_y=use_sparse,
    )


def _pf_loss_if_enabled(
    pf: PfPhysicsState,
    v_n: torch.Tensor,
    batch: Data,
    *,
    n_nodes: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_mean: torch.Tensor | None,
    x_std: torch.Tensor | None,
    cap_logits: torch.Tensor,
    reg_pred: torch.Tensor,
    reg_loss: str,
    reg_mean: torch.Tensor | None,
    reg_std: torch.Tensor | None,
    reg_logits: list[torch.Tensor] | None = None,
    reg_class_values: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if pf.weight <= 0.0 or pf.Y_re_base is None:
        return None
    with _pf_physics_fp32_ctx():
        return _power_balance_loss_from_batch(
            v_n,
            batch,
            n_nodes=n_nodes,
            y_mean=y_mean,
            y_std=y_std,
            pf=pf,
            cap_logits=cap_logits,
            reg_pred=reg_pred,
            x_mean=x_mean,
            x_std=x_std,
            reg_loss=reg_loss,
            reg_mean=reg_mean,
            reg_std=reg_std,
            reg_logits=reg_logits,
            reg_class_values=reg_class_values,
        )


def _resolve_pf_reg_catalog(
    edges_path: Path, data_root: Path, args: argparse.Namespace, repo: Path
) -> Path | None:
    raw = str(getattr(args, "pf_reg_edge_catalog", "") or "").strip()
    if raw:
        pf_root = _effective_pf_data_root(
            data_root=data_root,
            args=args,
            repo=repo,
            chunk_parent=edges_path.parent.parent
            if edges_path.parent.name.startswith("run_")
            else None,
        )
        p = Path(raw)
        if not p.is_absolute():
            p = (pf_root / p).resolve()
        return p if p.is_file() else None
    from gnn2_pf_data_paths import resolve_pf_catalog_paths

    chunk_parent = edges_path.parent.parent if edges_path.parent.name.startswith("run_") else None
    pf_root = _effective_pf_data_root(
        data_root=data_root, args=args, repo=repo, chunk_parent=chunk_parent
    )
    try:
        reg, _, _ = resolve_pf_catalog_paths(
            repo=repo,
            preferred_root=pf_root,
            chunk_parent=chunk_parent,
        )
    except FileNotFoundError:
        return None
    return reg


def _setup_pf_physics(
    *,
    edges_path: Path,
    nodes_path: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
    args: argparse.Namespace,
    device: torch.device,
    data_root: Path,
    cap_cols: list[str],
    reg_cols: list[str],
    meta_aux_cols: list[str],
    node_feature_cols: list[str],
    node_pe_csv: Path | None = None,
) -> PfPhysicsState:
    w = float(getattr(args, "loss_power_balance_weight", 0.0) or 0.0)
    if w <= 0.0:
        return PfPhysicsState(weight=0.0)
    s_base = float(getattr(args, "pf_s_base_kva", 5000.0))
    kv_base = float(getattr(args, "pf_kv_base", 12.47))
    z_base = (kv_base * 1000.0) ** 2 / (s_base * 1000.0)
    detach = bool(getattr(args, "pf_detach_controls", False))
    huber_delta_kw = float(getattr(args, "pf_huber_delta_kw", 10.0) or 10.0)

    repo = Path(__file__).resolve().parent
    chunk_parent = nodes_path.parent.parent if nodes_path.parent.name.startswith("run_") else None
    pf_root = _effective_pf_data_root(
        data_root=data_root, args=args, repo=repo, chunk_parent=chunk_parent
    )

    reg_catalog = _resolve_pf_reg_catalog(edges_path, data_root, args, repo)
    if reg_catalog is None:
        raise FileNotFoundError(
            "Physics loss enabled but regulator edge catalog not found. "
            "Set --pf_reg_edge_catalog or place hetero_mv_edge_catalog.csv under data_root."
        )
    reg_edges = _load_regulator_edges_for_pf(reg_catalog, node_to_local, reg_cols, None)
    skip_reg_pairs = {_undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re, y_im = _build_ybus_siemens_from_edge_csv(
        edges_path, node_to_local, n_nodes, skip_undirected=skip_reg_pairs
    )
    balance_mode = str(args.pf_balance_nodes)
    raw_explicit = str(getattr(args, "pf_balance_node_list_csv", "") or "").strip()
    if raw_explicit:
        explicit_csv = Path(raw_explicit)
        if not explicit_csv.is_absolute():
            for base in (pf_root, repo):
                candidate = (base / explicit_csv).resolve()
                if candidate.is_file():
                    explicit_csv = candidate
                    break
            else:
                explicit_csv = (pf_root / explicit_csv).resolve()
        pf_mask = _load_pf_balance_mask_from_explicit_list(
            explicit_csv, node_to_local, n_nodes
        )
        idx_to_node = {int(li): str(node) for node, li in node_to_local.items()}
        hetero_nodes = _load_pf_hetero_node_indices(pf_root, node_to_local)
        _warn_pf_balance_node_issues(
            pf_mask,
            idx_to_node,
            y_re,
            y_im,
            hetero_nodes=hetero_nodes or None,
        )
    else:
        distance_csv, distance_tried = _resolve_pf_electrical_distance_csv(
            nodes_csv=nodes_path,
            node_pe_csv=node_pe_csv,
            data_root=data_root,
            repo=repo,
            mode=balance_mode,
            pf_preferred=pf_root,
        )
        use_distance_csv = (
            distance_csv
            if distance_csv is not None and distance_csv != nodes_path
            else None
        )
        mv_fallback = balance_mode.strip().lower() == "mv" and distance_csv is None
        pf_mask = _load_pf_balance_mask(
            nodes_path,
            node_to_local,
            n_nodes,
            balance_mode,
            distance_csv=use_distance_csv,
            mv_fallback_all_non_slack=mv_fallback,
            distance_tried=distance_tried,
        )
        if balance_mode.strip().lower() == "mv":
            pf_mask = _apply_pf_balance_mask_refinement(
                pf_mask,
                node_to_local,
                pf_root,
                y_re,
                y_im,
                args,
                label="PF MV",
            )

    raw_cap = str(getattr(args, "pf_cap_nodes_csv", "") or "").strip()
    if raw_cap:
        cap_nodes_csv = Path(raw_cap)
        if not cap_nodes_csv.is_absolute():
            cap_nodes_csv = (pf_root / cap_nodes_csv).resolve()
        if not cap_nodes_csv.is_file():
            raise FileNotFoundError(f"--pf_cap_nodes_csv not found: {cap_nodes_csv}")
    else:
        from gnn2_pf_data_paths import resolve_pf_catalog_paths

        try:
            _, cap_nodes_csv, _ = resolve_pf_catalog_paths(
                repo=repo,
                preferred_root=pf_root,
                chunk_parent=chunk_parent,
            )
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                "Physics loss enabled but capacitor bus map not found. "
                "Set --pf_cap_nodes_csv."
            ) from ex

    load_cols = {"p_load_kw", "q_load_kvar", "p_pv_kw"}
    missing_load = load_cols - {str(c).lower() for c in node_feature_cols}
    if missing_load:
        raise ValueError(
            f"Physics loss requires node feature columns {sorted(missing_load)} in --node_feature_cols"
        )
    if cap_cols and len(cap_cols) != len({_cap_col_stem(c) for c in cap_cols if _cap_col_stem(c)}):
        raise ValueError(f"Duplicate or unparseable cap columns in meta: {cap_cols!r}")

    meta_csv = nodes_path.parent / str(getattr(args, "meta_csv", "gnn_sample_meta.csv"))
    cap_dss = repo / "8500-node" / "Capacitors.dss"
    cap_banks = _resolve_cap_bus_nodes(
        cap_cols,
        node_to_local,
        cap_nodes_csv=cap_nodes_csv,
        meta_csv=meta_csv if meta_csv.is_file() else None,
        capacitors_dss=cap_dss,
    )
    if cap_cols:
        n_cap = len(cap_cols)
        mapped_cj = {int(cj) for _, _, cj in cap_banks}
        if len(mapped_cj) != n_cap:
            missing = [cap_cols[j] for j in range(n_cap) if j not in mapped_cj]
            raise ValueError(
                f"Capacitor meta columns not mapped to any stamped node: {missing!r} "
                f"(stamped {len(cap_banks)} node(s) for {n_cap} logical bank(s))."
            )

    from gnn2_pf_bus_kv import load_or_build_bus_kv_tensors

    raw_kv_csv = str(getattr(args, "pf_bus_kv_base_csv", "") or "").strip()
    cache_csv = Path(raw_kv_csv) if raw_kv_csv else None
    if cache_csv is not None and not cache_csv.is_absolute():
        cache_csv = (pf_root / cache_csv).resolve()
    v_scale_np, _, kv_cache_path = load_or_build_bus_kv_tensors(
        repo=repo,
        data_root=pf_root,
        node_to_local=node_to_local,
        n_nodes=n_nodes,
        cache_csv=cache_csv,
    )
    v_scale_t = torch.tensor(v_scale_np, dtype=torch.float32, device=device)

    use_sparse_y = bool(int(getattr(args, "pf_sparse_y", 1) or 1))
    y_coo = _dense_y_to_coo(y_re, y_im).to(device) if use_sparse_y else None
    nnz = int(y_coo.row.numel()) if y_coo is not None else 0
    dense_elems = int(n_nodes) * int(n_nodes)
    pf_debug_nan = _resolve_pf_debug_nan(args)
    print(
        f"Power-balance physics: weight={w}, units=physical (V volts, Y Siemens, kW Huber), "
        f"pf_data_root={pf_root}, nodes={args.pf_balance_nodes}, "
        f"S_base={s_base} kVA, V_base={kv_base} kV, Z_base={z_base:.4f} ohm, "
        f"huber_delta_kw={huber_delta_kw}, "
        f"masked_nodes={int(pf_mask.sum())}/{n_nodes} (slack excluded), "
        f"line_base_edges={int((y_re.abs() > 0).sum().item())}, "
        f"reg_branches={len(reg_edges)}, cap_banks={len(cap_banks)}, "
        f"sparse_y={use_sparse_y} (nnz={nnz} vs dense {dense_elems}, "
        f"~{100.0 * (1.0 - nnz / max(dense_elems, 1)):.2f}% zero skip)"
        + (f", kv_cache={kv_cache_path}" if kv_cache_path else "")
        + (", pf_debug_nan=1 (epoch-1 first train batch + any non-finite loss)" if pf_debug_nan else ""),
        flush=True,
    )

    idx_to_node = {int(li): str(node) for node, li in node_to_local.items()}
    return PfPhysicsState(
        weight=w,
        s_base_kva=s_base,
        huber_delta_kw=huber_delta_kw,
        v_scale_volts=v_scale_t,
        Y_re_base=y_re.to(device),
        Y_im_base=y_im.to(device),
        y_coo=y_coo,
        use_sparse_y=use_sparse_y,
        mask=pf_mask.to(device),
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        detach_controls=detach,
        node_feature_cols=list(node_feature_cols),
        pf_debug_nan=pf_debug_nan,
        idx_to_node=idx_to_node,
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


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_reg_loss(spec: str) -> str:
    """``mse`` / ``mae`` on z-scored taps, or ``ce`` / ``cce`` (categorical cross-entropy on tap classes)."""
    s = str(spec).strip().lower()
    if s in ("mse", "l2"):
        return "mse"
    if s in ("mae", "l1"):
        return "mae"
    if s in ("ce", "cce", "cross_entropy", "cross-entropy", "nll"):
        return "ce"
    raise ValueError(f"--reg_loss must be 'mse', 'mae', or 'ce', got {spec!r}")


def _reg_loss_slug(reg_loss: str) -> str:
    if reg_loss == "ce":
        return "regce"
    if reg_loss == "mae":
        return "regmae"
    return ""


def _build_reg_class_tables(reg_cols: list[str], reg_raw: np.ndarray) -> list[dict]:
    """One discrete tap class set per regulator column (rounded unique tap_pu values)."""
    if reg_raw.ndim != 2 or reg_raw.shape[1] != len(reg_cols):
        raise ValueError(f"reg_raw shape {reg_raw.shape} vs n_reg={len(reg_cols)}")
    tables: list[dict] = []
    for j, col in enumerate(reg_cols):
        yq = np.round(reg_raw[:, j].astype(np.float64), 6)
        classes = np.unique(yq)
        classes = np.sort(classes)
        cls_to_i = {float(c): int(i) for i, c in enumerate(classes.tolist())}
        tables.append(
            {
                "col": str(col),
                "classes": [float(c) for c in classes.tolist()],
                "n_classes": int(len(classes)),
                "class_to_index": {str(float(k)): int(v) for k, v in cls_to_i.items()},
            }
        )
    return tables


def _reg_class_tables_digest(reg_class_tables: list[dict]) -> str:
    """Stable short hash of per-regulator tap class lists (for CE cache keys)."""
    payload = [
        {"col": str(t["col"]), "classes": [float(c) for c in t["classes"]], "n_classes": int(t["n_classes"])}
        for t in reg_class_tables
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]


def _reg_ce_targets_in_range(y_reg: torch.Tensor, reg_class_tables: list[dict]) -> bool:
    if y_reg.dim() != 2 or int(y_reg.shape[1]) != len(reg_class_tables):
        return False
    y = y_reg.detach().cpu()
    for j, tab in enumerate(reg_class_tables):
        nc = int(tab["n_classes"])
        col = y[:, j]
        if col.numel() == 0:
            continue
        lo = int(col.min().item())
        hi = int(col.max().item())
        if lo < 0 or hi >= nc:
            return False
    return True


def _validate_reg_ce_targets(
    target: torch.Tensor,
    reg_logits: list[torch.Tensor],
    *,
    reg_class_tables: list[dict] | None = None,
) -> None:
    """Fail fast before CUDA CE when cached/stale class indices exceed head width."""
    for j, logits in enumerate(reg_logits):
        nc = int(logits.shape[1])
        col = target[:, j].long()
        if col.numel() == 0:
            continue
        lo = int(col.min().item())
        hi = int(col.max().item())
        if lo < 0 or hi >= nc:
            col_name = reg_class_tables[j]["col"] if reg_class_tables and j < len(reg_class_tables) else f"reg#{j}"
            raise ValueError(
                f"Regulator CE target out of range for {col_name!r}: "
                f"min={lo}, max={hi}, n_classes={nc}. "
                "Chunk tensor caches for reg_loss=ce are keyed by the global tap-class table "
                "(all visible chunks). Re-run with a fresh --cache_dir or delete stale "
                "*__regce__*.pt caches when changing --chunk_subdir_glob or chunk set."
            )


def _encode_reg_class_indices(reg_raw: np.ndarray, reg_class_tables: list[dict]) -> np.ndarray:
    out = np.zeros((reg_raw.shape[0], len(reg_class_tables)), dtype=np.int64)
    for j, tab in enumerate(reg_class_tables):
        cls_to_i = {float(c): i for i, c in enumerate(tab["classes"])}
        yq = np.round(reg_raw[:, j].astype(np.float64), 6)
        try:
            out[:, j] = np.array([cls_to_i[float(v)] for v in yq.tolist()], dtype=np.int64)
        except KeyError as ex:
            raise KeyError(
                f"Unseen tap value for {tab['col']!r} (not in training class list). "
                f"Rebuild caches with --reg_loss ce after all chunks are visible."
            ) from ex
    return out


def _collect_reg_raw_all_chunks(
    chunk_dirs: list[Path],
    meta_name: str,
    reg_cols: list[str],
    selected_ids_list: list[list[int] | None],
) -> np.ndarray:
    import pandas as pd

    parts: list[np.ndarray] = []
    usecols = ["sample_id", *reg_cols]
    for ch, sel in zip(chunk_dirs, selected_ids_list):
        df = pd.read_csv(ch / meta_name, usecols=usecols)
        ren = {}
        for c in df.columns:
            cs = str(c)
            if cs.startswith("reg_") and cs.lower() != cs:
                ren[c] = cs.lower()
        if ren:
            df = df.rename(columns=ren)
        if sel is not None:
            want = {_norm_sid(s) for s in sel}
            df = df[df["sample_id"].map(_norm_sid).isin(want)]
        parts.append(df[list(reg_cols)].to_numpy(dtype=np.float64))
    if not parts:
        raise RuntimeError("No regulator tap rows collected from chunk meta CSVs.")
    return np.concatenate(parts, axis=0)


def _classes_to_tensor(reg_class_tables: list[dict]) -> torch.Tensor:
    """Pad class tap values to (n_reg, max_classes); unused entries are NaN."""
    n_reg = len(reg_class_tables)
    max_c = max(int(t["n_classes"]) for t in reg_class_tables) if n_reg else 0
    out = torch.full((n_reg, max_c), float("nan"), dtype=torch.float32)
    for j, tab in enumerate(reg_class_tables):
        for i, c in enumerate(tab["classes"]):
            out[j, i] = float(c)
    return out


def _reg_indices_to_tap_pu(
    pred_idx: torch.Tensor, tgt_idx: torch.Tensor, class_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map class indices to tap pu using ``class_values[j, k]``."""
    dev = pred_idx.device
    cv = class_values.to(device=dev, dtype=torch.float32)
    n_reg = int(cv.shape[0])
    pred_tap = torch.empty(pred_idx.shape, device=dev, dtype=torch.float32)
    tgt_tap = torch.empty_like(pred_tap)
    for j in range(n_reg):
        pred_tap[:, j] = cv[j].index_select(0, pred_idx[:, j].long().clamp(min=0))
        tgt_tap[:, j] = cv[j].index_select(0, tgt_idx[:, j].long().clamp(min=0))
    return pred_tap, tgt_tap


def _reg_loss_scalar(
    pred: torch.Tensor | None,
    target: torch.Tensor,
    reg_loss: str,
    *,
    reg_logits: list[torch.Tensor] | None = None,
    reg_class_tables: list[dict] | None = None,
) -> torch.Tensor:
    if reg_loss == "ce":
        if not reg_logits:
            raise ValueError("reg_loss=ce requires reg_logits from the model forward pass.")
        _validate_reg_ce_targets(target, reg_logits, reg_class_tables=reg_class_tables)
        losses = [F.cross_entropy(reg_logits[j], target[:, j].long()) for j in range(len(reg_logits))]
        return torch.stack(losses).mean()
    assert pred is not None
    if reg_loss == "mae":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


def _reg_loss_elementwise(
    pred: torch.Tensor | None,
    target: torch.Tensor,
    reg_loss: str,
    *,
    reg_logits: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if reg_loss == "ce":
        if not reg_logits:
            raise ValueError("reg_loss=ce requires reg_logits from the model forward pass.")
        return torch.stack(
            [
                F.cross_entropy(reg_logits[j], target[:, j].long(), reduction="none")
                for j in range(len(reg_logits))
            ],
            dim=1,
        )
    assert pred is not None
    if reg_loss == "mae":
        return (pred - target).abs()
    return (pred - target) ** 2


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
        reg_nclasses: list[int] | None = None,
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
        self.reg_nclasses = [int(c) for c in (reg_nclasses or [])]
        self.reg_classification = (
            len(self.reg_nclasses) == self.n_reg
            and self.n_reg > 0
            and all(int(c) >= 2 for c in self.reg_nclasses)
        )
        self._last_reg_logits: list[torch.Tensor] | None = None
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

        if self.reg_classification:
            if not self.per_device_reg_head:
                raise ValueError("reg_loss=ce requires --per_device_reg_head (one classifier per regulator).")
            self.reg_ce_heads = nn.ModuleList([nn.Linear(hidden, int(c)) for c in self.reg_nclasses])
            self.reg_W = None
            self.reg_b = None
            self.reg_head = None
        elif self.per_device_reg_head:
            self.reg_W = nn.Parameter(torch.randn(self.n_reg, self.hidden) * 0.02)
            self.reg_b = nn.Parameter(torch.zeros(self.n_reg))
            self.reg_head = None
            self.reg_ce_heads = None
        else:
            self.reg_head = nn.Linear(hidden, 1, bias=False)
            self.reg_W = None
            self.reg_b = None
            self.reg_ce_heads = None

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

    def _predict_reg(self, T_reg: torch.Tensor) -> torch.Tensor:
        if self.reg_classification:
            assert self.reg_ce_heads is not None
            logits = [head(T_reg[:, j, :]) for j, head in enumerate(self.reg_ce_heads)]
            self._last_reg_logits = logits
            return torch.stack([lg.argmax(dim=-1) for lg in logits], dim=1).to(torch.float32)
        self._last_reg_logits = None
        if self.per_device_reg_head:
            assert self.reg_W is not None and self.reg_b is not None
            return (T_reg * self.reg_W.unsqueeze(0)).sum(-1) + self.reg_b.unsqueeze(0)
        assert self.reg_head is not None
        return self.reg_head(T_reg).squeeze(-1)

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
        reg_pred = self._predict_reg(T_reg)
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
        reg_pred = self._predict_reg(T_reg)
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
    *,
    reg_class_tables: list[dict] | None = None,
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
    if reg_class_tables is not None:
        y_reg_idx = _encode_reg_class_indices(reg_raw, reg_class_tables)
        return torch.from_numpy(y_cap), torch.from_numpy(y_reg_idx)
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
        if batch.y_reg.dtype in (torch.int64, torch.int32, torch.long):
            batch.y_reg = batch.y_reg.long()
        else:
            batch.y_reg = batch.y_reg.float()
    if hasattr(batch, "y_pv") and batch.y_pv is not None:
        batch.y_pv = batch.y_pv.float()
    return batch


def _metric_key_segment(s: str) -> str:
    """Safe single-segment key fragment (no spaces)."""
    import re as _re

    return _re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")[:120]


def _print_per_head_two_lines(
    prefix: str,
    label: str,
    names: list[str],
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    *,
    max_tokens_per_line: int = 6,
) -> None:
    """Pretty-print name=value pairs, wrapping long lists across lines."""
    if not names:
        return

    def _pack(vals: np.ndarray, phase: str) -> None:
        buf: list[str] = []
        ntok = 0
        for i, nm in enumerate(names):
            piece = f"{nm}={float(vals[i]):.4f}"
            if ntok >= max_tokens_per_line and buf:
                print(f"{prefix}  {label} {phase}: " + "  ".join(buf), flush=True)
                buf = []
                ntok = 0
            buf.append(piece)
            ntok += 1
        if buf:
            print(f"{prefix}  {label} {phase}: " + "  ".join(buf), flush=True)

    _pack(train_vals, "train")
    _pack(val_vals, "val")


def _print_test_per_head_block(tag: str, met: dict[str, float], cap_cols: list[str], reg_cols: list[str], meta_cols: list[str]) -> None:
    print(f"{tag} Test per-head cap_BCE:", flush=True)
    for nm in cap_cols:
        k = f"cap_bce__{_metric_key_segment(nm)}"
        v = met.get(k, float("nan"))
        print(f"  {nm}={v:.6f}", flush=True)
    print(f"{tag} Test per-head reg_MSE / reg_MAE (nrm / tap pu):", flush=True)
    for nm in reg_cols:
        seg = _metric_key_segment(nm)
        kn = f"reg_mse_nrm__{seg}"
        kp = f"reg_mse_pu__{seg}"
        kan = f"reg_mae_nrm__{seg}"
        kap = f"reg_mae_pu__{seg}"
        print(
            f"  {nm}: MSE nrm={met.get(kn, float('nan')):.6f} pu={met.get(kp, float('nan')):.6f}  "
            f"MAE nrm={met.get(kan, float('nan')):.6f} pu={met.get(kap, float('nan')):.6f}",
            flush=True,
        )
    if meta_cols:
        print(f"{tag} Test per-head meta_aux_MSE (nrm / raw):", flush=True)
        for nm in meta_cols:
            seg = _metric_key_segment(nm)
            kn = f"meta_aux_mse_nrm__{seg}"
            kr = f"meta_aux_mse_raw__{seg}"
            print(f"  {nm}: nrm={met.get(kn, float('nan')):.6f}  raw={met.get(kr, float('nan')):.6f}", flush=True)


def _empty_eval_metrics(cap_cols: list[str], reg_cols: list[str], meta_aux_cols: list[str]) -> dict[str, float]:
    """All-NaN template including per-head keys (for zero-test-samples edge case)."""
    out: dict[str, float] = {
        "mae_vmag_pu": float("nan"),
        "rmse_vmag_pu": float("nan"),
        "mae_angle_deg": float("nan"),
        "rmse_angle_deg": float("nan"),
        "r2_vmag_mean": float("nan"),
        "r2_vmag_min": float("nan"),
        "mae_vmag_worst_node": float("nan"),
        "cap_bce": float("nan"),
        "reg_mse_normalized": float("nan"),
        "reg_mae_normalized": float("nan"),
        "reg_mse_tap_pu": float("nan"),
        "reg_mae_tap_pu": float("nan"),
        "pv_mse_normalized": float("nan"),
        "pv_mse_raw": float("nan"),
    }
    for nm in cap_cols:
        out[f"cap_bce__{_metric_key_segment(nm)}"] = float("nan")
    for nm in reg_cols:
        seg = _metric_key_segment(nm)
        out[f"reg_mse_nrm__{seg}"] = float("nan")
        out[f"reg_mse_pu__{seg}"] = float("nan")
        out[f"reg_mae_nrm__{seg}"] = float("nan")
        out[f"reg_mae_pu__{seg}"] = float("nan")
    for nm in meta_aux_cols:
        seg = _metric_key_segment(nm)
        out[f"meta_aux_mse_nrm__{seg}"] = float("nan")
        out[f"meta_aux_mse_raw__{seg}"] = float("nan")
    return out


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
    cap_cols: list[str] | None = None,
    reg_cols: list[str] | None = None,
    meta_aux_cols: list[str] | None = None,
    reg_loss: str = "mse",
    reg_class_values: torch.Tensor | None = None,
    base_model: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    core = base_model if base_model is not None else model
    preds, tgts = [], []
    cap_logits_all, cap_tgt_all = [], []
    reg_pred_all, reg_tgt_all = [], []
    pv_pred_all, pv_tgt_all = [], []
    reg_ce_losses: list[torch.Tensor] = []
    reg_acc_hits = 0
    reg_acc_n = 0
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
        if reg_loss == "ce" and getattr(core, "_last_reg_logits", None):
            logits = core._last_reg_logits
            tgt_l = y_reg.long()
            reg_ce_losses.append(
                torch.stack(
                    [F.cross_entropy(logits[j], tgt_l[:, j]) for j in range(len(logits))],
                    dim=0,
                ).mean()
            )
            reg_acc_hits += int((r_p.long() == tgt_l).sum().item())
            reg_acc_n += int(tgt_l.numel())
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
    if reg_loss == "ce" and reg_class_values is not None:
        met["reg_ce_loss"] = float(torch.stack(reg_ce_losses).mean().item()) if reg_ce_losses else float("nan")
        met["reg_accuracy"] = float(reg_acc_hits / max(reg_acc_n, 1))
        pred_tap, true_tap = _reg_indices_to_tap_pu(
            rp.long(), rt.long(), reg_class_values.to(device=rp.device)
        )
        met["reg_mse_normalized"] = float(F.mse_loss(rp, rt.to(rp.dtype)).item())
        met["reg_mae_normalized"] = float(F.l1_loss(rp, rt.to(rp.dtype)).item())
        met["reg_mse_tap_pu"] = float(F.mse_loss(pred_tap, true_tap).item())
        met["reg_mae_tap_pu"] = float(F.l1_loss(pred_tap, true_tap).item())
        rp_denorm, rt_denorm = pred_tap, true_tap
    else:
        met["reg_ce_loss"] = float("nan")
        met["reg_accuracy"] = float("nan")
        met["reg_mse_normalized"] = float(F.mse_loss(rp, rt.to(rp.dtype)).item())
        met["reg_mae_normalized"] = float(F.l1_loss(rp, rt.to(rp.dtype)).item())
        rp_denorm = rp * reg_std.to(rp.device) + reg_mean.to(rp.device)
        rt_denorm = rt * reg_std.to(rt.device) + reg_mean.to(rt.device)
        met["reg_mse_tap_pu"] = float(F.mse_loss(rp_denorm, rt_denorm.to(rp_denorm.dtype)).item())
        met["reg_mae_tap_pu"] = float(F.l1_loss(rp_denorm, rt_denorm.to(rp_denorm.dtype)).item())
    cap_list = list(cap_cols or [])
    reg_list = list(reg_cols or [])
    meta_list = list(meta_aux_cols or [])
    for nm in cap_list:
        met[f"cap_bce__{_metric_key_segment(nm)}"] = float("nan")
    for nm in reg_list:
        seg = _metric_key_segment(nm)
        met[f"reg_mse_nrm__{seg}"] = float("nan")
        met[f"reg_mse_pu__{seg}"] = float("nan")
        met[f"reg_mae_nrm__{seg}"] = float("nan")
        met[f"reg_mae_pu__{seg}"] = float("nan")
    for nm in meta_list:
        seg = _metric_key_segment(nm)
        met[f"meta_aux_mse_nrm__{seg}"] = float("nan")
        met[f"meta_aux_mse_raw__{seg}"] = float("nan")
    if cap_list and cap_log.shape[1] == len(cap_list):
        for j, nm in enumerate(cap_list):
            met[f"cap_bce__{_metric_key_segment(nm)}"] = float(
                F.binary_cross_entropy_with_logits(cap_log[:, j], cap_t[:, j]).item()
            )
    if reg_list and rp.shape[1] == len(reg_list):
        rm = reg_mean.to(rp.device)
        rs = reg_std.to(rp.device)
        for j, nm in enumerate(reg_list):
            seg = _metric_key_segment(nm)
            met[f"reg_mse_nrm__{seg}"] = float(F.mse_loss(rp[:, j], rt[:, j].to(rp.dtype)).item())
            met[f"reg_mae_nrm__{seg}"] = float(F.l1_loss(rp[:, j], rt[:, j].to(rp.dtype)).item())
            rpj = rp[:, j] * rs[0, j] + rm[0, j]
            rtj = rt[:, j] * rs[0, j] + rm[0, j]
            met[f"reg_mse_pu__{seg}"] = float(F.mse_loss(rpj, rtj.to(rpj.dtype)).item())
            met[f"reg_mae_pu__{seg}"] = float(F.l1_loss(rpj, rtj.to(rpj.dtype)).item())
    if use_pv and pv_pred_all:
        pp = torch.cat(pv_pred_all, dim=0)
        pt = torch.cat(pv_tgt_all, dim=0)
        met["pv_mse_normalized"] = float(F.mse_loss(pp, pt.to(pp.dtype)).item())
        pp_den = pp * pv_std.to(pp.device) + pv_mean.to(pp.device)
        pt_den = pt * pv_std.to(pt.device) + pv_mean.to(pt.device)
        met["pv_mse_raw"] = float(F.mse_loss(pp_den, pt_den.to(pp_den.dtype)).item())
        n_m = int(pp.shape[1])
        names_m = [meta_list[j] if j < len(meta_list) else f"meta_aux_{j}" for j in range(n_m)]
        pvm = pv_mean.to(pp.device)
        pvs = pv_std.to(pp.device)
        for j, nm in enumerate(names_m[:n_m]):
            seg = _metric_key_segment(nm)
            met[f"meta_aux_mse_nrm__{seg}"] = float(F.mse_loss(pp[:, j], pt[:, j].to(pp.dtype)).item())
            pdj = pp[:, j] * pvs[0, j] + pvm[0, j]
            tdj = pt[:, j] * pvs[0, j] + pvm[0, j]
            met[f"meta_aux_mse_raw__{seg}"] = float(F.mse_loss(pdj, tdj.to(pdj.dtype)).item())
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
    reg_slug: str = "",
    reg_classes_digest: str = "",
) -> Path:
    if float(sample_frac) >= 1.0:
        tag = "full"
    else:
        tag = f"sf{float(sample_frac):.6f}_s{int(seed)}_c{int(chunk_idx)}"
    base = f"{chunk_name}__{tag}"
    if str(feat_slug).strip():
        base = f"{base}__{str(feat_slug).strip()}"
    if str(reg_slug).strip():
        base = f"{base}__{str(reg_slug).strip()}"
    if str(reg_classes_digest).strip():
        base = f"{base}__rc{str(reg_classes_digest).strip()}"
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
    reg_class_tables: list[dict] | None = None,
    reg_target_mode: str = "regression",
    reg_classes_digest: str = "",
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

    want_reg_mode = str(reg_target_mode).strip().lower()
    want_reg_digest = ""
    if want_reg_mode == "class":
        if reg_class_tables is None:
            raise ValueError("reg_target_mode=class requires reg_class_tables.")
        want_reg_digest = str(reg_classes_digest or _reg_class_tables_digest(reg_class_tables)).strip()

    if cache_pt.is_file():
        z = torch.load(cache_pt, map_location="cpu", weights_only=False)
        cache_ok = str(z.get("reg_target_mode", "regression")).lower() == want_reg_mode
        if cache_ok and want_reg_mode == "class":
            stored_digest = str(z.get("reg_class_tables_digest", "") or "").strip()
            y_reg_cached = z.get("y_reg")
            if stored_digest != want_reg_digest:
                cache_ok = False
                print(
                    f"chunk cache reg_class_tables_digest mismatch "
                    f"({stored_digest!r} != {want_reg_digest!r}); rebuilding: {cache_pt}",
                    flush=True,
                )
            elif y_reg_cached is None or not _reg_ce_targets_in_range(y_reg_cached, reg_class_tables):
                cache_ok = False
                print(
                    f"chunk cache y_reg class indices out of range for current tap tables; "
                    f"rebuilding: {cache_pt}",
                    flush=True,
                )
        elif not cache_ok:
            print(
                f"chunk cache reg_target_mode={z.get('reg_target_mode')!r} != {want_reg_mode!r}; "
                f"rebuilding: {cache_pt}",
                flush=True,
            )
        if cache_ok:
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
            if want_reg_mode == "class":
                y_reg = z["y_reg"].to(dtype=torch.long)
            else:
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
            y_cap, y_reg = _load_meta_aux(
                mp_, sample_ids, cap_cols, reg_cols, reg_class_tables=reg_class_tables
            )
            y_cap = y_cap.to(dtype=torch.float32)
            if want_reg_mode == "class":
                y_reg = y_reg.to(dtype=torch.long)
            else:
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
                "reg_target_mode": want_reg_mode,
            }
            if want_reg_digest:
                row["reg_class_tables_digest"] = want_reg_digest
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
    y_cap, y_reg = _load_meta_aux(mp_, sample_ids, cap_cols, reg_cols, reg_class_tables=reg_class_tables)
    y_cap = y_cap.to(dtype=torch.float32)
    if want_reg_mode == "class":
        y_reg = y_reg.to(dtype=torch.long)
    else:
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
        "reg_target_mode": want_reg_mode,
    }
    if want_reg_digest:
        row["reg_class_tables_digest"] = want_reg_digest
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
    reg_loss: str = "mse",
    reg_class_tables: list[dict] | None = None,
    reg_target_mode: str = "regression",
    reg_classes_digest: str = "",
    reg_class_values: torch.Tensor | None = None,
    base_model: nn.Module | None = None,
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
            reg_class_tables=reg_class_tables,
            reg_target_mode=reg_target_mode,
            reg_classes_digest=reg_classes_digest,
        )
        if reg_loss == "ce":
            y_reg_n = y_reg.to(dtype=torch.long)
        else:
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
            cap_cols=cap_cols,
            reg_cols=reg_cols,
            meta_aux_cols=list(pv_aux_cols) if pv_aux_cols else [],
            reg_loss=reg_loss,
            reg_class_values=reg_class_values,
            base_model=base_model,
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
        return _empty_eval_metrics(cap_cols, reg_cols, list(pv_aux_cols or []))
    return {k: met_acc[k] / float(wtot) for k in met_acc}


def _da_gps_ckpt_meta(
    *,
    n_nodes: int,
    hidden: int,
    layers: int,
    heads: int,
    n_cap: int,
    n_reg: int,
    n_system_tokens: int,
    node_emb_dim: int,
    edge_emb_dim: int,
    per_node_heads: bool,
    per_device_cap_head: bool,
    per_device_reg_head: bool,
    n_pv_aux: int,
    pv_target_cols: list[str],
    meta_aux_target_cols: list[str],
    cap_target_cols: list[str],
    reg_target_cols: list[str],
    reg_loss: str,
    reg_nclasses: list[int] | None = None,
    chunk_parent: str | None = None,
    chunk_folders: list[str] | None = None,
) -> dict[str, object]:
    """Architecture / target metadata shared by ``da_gps_multitask_best.pt`` and ``training_last.pt``."""
    meta: dict[str, object] = {
        "n_nodes": int(n_nodes),
        "hidden": int(hidden),
        "layers": int(layers),
        "heads": int(heads),
        "n_cap": int(n_cap),
        "n_reg": int(n_reg),
        "n_system_tokens": int(n_system_tokens),
        "node_emb_dim": int(node_emb_dim),
        "edge_emb_dim": int(edge_emb_dim),
        "per_node_heads": bool(per_node_heads),
        "per_device_cap_head": bool(per_device_cap_head),
        "per_device_reg_head": bool(per_device_reg_head),
        "n_pv_aux": int(n_pv_aux),
        "pv_target_cols": list(pv_target_cols),
        "meta_aux_target_cols": list(meta_aux_target_cols),
        "cap_target_cols": list(cap_target_cols),
        "reg_target_cols": list(reg_target_cols),
        "reg_loss": str(reg_loss),
        "reg_nclasses": list(reg_nclasses) if reg_nclasses is not None else [],
    }
    if chunk_parent is not None:
        meta["chunk_parent"] = str(chunk_parent)
    if chunk_folders is not None:
        meta["chunk_folders"] = [str(p) for p in chunk_folders]
    return meta


def _da_gps_checkpoint_payload(
    base_model: nn.Module,
    ckpt_meta: dict[str, object],
) -> dict[str, object]:
    """Full inference bundle: ``model_state_dict`` plus all keys in ``da_gps_multitask_best.pt``."""
    return {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
        **dict(ckpt_meta),
    }


def _write_da_gps_run_manifest(
    out_dir: Path,
    *,
    task: str,
    chunk_parent: str | None,
    chunks: list[str],
    cache_dir: str | None,
    args: argparse.Namespace,
    cap_cols: list[str],
    reg_cols: list[str],
    meta_aux_cols: list[str],
    reg_loss: str,
    n_chunks: int | None = None,
) -> None:
    """Written before the training loop so daily compare / snapshots work without ``da_gps_report.json``."""
    manifest: dict[str, object] = {
        "task": str(task),
        "chunk_parent": str(chunk_parent) if chunk_parent else "",
        "chunks": list(chunks),
        "hyperparameters": vars(args),
        "cap_target_cols": list(cap_cols),
        "reg_target_cols": list(reg_cols),
        "meta_aux_target_cols": list(meta_aux_cols),
        "reg_loss": str(reg_loss),
        "manifest_stage": "pre_train",
    }
    if cache_dir is not None:
        manifest["chunk_tensor_cache_dir"] = str(cache_dir)
    if n_chunks is not None:
        manifest["n_chunks"] = int(n_chunks)
    path = out_dir / "da_gps_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote run manifest (for daily compare / mid-train snapshots): {path}", flush=True)


def _save_periodic_training_checkpoint(
    path: Path,
    base_model: nn.Module,
    opt: torch.optim.Optimizer,
    sch: object,
    scaler: object | None,
    ckpt_meta: dict[str, object],
    *,
    epoch: int,
    bad: int,
    best_val: float,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    """Atomic write: same architecture metadata as ``da_gps_multitask_best.pt`` plus resume fields."""
    payload: dict[str, object] = {
        **_da_gps_checkpoint_payload(base_model, ckpt_meta),
        "checkpoint_type": "training_last",
        "epoch": int(epoch),
        "bad": int(bad),
        "best_val": float(best_val),
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



def _chunk_dirs_from_subdir_glob(chunk_parent: Path, glob_pat: str) -> list[Path]:
    """Resolve chunk folders: fnmatch pattern or comma-separated exact names."""
    glob_pat = str(glob_pat).strip()
    if "," in glob_pat:
        allowed = {s.strip() for s in glob_pat.split(",") if s.strip()}
        chunk_dirs = sorted(
            [p for p in chunk_parent.iterdir() if p.is_dir() and p.name in allowed],
            key=lambda p: p.name,
        )
        missing = allowed - {p.name for p in chunk_dirs}
        if missing:
            raise FileNotFoundError(
                f"--chunk_subdir_glob comma-list missing under {chunk_parent}: {sorted(missing)}"
            )
        return chunk_dirs
    return sorted(
        [p for p in chunk_parent.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob_pat)],
        key=lambda p: p.name,
    )

def main_multi_chunk(args: argparse.Namespace, repo: Path) -> None:
    """Train on many chunk folders (no merged mega-CSV). One chunk loaded at a time."""
    _set_seed(args.seed)
    dropout = 0.0 if args.disable_dropout else float(args.dropout)
    reg_loss = _parse_reg_loss(args.reg_loss)
    reg_target_mode = "class" if reg_loss == "ce" else "regression"
    if reg_loss == "ce" and not bool(args.per_device_reg_head):
        raise ValueError("--reg_loss ce requires --per_device_reg_head (one softmax head per regulator).")
    reg_slug = _reg_loss_slug(reg_loss)
    print(
        f"regulator tap training loss: {reg_loss} "
        f"({'discrete tap classes + cross-entropy' if reg_loss == 'ce' else 'z-scored continuous taps'})",
        flush=True,
    )

    chunk_parent = Path(args.chunk_parent).resolve()
    if not chunk_parent.is_dir():
        raise FileNotFoundError(chunk_parent)

    glob_pat = str(args.chunk_subdir_glob)
    chunk_dirs = _chunk_dirs_from_subdir_glob(chunk_parent, glob_pat)
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

    reg_class_tables: list[dict] | None = None
    reg_nclasses: list[int] = []
    if reg_loss == "ce":
        _sel_for_classes: list[list[int] | None] = []
        for ci, ch in enumerate(chunk_dirs):
            _sel_for_classes.append(
                _select_sample_ids_from_meta(ch / meta_name, float(args.sample_frac), int(args.seed), ci)
            )
        reg_raw_all = _collect_reg_raw_all_chunks(chunk_dirs, meta_name, reg_cols, _sel_for_classes)
        reg_class_tables = _build_reg_class_tables(reg_cols, reg_raw_all)
        reg_nclasses = [int(t["n_classes"]) for t in reg_class_tables]
        (out_dir / "reg_class_tables.json").write_text(
            json.dumps(reg_class_tables, indent=2), encoding="utf-8"
        )
        print("reg_loss=ce: per-regulator tap classes (rounded unique tap_pu):", flush=True)
        for tab in reg_class_tables:
            print(f"  {tab['col']}: n_classes={tab['n_classes']}", flush=True)

    reg_classes_digest = _reg_class_tables_digest(reg_class_tables) if reg_class_tables else ""

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
            cache_dir,
            ch.name,
            float(args.sample_frac),
            int(args.seed),
            ci,
            feat_slug=feat_slug,
            meta_aux_slug=maux_slug,
            reg_slug=reg_slug,
            reg_classes_digest=reg_classes_digest,
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
            reg_class_tables=reg_class_tables,
            reg_target_mode=reg_target_mode,
            reg_classes_digest=reg_classes_digest,
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

        if reg_loss != "ce":
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

    if reg_loss == "ce":
        reg_mean = torch.zeros(1, n_reg, dtype=torch.float32)
        reg_std = torch.ones(1, n_reg, dtype=torch.float32)
    else:
        reg_mean = (sum_reg / float(cnt_reg)).view(1, -1).float()
        reg_var = sum_reg2 / float(cnt_reg) - (sum_reg / float(cnt_reg)) ** 2
        reg_std = torch.sqrt(reg_var.clamp_min(1e-24)).view(1, -1).clamp_min(1e-6).float()
    reg_class_values = _classes_to_tensor(reg_class_tables) if reg_class_tables else None

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
    if reg_class_values is not None:
        torch.save(reg_class_values, out_dir / "reg_class_values.pt")

    _write_da_gps_run_manifest(
        out_dir,
        task="DA-GPS multitask chunk_parent",
        chunk_parent=str(chunk_parent),
        chunks=[str(p) for p in chunk_dirs],
        cache_dir=str(cache_dir),
        args=args,
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        meta_aux_cols=list(pv_aux_cols),
        reg_loss=reg_loss,
        n_chunks=len(chunk_dirs),
    )

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
        reg_nclasses=reg_nclasses if reg_loss == "ce" else None,
    ).to(device)
    ckpt_meta = _da_gps_ckpt_meta(
        n_nodes=n_nodes,
        hidden=int(args.hidden),
        layers=int(args.layers),
        heads=int(args.heads),
        n_cap=n_cap,
        n_reg=n_reg,
        n_system_tokens=n_sys,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        per_node_heads=bool(args.per_node_heads),
        per_device_cap_head=bool(args.per_device_cap_head),
        per_device_reg_head=bool(args.per_device_reg_head),
        n_pv_aux=int(n_pv_aux),
        pv_target_cols=list(pv_aux_cols) if n_pv_aux > 0 else [],
        meta_aux_target_cols=list(pv_aux_cols) if n_pv_aux > 0 else [],
        cap_target_cols=cap_cols,
        reg_target_cols=reg_cols,
        reg_loss=reg_loss,
        reg_nclasses=list(reg_nclasses) if reg_loss == "ce" else None,
        chunk_parent=str(chunk_parent),
        chunk_folders=[str(p) for p in chunk_dirs],
    )
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
    reg_class_values_d = reg_class_values.to(device) if reg_class_values is not None else None

    y_mean_d = y_mean.to(device).float()
    y_std_d = y_std.to(device).float()
    reg_mean_d = reg_mean.to(device).float()
    reg_std_d = reg_std.to(device).float()
    pv_mean_d = pv_mean.to(device).float() if pv_mean is not None else None
    pv_std_d = pv_std.to(device).float() if pv_std is not None else None
    x_mean_d = x_mean.to(device).float()
    x_std_d = x_std.to(device).float()
    pf_state = _setup_pf_physics(
        edges_path=chunk_dirs[0] / edge_name,
        nodes_path=chunk_dirs[0] / nodes_name,
        node_to_local=ref_ntl,
        n_nodes=n_nodes,
        args=args,
        device=device,
        data_root=chunk_dirs[0],
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        meta_aux_cols=list(pv_aux_cols),
        node_feature_cols=node_feature_cols,
        node_pe_csv=node_pe_csv,
    )
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
        train_loss_sum = train_v_sum = train_c_sum = train_r_sum = train_pv_sum = train_pf_sum = 0.0
        train_n = 0
        train_cap_dim = torch.zeros(n_cap, dtype=torch.float64)
        train_reg_dim = torch.zeros(n_reg, dtype=torch.float64)
        train_meta_dim = torch.zeros(n_pv_aux, dtype=torch.float64) if n_pv_aux > 0 else None
        val_cap_dim = torch.zeros(n_cap, dtype=torch.float64)
        val_reg_dim = torch.zeros(n_reg, dtype=torch.float64)
        val_meta_dim = torch.zeros(n_pv_aux, dtype=torch.float64) if n_pv_aux > 0 else None
        pf_dbg_first_batch = ep == 1 and pf_state.pf_debug_nan
        train_batch_idx = 0
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
                reg_class_tables=reg_class_tables,
                reg_target_mode=reg_target_mode,
                reg_classes_digest=reg_classes_digest,
            )
            if reg_loss == "ce":
                y_reg_n = y_reg.to(dtype=torch.long)
            else:
                y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)
            x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
            y_pv_n = None
            if n_pv_aux > 0 and y_pv is not None and pv_mean is not None and pv_std is not None:
                y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
            ds = DAGPSDataset(
                x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n
            )
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
                train_batch_idx += 1
                yb = batch.y.view(batch.num_graphs, -1)
                y_cap_b = batch.y_cap.view(batch.num_graphs, -1)
                y_reg_b = batch.y_reg.view(batch.num_graphs, -1)
                yb_n = (yb - y_mean_d) / y_std_d
                opt.zero_grad(set_to_none=True)
                with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                    v_n, c_log, r_p, pv_p = model(batch)
                    loss_v = mse(v_n.view_as(yb_n), yb_n)
                    loss_c = bce(c_log, y_cap_b)
                    reg_logits = base_model._last_reg_logits if reg_loss == "ce" else None
                    loss_r = _reg_loss_scalar(
                        r_p if reg_loss != "ce" else None,
                        y_reg_b,
                        reg_loss,
                        reg_logits=reg_logits,
                        reg_class_tables=reg_class_tables,
                    )
                    loss = loss_v + float(args.lambda_cap) * loss_c + float(args.lambda_reg) * loss_r
                    loss_pf = _pf_loss_if_enabled(
                        pf_state,
                        v_n,
                        batch,
                        n_nodes=n_nodes,
                        y_mean=y_mean_d,
                        y_std=y_std_d,
                        x_mean=x_mean_d,
                        x_std=x_std_d,
                        cap_logits=c_log,
                        reg_pred=r_p,
                        reg_loss=reg_loss,
                        reg_mean=reg_mean_d,
                        reg_std=reg_std_d,
                        reg_logits=reg_logits,
                        reg_class_values=reg_class_values_d,
                    )
                    if loss_pf is not None:
                        if _pf_should_emit_debug(
                            pf_state,
                            epoch=ep,
                            first_batch_of_epoch=pf_dbg_first_batch,
                            loss_pf=loss_pf,
                        ):
                            _pf_debug_nan_report(
                                loss_pf=loss_pf,
                                v_n=v_n,
                                batch=batch,
                                n_nodes=n_nodes,
                                y_mean=y_mean_d,
                                y_std=y_std_d,
                                pf=pf_state,
                                cap_logits=c_log,
                                reg_pred=r_p,
                                x_mean=x_mean_d,
                                x_std=x_std_d,
                                reg_loss=reg_loss,
                                reg_mean=reg_mean_d,
                                reg_std=reg_std_d,
                                reg_logits=reg_logits,
                                reg_class_values=reg_class_values_d,
                                use_amp=use_amp,
                                epoch=ep,
                                batch_idx=train_batch_idx,
                                trigger=(
                                    "non-finite loss_pf"
                                    if not torch.isfinite(loss_pf).all()
                                    else "pf_debug_nan epoch-1 first train batch"
                                ),
                            )
                            pf_dbg_first_batch = False
                        loss = loss + pf_state.weight * loss_pf
                        train_pf_sum += float(loss_pf.item()) * batch.num_graphs
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
                    bce_e = F.binary_cross_entropy_with_logits(c_log, y_cap_b, reduction="none")
                    train_cap_dim += bce_e.sum(dim=0).detach().float().cpu().double()
                    reg_e = _reg_loss_elementwise(
                        r_p if reg_loss != "ce" else None,
                        y_reg_b,
                        reg_loss,
                        reg_logits=reg_logits,
                    )
                    train_reg_dim += reg_e.sum(dim=0).detach().float().cpu().double()
                    if train_meta_dim is not None and n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                        y_pv_b = batch.y_pv.view(batch.num_graphs, -1)
                        mse_p = F.mse_loss(pv_p, y_pv_b, reduction="none")
                        train_meta_dim += mse_p.sum(dim=0).detach().float().cpu().double()
            del x, y_ri, y_cap, y_reg, y_pv, y_reg_n, y_pv_n, x_n, ds, dl_tr
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model.eval()
        val_tot = val_v = 0.0
        val_c_sum = val_r_sum = val_pv_sum = val_pf_sum = 0.0
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
                    reg_class_tables=reg_class_tables,
                    reg_target_mode=reg_target_mode,
                    reg_classes_digest=reg_classes_digest,
                )
                if reg_loss == "ce":
                    y_reg_n = y_reg.to(dtype=torch.long)
                else:
                    y_reg_n = ((y_reg - reg_mean) / reg_std).to(dtype=torch.float32)
                x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
                y_pv_n = None
                if n_pv_aux > 0 and y_pv is not None and pv_mean is not None and pv_std is not None:
                    y_pv_n = ((y_pv - pv_mean) / pv_std).to(dtype=torch.float32)
                ds = DAGPSDataset(
                    x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n
                )
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
                        reg_logits_v = base_model._last_reg_logits if reg_loss == "ce" else None
                        lr_ = _reg_loss_scalar(
                            r_p if reg_loss != "ce" else None,
                            y_reg_b,
                            reg_loss,
                            reg_logits=reg_logits_v,
                            reg_class_tables=reg_class_tables,
                        )
                        lt = lv + float(args.lambda_cap) * lc + float(args.lambda_reg) * lr_
                        lpf = _pf_loss_if_enabled(
                            pf_state,
                            v_n,
                            batch,
                            n_nodes=n_nodes,
                            y_mean=y_mean_d,
                            y_std=y_std_d,
                            x_mean=x_mean_d,
                            x_std=x_std_d,
                            cap_logits=c_log,
                            reg_pred=r_p,
                            reg_loss=reg_loss,
                            reg_mean=reg_mean_d,
                            reg_std=reg_std_d,
                            reg_logits=reg_logits_v,
                            reg_class_values=reg_class_values_d,
                        )
                        if lpf is not None:
                            lt = lt + pf_state.weight * lpf
                            val_pf_sum += float(lpf.item()) * batch.num_graphs
                        if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                            lpv = mse(pv_p, batch.y_pv.view(batch.num_graphs, -1))
                            lt = lt + float(args.lambda_pv) * lpv
                            val_pv_sum += float(lpv.item()) * batch.num_graphs
                    val_tot += float(lt.item()) * batch.num_graphs
                    val_v += float(lv.item()) * batch.num_graphs
                    val_c_sum += float(lc.item()) * batch.num_graphs
                    val_r_sum += float(lr_.item()) * batch.num_graphs
                    nv += int(batch.num_graphs)
                    bce_ev = F.binary_cross_entropy_with_logits(c_log, y_cap_b, reduction="none")
                    val_cap_dim += bce_ev.sum(dim=0).detach().float().cpu().double()
                    reg_ev = _reg_loss_elementwise(
                        r_p if reg_loss != "ce" else None,
                        y_reg_b,
                        reg_loss,
                        reg_logits=reg_logits_v,
                    )
                    val_reg_dim += reg_ev.sum(dim=0).detach().float().cpu().double()
                    if val_meta_dim is not None and n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                        lpv_e = F.mse_loss(pv_p, batch.y_pv.view(batch.num_graphs, -1), reduction="none")
                        val_meta_dim += lpv_e.sum(dim=0).detach().float().cpu().double()
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
        val_pf = val_pf_sum / max(nv, 1) if pf_state.weight > 0 else float("nan")
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
        train_pf = train_pf_sum / max(train_n, 1) if pf_state.weight > 0 else float("nan")
        train_tot = train_loss_sum / max(train_n, 1)
        train_cap_mean = (train_cap_dim / max(train_n, 1)).numpy()
        train_reg_mean = (train_reg_dim / max(train_n, 1)).numpy()
        train_meta_mean = (train_meta_dim / max(train_n, 1)).numpy() if train_meta_dim is not None else np.zeros(0)
        if nv > 0:
            val_cap_mean = (val_cap_dim / float(nv)).numpy()
            val_reg_mean = (val_reg_dim / float(nv)).numpy()
            val_meta_mean = (val_meta_dim / float(nv)).numpy() if val_meta_dim is not None else np.zeros(0)
        else:
            val_cap_mean = np.full(n_cap, np.nan)
            val_reg_mean = np.full(n_reg, np.nan)
            val_meta_mean = np.full(n_pv_aux, np.nan) if n_pv_aux > 0 else np.zeros(0)
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
            if pf_state.weight > 0:
                _log += (
                    f" train_pf={train_pf:.4e} val_pf={val_pf:.4e}"
                    f" pf_wt={pf_state.weight:g}"
                )
            _log += (
                f" | val_tot={val_tot:.4f} val_volt={val_v:.4f} val_cap={val_c:.4f} val_reg={val_r:.4f} "
                f"| val_r2_mean={val_r2_mean:.4f} val_r2_min={val_r2_min:.4f} val_worst_mae={val_worst_node_mae:.4f} "
                f"| best={best_val:.4f}"
            )
            print(_log, flush=True)
            _print_per_head_two_lines("[da_gps chunk_parent]", "cap_BCE", cap_cols, train_cap_mean, val_cap_mean)
            if reg_loss == "ce":
                _reg_head_label = "reg_CE"
            elif reg_loss == "mae":
                _reg_head_label = "reg_MAE_nrm"
            else:
                _reg_head_label = "reg_MSE_nrm"
            _print_per_head_two_lines(
                "[da_gps chunk_parent]", _reg_head_label, reg_cols, train_reg_mean, val_reg_mean
            )
            if n_pv_aux > 0:
                _print_per_head_two_lines("[da_gps chunk_parent]", "meta_aux_MSE_nrm", pv_aux_cols, train_meta_mean, val_meta_mean)
        _ce = int(args.checkpoint_every)
        if _ce > 0 and ep % _ce == 0:
            _ck = out_dir / "training_last.pt"
            _save_periodic_training_checkpoint(
                _ck,
                base_model,
                opt,
                sch,
                scaler,
                ckpt_meta,
                epoch=ep,
                bad=bad,
                best_val=best_val,
                best_state=best_state,
            )
            print(f"  periodic checkpoint -> {_ck}", flush=True)
        if bad >= args.patience:
            print(f"[da_gps chunk_parent] early stop at epoch {ep}", flush=True)
            if int(args.checkpoint_every) > 0:
                _ck = out_dir / "training_last.pt"
                _save_periodic_training_checkpoint(
                    _ck,
                    base_model,
                    opt,
                    sch,
                    scaler,
                    ckpt_meta,
                    epoch=ep,
                    bad=bad,
                    best_val=best_val,
                    best_state=best_state,
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
        reg_loss=reg_loss,
        reg_class_tables=reg_class_tables,
        reg_target_mode=reg_target_mode,
        reg_classes_digest=reg_classes_digest,
        reg_class_values=reg_class_values,
        base_model=base_model,
    )

    ckpt = out_dir / "da_gps_multitask_best.pt"
    torch.save(_da_gps_checkpoint_payload(base_model, ckpt_meta), ckpt)
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
    _reg_tail = (
        f"  reg_CE={met.get('reg_ce_loss', float('nan')):.6f}  reg_acc={met.get('reg_accuracy', float('nan')):.4f}"
        if reg_loss == "ce"
        else f"  reg_MSE(pu)={met['reg_mse_tap_pu']:.6f}  reg_MAE(pu)={met.get('reg_mae_tap_pu', float('nan')):.6f}"
    )
    print(
        f"Test |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"cap_BCE={met['cap_bce']:.6f}{_reg_tail}{_pv_tail}  time={train_seconds:.1f}s",
        flush=True,
    )
    _print_test_per_head_block("[da_gps chunk_parent]", met, cap_cols, reg_cols, list(pv_aux_cols) if n_pv_aux > 0 else [])
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
    p.add_argument(
        "--reg_loss",
        type=str,
        default="mse",
        help="Regulator tap loss: mse or mae on z-scored tap_pu; ce (cce) = cross-entropy on discrete tap "
        "classes (rounded unique tap_pu per reg column, requires --per_device_reg_head). Caps use BCE.",
    )
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
        help="Every N epochs save out_dir/training_last.pt (same arch metadata as da_gps_multitask_best.pt "
        "+ optimizer/scheduler/epoch); 0 disables.",
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
        help="Only used with --chunk_parent: fnmatch pattern (e.g. run_*) or comma-separated exact folder names for smoke subsets (e.g. run_001_...,run_002_...).",
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
    p.add_argument(
        "--loss_power_balance_weight",
        type=float,
        default=0.0,
        help="Weight for nodal P/Q power-balance physics loss on denormalized voltage (0 = disabled).",
    )
    p.add_argument(
        "--pf_data_root",
        type=str,
        default="",
        help="MVagg PF topology root (hetero_mv_edge_catalog.csv, capacitor_involved_nodes.csv, "
        "electrical_distance_from_substation.csv). Default: auto (repo colab_pf_data/ or dailyagg).",
    )
    p.add_argument(
        "--pf_balance_nodes",
        type=str,
        default="mv",
        choices=("mv", "all"),
        help="Nodes included in power-balance residual: mv=electrical_distance>0 (from nodes CSV, else --node_pe_csv / PF_DATA_ROOT), all=every node.",
    )
    p.add_argument(
        "--pf_balance_node_list_csv",
        type=str,
        default="",
        help="Optional CSV of explicit balance nodes (columns: node and/or bus and/or node_idx; "
        "node preferred when present). Used as-is; only slack/source nodes are excluded.",
    )
    p.add_argument(
        "--pf_s_base_kva",
        type=float,
        default=5000.0,
        help="System base kVA for converting pu power S=V*conj(I) to kW/kvar in physics loss.",
    )
    p.add_argument(
        "--pf_kv_base",
        type=float,
        default=12.47,
        help="Nominal line-line kV (metadata for logging; Y-bus built in Siemens from line ohms).",
    )
    p.add_argument(
        "--pf_detach_controls",
        action="store_true",
        help="Stop gradients through predicted taps/cap states inside physics loss (stability).",
    )
    p.add_argument(
        "--pf_huber_delta_kw",
        type=float,
        default=10.0,
        help="Huber delta (kW/kvar) for physics residual (default 10 kW).",
    )
    p.add_argument(
        "--pf_bus_kv_base_csv",
        type=str,
        default="",
        help="Optional cache CSV for per-bus kVBase (default: bus_kv_base_by_node.csv under data_root).",
    )
    p.add_argument(
        "--pf_reg_edge_catalog",
        type=str,
        default="",
        help="CSV with edge_type=regulator rows (default: auto-find hetero_mv_edge_catalog.csv under data_root).",
    )
    p.add_argument(
        "--pf_cap_nodes_csv",
        type=str,
        default="",
        help="Capacitor bus map CSV (default: capacitor_involved_nodes.csv under data_root).",
    )
    p.add_argument(
        "--pf_exclude_interface_buses",
        type=int,
        default=1,
        choices=(0, 1),
        help="1=exclude regxfmr/190-/m/p/n interface buses from MV balance mask (default 1).",
    )
    p.add_argument(
        "--pf_hetero_y_neighbors_only",
        type=int,
        default=1,
        choices=(0, 1),
        help="1=MV mask only hetero load nodes whose Y-neighbors are also hetero catalog nodes (default 1).",
    )
    p.add_argument(
        "--pf_sparse_y",
        type=int,
        default=1,
        choices=(0, 1),
        help="1=O(E) sparse edge-local Y@V for physics loss (default). 0=dense (B,N,N) debug path.",
    )
    p.add_argument(
        "--pf_debug_nan",
        type=int,
        default=0,
        choices=(0, 1),
        help="1=print one-shot PF physics diagnostics on epoch-1 first train batch and when loss_pf is non-finite "
        "(also GNN2_PF_DEBUG_NAN=1).",
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
    reg_loss = _parse_reg_loss(args.reg_loss)
    print(f"regulator tap training loss: {reg_loss} (z-scored tap targets)", flush=True)

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

    _write_da_gps_run_manifest(
        out_dir,
        task="DA-GPS multitask single dataset",
        chunk_parent=str(data_root),
        chunks=[str(nodes_path.parent)],
        cache_dir=str(Path(args.cache_tensor).resolve().parent) if args.cache_tensor else None,
        args=args,
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        meta_aux_cols=list(pv_aux_cols),
        reg_loss=reg_loss,
        n_chunks=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pf_state = _setup_pf_physics(
        edges_path=edges_path,
        nodes_path=nodes_path,
        node_to_local=node_to_local,
        n_nodes=n_nodes,
        args=args,
        device=device,
        data_root=data_root,
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        meta_aux_cols=list(pv_aux_cols),
        node_feature_cols=node_feature_cols,
        node_pe_csv=node_pe_csv,
    )
    ds = DAGPSDataset(
        x_n, y_ri, y_cap, y_reg_n, edge_index, edge_attr, y_pv=y_pv_n
    )
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
    ckpt_meta = _da_gps_ckpt_meta(
        n_nodes=n_nodes,
        hidden=int(args.hidden),
        layers=int(args.layers),
        heads=int(args.heads),
        n_cap=n_cap,
        n_reg=n_reg,
        n_system_tokens=n_sys,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        per_node_heads=bool(args.per_node_heads),
        per_device_cap_head=bool(args.per_device_cap_head),
        per_device_reg_head=bool(args.per_device_reg_head),
        n_pv_aux=int(n_pv_aux),
        pv_target_cols=list(pv_aux_cols) if n_pv_aux > 0 else [],
        meta_aux_target_cols=list(pv_aux_cols) if n_pv_aux > 0 else [],
        cap_target_cols=cap_cols,
        reg_target_cols=reg_cols,
        reg_loss=reg_loss,
    )
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
    x_mean_d = x_mean.to(device).float()
    x_std_d = x_std.to(device).float()
    reg_class_values_d = None
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
        train_loss_sum = train_v_sum = train_c_sum = train_r_sum = train_pv_sum = train_pf_sum = 0.0
        train_n = 0
        train_cap_dim = torch.zeros(n_cap, dtype=torch.float64)
        train_reg_dim = torch.zeros(n_reg, dtype=torch.float64)
        train_meta_dim = torch.zeros(n_pv_aux, dtype=torch.float64) if n_pv_aux > 0 else None
        val_cap_dim = torch.zeros(n_cap, dtype=torch.float64)
        val_reg_dim = torch.zeros(n_reg, dtype=torch.float64)
        val_meta_dim = torch.zeros(n_pv_aux, dtype=torch.float64) if n_pv_aux > 0 else None
        pf_dbg_first_batch = ep == 1 and pf_state.pf_debug_nan
        train_batch_idx = 0
        for batch in dl_tr:
            batch = batch.to(device)
            batch = _cast_batch_float_tensors(batch)
            train_batch_idx += 1
            yb = batch.y.view(batch.num_graphs, -1)
            y_cap = batch.y_cap.view(batch.num_graphs, -1)
            y_reg = batch.y_reg.view(batch.num_graphs, -1)
            yb_n = (yb - y_mean_d) / y_std_d
            opt.zero_grad(set_to_none=True)
            with (torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()):
                v_n, c_log, r_p, pv_p = model(batch)
                loss_v = mse(v_n.view_as(yb_n), yb_n)
                loss_c = bce(c_log, y_cap)
                loss_r = _reg_loss_scalar(r_p, y_reg, reg_loss)
                loss = loss_v + float(args.lambda_cap) * loss_c + float(args.lambda_reg) * loss_r
                loss_pf = _pf_loss_if_enabled(
                    pf_state,
                    v_n,
                    batch,
                    n_nodes=n_nodes,
                    y_mean=y_mean_d,
                    y_std=y_std_d,
                    x_mean=x_mean_d,
                    x_std=x_std_d,
                    cap_logits=c_log,
                    reg_pred=r_p,
                    reg_loss=reg_loss,
                    reg_mean=reg_mean_d,
                    reg_std=reg_std_d,
                    reg_logits=None,
                    reg_class_values=reg_class_values_d,
                )
                if loss_pf is not None:
                    if _pf_should_emit_debug(
                        pf_state,
                        epoch=ep,
                        first_batch_of_epoch=pf_dbg_first_batch,
                        loss_pf=loss_pf,
                    ):
                        _pf_debug_nan_report(
                            loss_pf=loss_pf,
                            v_n=v_n,
                            batch=batch,
                            n_nodes=n_nodes,
                            y_mean=y_mean_d,
                            y_std=y_std_d,
                            pf=pf_state,
                            cap_logits=c_log,
                            reg_pred=r_p,
                            x_mean=x_mean_d,
                            x_std=x_std_d,
                            reg_loss=reg_loss,
                            reg_mean=reg_mean_d,
                            reg_std=reg_std_d,
                            reg_logits=None,
                            reg_class_values=reg_class_values_d,
                            use_amp=use_amp,
                            epoch=ep,
                            batch_idx=train_batch_idx,
                            trigger=(
                                "non-finite loss_pf"
                                if not torch.isfinite(loss_pf).all()
                                else "pf_debug_nan epoch-1 first train batch"
                            ),
                        )
                        pf_dbg_first_batch = False
                    loss = loss + pf_state.weight * loss_pf
                    train_pf_sum += float(loss_pf.item()) * batch.num_graphs
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
                bce_e = F.binary_cross_entropy_with_logits(c_log, y_cap, reduction="none")
                train_cap_dim += bce_e.sum(dim=0).detach().float().cpu().double()
                reg_e = _reg_loss_elementwise(r_p, y_reg, reg_loss)
                train_reg_dim += reg_e.sum(dim=0).detach().float().cpu().double()
                if train_meta_dim is not None and n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                    y_pv_b = batch.y_pv.view(batch.num_graphs, -1)
                    mse_p = F.mse_loss(pv_p, y_pv_b, reduction="none")
                    train_meta_dim += mse_p.sum(dim=0).detach().float().cpu().double()

        model.eval()
        val_tot = val_v = 0.0
        val_c_sum = val_r_sum = val_pv_sum = val_pf_sum = 0.0
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
                    lr_ = _reg_loss_scalar(r_p, y_reg, reg_loss)
                    lt = lv + float(args.lambda_cap) * lc + float(args.lambda_reg) * lr_
                    lpf = _pf_loss_if_enabled(
                        pf_state,
                        v_n,
                        batch,
                        n_nodes=n_nodes,
                        y_mean=y_mean_d,
                        y_std=y_std_d,
                        x_mean=x_mean_d,
                        x_std=x_std_d,
                        cap_logits=c_log,
                        reg_pred=r_p,
                        reg_loss=reg_loss,
                        reg_mean=reg_mean_d,
                        reg_std=reg_std_d,
                        reg_logits=None,
                        reg_class_values=reg_class_values_d,
                    )
                    if lpf is not None:
                        lt = lt + pf_state.weight * lpf
                        val_pf_sum += float(lpf.item()) * batch.num_graphs
                    if n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                        lpv = mse(pv_p, batch.y_pv.view(batch.num_graphs, -1))
                        lt = lt + float(args.lambda_pv) * lpv
                        val_pv_sum += float(lpv.item()) * batch.num_graphs
                val_tot += float(lt.item()) * batch.num_graphs
                val_v += float(lv.item()) * batch.num_graphs
                val_c_sum += float(lc.item()) * batch.num_graphs
                val_r_sum += float(lr_.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
                bce_ev = F.binary_cross_entropy_with_logits(c_log, y_cap, reduction="none")
                val_cap_dim += bce_ev.sum(dim=0).detach().float().cpu().double()
                reg_ev = _reg_loss_elementwise(r_p, y_reg, reg_loss)
                val_reg_dim += reg_ev.sum(dim=0).detach().float().cpu().double()
                if val_meta_dim is not None and n_pv_aux > 0 and hasattr(batch, "y_pv") and batch.y_pv is not None:
                    lpv_e = F.mse_loss(pv_p, batch.y_pv.view(batch.num_graphs, -1), reduction="none")
                    val_meta_dim += lpv_e.sum(dim=0).detach().float().cpu().double()
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
        val_pf = val_pf_sum / max(nv, 1) if pf_state.weight > 0 else float("nan")
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
        train_pf = train_pf_sum / max(train_n, 1) if pf_state.weight > 0 else float("nan")
        train_tot = train_loss_sum / max(train_n, 1)
        train_cap_mean = (train_cap_dim / max(train_n, 1)).numpy()
        train_reg_mean = (train_reg_dim / max(train_n, 1)).numpy()
        train_meta_mean = (train_meta_dim / max(train_n, 1)).numpy() if train_meta_dim is not None else np.zeros(0)
        if nv > 0:
            val_cap_mean = (val_cap_dim / float(nv)).numpy()
            val_reg_mean = (val_reg_dim / float(nv)).numpy()
            val_meta_mean = (val_meta_dim / float(nv)).numpy() if val_meta_dim is not None else np.zeros(0)
        else:
            val_cap_mean = np.full(n_cap, np.nan)
            val_reg_mean = np.full(n_reg, np.nan)
            val_meta_mean = np.full(n_pv_aux, np.nan) if n_pv_aux > 0 else np.zeros(0)
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
            if pf_state.weight > 0:
                _log += (
                    f" train_pf={train_pf:.4e} val_pf={val_pf:.4e}"
                    f" pf_wt={pf_state.weight:g}"
                )
            _log += (
                f" | val_tot={val_tot:.4f} val_volt={val_v:.4f} val_cap={val_c:.4f} val_reg={val_r:.4f} "
                f"| val_r2_mean={val_r2_mean:.4f} val_r2_min={val_r2_min:.4f} val_worst_mae={val_worst_node_mae:.4f} "
                f"| best={best_val:.4f}"
            )
            print(_log, flush=True)
            _print_per_head_two_lines("[da_gps]", "cap_BCE", cap_cols, train_cap_mean, val_cap_mean)
            _reg_head_label = "reg_MAE_nrm" if reg_loss == "mae" else "reg_MSE_nrm"
            _print_per_head_two_lines("[da_gps]", _reg_head_label, reg_cols, train_reg_mean, val_reg_mean)
            if n_pv_aux > 0:
                _print_per_head_two_lines("[da_gps]", "meta_aux_MSE_nrm", pv_aux_cols, train_meta_mean, val_meta_mean)
        _ce = int(args.checkpoint_every)
        if _ce > 0 and ep % _ce == 0:
            _ck = out_dir / "training_last.pt"
            _save_periodic_training_checkpoint(
                _ck,
                base_model,
                opt,
                sch,
                scaler,
                ckpt_meta,
                epoch=ep,
                bad=bad,
                best_val=best_val,
                best_state=best_state,
            )
            print(f"  periodic checkpoint -> {_ck}", flush=True)
        if bad >= args.patience:
            print(f"[da_gps] early stop at epoch {ep}", flush=True)
            if int(args.checkpoint_every) > 0:
                _ck = out_dir / "training_last.pt"
                _save_periodic_training_checkpoint(
                    _ck,
                    base_model,
                    opt,
                    sch,
                    scaler,
                    ckpt_meta,
                    epoch=ep,
                    bad=bad,
                    best_val=best_val,
                    best_state=best_state,
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
        cap_cols=cap_cols,
        reg_cols=reg_cols,
        meta_aux_cols=list(pv_aux_cols) if n_pv_aux > 0 else [],
    )
    ckpt = out_dir / "da_gps_multitask_best.pt"
    torch.save(_da_gps_checkpoint_payload(base_model, ckpt_meta), ckpt)
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
        f"cap_BCE={met['cap_bce']:.6f}  reg_MSE(pu)={met['reg_mse_tap_pu']:.6f}  "
        f"reg_MAE(pu)={met.get('reg_mae_tap_pu', float('nan')):.6f}{_pv_tail}  time={train_seconds:.1f}s",
        flush=True,
    )
    _print_test_per_head_block("[da_gps]", met, cap_cols, reg_cols, list(pv_aux_cols) if n_pv_aux > 0 else [])
    print(f"Saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
