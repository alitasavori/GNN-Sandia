"""Convert DA-GPS phase-node chunk CSVs → PowerFlowMultiNet physical-bus tensors.

Oracle device-state baseline (arXiv:2403.00892v3 framing — not a full paper clone):
settled regulator taps and capacitor states are inputs (edge tap attrs + bus cap
features + flattened ``device_state``), not targets. Targets include bus V/φ and
substation P/Q.

Gaps vs paper: ieee34/8500 are not paper cases; 8500 secondary / split-phase is
not modeled beyond A/B/C phase edges in ``gnn_edges_phase_static.csv``; node
feature vector adds masks/source/caps beyond paper's P,Q-per-phase description.
If a nodes CSV omits ``bus``/``phase``, they are derived from ``node`` (``bus.phase``).
``p_pv_kw`` is optional (treated as 0 when absent).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Node feature layout (documented implementation choice):
# [P_A,Q_A,P_B,Q_B,P_C,Q_C, m_A,m_B,m_C, source, cap_A,cap_B,cap_C]
NODE_FEAT_DIM = 13
NODE_CONT_IDX = (0, 1, 2, 3, 4, 5)  # z-score these
# Edge: [phA,phB,phC, is_line,is_xfmr,is_reg,is_sw, tap, switch_closed]
EDGE_FEAT_DIM = 9
EDGE_TAP_IDX = 7
EDGE_CONT_IDX = (7,)  # tap only (one-hots / binary left raw)

_PHASE_TO_I = {1: 0, 2: 1, 3: 2, "1": 0, "2": 1, "3": 2, "a": 0, "b": 1, "c": 2}


def _norm_sid(v) -> int:
    return int(float(v))


def _bus_phase_from_node(node: str) -> tuple[str | None, int | None]:
    """Parse OpenDSS ``bus.phase`` node labels → (bus, phase 1..3 | None)."""
    s = str(node).strip().lower()
    if "." not in s:
        return (s or None), None
    bus, phs = s.rsplit(".", 1)
    bus = bus.strip() or None
    letter = {"a": 1, "b": 2, "c": 3, "1": 1, "2": 2, "3": 3}
    if phs in letter:
        return bus, letter[phs]
    try:
        ph = int(phs)
    except ValueError:
        return bus, None
    if ph in (1, 2, 3):
        return bus, ph
    return bus, None


def _csv_header_lower(path: Path) -> dict[str, str]:
    """Map lowercased header → original column name."""
    hdr = [str(c) for c in pd.read_csv(path, nrows=0).columns.tolist()]
    return {c.lower(): c for c in hdr}


def _ensure_bus_phase_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``bus`` / ``phase`` from ``node`` (``bus.phase`` convention).

    Some Drive chunk exports omit ``bus`` (and occasionally ``phase``) even though
    ``node`` is always ``bus.phase``. Edge CSVs still use ``from_bus`` / ``to_bus``.
    """
    out = df.copy()
    cols_l = {str(c).lower(): c for c in out.columns}
    has_bus = "bus" in cols_l
    has_phase = "phase" in cols_l
    if has_bus and cols_l["bus"] != "bus":
        out = out.rename(columns={cols_l["bus"]: "bus"})
        cols_l = {str(c).lower(): c for c in out.columns}
    if has_phase and cols_l["phase"] != "phase":
        out = out.rename(columns={cols_l["phase"]: "phase"})
        cols_l = {str(c).lower(): c for c in out.columns}
    if "bus" in out.columns and "phase" in out.columns:
        return out
    if "node" not in cols_l:
        raise ValueError("nodes CSV needs 'bus'+'phase' or a 'node' column to derive them")
    parsed = out[cols_l["node"]].map(_bus_phase_from_node)
    if "bus" not in out.columns:
        out["bus"] = [p[0] if p[0] is not None else "" for p in parsed]
    if "phase" not in out.columns:
        out["phase"] = [p[1] if p[1] is not None else -1 for p in parsed]
    return out


def _device_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _cap_col_stem(col: str) -> str | None:
    m = re.match(r"^cap_(.+)_n_steps_on$", str(col).strip(), flags=re.IGNORECASE)
    return _device_stem(m.group(1)) if m else None


def _reg_col_stem(col: str) -> str | None:
    m = re.match(r"^reg_(.+)_tap_pu$", str(col).strip(), flags=re.IGNORECASE)
    return _device_stem(m.group(1)) if m else None


def _line_component_stem(line_name: str) -> str:
    s = str(line_name).strip().lower()
    if "." in s:
        s = s.split(".", 1)[1]
    return _device_stem(s)


def _classify_line_type(line_name: str) -> str:
    s = str(line_name).strip().lower()
    if "reg" in s:
        return "regulator"
    if "transformer" in s or s.startswith("xfmr") or ".xfm" in s:
        return "transformer"
    if "switch" in s or s.startswith("swt"):
        return "switch"
    return "line"


def _guess_cap_bus_and_phase(stem: str, bus_set: set[str]) -> tuple[str | None, int | None]:
    """Best-effort map capacitor stem → (physical bus, phase_idx|None=all).

    Examples: ``c844`` → (``844``, None); ``capbank0a`` → (None, 0).
    """
    st = str(stem).lower()
    m = re.match(r"^c(\d+[a-z]*)$", st)
    if m:
        bus = m.group(1)
        # strip trailing phase letter if bus not found
        if bus in bus_set:
            return bus, None
        if bus[:-1] in bus_set and bus[-1] in "abc":
            return bus[:-1], "abc".index(bus[-1])
    m = re.match(r"^(.*?)([abc])$", st)
    phase = None
    core = st
    if m:
        core, ph = m.group(1), m.group(2)
        phase = "abc".index(ph)
    # numeric bus embedded
    m2 = re.search(r"(\d{2,})", core)
    if m2:
        cand = m2.group(1)
        if cand in bus_set:
            return cand, phase
    return None, phase


@dataclass
class PfmnGraphStatic:
    bus_order: list[str]
    bus_to_local: dict[str, int]
    edge_index: torch.Tensor  # [2, E]
    edge_attr_static: torch.Tensor  # [E, EDGE_FEAT_DIM] tap column zeroed
    edge_tap_reg_idx: torch.Tensor  # [E] int64, -1 if no tap
    phase_present: torch.Tensor  # [N, 3] bool/float
    source_bus_local: int
    reg_cols: list[str]
    cap_cols: list[str]
    cap_bus_phase: list[tuple[int, int | None]]  # per cap_col: (bus_local|-1, phase|None)


def discover_meta_device_cols(meta_csv: Path) -> tuple[list[str], list[str]]:
    hdr = [str(c) for c in pd.read_csv(meta_csv, nrows=0).columns.tolist()]
    # normalize case for matching but return actual column names lowercased
    lower_map = {c.lower(): c for c in hdr}
    reg_cols = sorted(
        lower_map[c].lower()
        for c in lower_map
        if re.match(r"^reg_.+_tap_pu$", c)
    )
    cap_cols = sorted(
        lower_map[c].lower()
        for c in lower_map
        if re.match(r"^cap_.+_n_steps_on$", c)
    )
    return reg_cols, cap_cols


def build_physical_bus_static(
    nodes_csv: Path,
    edges_csv: Path,
    meta_csv: Path,
    *,
    sample_id_probe: int | None = None,
) -> PfmnGraphStatic:
    """Build fixed topology + device column mapping from one chunk."""
    hdr = _csv_header_lower(nodes_csv)
    required = ["sample_id", "node", "node_idx"]
    for c in required:
        if c not in hdr:
            raise ValueError(f"{nodes_csv} missing required column {c!r} (have {sorted(hdr)})")
    usecols = [hdr[c] for c in required]
    for opt in ("bus", "phase"):
        if opt in hdr:
            usecols.append(hdr[opt])
    # Probe first sample for bus roster
    sids: list[int] = []
    for ch in pd.read_csv(nodes_csv, usecols=[hdr["sample_id"]], chunksize=200_000):
        sids.extend(int(_norm_sid(s)) for s in ch[hdr["sample_id"]].tolist())
        if sids:
            break
    if not sids:
        raise RuntimeError(f"No samples in {nodes_csv}")
    sid0 = int(sample_id_probe) if sample_id_probe is not None else int(sorted(set(sids))[0])

    parts = []
    for ch in pd.read_csv(nodes_csv, usecols=usecols, chunksize=500_000):
        # Normalize column names used below
        rename = {}
        for want in ("sample_id", "node", "node_idx", "bus", "phase"):
            if want in hdr and hdr[want] != want and hdr[want] in ch.columns:
                rename[hdr[want]] = want
        if rename:
            ch = ch.rename(columns=rename)
        ch = _ensure_bus_phase_cols(ch)
        sid = ch["sample_id"].map(_norm_sid)
        sub = ch.loc[sid == sid0]
        if len(sub):
            parts.append(sub)
    if not parts:
        raise RuntimeError(f"sample_id={sid0} missing in {nodes_csv}")
    first = pd.concat(parts, ignore_index=True)
    first = _ensure_bus_phase_cols(first)
    first["bus"] = first["bus"].astype(str).str.strip().str.lower()
    first["phase"] = first["phase"].map(lambda p: _PHASE_TO_I.get(p, _PHASE_TO_I.get(str(p).lower(), -1)))
    first = first[first["phase"] >= 0].copy()
    # Stable bus order: by min node_idx
    bus_min = first.groupby("bus")["node_idx"].min().sort_values()
    bus_order = [str(b) for b in bus_min.index.tolist()]
    bus_to_local = {b: i for i, b in enumerate(bus_order)}
    n_bus = len(bus_order)

    phase_present = np.zeros((n_bus, 3), dtype=np.float32)
    for _, row in first.iterrows():
        bi = bus_to_local[str(row["bus"])]
        phase_present[bi, int(row["phase"])] = 1.0

    source_bus_local = 0  # bus of smallest node_idx (implementation choice)

    edges = pd.read_csv(edges_csv)
    for c in ("from_bus", "to_bus", "phase", "line_name"):
        if c not in edges.columns:
            raise ValueError(f"{edges_csv} missing {c}")

    reg_cols_all, cap_cols_all = discover_meta_device_cols(meta_csv)
    reg_stem_to_j = {}
    for j, col in enumerate(reg_cols_all):
        st = _reg_col_stem(col)
        if st:
            reg_stem_to_j[st] = j

    # Keep regulator columns that match at least one edge stem; if none match, keep all.
    matched_reg: set[int] = set()
    src: list[int] = []
    dst: list[int] = []
    attrs: list[list[float]] = []
    tap_idx: list[int] = []

    type_onehot = {
        "line": [1, 0, 0, 0],
        "transformer": [0, 1, 0, 0],
        "regulator": [0, 0, 1, 0],
        "switch": [0, 0, 0, 1],
    }

    for _, r in edges.iterrows():
        fb = str(r["from_bus"]).strip().lower()
        tb = str(r["to_bus"]).strip().lower()
        if fb not in bus_to_local or tb not in bus_to_local:
            continue
        ph = _PHASE_TO_I.get(r["phase"], _PHASE_TO_I.get(str(r["phase"]).lower(), -1))
        if ph < 0:
            continue
        iu, iv = bus_to_local[fb], bus_to_local[tb]
        # Edge CSV is already bidirectional for phase-nodes; keep as-is at bus level
        # (each directed phase edge once). Avoid double-doubling.
        line_name = str(r["line_name"])
        ctype = _classify_line_type(line_name)
        stem = _line_component_stem(line_name)
        rj = reg_stem_to_j.get(stem, -1)
        if rj >= 0:
            matched_reg.add(rj)
            ctype = "regulator"
        phase_oh = [0.0, 0.0, 0.0]
        phase_oh[ph] = 1.0
        ea = phase_oh + type_onehot[ctype] + [0.0, 1.0]  # tap=0 placeholder, switch_closed=1
        src.append(iu)
        dst.append(iv)
        attrs.append(ea)
        tap_idx.append(int(rj))

    if not src:
        raise RuntimeError(f"No usable edges from {edges_csv}")

    if matched_reg:
        # Remap tap indices to compact active reg list
        active_reg = [reg_cols_all[j] for j in sorted(matched_reg)]
        old_to_new = {j: i for i, j in enumerate(sorted(matched_reg))}
        tap_idx = [old_to_new[t] if t >= 0 else -1 for t in tap_idx]
        reg_cols = active_reg
    else:
        reg_cols = list(reg_cols_all)
        # remap: keep global indices as-is (edge tap_idx already indexes reg_cols_all)
        pass

    # Capacitors: prefer those that map onto a bus or have a clear phase tag.
    cap_cols: list[str] = []
    cap_bus_phase: list[tuple[int, int | None]] = []
    bus_set = set(bus_order)
    for col in cap_cols_all:
        st = _cap_col_stem(col)
        if not st:
            continue
        bus, phase = _guess_cap_bus_and_phase(st, bus_set)
        # Always keep for device_state; bus attach when possible
        cap_cols.append(col)
        if bus is not None and bus in bus_to_local:
            cap_bus_phase.append((bus_to_local[bus], phase))
        else:
            cap_bus_phase.append((-1, phase))

    return PfmnGraphStatic(
        bus_order=bus_order,
        bus_to_local=bus_to_local,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr_static=torch.tensor(np.asarray(attrs, dtype=np.float32), dtype=torch.float32),
        edge_tap_reg_idx=torch.tensor(tap_idx, dtype=torch.long),
        phase_present=torch.from_numpy(phase_present),
        source_bus_local=int(source_bus_local),
        reg_cols=reg_cols,
        cap_cols=cap_cols,
        cap_bus_phase=cap_bus_phase,
    )


def load_pfmn_chunk_tensors(
    nodes_csv: Path,
    edges_csv: Path,
    meta_csv: Path,
    *,
    graph: PfmnGraphStatic | None = None,
    csv_chunksize: int = 500_000,
) -> dict:
    """Load one chunk into PowerFlowMultiNet sample tensors."""
    g = graph or build_physical_bus_static(nodes_csv, edges_csv, meta_csv)
    n_bus = len(g.bus_order)
    phase_present = g.phase_present.numpy()

    # Collect sample ids
    sid_set: set[int] = set()
    for ch in pd.read_csv(nodes_csv, usecols=["sample_id"], chunksize=csv_chunksize):
        sid_set.update(int(_norm_sid(s)) for s in ch["sample_id"].tolist())
    sample_ids = sorted(sid_set)
    if not sample_ids:
        raise RuntimeError(f"No sample_ids in {nodes_csv}")
    sid_to_i = {s: i for i, s in enumerate(sample_ids)}
    S = len(sample_ids)

    # x: P/Q + masks + source + caps
    x_np = np.zeros((S, n_bus, NODE_FEAT_DIM), dtype=np.float32)
    y_np = np.zeros((S, n_bus, 6), dtype=np.float32)
    mask_np = np.zeros((S, n_bus, 6), dtype=np.float32)

    # static masks / source
    for bi in range(n_bus):
        x_np[:, bi, 6:9] = phase_present[bi]
        if bi == g.source_bus_local:
            x_np[:, bi, 9] = 1.0
        for ph in range(3):
            if phase_present[bi, ph] > 0.5:
                mask_np[:, bi, 2 * ph] = 1.0
                mask_np[:, bi, 2 * ph + 1] = 1.0

    hdr = _csv_header_lower(nodes_csv)
    must = ["sample_id", "p_load_kw", "q_load_kvar", "vmag_pu", "vang_deg"]
    for c in must:
        if c not in hdr:
            raise ValueError(f"{nodes_csv} missing required column {c!r} (have {sorted(hdr)})")
    usecols = [hdr[c] for c in must]
    for opt in ("bus", "phase", "node", "p_pv_kw"):
        if opt in hdr and hdr[opt] not in usecols:
            usecols.append(hdr[opt])
    if "bus" not in hdr and "phase" not in hdr and "node" not in hdr:
        raise ValueError(f"{nodes_csv} needs bus+phase or node to derive them (have {sorted(hdr)})")

    fill = np.zeros((S, n_bus, 3), dtype=np.int8)
    for ch in pd.read_csv(nodes_csv, usecols=usecols, chunksize=csv_chunksize):
        # Normalize to lower-case canonical names
        ren = {hdr[k]: k for k in must if hdr[k] != k}
        for opt in ("bus", "phase", "node", "p_pv_kw"):
            if opt in hdr and hdr[opt] != opt:
                ren[hdr[opt]] = opt
        if ren:
            ch = ch.rename(columns=ren)
        ch = _ensure_bus_phase_cols(ch)
        sid_arr = ch["sample_id"].map(_norm_sid).to_numpy(dtype=np.int64)
        bus_arr = ch["bus"].astype(str).str.strip().str.lower().map(g.bus_to_local).fillna(-1).to_numpy(
            dtype=np.int64
        )
        ph_arr = ch["phase"].map(lambda p: _PHASE_TO_I.get(p, _PHASE_TO_I.get(str(p).lower(), -1))).to_numpy(
            dtype=np.int64
        )
        valid = (bus_arr >= 0) & (ph_arr >= 0) & np.isin(sid_arr, sample_ids)
        if not np.any(valid):
            continue
        s_loc = np.array([sid_to_i[int(s)] for s in sid_arr[valid]], dtype=np.int64)
        b_loc = bus_arr[valid]
        p_loc = ph_arr[valid]
        p_load = ch.loc[valid, "p_load_kw"].to_numpy(dtype=np.float32)
        q_load = ch.loc[valid, "q_load_kvar"].to_numpy(dtype=np.float32)
        if "p_pv_kw" in ch.columns:
            p_pv = ch.loc[valid, "p_pv_kw"].to_numpy(dtype=np.float32)
        else:
            p_pv = np.zeros(int(np.count_nonzero(valid)), dtype=np.float32)
        # Net load convention (documented): P = p_load - p_pv, Q = q_load
        x_np[s_loc, b_loc, 2 * p_loc] = p_load - p_pv
        x_np[s_loc, b_loc, 2 * p_loc + 1] = q_load
        vmag = ch.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
        vang = np.deg2rad(ch.loc[valid, "vang_deg"].to_numpy(dtype=np.float32))
        y_np[s_loc, b_loc, 2 * p_loc] = vmag
        y_np[s_loc, b_loc, 2 * p_loc + 1] = vang
        fill[s_loc, b_loc, p_loc] = 1

    # Meta device states
    meta = pd.read_csv(meta_csv)
    ren = {c: str(c).lower() for c in meta.columns if str(c).lower() != str(c)}
    if ren:
        meta = meta.rename(columns=ren)
    meta["sample_id"] = meta["sample_id"].map(_norm_sid)
    lk = {int(s): j for j, s in enumerate(meta["sample_id"].tolist())}
    order = [lk[s] for s in sample_ids]

    n_reg = len(g.reg_cols)
    n_cap = len(g.cap_cols)
    reg_np = np.ones((S, max(1, n_reg)), dtype=np.float32)
    cap_np = np.zeros((S, max(1, n_cap)), dtype=np.float32)
    if n_reg:
        for j, col in enumerate(g.reg_cols):
            if col not in meta.columns:
                continue
            reg_np[:, j] = meta[col].to_numpy(dtype=np.float32)[order]
    if n_cap:
        for j, col in enumerate(g.cap_cols):
            if col not in meta.columns:
                continue
            cap_np[:, j] = (meta[col].to_numpy(dtype=np.float64)[order] > 0.5).astype(np.float32)

    # Stamp caps onto bus features
    for j, (bi, phase) in enumerate(g.cap_bus_phase):
        if bi < 0 or j >= n_cap:
            continue
        if phase is None:
            for ph in range(3):
                x_np[:, bi, 10 + ph] = np.maximum(x_np[:, bi, 10 + ph], cap_np[:, j])
        else:
            x_np[:, bi, 10 + int(phase)] = np.maximum(x_np[:, bi, 10 + int(phase)], cap_np[:, j])

    # device_state = capacitor (and future switch) vector
    if n_cap:
        device_state = cap_np[:, :n_cap]
    else:
        device_state = np.zeros((S, 1), dtype=np.float32)

    # Optional substation targets: total grid P/Q split equally across present source phases
    y_sub = np.zeros((S, 6), dtype=np.float32)
    if "p_grid_upstream_post_kw" in meta.columns and "q_grid_upstream_post_kvar" in meta.columns:
        p_tot = meta["p_grid_upstream_post_kw"].to_numpy(dtype=np.float32)[order]
        q_tot = meta["q_grid_upstream_post_kvar"].to_numpy(dtype=np.float32)[order]
        n_ph = max(1.0, float(phase_present[g.source_bus_local].sum()))
        for ph in range(3):
            if phase_present[g.source_bus_local, ph] > 0.5:
                y_sub[:, 2 * ph] = p_tot / n_ph
                y_sub[:, 2 * ph + 1] = q_tot / n_ph

    if n_reg:
        reg_out = reg_np[:, :n_reg]
    else:
        reg_out = np.ones((S, 1), dtype=np.float32)

    return {
        "sample_ids": sample_ids,
        "bus_order": g.bus_order,
        "edge_index": g.edge_index,
        "edge_attr_static": g.edge_attr_static,
        "edge_tap_reg_idx": g.edge_tap_reg_idx,
        "phase_present": g.phase_present,
        "source_bus_local": g.source_bus_local,
        "reg_cols": g.reg_cols,
        "cap_cols": g.cap_cols,
        "x": torch.from_numpy(x_np),
        "y_voltage": torch.from_numpy(y_np),
        "y_voltage_mask": torch.from_numpy(mask_np),
        "y_substation": torch.from_numpy(y_sub),
        "device_state": torch.from_numpy(device_state),
        "reg_taps": torch.from_numpy(reg_out),
        "node_dim": NODE_FEAT_DIM,
        "edge_dim": EDGE_FEAT_DIM,
        "state_dim": int(device_state.shape[1]),
    }


def materialize_edge_attr(
    edge_attr_static: torch.Tensor,
    edge_tap_reg_idx: torch.Tensor,
    reg_taps: torch.Tensor,
) -> torch.Tensor:
    """Fill tap column from oracle regulator taps for one sample ``reg_taps [n_reg]``."""
    ea = edge_attr_static.clone()
    idx = edge_tap_reg_idx
    valid = idx >= 0
    if valid.any():
        # Normalize tap around typical OpenDSS range [0.9, 1.1] → [0, 1]
        taps = reg_taps[idx[valid]].clamp(0.9, 1.1)
        ea[valid, EDGE_TAP_IDX] = (taps - 0.9) / 0.2
    return ea


__all__ = [
    "NODE_FEAT_DIM",
    "EDGE_FEAT_DIM",
    "EDGE_TAP_IDX",
    "NODE_CONT_IDX",
    "EDGE_CONT_IDX",
    "PfmnGraphStatic",
    "build_physical_bus_static",
    "discover_meta_device_cols",
    "load_pfmn_chunk_tensors",
    "materialize_edge_attr",
]
