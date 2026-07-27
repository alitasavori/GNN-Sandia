"""
Map DA-GPS ``reg_*_tap_pu`` targets to hop CSV columns (``compute_hop_distance_all_index_nodes.py`` output)
and compute downstream vs rest attention ratios.

Hop CSV convention:
- ``0`` = node appears in ``hetero_mv_edge_catalog`` but is not strictly downstream (``hop > 0``) for that
  regulator — includes reference ``terminal_2`` and buses outside that regulator's downstream subtree.
- ``-1`` = node does not appear on any edge of the hop topology (``hetero_mv_edge_catalog`` or
  ``gnn_edges_phase_static`` per ``compute_hop_distance_all_index_nodes.py``);
  excluded from downstream vs "other" ratio splits (see ``downstream_mask``).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from train_da_gps_multitask_complex_voltage import TARGET_REG_COLS

# Hop CSV sentinel (``compute_hop_distance_all_index_nodes.py`` convention): node absent from topology graph.
HOP_NOT_IN_MV_CATALOG = -1

# Output columns from compute_hop_distance_all_index_nodes.py (regulator names in CSV)
REG_COL_TO_HOP_COL: dict[str, str] = {  # keys must match checkpoint ``reg_target_cols`` / TARGET_REG_COLS
    "reg_feeder_rega_tap_pu": "FEEDER_REGA",
    "reg_feeder_regb_tap_pu": "FEEDER_REGB",
    "reg_feeder_regc_tap_pu": "FEEDER_REGC",
    "reg_vreg2_a_tap_pu": "VREG2_A",
    "reg_vreg2_b_tap_pu": "VREG2_B",
    "reg_vreg2_c_tap_pu": "VREG2_C",
    "reg_vreg3_a_tap_pu": "VREG3_A",
    "reg_vreg3_b_tap_pu": "VREG3_B",
    "reg_vreg3_c_tap_pu": "VREG3_C",
    "reg_vreg4_a_tap_pu": "VREG4_A",
    "reg_vreg4_b_tap_pu": "VREG4_B",
    "reg_vreg4_c_tap_pu": "VREG4_C",
}

if set(REG_COL_TO_HOP_COL.keys()) != set(TARGET_REG_COLS):
    raise RuntimeError("REG_COL_TO_HOP_COL keys must match TARGET_REG_COLS")

_PHASE_FROM_REG_COL_RE = re.compile(r"_(rega|regb|regc|[abc])_tap_pu$", re.IGNORECASE)


def _phase_letter_from_reg_col(reg_col: str) -> str:
    m = _PHASE_FROM_REG_COL_RE.search(reg_col)
    if not m:
        raise ValueError(f"Cannot parse phase from regulator column {reg_col!r}")
    return {"rega": "A", "regb": "B", "regc": "C", "a": "A", "b": "B", "c": "C"}[m.group(1).lower()]


def _reg_unit_and_phase_from_reg_col(reg_col: str) -> tuple[str, str]:
    phase = _phase_letter_from_reg_col(reg_col)
    stem = reg_col.replace("reg_", "").replace("_tap_pu", "")
    for suf in ("_rega", "_regb", "_regc", "_a", "_b", "_c"):
        if stem.endswith(suf):
            unit = stem[: -len(suf)]
            break
    else:
        raise ValueError(f"Cannot parse regulator unit from column {reg_col!r}")
    if unit == "feeder":
        return "FEEDER", phase
    if unit.startswith("vreg"):
        return unit.upper(), phase
    raise ValueError(f"Unknown regulator unit in column {reg_col!r}")


def _reg_unit_and_phase_from_hop_col(hop_col: str) -> tuple[str, str]:
    if hop_col.startswith("FEEDER_REG") and len(hop_col) == len("FEEDER_REGA"):
        return "FEEDER", hop_col[-1]
    if "_" in hop_col and hop_col.startswith("VREG"):
        unit, phase = hop_col.rsplit("_", 1)
        if len(phase) == 1 and phase in "ABC":
            return unit, phase
    raise ValueError(f"Cannot parse regulator unit/phase from hop column {hop_col!r}")


def validate_reg_hop_csv(
    hop_csv: Path | str,
    reg_cols: list[str],
    *,
    node_names: list[str] | None = None,
    reg_col_to_hop_col: dict[str, str] | None = None,
    regulator_csv: Path | str | None = None,
    max_missing_node_frac: float = 0.0,
) -> dict[str, str]:
    """Preflight: map ``reg_cols`` to hop CSV columns; raise on missing/unmapped/phase mismatch.

  When ``node_names`` is set, also checks that every training node appears in the hop CSV.
  ``max_missing_node_frac=0`` (default) errors on any missing node; increase to warn only.
  """
    mapping = dict(reg_col_to_hop_col or REG_COL_TO_HOP_COL)
    p = Path(hop_csv).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Hop CSV not found: {p}")

    hop_df = load_hop_frame(p)
    hop_cols = set(hop_df.columns)

    unmapped = [c for c in reg_cols if c not in mapping]
    if unmapped:
        raise ValueError(
            f"Regulator target columns have no hop mapping: {unmapped}. "
            f"Update REG_COL_TO_HOP_COL in da_gps_hop_attention_ratios.py."
        )

    pairs: dict[str, str] = {}
    missing_hop_cols: list[str] = []
    phase_mismatches: list[tuple[str, str, str, str]] = []
    unit_mismatches: list[tuple[str, str, str, str]] = []

    for reg_col in reg_cols:
        hop_col = mapping[reg_col]
        pairs[reg_col] = hop_col
        if hop_col not in hop_cols:
            missing_hop_cols.append(hop_col)
            continue
        reg_unit, reg_phase = _reg_unit_and_phase_from_reg_col(reg_col)
        hop_unit, hop_phase = _reg_unit_and_phase_from_hop_col(hop_col)
        if reg_phase != hop_phase:
            phase_mismatches.append((reg_col, hop_col, reg_phase, hop_phase))
        if reg_unit != hop_unit:
            unit_mismatches.append((reg_col, hop_col, reg_unit, hop_unit))

    if missing_hop_cols:
        raise ValueError(
            f"Hop CSV {p} missing regulator columns {sorted(set(missing_hop_cols))}. "
            f"Have: {[c for c in hop_df.columns if c != 'node']}"
        )
    if phase_mismatches:
        detail = "; ".join(
            f"{rc}->{hc} reg_phase={rp} hop_phase={hp}" for rc, hc, rp, hp in phase_mismatches
        )
        raise ValueError(f"Regulator/hop column phase mismatch (A/B/C swapped?): {detail}")
    if unit_mismatches:
        detail = "; ".join(
            f"{rc}->{hc} reg_unit={ru} hop_unit={hu}" for rc, hc, ru, hu in unit_mismatches
        )
        raise ValueError(f"Regulator/hop column unit mismatch: {detail}")

    if regulator_csv is not None:
        reg_path = Path(regulator_csv).resolve()
        if reg_path.is_file():
            reg_df = pd.read_csv(reg_path)
            reg_name_col = "Regulator" if "Regulator" in reg_df.columns else "regulator"
            if reg_name_col not in reg_df.columns:
                raise ValueError(f"{reg_path} must have a Regulator column")
            catalog_hop_cols = {str(x).strip() for x in reg_df[reg_name_col].astype(str)}
            extra = sorted(set(pairs.values()) - catalog_hop_cols)
            if extra:
                raise ValueError(
                    f"Hop columns {extra} not listed in {reg_path} ({reg_name_col}); "
                    "territory mask may target wrong regulators."
                )

    if node_names is not None:
        hop_nodes = set(hop_df["node"].astype(str).str.strip().str.lower())
        miss = [str(n).strip().lower() for n in node_names if str(n).strip().lower() not in hop_nodes]
        if miss:
            frac = len(miss) / max(len(node_names), 1)
            msg = (
                f"Hop CSV missing {len(miss)}/{len(node_names)} training nodes "
                f"({frac:.4%}); first examples: {miss[:5]}"
            )
            if frac > float(max_missing_node_frac):
                raise ValueError(
                    msg + " — regenerate hop CSV with the same node index as training "
                    "(compute_hop_distance_all_index_nodes.py --node-index ...)."
                )
            print(f"WARNING: {msg}", flush=True)

    return pairs


def load_hop_frame(hop_csv: Path | str) -> pd.DataFrame:
    p = Path(hop_csv).resolve()
    df = pd.read_csv(p)
    if "node" not in df.columns:
        raise ValueError(f"{p} must have a 'node' column")
    df = df.copy()
    df["node"] = df["node"].astype(str).str.strip().str.lower()
    return df


def hops_for_manifest_nodes(
    hop_df: pd.DataFrame,
    node_names: list[str],
    hop_col: str,
) -> tuple[np.ndarray, list[str]]:
    """Return hop counts aligned to ``node_names`` order; fill 0 for missing nodes."""
    col = hop_col
    if col not in hop_df.columns:
        raise KeyError(f"Hop CSV missing column {col!r}; have {list(hop_df.columns)}")
    m = hop_df.set_index("node")[col]
    miss: list[str] = []
    out = np.zeros(len(node_names), dtype=np.int32)
    for i, n in enumerate(node_names):
        k = str(n).strip().lower()
        if k not in m.index:
            miss.append(k)
            continue
        out[i] = int(m.loc[k])
    return out, miss


def downstream_mask(hops: np.ndarray, *, rule: str) -> np.ndarray:
    """Strictly downstream nodes; never true for ``HOP_NOT_IN_MV_CATALOG`` (-1)."""
    unk = hops == HOP_NOT_IN_MV_CATALOG
    if rule == "hop_gt_0":
        return (hops > 0) & ~unk
    if rule == "hop_ge_1":
        return (hops >= 1) & ~unk
    raise ValueError(f"Unknown rule {rule!r} (use 'hop_gt_0' or 'hop_ge_1')")


def non_downstream_catalog_mask(hops: np.ndarray, *, rule: str) -> np.ndarray:
    """In-catalog, not strictly downstream — complements ``downstream_mask`` (excludes -1)."""
    unk = hops == HOP_NOT_IN_MV_CATALOG
    return ~downstream_mask(hops, rule=rule) & ~unk


def node_regulator_hop_dataframe(
    hop_df: pd.DataFrame,
    node_names: list[str],
    reg_target_cols: list[str],
    *,
    reg_col_to_hop_col: dict[str, str],
) -> pd.DataFrame:
    """Hop counts per regulator column, aligned to ``node_names`` order.

    Index: node names (exact strings from ``node_names``). Columns: short regulator labels
    (``feeder_rega``, ``vreg2_a``, …). Missing hop CSV mapping for a ``reg_target_cols`` entry
    is skipped. Values are integers from the hop CSV (0 = in-catalog but not hop>0; -1 = not in MV catalog).
    """
    cols: dict[str, np.ndarray] = {}
    for reg_col in reg_target_cols:
        hop_col = reg_col_to_hop_col.get(reg_col)
        if hop_col is None:
            continue
        hvec, _ = hops_for_manifest_nodes(hop_df, node_names, hop_col)
        short = reg_col.replace("reg_", "").replace("_tap_pu", "")
        cols[short] = hvec.astype(np.int32)
    if not cols:
        raise ValueError("No regulator columns matched REG_COL_TO_HOP_COL")
    return pd.DataFrame(cols, index=[str(n) for n in node_names])


def worst_nodes_downstream_regulator_table(
    worst_nodes_df: pd.DataFrame,
    hop_df: pd.DataFrame,
    node_manifest_names: list[str],
    reg_target_cols: list[str],
    *,
    reg_col_to_hop_col: dict[str, str],
    downstream_rule: str = "hop_gt_0",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join worst-bus rows with downstream-regulator labels and hop submatrix for heatmaps.

    ``worst_nodes_df`` must have a ``node`` column (bus names matching ``node_manifest_names``).

    Returns:
        (annotated, hop_sub) where ``annotated`` adds ``downstream_regs``, ``n_downstream_regs``,
        and ``hop_*`` columns for each mapped regulator; ``hop_sub`` is (K, R) aligned to worst
        row order for plotting (integer hops, NaN if node cannot be resolved).
    """
    hop_mat = node_regulator_hop_dataframe(
        hop_df, node_manifest_names, reg_target_cols, reg_col_to_hop_col=reg_col_to_hop_col
    )
    idx_map = {str(k).strip().lower(): str(k) for k in hop_mat.index}

    def _resolve(key: str) -> str | None:
        s = str(key).strip()
        if s in hop_mat.index:
            return s
        return idx_map.get(s.lower())

    out = worst_nodes_df.copy()
    reg_short_cols = list(hop_mat.columns)
    downstream_regs: list[str] = []
    n_down: list[int] = []
    hop_rows: list[np.ndarray] = []
    for _, row in worst_nodes_df.iterrows():
        key = _resolve(row["node"])
        if key is None:
            downstream_regs.append("")
            n_down.append(0)
            hop_rows.append(np.full(len(reg_short_cols), np.nan, dtype=np.float64))
            continue
        hrow = hop_mat.loc[key]
        hvals = hrow.to_numpy(dtype=np.int32)
        if np.all(hvals == HOP_NOT_IN_MV_CATALOG):
            downstream_regs.append("(not in hop topology graph / edges CSV)")
            n_down.append(0)
            hop_rows.append(hvals.astype(np.float64))
            continue
        dmask = downstream_mask(hvals, rule=downstream_rule)
        names = [reg_short_cols[j] for j in range(len(reg_short_cols)) if dmask[j]]
        downstream_regs.append(", ".join(names))
        n_down.append(len(names))
        hop_rows.append(hrow.to_numpy(dtype=np.float64))
    out["downstream_regs"] = downstream_regs
    out["n_downstream_regs"] = n_down
    H = np.stack(hop_rows, axis=0)
    hop_sub = pd.DataFrame(H, columns=reg_short_cols, index=worst_nodes_df["node"].astype(str).tolist())
    for j, c in enumerate(reg_short_cols):
        out[f"hop_{c}"] = H[:, j].astype(np.float64)
    return out, hop_sub


def _normalize_mean_heads_to_lnt(mean_heads: object) -> np.ndarray:
    """(L,N,T) float32; drop singleton batch (L,1,N,T). Accepts NumPy or torch.Tensor."""
    if hasattr(mean_heads, "detach"):
        x = mean_heads.detach().float().cpu().numpy()
    else:
        x = np.asarray(mean_heads)
    x = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = tuple(int(s) for s in x.shape)
    while x.ndim == 4 and int(x.shape[1]) == 1:
        x = x[:, 0, :, :]
    if x.ndim != 3:
        raise ValueError(
            "mean_heads must be (L, N, T), optionally (L, 1, N, T) with a batch dimension; "
            f"got raw shape {orig_shape} -> {tuple(x.shape)}"
        )
    return x


def attention_downstream_ratio_table(
    mean_heads: np.ndarray,
    *,
    reg_target_cols: list[str],
    n_cap: int,
    node_names: list[str],
    hop_df: pd.DataFrame,
    downstream_rule: str = "hop_gt_0",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """mean_heads: (L, N, T) from extract script (mean over heads).

    Accepts legacy (L, 1, N, T) or a torch.Tensor with the same layout.
    """
    mean_heads = _normalize_mean_heads_to_lnt(mean_heads)
    L, N, T = mean_heads.shape
    if N != len(node_names):
        raise ValueError(f"N={N} vs len(node_names)={len(node_names)}")

    rows: list[dict] = []
    for reg_col in reg_target_cols:
        hop_col = REG_COL_TO_HOP_COL.get(reg_col)
        if hop_col is None:
            continue
        tok = n_cap + int(reg_target_cols.index(reg_col))
        hops, miss = hops_for_manifest_nodes(hop_df, node_names, hop_col)
        if miss:
            # one line in notebook can print; keep quiet here or use logging
            pass
        dmask = downstream_mask(hops, rule=downstream_rule)
        omask = non_downstream_catalog_mask(hops, rule=downstream_rule)
        if not np.any(dmask) or not np.any(omask):
            continue
        for layer in range(L):
            a = mean_heads[layer, :, tok]
            mu_d = float(a[dmask].mean())
            mu_o = float(a[omask].mean())
            rows.append(
                {
                    "layer": layer,
                    "reg_col": reg_col,
                    "hop_col": hop_col,
                    "token_idx": tok,
                    "mu_downstream": mu_d,
                    "mu_other": mu_o,
                    "ratio": mu_d / (mu_o + eps),
                    "n_downstream": int(dmask.sum()),
                    "n_other": int(omask.sum()),
                    "n_hop_missing": len(miss),
                }
            )
    return pd.DataFrame(rows)


def _normalize_mean_heads_to_ltn(mean_heads: object) -> np.ndarray:
    """(L, T, N) token→node, mean over heads; drop singleton batch (L, 1, T, N)."""
    if hasattr(mean_heads, "detach"):
        x = mean_heads.detach().float().cpu().numpy()
    else:
        x = np.asarray(mean_heads)
    x = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = tuple(int(s) for s in x.shape)
    while x.ndim == 4 and int(x.shape[1]) == 1:
        x = x[:, 0, :, :]
    if x.ndim != 3:
        raise ValueError(
            "mean_heads_tn must be (L, T, N), optionally (L, 1, T, N); "
            f"got raw shape {orig_shape} -> {tuple(x.shape)}"
        )
    return x


def attention_downstream_ratio_table_tn(
    mean_heads_tn: np.ndarray,
    *,
    reg_target_cols: list[str],
    n_cap: int,
    node_names: list[str],
    hop_df: pd.DataFrame,
    downstream_rule: str = "hop_gt_0",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """First cross-attn (token→node): for each regulator **token**, compare mean attention
    mass on downstream vs other **nodes** (same hop masks as node→token analysis).

    ``mean_heads_tn``: (L, T, N) after mean over heads.
    """
    x = _normalize_mean_heads_to_ltn(mean_heads_tn)
    L, T, Nn = x.shape
    if Nn != len(node_names):
        raise ValueError(f"N={Nn} vs len(node_names)={len(node_names)}")

    rows: list[dict] = []
    for reg_col in reg_target_cols:
        hop_col = REG_COL_TO_HOP_COL.get(reg_col)
        if hop_col is None:
            continue
        tok = n_cap + int(reg_target_cols.index(reg_col))
        hops, miss = hops_for_manifest_nodes(hop_df, node_names, hop_col)
        dmask = downstream_mask(hops, rule=downstream_rule)
        omask = non_downstream_catalog_mask(hops, rule=downstream_rule)
        if not np.any(dmask) or not np.any(omask):
            continue
        for layer in range(L):
            a = x[layer, tok, :]
            mu_d = float(a[dmask].mean())
            mu_o = float(a[omask].mean())
            rows.append(
                {
                    "layer": layer,
                    "reg_col": reg_col,
                    "hop_col": hop_col,
                    "token_idx": tok,
                    "mu_downstream": mu_d,
                    "mu_other": mu_o,
                    "ratio": mu_d / (mu_o + eps),
                    "n_downstream": int(dmask.sum()),
                    "n_other": int(omask.sum()),
                    "n_hop_missing": len(miss),
                }
            )
    return pd.DataFrame(rows)


def attention_ratio_vs_hop_distance(
    mean_heads: np.ndarray,
    *,
    reg_target_cols: list[str],
    n_cap: int,
    node_names: list[str],
    hop_df: pd.DataFrame,
    layer: int | None = None,
    max_hop: int | None = None,
    eps: float = 1e-8,
    direction: str = "node_to_token",
) -> pd.DataFrame:
    """Downstream-to-non-downstream attention ratio as a function of hop distance.

    For each regulator and hop ``h > 0``, ratio =
    ``mean(attn | hop==h) / (mean(attn | non-downstream hop≤0) + eps)``.

    ``mean_heads``: node→token ``(L,N,T)`` when ``direction='node_to_token'``,
    else token→node ``(L,T,N)``.
    """
    if direction == "node_to_token":
        attn = _normalize_mean_heads_to_lnt(mean_heads)
        L = int(attn.shape[0])
    elif direction == "token_to_node":
        attn = _normalize_mean_heads_to_ltn(mean_heads)
        L = int(attn.shape[0])
    else:
        raise ValueError(f"direction must be node_to_token or token_to_node, got {direction!r}")

    layers = [int(layer)] if layer is not None else list(range(L))
    rows: list[dict] = []
    for reg_col in reg_target_cols:
        hop_col = REG_COL_TO_HOP_COL.get(reg_col)
        if hop_col is None:
            continue
        tok = int(n_cap) + int(reg_target_cols.index(reg_col))
        hops, _miss = hops_for_manifest_nodes(hop_df, node_names, hop_col)
        omask = non_downstream_catalog_mask(hops, rule="hop_gt_0")
        if not np.any(omask):
            continue
        hop_pos = hops[(hops > 0) & np.isfinite(hops)]
        if hop_pos.size == 0:
            continue
        h_max = int(np.nanmax(hop_pos)) if max_hop is None else int(max_hop)
        for lyr in layers:
            if direction == "node_to_token":
                a = attn[lyr, :, tok]
            else:
                a = attn[lyr, tok, :]
            mu_o = float(a[omask].mean())
            for h in range(1, h_max + 1):
                m = hops == h
                if not np.any(m):
                    continue
                mu_h = float(a[m].mean())
                rows.append(
                    {
                        "layer": int(lyr),
                        "reg_col": reg_col,
                        "hop_col": hop_col,
                        "hop": int(h),
                        "mu_at_hop": mu_h,
                        "mu_other": mu_o,
                        "ratio": mu_h / (mu_o + float(eps)),
                        "n_at_hop": int(m.sum()),
                        "n_other": int(omask.sum()),
                        "direction": direction,
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "TARGET_REG_COLS",
    "REG_COL_TO_HOP_COL",
    "HOP_NOT_IN_MV_CATALOG",
    "validate_reg_hop_csv",
    "load_hop_frame",
    "hops_for_manifest_nodes",
    "downstream_mask",
    "non_downstream_catalog_mask",
    "node_regulator_hop_dataframe",
    "worst_nodes_downstream_regulator_table",
    "attention_downstream_ratio_table",
    "attention_downstream_ratio_table_tn",
    "attention_ratio_vs_hop_distance",
]
