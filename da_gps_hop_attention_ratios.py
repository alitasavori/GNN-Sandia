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

from pathlib import Path

import numpy as np
import pandas as pd

from compute_hop_distance_all_index_nodes import HOP_NOT_IN_MV_CATALOG
from train_da_gps_multitask_complex_voltage import TARGET_REG_COLS

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


__all__ = [
    "TARGET_REG_COLS",
    "REG_COL_TO_HOP_COL",
    "HOP_NOT_IN_MV_CATALOG",
    "load_hop_frame",
    "hops_for_manifest_nodes",
    "downstream_mask",
    "non_downstream_catalog_mask",
    "node_regulator_hop_dataframe",
    "worst_nodes_downstream_regulator_table",
    "attention_downstream_ratio_table",
    "attention_downstream_ratio_table_tn",
]
