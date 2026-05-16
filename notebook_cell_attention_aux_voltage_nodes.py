# Full notebook cell: aux + reg bars + per-bus voltage rank + attention / hop / hist.
# Also prints **per meta-aux / system token** the top graph nodes by cross-attention (mean over GPS layers,
# heads in extract, cache samples): **node→token** (nodes attending *to* each aux) and **token→node**
# (which nodes each aux token attends to). Writes ``aux_sys_node_to_token_mean_layers_avg*.csv`` and
# ``aux_sys_token_to_node_mean_layers_avg*.csv``.
# Source of truth: this file (copy entire file into one Jupyter cell, or %run it).
#
# --- Two training styles (same script) ---
# 1) **GINE / meta-aux / ``--exclude_bess_features``** — cache often
#    ``run_*__full__nobess__maux<md5>.pt``; ``x`` has fewer columns; ``augment_da_gps_pack_for_eval`` in
#    ``extract_da_gps_attention.py`` fills missing periodic-ckpt metadata from ``da_gps_multitask_best.pt``
#    or from the state dict.
# 2) **BESS in node features** (no ``__nobess`` slug) — cache ``run_*__full.pt`` or ``run_*__full__maux<md5>.pt``;
#    ``x`` is wider (``p_bess_kw``, ``q_bess_kvar``); same eval code; set ``RUN_DIR`` / ``CACHE_PT`` / ``CKPT_PATH``
#    to that run. Use ``NOTEBOOK_PRESET`` (see below) or full paths in ``__NOTEBOOK_KNOBS__``.
#    If ``gnn_edges_phase_static.csv`` is **not** under ``run_*\``, set ``"EDGES_CSV": Path(r"...")`` in knobs
#    (same for ``HOP_CSV`` if needed). The script auto-falls back to ``datasets_gnn2_from pc\gnn_edges_phase_static.csv``
#    when the chunk-subdir path is missing. Preset ``bess_l3_chunk`` sets ``CACHE_RESOLVER_REJECT_SUBSTR`` so
#    ``__nobess__`` tensor caches are not auto-picked against a with-BESS ``RUN_DIR`` (set ``CACHE_PT`` explicitly
#    if no BESS-aligned sibling ``.pt`` exists yet).
#
# **Presets:** set ``NOTEBOOK_PRESET = "bess_l3_chunk"`` before ``exec``, or add
# ``"NOTEBOOK_PRESET": "bess_l3_chunk"`` to ``__NOTEBOOK_KNOBS__`` (applied before other knob keys).
# Edit ``RUN_DIR`` inside ``NOTEBOOK_PRESETS["bess_l3_chunk"]`` once, or override paths in knobs.
#
# Example — **BESS** (after editing ``RUN_DIR`` in ``NOTEBOOK_PRESETS`` or overriding here)::
#
#   from pathlib import Path
#   __NOTEBOOK_KNOBS__ = {
#       "NOTEBOOK_PRESET": "bess_l3_chunk",
#       "RUN_DIR": Path(r"C:\...\your_DA_GPS_out_with_x_mean"),
#       "CKPT_PATH": Path(r"C:\...\training_last.pt"),
#       "EDGES_CSV": Path(r"C:\...\gnn_edges_phase_static.csv"),
#       "HOP_CSV": Path(r"C:\...\load_hop_distance_to_each_regulator_all_index_nodes.csv"),
#       "N_SAMPLES_AVG": 1000,
#   }
#   _p = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\notebook_cell_attention_aux_voltage_nodes.py")
#   exec(compile(_p.read_text(encoding="utf-8"), str(_p), "exec"), globals())
#
# Example — **GINE / nobess / meta-aux** (defaults in file; optional preset name for clarity)::
#
#   __NOTEBOOK_KNOBS__ = {"NOTEBOOK_PRESET": "gine_metaaux_default", "N_SAMPLES_AVG": 1000}

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2").resolve()
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib

import da_gps_hop_attention_ratios
import extract_da_gps_attention

importlib.reload(extract_da_gps_attention)
importlib.reload(da_gps_hop_attention_ratios)

from da_gps_hop_attention_ratios import (
    REG_COL_TO_HOP_COL,
    attention_downstream_ratio_table,
    attention_downstream_ratio_table_tn,
    load_hop_frame,
    worst_nodes_downstream_regulator_table,
)
from extract_da_gps_attention import (
    eval_aux_per_device_on_cache_indices,
    eval_voltage_per_node_errors_on_cache_indices,
    run_attention_extract,
)
from plot_da_gps_attention_hop_histograms_all import plot_all_regulator_layer_hop_histograms


def _regulator_cross_attn_node_tables(
    mh: np.ndarray,
    mhtn: np.ndarray,
    *,
    n_cap: int,
    reg_cols: list[str],
    node_names: list[str],
    top_k_global: int,
    top_k_per_reg: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Rank buses by regulator-related cross-attention (mean over heads, max over layers).

    - **First cross-attn (token→node):** ``mhtn[l, reg_token, n]`` — regulator token attends to node.
    - **Second cross-attn (node→token):** ``mh[l, n, reg_token]`` — node attends to regulator token.

    For each node, pool over all regulator tokens and GPS layers, then take
    ``max(max_nt, max_tn)`` so a bus is highlighted if it is strong in **either** direction.
    """
    _L, N, _T = mh.shape
    R = len(reg_cols)
    rt0 = int(n_cap)
    w_nt = mh[:, :, rt0 : rt0 + R]
    w_tn = mhtn[:, rt0 : rt0 + R, :]
    max_nt = np.max(w_nt, axis=(0, 2))
    max_tn = np.max(w_tn, axis=(0, 1))
    comb = np.maximum(max_nt, max_tn)
    attn_map = {str(node_names[i]): float(comb[i]) for i in range(N)}

    order = np.argsort(-comb)
    topk = min(int(top_k_global), N)
    rows_g: list[dict] = []
    for rank, ni in enumerate(order[:topk], start=1):
        ni = int(ni)
        rows_g.append(
            {
                "rank_attn_reg_pool": rank,
                "node": str(node_names[ni]),
                "max_node_to_token_over_regs_layers": float(max_nt[ni]),
                "max_token_to_node_over_regs_layers": float(max_tn[ni]),
                "max_either_dir": float(comb[ni]),
            }
        )
    df_g = pd.DataFrame(rows_g)

    rows_l: list[dict] = []
    for j, rname in enumerate(reg_cols):
        nt_j = np.max(w_nt[:, :, j], axis=0)
        tn_j = np.max(w_tn[:, j, :], axis=0)
        cj = np.maximum(nt_j, tn_j)
        ordj = np.argsort(-cj)
        tk = min(int(top_k_per_reg), N)
        for rank, ni in enumerate(ordj[:tk], start=1):
            ni = int(ni)
            rows_l.append(
                {
                    "reg_col": str(rname),
                    "rank_within_reg": rank,
                    "node": str(node_names[ni]),
                    "max_node_to_token": float(nt_j[ni]),
                    "max_token_to_node": float(tn_j[ni]),
                    "max_either_dir": float(cj[ni]),
                }
            )
    df_l = pd.DataFrame(rows_l)
    return df_g, df_l, attn_map


def _meta_aux_token_node_rankings_mean_layers(
    attn: np.ndarray,
    *,
    cross_attn: str,
    token_names: list[str],
    node_names: list[str],
    n_cap: int,
    n_reg: int,
    top_k_per_token: int,
    meta_aux_target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Rank graph nodes per **meta-aux / system** token (indices after cap+reg), mean over GPS layers.

    ``cross_attn``:

    - ``node_to_token``: ``attn`` is ``mh`` (L, N, T); top nodes that **attend to** each aux token.
    - ``token_to_node``: ``attn`` is ``mhtn`` (L, T, N); top nodes each aux token **attends to**.

    ``attn`` is head-meaned in ``run_attention_extract`` and sample-averaged in this notebook.
    """
    attn = np.asarray(attn, dtype=np.float64)
    if attn.ndim != 3:
        return pd.DataFrame()
    L = int(attn.shape[0])
    if cross_attn == "node_to_token":
        _L, N, T = attn.shape
    elif cross_attn == "token_to_node":
        _L, T, N = attn.shape
    else:
        raise ValueError(f"cross_attn must be 'node_to_token' or 'token_to_node', got {cross_attn!r}")
    t0 = int(n_cap) + int(n_reg)
    if t0 >= T:
        return pd.DataFrame()
    meta = list(meta_aux_target_cols or [])
    names_t = [str(token_names[i]) if i < len(token_names) else f"tok_{i}" for i in range(T)]
    names_n = [str(node_names[i]) if i < len(node_names) else f"node_{i}" for i in range(N)]
    n_sys = T - t0
    tk = max(1, min(int(top_k_per_token), N))
    w_mean = np.mean(attn, axis=0)
    rows: list[dict] = []
    for j, ti in enumerate(range(t0, T)):
        ti = int(ti)
        meta_label = meta[j] if j < len(meta) else ""
        tok = names_t[ti]
        label = str(meta_label) if meta_label and len(meta) == n_sys else tok
        if cross_attn == "node_to_token":
            vec = w_mean[:, ti]
        else:
            vec = w_mean[ti, :]
        order = np.argsort(-vec)[:tk]
        for rank, ni in enumerate(order, start=1):
            ni = int(ni)
            rows.append(
                {
                    "n_gps_layers": L,
                    "layer_aggregate": "mean",
                    "cross_attn": cross_attn,
                    "token_index": int(ti),
                    "token_name": tok,
                    "meta_aux_head": str(meta_label) if meta_label else "",
                    "token_label": label,
                    "rank_node": int(rank),
                    "node": names_n[ni],
                    "attn_mass": float(vec[ni]),
                }
            )
    return pd.DataFrame(rows)


def _system_token_to_node_rankings_mean_layers(
    mhtn: np.ndarray,
    *,
    token_names: list[str],
    node_names: list[str],
    n_cap: int,
    n_reg: int,
    top_k_per_token: int,
    meta_aux_target_cols: list[str] | None = None,
) -> pd.DataFrame:
    return _meta_aux_token_node_rankings_mean_layers(
        mhtn,
        cross_attn="token_to_node",
        token_names=token_names,
        node_names=node_names,
        n_cap=n_cap,
        n_reg=n_reg,
        top_k_per_token=top_k_per_token,
        meta_aux_target_cols=meta_aux_target_cols,
    )


# =============================================================================
# Knobs — edit here
# =============================================================================
# How many cache rows to average for aux / voltage / attention (max = cache size).
#   None → use every sample in the cache .pt (batch dim), from SAMPLE_IDX_START.
#   int  → use at most that many consecutive rows (e.g. 50 for a quick run).
N_SAMPLES_AVG = None

SAMPLE_IDX_START = 0  # starting cache row index when slicing samples

# Worst-bus bar chart, hop heatmap, downstream-reg CSV: at most this many buses
# (clipped to n_nodes in the graph).
TOP_K_WORST_BUSES = 1000

# Regulator-related cross-attention: how many nodes to list globally and per regulator token.
TOP_K_ATTENTION_NODES = 200
TOP_K_ATTENTION_NODES_PER_REG = 50
# Meta-aux / system tokens (``sys_*`` after cap+reg): top nodes per aux for node→token and token→node;
# mean over all GPS layers (and heads in extract, cache samples in this cell).
TOP_K_ATTENTION_NODES_PER_AUX_TOKEN = 10

# Notebook: how many rows to print / display for the R²-sorted table (lowest R² first).
N_ROWS_PRINT_R2_TABLE = 30

# Max matplotlib figure height (inches) for worst-bus bar charts when TOP_K is large.
TOP_K_PLOT_H_CAP_IN = 42.0

# Shared chunk topology (DA cache ``.pt`` often sits next to ``datasets_gnn2_from pc``; edges may be
# **flat** ``…\gnn_edges_phase_static.csv`` or under ``…\run_*\gnn_edges_phase_static.csv`` — see auto-pick below).
_DATASET_PC = REPO_ROOT / r"datasets_gnn2_from pc"
_CHUNK_RUN = _DATASET_PC / "run_001_scen_0000_0049_seed_20360143"
_CHUNK_EDGES_CSV = _CHUNK_RUN / "gnn_edges_phase_static.csv"
_EDGES_FLAT_CSV = _DATASET_PC / "gnn_edges_phase_static.csv"
_CACHE_STEM = _CHUNK_RUN.name
CACHE_PT = _CHUNK_RUN.parent / f"{_CACHE_STEM}__full.pt"
EDGES_CSV = _CHUNK_EDGES_CSV

# --- paths (defaults: GINE meta-aux run + same chunk topology as training; edit CACHE_PT if needed) ---
RUN_DIR = REPO_ROOT / r"gnn2_architecture_search\da_gps_chunked_l4_mvagg_gine_metaaux_20260513_140037"
# DA multitask cache .pt for one chunk (must match checkpoint n_nodes / edges topology).
# Use the same ``run_*`` chunk as training (here: seed 20360143 from your Colab NODE_PE_CSV).
_HOP_FLAT = _DATASET_PC / "load_hop_distance_to_each_regulator_all_index_nodes.csv"
HOP_CSV = _HOP_FLAT
# ``RUN_DIR``: training output folder that holds ``x_mean.pt``, ``reg_mean.pt``, etc. (not necessarily the
# same folder as a copied checkpoint under ``attention checkpoints\``).
# ``OUT_DIR``: where this notebook writes CSV/PNG/NPZ. Default targets your ``attention checkpoints\…_epoch20``
# tree while ``RUN_DIR`` (above) stays the original training folder for ``*.pt`` norms. If you change ``RUN_DIR``
# to another experiment, set ``OUT_DIR`` in ``__NOTEBOOK_KNOBS__`` as well.
OUT_DIR = (
    REPO_ROOT
    / r"gnn2_architecture_search\attention checkpoints\da_gps_chunked_l4_mvagg_gine_metaaux_20260513_140037_epoch20"
    / "attention_extract"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNSTREAM_RULE = "hop_gt_0"
# Optional: set to a specific ``.pt`` file. If None, uses ``da_gps_multitask_best.pt`` when present,
# otherwise ``training_last.pt`` (from ``--checkpoint_every`` saves during training).
CKPT_PATH = None
# When ``CACHE_PT`` is missing, ``_resolve_da_cache_pt`` globs siblings. For BESS-in-``x`` runs, set this to
# ``("__nobess__",)`` so ``__full__nobess__...`` caches are never auto-picked (avoids silent x_dim mismatch).
CACHE_RESOLVER_REJECT_SUBSTR: tuple[str, ...] = ()
# import matplotlib
# matplotlib.use("Agg")
# =============================================================================
# Optional: before exec(this file), set __NOTEBOOK_KNOBS__ = {"N_SAMPLES_AVG": 50, ...}
# in the parent cell to override any of the keys below (omit keys you keep from file).
#
# Presets: merged before ``__NOTEBOOK_KNOBS__`` (knobs override). Empty dict = no-op.
NOTEBOOK_PRESETS: dict[str, dict[str, object]] = {
    "gine_metaaux_default": {"CACHE_RESOLVER_REJECT_SUBSTR": ()},
    "bess_l3_chunk": {
        # --- replace ``_BESS_RUN`` with your real DA-GPS output dir (must contain x_mean.pt + ckpt) ---
        "RUN_DIR": REPO_ROOT
        / r"gnn2_architecture_search\REPLACE_WITH_YOUR_DA_GPS_BESS_RUN_DIR",
        # BESS cache: usually ``...__full.pt`` or ``...__full__maux<8hex>.pt`` (no ``__nobess`` when BESS kept).
        "CACHE_PT": _CHUNK_RUN.parent / f"{_CACHE_STEM}__full.pt",
        "CACHE_RESOLVER_REJECT_SUBSTR": ("__nobess__",),
        "OUT_DIR": REPO_ROOT
        / r"gnn2_architecture_search\attention checkpoints\da_gps_bess_l3_attention_extract\attention_extract",
        "CKPT_PATH": None,
    },
}

_KNOB_KEYS = (
    "N_SAMPLES_AVG",
    "SAMPLE_IDX_START",
    "TOP_K_WORST_BUSES",
    "TOP_K_ATTENTION_NODES",
    "TOP_K_ATTENTION_NODES_PER_REG",
    "TOP_K_ATTENTION_NODES_PER_AUX_TOKEN",
    "N_ROWS_PRINT_R2_TABLE",
    "TOP_K_PLOT_H_CAP_IN",
    "RUN_DIR",
    "CACHE_PT",
    "CACHE_RESOLVER_REJECT_SUBSTR",
    "EDGES_CSV",
    "HOP_CSV",
    "OUT_DIR",
    "DEVICE",
    "DOWNSTREAM_RULE",
    "CKPT_PATH",
)

_nbk_raw = globals().get("__NOTEBOOK_KNOBS__")
_nbk: dict = dict(_nbk_raw) if isinstance(_nbk_raw, dict) else {}
_preset_name = str(_nbk.pop("NOTEBOOK_PRESET", "") or "").strip()
if not _preset_name:
    _preset_name = str(globals().get("NOTEBOOK_PRESET", "") or "").strip()
if _preset_name:
    if _preset_name not in NOTEBOOK_PRESETS:
        raise KeyError(
            f"Unknown NOTEBOOK_PRESET={_preset_name!r}. "
            f"Valid: {sorted(NOTEBOOK_PRESETS.keys())!r}"
        )
    _pd = NOTEBOOK_PRESETS[_preset_name]
    if _pd:
        for _pk, _pv in _pd.items():
            if _pk in _KNOB_KEYS:
                globals()[_pk] = _pv
        print(f"applied NOTEBOOK_PRESET={_preset_name!r} ({len(_pd)} keys)", flush=True)
    else:
        print(f"NOTEBOOK_PRESET={_preset_name!r} (empty overlay; file defaults unchanged)", flush=True)

if isinstance(_nbk_raw, dict):
    for _k in _KNOB_KEYS:
        if _k in _nbk:
            globals()[_k] = _nbk[_k]
else:
    print(
        "note: __NOTEBOOK_KNOBS__ not set or not a dict — using file defaults for paths.",
        flush=True,
    )

print("RUN_DIR =", Path(RUN_DIR).resolve(), flush=True)
print("OUT_DIR =", Path(OUT_DIR).resolve(), flush=True)
print("DEVICE =", DEVICE)

# --- resolve edges / hop CSV (override with __NOTEBOOK_KNOBS__["EDGES_CSV"] / ["HOP_CSV"] if needed) ---
_edges_set = Path(EDGES_CSV)
if not _edges_set.is_file():
    if _EDGES_FLAT_CSV.is_file():
        EDGES_CSV = _EDGES_FLAT_CSV
        print(f"EDGES_CSV: using flat file (chunk subdir missing): {EDGES_CSV.resolve()}", flush=True)
    elif _CHUNK_EDGES_CSV.is_file():
        EDGES_CSV = _CHUNK_EDGES_CSV
        print(f"EDGES_CSV: using chunk subdir: {EDGES_CSV.resolve()}", flush=True)
    else:
        raise FileNotFoundError(
            f"EDGES_CSV not found at {_edges_set}\n"
            f"  Tried flat: {_EDGES_FLAT_CSV}\n"
            f"  Tried chunk: {_CHUNK_EDGES_CSV}\n"
            f"Set __NOTEBOOK_KNOBS__['EDGES_CSV'] = Path(r\"...\\gnn_edges_phase_static.csv\") to your real file."
        )
else:
    EDGES_CSV = _edges_set.resolve()
    print(f"EDGES_CSV = {EDGES_CSV}", flush=True)

_hop_set = Path(HOP_CSV)
if _hop_set.is_file():
    HOP_CSV = _hop_set.resolve()
elif _HOP_FLAT.is_file():
    HOP_CSV = _HOP_FLAT.resolve()
    print(f"HOP_CSV: explicit path missing; using: {HOP_CSV}", flush=True)
else:
    raise FileNotFoundError(
        f"HOP_CSV not found at {_hop_set}. Tried fallback {_HOP_FLAT}. "
        f"Set __NOTEBOOK_KNOBS__['HOP_CSV'] = Path(r\"...\") to your hop CSV."
    )
print(f"HOP_CSV = {HOP_CSV}", flush=True)


def _resolve_da_gps_ckpt(run_dir: Path, explicit) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CKPT_PATH not found: {p}")
        return p
    rd = Path(run_dir).resolve()
    best = rd / "da_gps_multitask_best.pt"
    last = rd / "training_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(
        f"No checkpoint in {rd}: expected {best.name} or {last.name} "
        "(or pass __NOTEBOOK_KNOBS__['CKPT_PATH'] = Path(...))."
    )


ckpt = _resolve_da_gps_ckpt(RUN_DIR, CKPT_PATH)
print("ckpt =", ckpt)


def _resolve_da_cache_pt(cache_pt: Path | str, run_dir: Path | str | None = None) -> Path:
    """Use ``CACHE_PT`` if it exists; else try sibling names from chunk DA-GPS training.

    Training may write ``<stem>__full__nobess__maux<md5>.pt`` (``--exclude_bess_features``) or
    ``<stem>__full__maux<md5>.pt`` / bare ``<stem>.pt`` when BESS columns stay in ``x``.

    When both exist, **BESS / maux-only** names (``__full__maux*.pt`` without ``__nobess__``) are tried **before**
    ``__nobess__`` variants so a missing bare ``__full.pt`` does not silently pick the wrong feature set.

    If ``CACHE_RESOLVER_REJECT_SUBSTR`` is set (e.g. ``(\"__nobess__\",)`` for preset ``bess_l3_chunk``), sibling
    paths whose **filename** contains any of those substrings are skipped. If that removes all candidates, a
    ``FileNotFoundError`` lists what was found so you can copy the correct ``.pt`` or set ``CACHE_PT`` explicitly.
    """
    p = Path(cache_pt).expanduser().resolve()
    if p.is_file():
        return p
    par = p.parent
    stem = p.stem
    rej = globals().get("CACHE_RESOLVER_REJECT_SUBSTR") or ()
    if isinstance(rej, str):
        rej_t = (rej,) if rej.strip() else ()
    elif isinstance(rej, (list, tuple)):
        rej_t = tuple(str(x) for x in rej if str(x).strip())
    else:
        rej_t = ()

    def _allowed(c: Path) -> bool:
        name = c.name
        return not any(s in name for s in rej_t)

    raw_all: list[Path] = []
    for pat in (
        f"{stem}__maux*.pt",
        f"{stem}__nobess__maux*.pt",
        f"{stem}__nobess.pt",
    ):
        raw_all.extend(sorted(par.glob(pat)))
    hits = [c for c in raw_all if _allowed(c)]
    uniq: list[Path] = []
    seen: set[str] = set()
    for c in hits:
        k = str(c.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    if len(uniq) == 1:
        print(f"CACHE_PT not at {p}; using resolved: {uniq[0]}", flush=True)
        return uniq[0].resolve()
    if len(uniq) > 1:
        msg = "\n  ".join(str(x) for x in uniq[:15])
        raise FileNotFoundError(
            f"CACHE_PT missing: {p}\nMultiple sibling caches; set CACHE_PT explicitly to one of:\n  {msg}"
        )
    if raw_all and rej_t:
        raw_u: list[Path] = []
        seen2: set[str] = set()
        for c in raw_all:
            k2 = str(c.resolve())
            if k2 not in seen2:
                seen2.add(k2)
                raw_u.append(c)
        msg_r = "\n  ".join(str(x) for x in raw_u[:20])
        raise FileNotFoundError(
            f"CACHE_PT missing: {p}\n"
            f"Found {len(raw_u)} sibling tensor cache(s), but all filenames match CACHE_RESOLVER_REJECT_SUBSTR={rej_t!r} "
            f"(skipped). For with-BESS ``RUN_DIR``, copy or build a cache **without** ``__nobess__`` in the name "
            f"(e.g. ``{stem}__full.pt`` or ``{stem}__full__maux....pt``), or set ``CACHE_PT`` / "
            f"``CACHE_RESOLVER_REJECT_SUBSTR`` explicitly.\n"
            f"Skipped candidates:\n  {msg_r}"
        )
    hint = ""
    if run_dir is not None:
        rp = Path(run_dir) / "da_gps_report.json"
        if rp.is_file():
            hint = f"\nSee {rp} for aux_meta_cols / exclude_bess (cache basename includes __nobess / __maux...)."
    raise FileNotFoundError(
        f"CACHE_PT not found: {p}{hint}\n"
        f"Searched under {par} for patterns from stem {stem!r} "
        f"(``__full__maux*.pt`` without ``__nobess__`` first, then ``__nobess__`` variants). "
        f"Copy the chunk ``.pt`` from your Colab ``--cache_dir`` next to the chunk CSVs, or point ``CACHE_PT`` "
        f"at the real file. For BESS-in-``x`` training, avoid ``__nobess__`` caches unless that is what you trained."
    )


CACHE_PT = _resolve_da_cache_pt(CACHE_PT, RUN_DIR)

_z = torch.load(CACHE_PT, map_location="cpu", weights_only=False)
_n_cache = int(_z["x"].shape[0])
if N_SAMPLES_AVG is None:
    _n_avg = _n_cache
else:
    _n_avg = max(1, min(int(N_SAMPLES_AVG), _n_cache))
_sample_indices = [SAMPLE_IDX_START + k for k in range(_n_avg)]
if not _sample_indices or _sample_indices[-1] >= _n_cache:
    raise IndexError(
        f"sample range {SAMPLE_IDX_START}..{SAMPLE_IDX_START + _n_avg - 1} out of cache [0,{_n_cache})"
    )
print(f"cache_pt={CACHE_PT}")
print(f"  total_samples_in_cache (x batch dim) = {_n_cache}")
print(
    f"  n_used_for_average = {_n_avg}  (indices {_sample_indices[0]}..{_sample_indices[-1]} inclusive)"
)
print(
    f"knobs: N_SAMPLES_AVG={N_SAMPLES_AVG!r} → n_used={_n_avg}  |  "
    f"TOP_K_WORST_BUSES={TOP_K_WORST_BUSES}  |  N_ROWS_PRINT_R2_TABLE={N_ROWS_PRINT_R2_TABLE}  |  "
    f"TOP_K_ATTENTION_NODES_PER_AUX_TOKEN={TOP_K_ATTENTION_NODES_PER_AUX_TOKEN}"
)
if _n_avg > 150:
    print(
        f"WARNING: attention section below runs ``run_attention_extract`` {_n_avg} times (full model + "
        f"edges each time). Expect a long run or use e.g. N_SAMPLES_AVG=50 for development; "
        f"KeyboardInterrupt is normal if you hit Stop.",
        flush=True,
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
hop_df = load_hop_frame(HOP_CSV)

# ----- same cache rows: per-regulator MAE/MSE/RMS in tap pu + per-cap BCE & accuracy -----
reg_df, cap_df, aux_meta, meta_aux_df = eval_aux_per_device_on_cache_indices(
    ckpt,
    RUN_DIR,
    CACHE_PT,
    EDGES_CSV,
    _sample_indices,
    device=DEVICE,
    dropout=0.0,
)
reg_df.to_csv(OUT_DIR / f"aux_reg_per_device_avg{_n_avg}.csv", index=False)
cap_df.to_csv(OUT_DIR / f"aux_cap_per_device_avg{_n_avg}.csv", index=False)
(OUT_DIR / f"aux_metrics_meta_avg{_n_avg}.json").write_text(
    json.dumps(aux_meta, indent=2), encoding="utf-8"
)
if len(meta_aux_df):
    _mcsv = OUT_DIR / f"aux_meta_per_col_avg{_n_avg}.csv"
    meta_aux_df.to_csv(_mcsv, index=False)
    print(f"wrote {_mcsv}")
print(f"wrote {OUT_DIR / f'aux_reg_per_device_avg{_n_avg}.csv'}")
print(f"wrote {OUT_DIR / f'aux_cap_per_device_avg{_n_avg}.csv'}")
print(f"wrote {OUT_DIR / f'aux_metrics_meta_avg{_n_avg}.json'}")
print(f"aggregate reg_mse_tap_pu_all={aux_meta['reg_mse_tap_pu_all']:.8f}  cap_bce_all={aux_meta['cap_bce_all']:.6f}")
if "pv_mse_raw_all" in aux_meta:
    print(
        f"meta_aux: pv_mse_nrm_all={aux_meta.get('pv_mse_nrm_all', float('nan')):.6f}  "
        f"pv_mse_raw_all={aux_meta['pv_mse_raw_all']:.6f}"
    )
print("\n--- per-regulator (tap pu) + hit-rate style accuracy ---")
print(reg_df.to_string(index=False))
print("\n--- per-cap (BCE + threshold accuracy) ---")
print(cap_df.to_string(index=False))
if len(meta_aux_df):
    print("\n--- per-meta-aux column (raw + normalized errors; R² across evaluated cache rows) ---")
    print(meta_aux_df.to_string(index=False))


def _short_reg_col(name: str) -> str:
    return str(name).replace("reg_", "").replace("_tap_pu", "")


# ----- regulator metrics: horizontal bars (MAE, RMSE, MSE in tap pu) -----
if len(reg_df):
    fig_reg, axes_reg = plt.subplots(1, 3, figsize=(15, max(4.0, 0.22 * len(reg_df))), constrained_layout=True)
    fig_reg.suptitle(
        f"Regulator tap vs cache (n={_n_avg} rows)  |  agg MSE(pu)={aux_meta['reg_mse_tap_pu_all']:.6e}",
        fontsize=11,
    )

    r_mae = reg_df.sort_values("mae_tap_pu", ascending=True)
    axes_reg[0].barh([_short_reg_col(c) for c in r_mae["reg_col"]], r_mae["mae_tap_pu"], color="steelblue")
    axes_reg[0].set_xlabel("MAE (tap pu)")
    axes_reg[0].set_title("Mean absolute error")
    axes_reg[0].grid(True, axis="x", alpha=0.3)

    r_rmse = reg_df.sort_values("rmse_tap_pu", ascending=True)
    axes_reg[1].barh([_short_reg_col(c) for c in r_rmse["reg_col"]], r_rmse["rmse_tap_pu"], color="darkorange")
    axes_reg[1].set_xlabel("RMSE (tap pu)")
    axes_reg[1].set_title("Root mean square error")
    axes_reg[1].grid(True, axis="x", alpha=0.3)

    r_mse = reg_df.sort_values("mse_tap_pu", ascending=True)
    axes_reg[2].barh([_short_reg_col(c) for c in r_mse["reg_col"]], r_mse["mse_tap_pu"], color="seagreen")
    axes_reg[2].set_xlabel("MSE (tap pu²)")
    axes_reg[2].set_title("Mean square error")
    axes_reg[2].grid(True, axis="x", alpha=0.3)

    _reg_png = OUT_DIR / f"aux_reg_metrics_bar_avg{_n_avg}.png"
    fig_reg.savefig(_reg_png, dpi=150)
    plt.show()
    print(f"wrote {_reg_png}")

# ----- per-bus voltage errors (mean over same samples); rank all nodes -----
node_err_df, node_err_meta = eval_voltage_per_node_errors_on_cache_indices(
    ckpt,
    RUN_DIR,
    CACHE_PT,
    EDGES_CSV,
    _sample_indices,
    device=DEVICE,
    dropout=0.0,
)
_node_csv = OUT_DIR / f"voltage_node_errors_ranked_avg{_n_avg}.csv"
node_err_df.to_csv(_node_csv, index=False)
(OUT_DIR / f"voltage_node_errors_meta_avg{_n_avg}.json").write_text(
    json.dumps(node_err_meta, indent=2), encoding="utf-8"
)
print(f"wrote {_node_csv}")
print(
    f"voltage pooled over all nodes×cache rows (n={node_err_meta.get('n_points_vmag_finite_overlap', 0)}): "
    f"|V| MAE={node_err_meta.get('mae_global_vmag_pu', float('nan')):.6f} pu  "
    f"RMSE={node_err_meta.get('rmse_global_vmag_pu', float('nan')):.6f} pu  "
    f"R²={node_err_meta.get('r2_global_vmag_pu', float('nan')):.6f}"
)
print(
    f"angle pooled (circular MAE, naive linear R² vs true angle in deg): "
    f"MAE={node_err_meta.get('mae_global_angle_deg', float('nan')):.4f} deg  "
    f"R²={node_err_meta.get('r2_global_vang_deg_naive', float('nan')):.6f}"
)
print(
    f"worst bus (mean |V| MAE over n={_n_avg}): {node_err_meta['worst_node']}  "
    f"mean_mae_vmag_pu={node_err_meta['worst_mean_mae_vmag_pu']:.6f}"
)
_r2s = node_err_df["r2_vmag"].to_numpy(dtype=np.float64)
_r2ok = _r2s[np.isfinite(_r2s)]
print(
    f"|V| R² over nodes (finite {len(_r2ok)}/{node_err_meta['n_nodes']}): "
    f"mean={node_err_meta.get('r2_vmag_mean', float('nan')):.4f}  "
    f"median={node_err_meta.get('r2_vmag_median', float('nan')):.4f}  "
    f"min={node_err_meta.get('r2_vmag_min', float('nan')):.4f}  "
    f"max={node_err_meta.get('r2_vmag_max', float('nan')):.4f}"
)
print(
    f"worst |V| R² bus (finite): {node_err_meta.get('worst_r2_node', '')!r}  "
    f"r2_vmag={node_err_meta.get('worst_r2_vmag', float('nan')):.4f}"
)
_r2_sorted = (
    node_err_df.sort_values("r2_vmag", ascending=True, na_position="last").reset_index(drop=True)
)
_r2_sorted.insert(0, "rank_r2_table", np.arange(1, len(_r2_sorted) + 1, dtype=np.int32))
print("--- lowest R² buses (first {0} rows; rank_r2_table 1 = lowest R²) ---".format(int(N_ROWS_PRINT_R2_TABLE)))
print(_r2_sorted.head(int(N_ROWS_PRINT_R2_TABLE)).to_string(index=False))

fig_nv, axes_nv = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes_nv[0].hist(node_err_df["mean_mae_vmag_pu"], bins=50, color="steelblue", alpha=0.85, edgecolor="white")
axes_nv[0].set_xlabel("mean |V| MAE (pu) per bus")
axes_nv[0].set_ylabel("count")
axes_nv[0].set_title(f"Distribution over {node_err_meta['n_nodes']} buses")
axes_nv[1].hist(node_err_df["mean_mae_angle_deg"], bins=50, color="darkorange", alpha=0.85, edgecolor="white")
axes_nv[1].set_xlabel("mean |Δangle| (deg) per bus")
axes_nv[1].set_ylabel("count")
axes_nv[1].set_title("Angle error distribution")
_fig_hist = OUT_DIR / f"voltage_node_error_histograms_avg{_n_avg}.png"
fig_nv.savefig(_fig_hist, dpi=150)
plt.show()
print(f"wrote {_fig_hist}")

fig_r2, ax_r2 = plt.subplots(figsize=(8, 4), constrained_layout=True)
if len(_r2ok):
    ax_r2.hist(_r2ok, bins=50, color="seagreen", alpha=0.85, edgecolor="white")
    ax_r2.axvline(float(np.mean(_r2ok)), color="k", ls="--", lw=1.2, label=f"mean={np.mean(_r2ok):.3f}")
    ax_r2.legend(loc="upper left", fontsize=8)
else:
    ax_r2.text(0.5, 0.5, "no finite r2_vmag", ha="center", va="center", transform=ax_r2.transAxes)
ax_r2.set_xlabel(r"per-node $R^2$ of $|V|$ over cache rows")
ax_r2.set_ylabel("count (buses)")
ax_r2.set_title(
    f"Distribution of |V| magnitude R² (n_nodes finite={node_err_meta.get('n_nodes_r2_vmag_finite', 0)}, "
    f"n_samples={_n_avg})"
)
_fig_r2 = OUT_DIR / f"voltage_node_r2_vmag_distribution_avg{_n_avg}.png"
fig_r2.savefig(_fig_r2, dpi=150)
plt.show()
print(f"wrote {_fig_r2}")

# --- std(|V|_true) vs R² / MAE: separate low-variance (unstable R²) from hard buses ---
_sy = node_err_df["std_vmag_true_pu"].to_numpy(dtype=np.float64)
_r2col = node_err_df["r2_vmag"].to_numpy(dtype=np.float64)
_mae_col = node_err_df["mean_mae_vmag_pu"].to_numpy(dtype=np.float64)
_xlog = np.maximum(_sy, 1e-12)
mask_r2 = np.isfinite(_r2col)
fig_diag, axes_diag = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axd0, axd1 = axes_diag
axd0.scatter(_xlog[mask_r2], _r2col[mask_r2], s=10, alpha=0.35, c="tab:blue", edgecolors="none")
axd0.set_xscale("log")
axd0.set_xlabel(r"std($|V|_{\mathrm{true}}$) (pu) over cache rows [log scale]")
axd0.set_ylabel(r"$R^2$ of $|V|$ (finite only)")
axd0.set_title("small std + low R² → touchy metric; large std + low R² → hard / misfit")
axd0.axhline(0.0, color="k", lw=0.5, alpha=0.4)
axd0.grid(True, which="both", alpha=0.25)
axd1.scatter(_xlog, _mae_col, s=10, alpha=0.35, c="tab:orange", edgecolors="none")
axd1.set_xscale("log")
axd1.set_xlabel(r"std($|V|_{\mathrm{true}}$) (pu) [log scale]")
axd1.set_ylabel("mean |V| MAE (pu)")
axd1.set_title("MAE vs spread: small MAE at tiny std can still give odd R²")
axd1.grid(True, which="both", alpha=0.25)
_fig_diag = OUT_DIR / f"voltage_node_std_vmag_vs_r2_mae_avg{_n_avg}.png"
fig_diag.savefig(_fig_diag, dpi=150)
plt.show()
print(f"wrote {_fig_diag}")
_sy_fin = _sy[mask_r2]
_r2_fin = _r2col[mask_r2]
_low_thr = 1e-4
_touchy = int(np.sum((_sy_fin < _low_thr) & (_r2_fin < 0.5)))
_hardish = int(np.sum((_sy_fin >= _low_thr) & (_r2_fin < 0.5)))
print(
    f"std-vs-R² hint (finite R² only): n(sy<{_low_thr:g} pu & R²<0.5)={_touchy} "
    f"(metric touchy); n(sy>={_low_thr:g} & R²<0.5)={_hardish} (likely hard / misfit)"
)

_top_k = min(int(TOP_K_WORST_BUSES), len(node_err_df))
print(
    f"worst-bus analysis: TOP_K_WORST_BUSES={TOP_K_WORST_BUSES}, rows_used={_top_k} (n_nodes={len(node_err_df)})"
)
worst = node_err_df.head(_top_k).iloc[::-1]
_h_bar = min(TOP_K_PLOT_H_CAP_IN, max(5.0, 0.06 * _top_k))
fig_top, ax_top = plt.subplots(figsize=(10, _h_bar), constrained_layout=True)
ax_top.barh(worst["node"].astype(str), worst["mean_mae_vmag_pu"], color="crimson", alpha=0.85)
ax_top.set_xlabel("mean |V| MAE (pu)")
ax_top.set_title(f"Top {_top_k} worst buses (mean over {_n_avg} samples)")
ax_top.grid(True, axis="x", alpha=0.3)
_fig_top = OUT_DIR / f"voltage_node_worst_{_top_k}_avg{_n_avg}.png"
fig_top.savefig(_fig_top, dpi=150)
plt.show()
print(f"wrote {_fig_top}")

# ----- worst buses: regulators they are strictly downstream of (hop CSV) + heatmap -----
worst_raw = node_err_df.head(_top_k).copy()
worst_ann, hop_sub_worst = worst_nodes_downstream_regulator_table(
    worst_raw,
    hop_df,
    node_err_meta["node_names"],
    node_err_meta["reg_target_cols"],
    reg_col_to_hop_col=REG_COL_TO_HOP_COL,
    downstream_rule=DOWNSTREAM_RULE,
)
_csv_w = OUT_DIR / f"voltage_worst_{_top_k}_with_downstream_regs_avg{_n_avg}.csv"
worst_ann.to_csv(_csv_w, index=False)
print(f"wrote {_csv_w}")
worst_ann2 = worst_ann.copy()
worst_ann2["max_reg_cross_attn_either"] = float("nan")

M = np.ma.masked_invalid(hop_sub_worst.to_numpy(dtype=float))
_h_hm = min(48.0, max(5.5, 0.038 * float(M.shape[0])))
fig_hm, ax_hm = plt.subplots(
    figsize=(max(10.0, 0.42 * M.shape[1]), _h_hm),
    constrained_layout=True,
)
im = ax_hm.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0)
ax_hm.set_xticks(np.arange(M.shape[1]))
ax_hm.set_xticklabels(list(hop_sub_worst.columns), rotation=55, ha="right", fontsize=8)
ax_hm.set_yticks(np.arange(M.shape[0]))
_labels_y = [f"#{int(r['rank_vmag'])}  {r['node']}" for _, r in worst_raw.iterrows()]
_ylab_fs = 6 if _top_k > 400 else 7
ax_hm.set_yticklabels(_labels_y, fontsize=_ylab_fs)
ax_hm.set_xlabel("regulator (hop CSV)")
ax_hm.set_ylabel("worst buses by mean |V| MAE")
fig_hm.colorbar(im, ax=ax_hm, label="hop count (0 = not downstream for hop>0 rule)")
fig_hm.suptitle(
    f"Hop distance to each regulator — top {_top_k} worst voltage buses\n"
    f"Same downstream rule as attention ratios: {DOWNSTREAM_RULE!r}",
    fontsize=10,
)
_png_hm = OUT_DIR / f"voltage_worst_{_top_k}_hop_heatmap_avg{_n_avg}.png"
fig_hm.savefig(_png_hm, dpi=150)
plt.show()
print(f"wrote {_png_hm}")

_ypos = np.arange(len(worst_ann))
fig_nd, ax_nd = plt.subplots(figsize=(10, _h_bar), constrained_layout=True)
ax_nd.barh(_ypos, worst_ann["n_downstream_regs"].to_numpy(), color="teal", alpha=0.85)
ax_nd.set_yticks(_ypos)
ax_nd.set_yticklabels(_labels_y, fontsize=_ylab_fs)
ax_nd.invert_yaxis()
ax_nd.set_xlabel("number of regulators with hop>0 (strictly downstream)")
ax_nd.set_title(f"Downstream regulator count (same {DOWNSTREAM_RULE!r} as hop ratios)")
_png_nd = OUT_DIR / f"voltage_worst_{_top_k}_n_downstream_bar_avg{_n_avg}.png"
fig_nd.savefig(_png_nd, dpi=150)
plt.show()
print(f"wrote {_png_nd}")

# ----- mean attention over same indices -----
mh_acc = None
mhtn_acc = None
res = None
for si in _sample_indices:
    res = run_attention_extract(
        ckpt,
        RUN_DIR,
        CACHE_PT,
        EDGES_CSV,
        sample_idx=si,
        sample_id=None,
        out_dir=OUT_DIR,
        device=DEVICE,
        dropout=0.0,
        head_mean=True,
        save_outputs=False,
    )
    mh = np.asarray(res["mean_heads"], dtype=np.float64)
    mhtn = np.asarray(res["mean_heads_tn"], dtype=np.float64)
    while mh.ndim == 4 and mh.shape[1] == 1:
        mh = mh[:, 0, :, :]
    while mhtn.ndim == 4 and mhtn.shape[1] == 1:
        mhtn = mhtn[:, 0, :, :]
    if mh_acc is None:
        mh_acc = mh
        mhtn_acc = mhtn
    else:
        mh_acc += mh
        mhtn_acc += mhtn

mh = (mh_acc / float(_n_avg)).astype(np.float32)
mhtn = (mhtn_acc / float(_n_avg)).astype(np.float32)
L, N, T = mh.shape
Lt, Tt, Nt = mhtn.shape

np.savez_compressed(
    OUT_DIR / f"attention_mean_heads_avg{_n_avg}.npz",
    probs_nt=mh,
    probs_tn=mhtn,
    n_samples_averaged=np.int64(_n_avg),
    sample_indices=np.asarray(_sample_indices, dtype=np.int64),
)
_manifest_avg = {
    **res["manifest"],
    "attention_averaging": {
        "n_samples_averaged": _n_avg,
        "sample_indices": _sample_indices,
        "note": "mean_heads are arithmetic mean of head-mean attention over listed cache indices",
    },
}
(OUT_DIR / f"manifest_attention_avg{_n_avg}.json").write_text(
    json.dumps(_manifest_avg, indent=2), encoding="utf-8"
)
print(f"wrote {OUT_DIR / f'attention_mean_heads_avg{_n_avg}.npz'}")
print(f"wrote {OUT_DIR / f'manifest_attention_avg{_n_avg}.json'}")

ratios = attention_downstream_ratio_table(
    mh,
    reg_target_cols=list(res["manifest"]["reg_target_cols"]),
    n_cap=int(res["n_cap"]),
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
)
ratios_tn = attention_downstream_ratio_table_tn(
    mhtn,
    reg_target_cols=list(res["manifest"]["reg_target_cols"]),
    n_cap=int(res["n_cap"]),
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
)

out_csv = OUT_DIR / f"attention_hop_ratios_avg{_n_avg}.csv"
out_csv_tn = OUT_DIR / f"attention_hop_ratios_token_to_node_avg{_n_avg}.csv"
ratios.to_csv(out_csv, index=False)
ratios_tn.to_csv(out_csv_tn, index=False)

reg_cols = list(res["manifest"]["reg_target_cols"])
n_cap = int(res["n_cap"])

node_names_attn = list(res["manifest"]["node_names"])
_df_g, _df_per_reg, _attn_map = _regulator_cross_attn_node_tables(
    np.asarray(mh, dtype=np.float64),
    np.asarray(mhtn, dtype=np.float64),
    n_cap=n_cap,
    reg_cols=reg_cols,
    node_names=node_names_attn,
    top_k_global=int(TOP_K_ATTENTION_NODES),
    top_k_per_reg=int(TOP_K_ATTENTION_NODES_PER_REG),
)
_csv_g = OUT_DIR / f"reg_cross_attn_node_rank_global_avg{_n_avg}.csv"
_csv_pr = OUT_DIR / f"reg_cross_attn_top_nodes_per_regulator_avg{_n_avg}.csv"
_df_g.to_csv(_csv_g, index=False)
_df_per_reg.to_csv(_csv_pr, index=False)
print(f"wrote {_csv_g}")
print(f"wrote {_csv_pr}")
worst_ann2 = worst_ann.copy()
worst_ann2["max_reg_cross_attn_either"] = worst_ann2["node"].astype(str).map(_attn_map)
_csv_w2 = OUT_DIR / f"voltage_worst_{_top_k}_with_downstream_regs_and_cross_attn_avg{_n_avg}.csv"
worst_ann2.to_csv(_csv_w2, index=False)
print(f"wrote {_csv_w2}")
print(
    "\n--- Regulator cross-attn (max over layers & reg tokens): "
    "node→token vs token→node, pooled by max either ---"
)
print(_df_g.head(15).to_string(index=False))

try:
    _bund_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    _meta_aux_cols = list(_bund_meta.get("meta_aux_target_cols") or _bund_meta.get("pv_target_cols") or [])
except Exception:
    _meta_aux_cols = []

def _print_meta_aux_attn_block(df: pd.DataFrame, *, title: str, csv_path: Path) -> None:
    if not len(df):
        return
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    print(title)
    print(
        "(``attn_mass`` = mean over **all GPS layers** × **heads** (extract) × **cache rows** in this run; "
        "``token_index`` = cap + reg + sys slot; ``sys_j`` ↔ ``meta_aux_target_cols[j]`` when lengths match.)"
    )
    for _tok, _sub in df.groupby("token_label", sort=False):
        print(f"\n  meta_aux={_tok!r}")
        print(_sub.drop(columns=["token_label"]).to_string(index=False))


_df_aux_nt = _meta_aux_token_node_rankings_mean_layers(
    np.asarray(mh, dtype=np.float64),
    cross_attn="node_to_token",
    token_names=list(res["manifest"]["token_names"]),
    node_names=node_names_attn,
    n_cap=n_cap,
    n_reg=len(reg_cols),
    top_k_per_token=int(TOP_K_ATTENTION_NODES_PER_AUX_TOKEN),
    meta_aux_target_cols=_meta_aux_cols,
)
_df_aux_tn = _system_token_to_node_rankings_mean_layers(
    np.asarray(mhtn, dtype=np.float64),
    token_names=list(res["manifest"]["token_names"]),
    node_names=node_names_attn,
    n_cap=n_cap,
    n_reg=len(reg_cols),
    top_k_per_token=int(TOP_K_ATTENTION_NODES_PER_AUX_TOKEN),
    meta_aux_target_cols=_meta_aux_cols,
)
if len(_df_aux_nt) or len(_df_aux_tn):
    _print_meta_aux_attn_block(
        _df_aux_nt,
        title=(
            f"\n--- Node→token (2nd cross-attn): top {TOP_K_ATTENTION_NODES_PER_AUX_TOKEN} **nodes** "
            f"attending **to** each meta-aux token (mean over layers) ---"
        ),
        csv_path=OUT_DIR / f"aux_sys_node_to_token_mean_layers_avg{_n_avg}.csv",
    )
    _print_meta_aux_attn_block(
        _df_aux_tn,
        title=(
            f"\n--- Token→node (1st cross-attn): top {TOP_K_ATTENTION_NODES_PER_AUX_TOKEN} **nodes** "
            f"each meta-aux token attends **to** (mean over layers) ---"
        ),
        csv_path=OUT_DIR / f"aux_sys_token_to_node_mean_layers_avg{_n_avg}.csv",
    )
else:
    print(
        "\n(no meta-aux / system tokens: token count equals cap+reg only — skip aux cross-attn rankings)",
        flush=True,
    )


def _print_ratio_block(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    if len(df) == 0:
        print("  No ratio rows.")
        return
    last_layer = int(df["layer"].max())
    r_last = df[df["layer"] == last_layer]
    print(f"  Final layer ({last_layer}): median={r_last['ratio'].median():.4f}  mean={r_last['ratio'].mean():.4f}")
    print(f"  share ratio>1: {(r_last['ratio'] > 1).mean():.1%}  ratio>1.5: {(r_last['ratio'] > 1.5).mean():.1%}")
    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer]
        gt1 = (sub["ratio"] > 1).sum()
        print(f"  layer {int(layer)}: median={sub['ratio'].median():.4f}  count(ratio>1)={gt1}/{len(sub)}")


print("\n========== DA-GPS attention + hop ratios ==========")
print(
    f"averaged over n={_n_avg} of {_n_cache} cache rows (indices {_sample_indices[0]}..{_sample_indices[-1]})"
)
print(f"out_dir={OUT_DIR}")
print(f"wrote {out_csv}")
print(f"wrote {out_csv_tn}")
print(f"node→token tensors: layers={L}  nodes={N}  tokens={T}")
print(f"token→node tensors: layers={Lt}  tokens={Tt}  nodes={Nt}  regulators={len(reg_cols)}")
print(f"downstream_rule={DOWNSTREAM_RULE!r}")

_print_ratio_block(ratios, "Hop ratios: node → token (each node attends regulators)")
_print_ratio_block(ratios_tn, "Hop ratios: token → node (each regulator token attends nodes)")

_ratio_title = "downstream vs non-downstream (hop≤0)"
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

ax00 = axes[0, 0]
if len(ratios):
    for reg_col in reg_cols:
        sub = ratios[ratios["reg_col"] == reg_col]
        if len(sub) == 0:
            continue
        lab = reg_col.replace("reg_", "").replace("_tap_pu", "")
        ax00.plot(sub["layer"], sub["ratio"], marker="o", ms=4, label=lab)
    ax00.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax00.set_xlabel("GPS block (layer)")
ax00.set_ylabel("ratio")
ax00.set_title(f"Node → token: {_ratio_title} (mean over {_n_avg} samples)")
ax00.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax00.grid(True, alpha=0.3)

ax01 = axes[0, 1]
if len(ratios_tn):
    for reg_col in reg_cols:
        sub = ratios_tn[ratios_tn["reg_col"] == reg_col]
        if len(sub) == 0:
            continue
        lab = reg_col.replace("reg_", "").replace("_tap_pu", "")
        ax01.plot(sub["layer"], sub["ratio"], marker="o", ms=4, label=lab)
    ax01.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax01.set_xlabel("GPS block (layer)")
ax01.set_ylabel("ratio")
ax01.set_title(f"Token → node: {_ratio_title} (mean over {_n_avg} samples)")
ax01.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax01.grid(True, alpha=0.3)

ax10 = axes[1, 0]
_last_nt = "—"
if len(ratios):
    _last_nt = str(int(ratios["layer"].max()))
    r_last = ratios[ratios["layer"] == int(_last_nt)].sort_values("ratio", ascending=True)
    if len(r_last):
        labels = [c.replace("reg_", "").replace("_tap_pu", "") for c in r_last["reg_col"]]
        ax10.barh(labels, r_last["ratio"], color="steelblue")
        ax10.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax10.set_xlabel("ratio")
ax10.set_title(f"Node→token final layer ({_last_nt}), mean n={_n_avg}")

ax11 = axes[1, 1]
_last_tn = "—"
if len(ratios_tn):
    _last_tn = str(int(ratios_tn["layer"].max()))
    r2 = ratios_tn[ratios_tn["layer"] == int(_last_tn)].sort_values("ratio", ascending=True)
    if len(r2):
        labels2 = [c.replace("reg_", "").replace("_tap_pu", "") for c in r2["reg_col"]]
        ax11.barh(labels2, r2["ratio"], color="darkorange")
        ax11.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax11.set_xlabel("ratio")
ax11.set_title(f"Token→node final layer ({_last_tn}), mean n={_n_avg}")

plt.show()

_hist_suffix = f"mean over {_n_avg} cache samples"
plot_all_regulator_layer_hop_histograms(
    mh,
    mhtn,
    reg_target_cols=reg_cols,
    reg_col_to_hop_col=REG_COL_TO_HOP_COL,
    n_cap=n_cap,
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
    downstream_label=f"downstream ({DOWNSTREAM_RULE})",
    non_downstream_label="non-downstream (hop≤0): ref bus / outside subtree / other laterals / per CSV",
    suptitle_suffix=_hist_suffix,
    out_dir=OUT_DIR,
    show=True,
)
print(f"wrote {OUT_DIR / 'attention_hop_hist_all_layers_regs_nt.png'}")
print(f"wrote {OUT_DIR / 'attention_hop_hist_all_layers_regs_tn.png'}")

try:
    from IPython.display import display

    print("\n--- lowest R² buses (sorted, first {0}) ---".format(int(N_ROWS_PRINT_R2_TABLE)))
    display(_r2_sorted.head(int(N_ROWS_PRINT_R2_TABLE)))
    print("\n--- worst buses + downstream regs + cross-attn column (head) ---")
    display(worst_ann2.head(15))
    print("\n--- meta aux per column (if trained) ---")
    if len(meta_aux_df):
        display(meta_aux_df)
    print("\n--- top buses by regulator cross-attn pool (head) ---")
    display(_df_g.head(20))
    print("\n--- system/aux tokens: token→node top nodes, **mean over layers+heads**, mean over samples (head) ---")
    if len(_df_aux_tn):
        display(_df_aux_tn)
    print("\n--- worst buses voltage only (head) ---")
    display(node_err_df.head(20))
    print("\n--- node→token ratios (head) ---")
    display(ratios.head(24))
    print("\n--- token→node ratios (head) ---")
    display(ratios_tn.head(24))
except Exception:
    print(_r2_sorted.head(int(N_ROWS_PRINT_R2_TABLE)).to_string())
    print(worst_ann2.head(15).to_string())
    if len(meta_aux_df):
        print(meta_aux_df.to_string(index=False))
    print(_df_g.head(20).to_string(index=False))
    if len(_df_aux_tn):
        print("\n--- system/aux tokens: token→node, mean layers+heads, mean samples (full table) ---")
        print(_df_aux_tn.to_string(index=False))
    print(node_err_df.head(20).to_string())
    print(ratios.head(24).to_string())
    print(ratios_tn.head(24).to_string())
