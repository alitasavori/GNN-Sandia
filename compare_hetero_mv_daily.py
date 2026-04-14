"""
Daily OpenDSS vs hetero-MV checkpoint comparison (8500 feeder).

Outputs:
  - Per-node 24h plots for requested nodes.
  - Per-node MAE CSV.
  - `daily_gnn_variation_load_nodes_{cfg}.csv` — per hetero **load** bus: std / range of GNN |V| over the day; ranks by **range** (max−min). Warns if predictions are flat (<1e-6 pu).
  - All-node error histogram.
  - Printed global MAE/RMSE vs OpenDSS.
  - **Timing (`run_compare` / SAGE & GINE wrappers):** wall-clock totals and mean ms per **converged**
    step for OpenDSS load apply, solve, voltage collection, hetero feature build, and GNN forward
    (CUDA sync before stopping the timer). Progress logs every ~`npts/12` steps.
    OpenDSS **solve** times use **snapshot** mode per step (after compiling the daily circuit);
    ``GNN_TORCH_COMPILE=0`` disables optional ``torch.compile`` on the GNN (default off on Windows).

Interpretation:
  - Each PyG node type (upstream, downstream, capacitor, load) has its own readout head.
    If training used --target-node-types load, only the load head is trained; other heads
    receive no gradient and stay near ~0, so predictions at upstream/capacitor/downstream
    buses look "flat" near the bottom of the plot.
  - Use --nodes with bus names that are in gnn_node_index_master and appear in the hetero
    node CSVs; the script maps each name to its storage type for the GNN.
  - **hetero_gine_*** checkpoints: after each OpenDSS solve, regulator **tap pu** on `reg` edges
    is filled from the same `RegControls` + `Transformers.Tap()` path as training
    (`_reg_attr_dict_per_keys` + `REGULATOR_TO_TAP_COL` from `build_hetero_mv_edge_dataset`).
  - **Load P/Q:** hetero **load** rows use **MV** node names (`l….phase`); OpenDSS `Load` elements
    sit on **sx…** (secondary) buses. Training uses `aggregate_mv_node_dataset_8500` rules from
    `8500-node/mv_x_sx_node_mapping_8500.csv` (sum of two sx/lv leaf nodes per MV). This script
    applies the same mapping so `x["load"]` receives time-varying P/Q; without that file, P/Q stay 0.
  - **Two checkpoints:** pass `--vs-checkpoint` to run one OpenDSS trajectory, evaluate both models.
    Default `--juxtapose-mode disagree` ranks by mean |V_a−V_b|, saves CSV, plots top‑K.
    Use `--juxtapose-mode both-fail-dss` to rank nodes where **both** models track OpenDSS poorly:
    sort by **min**(MAE_a, MAE_b) vs DSS (high `min` ⇒ neither model is accurate).
    Use `--juxtapose-mode lowest-min-v-dss` to rank by **lowest daily minimum** OpenDSS |V| (stressed nodes).
    Optional `--also-nodes`.
    Use `--disagree-scope load` (default) to restrict to hetero **load** buses (fairer when one checkpoint
    supervised load only).
  - **Regulator tap diagnostics (GINE / juxtapose with a GINE):** prints catalog vs REGULATOR_TO_TAP_COL,
    training CSV tap ranges from `edges/hetero_mv_regulator_edge_features.csv`, OpenDSS `tap_raw` key
    coverage, first-step resolved tap per regulator edge, and after the day writes
    `daily_reg_tap_timeseries_*.csv` (per-edge min/max/range/std). Use `--no-reg-tap-diag` to disable.
"""

from __future__ import annotations

import argparse
import time
from typing import Literal
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss

import run_injection_dataset as inj
import search_hetero_mv_gnn_architectures as hm
import run_daily_aggregate_dataset_8500 as rd8500
from build_hetero_mv_edge_dataset import REGULATOR_TO_TAP_COL
from compare_gnn_inference_utils import maybe_torch_compile
from compare_mv_daily_timing import print_mv_daily_timing_summary, resolve_inference_device, sync_inference_device
from compare_opendss_snapshot_helpers import (
    force_snapshot_mode_for_compare_timing,
    reassert_snapshot_before_each_solve,
)


def _load_model(cfg_name: str, state_dict: dict, edge_index_dict: dict, device: torch.device) -> tuple[torch.nn.Module, bool]:
    use_gine = "gine" in cfg_name
    if use_gine:
        core = hm.HeteroTypedGINE(hm.NODE_TYPES, hm.IN_DIMS, 80, 3, 0.1, edge_index_dict).to(device)
        clean_sd = {k.replace("core.", "", 1): v for k, v in state_dict.items()}
        core.load_state_dict(clean_sd, strict=False)
        model = core
    elif "gat" in cfg_name:
        model = hm.HeteroTypedGAT(hm.NODE_TYPES, hm.IN_DIMS, 128, 4, 2, 0.1, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    elif "4x64" in cfg_name:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 64, 4, 0.15, True, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    elif "3x112" in cfg_name:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 112, 3, 0.05, False, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 96, 2, 0.0, False, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, use_gine


def _load_mv_sx_mapping(path: Path) -> list[dict[str, str]]:
    """Same mv→(sx or lv) leaf pairing as `aggregate_mv_node_dataset_8500._load_mapping`."""
    rules: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mv = (row.get("mv_node") or "").strip()
            lv1 = (row.get("lv_x_node_1") or "").strip()
            lv2 = (row.get("lv_x_node_2") or "").strip()
            sx1 = (row.get("sx_node_1") or "").strip()
            sx2 = (row.get("sx_node_2") or "").strip()
            if not mv or not lv1 or not lv2:
                continue
            if sx1 and sx2:
                la, lb = sx1, sx2
            else:
                la, lb = lv1, lv2
            rules.append(
                {
                    "mv_key": mv.lower(),
                    "load_a": la.lower(),
                    "load_b": lb.lower(),
                }
            )
    return rules


def _expected_dss_tap_keys(reg_control_names: list[str]) -> list[str]:
    """Keys produced by `run_daily_aggregate_dataset_8500._read_reg_control_state`."""
    return [f"reg_{nm}_tap_pu" for nm in reg_control_names]


def _print_reg_edge_catalog_diag(catalog: pd.DataFrame) -> None:
    """hetero_mv_edge_catalog regulator rows vs REGULATOR_TO_TAP_COL and tap_column."""
    reg = catalog[catalog["edge_type"].astype(str).str.strip().str.lower() == "regulator"].copy()
    if reg.empty:
        print("[reg_tap_diag] no regulator rows in hetero_mv_edge_catalog.csv")
        return
    n_bad_col = 0
    n_missing_map = 0
    for _, row in reg.iterrows():
        rlab = str(row.get("Regulator", "")).strip()
        tcol = str(row.get("tap_column", "")).strip()
        exp = REGULATOR_TO_TAP_COL.get(rlab)
        if exp is None:
            n_missing_map += 1
        elif tcol and exp != tcol:
            n_bad_col += 1
    print(
        f"[reg_tap_diag] catalog: {len(reg)} regulator edges; "
        f"Regulator labels missing from REGULATOR_TO_TAP_COL: {n_missing_map}; "
        f"tap_column != map: {n_bad_col}"
    )
    if n_missing_map:
        miss = sorted({str(r.get("Regulator", "")).strip() for _, r in reg.iterrows() if not REGULATOR_TO_TAP_COL.get(str(r.get("Regulator", "")).strip())})
        print(f"[reg_tap_diag]   unmapped Regulator names (first 12): {miss[:12]}")


def _print_reg_training_csv_diag(edges_dir: Path) -> None:
    """hetero_mv_regulator_edge_features.csv: training-time tap ranges per edge_id (dataset artifact)."""
    path = edges_dir / "hetero_mv_regulator_edge_features.csv"
    if not path.is_file():
        print(f"[reg_tap_diag] no {path.name} — skip training-CSV tap range check")
        return
    try:
        df = pd.read_csv(path, usecols=["edge_id", "reg_tap_pu"])
    except (ValueError, KeyError):
        df = pd.read_csv(path)
        if "edge_id" not in df.columns or "reg_tap_pu" not in df.columns:
            print(f"[reg_tap_diag] {path.name}: missing edge_id/reg_tap_pu columns")
            return
    g = df.groupby("edge_id")["reg_tap_pu"].agg(["min", "max", "count"])
    g = g.sort_index()
    print(
        f"[reg_tap_diag] {path.name}: {len(g)} regulator edge_ids; "
        "global reg_tap_pu min/max over all samples:"
        f" {float(df['reg_tap_pu'].min()):.6f} / {float(df['reg_tap_pu'].max()):.6f} pu"
    )
    # Edges incident to l2879064.1 (user-facing example): catalog has line from 190-7361.1
    for eid in [4, 11, 21]:
        if eid not in g.index:
            continue
        row = g.loc[eid]
        print(
            f"[reg_tap_diag]   edge_id={int(eid)} training reg_tap_pu: "
            f"min={float(row['min']):.6f} max={float(row['max']):.6f} n={int(row['count'])}"
        )


def _accumulate_reg_tap_history(
    hist: dict[int, list[float]],
    reg_tap_map_step: dict[tuple[int, int], float],
) -> None:
    for (_sid, eid), val in reg_tap_map_step.items():
        hist.setdefault(int(eid), []).append(float(val))


def _print_reg_tap_open_dss_key_coverage(
    tap_raw: dict[str, float | int],
    reg_control_names: list[str],
    catalog: pd.DataFrame,
) -> None:
    """Ensure OpenDSS keys reg_{RegControlName}_tap_pu exist; compare to catalog tap_column names."""
    expected = set(_expected_dss_tap_keys(reg_control_names))
    present = {k for k in tap_raw if str(k).startswith("reg_") and str(k).endswith("_tap_pu")}
    missing_dss = sorted(expected - set(tap_raw.keys()))
    extra = sorted(present - expected)
    nz = sum(1 for k in expected & set(tap_raw.keys()) if np.isfinite(float(tap_raw[k])) and abs(float(tap_raw[k])) > 1e-12)
    print(
        f"[reg_tap_diag] OpenDSS tap_raw: {len(tap_raw)} keys; "
        f"expected {len(expected)} reg_*_tap_pu from RegControls; "
        f"nonzero finite: {nz}/{len(expected)}"
    )
    if missing_dss:
        print(f"[reg_tap_diag]   WARNING: missing OpenDSS keys (first 8): {missing_dss[:8]}")
    if extra:
        print(f"[reg_tap_diag]   extra tap-like keys (first 8): {extra[:8]}")
    cols_needed: set[str] = set()
    reg_rows = catalog[catalog["edge_type"].astype(str).str.strip().str.lower() == "regulator"]
    for _, row in reg_rows.iterrows():
        rlab = str(row.get("Regulator", "")).strip()
        c = REGULATOR_TO_TAP_COL.get(rlab) or str(row.get("tap_column", "")).strip()
        if c:
            cols_needed.add(c.lower())
    tap_keys_lower = {str(k).lower() for k in tap_raw}
    unresolved = sorted(cols_needed - tap_keys_lower)
    if unresolved:
        # fuzzy: underscore-insensitive
        still: list[str] = []
        for c in unresolved:
            cn = c.replace("_", "")
            if not any(str(k).lower().replace("_", "") == cn for k in tap_raw):
                still.append(c)
        if still:
            print(
                f"[reg_tap_diag]   WARNING: catalog tap columns with no exact tap_raw key (first 8): {still[:8]}"
            )


def _print_reg_tap_inference_first_step(
    catalog: pd.DataFrame,
    reg_tap_map_step: dict[tuple[int, int], float],
    tap_raw: dict[str, float | int],
) -> None:
    """Per regulator edge: resolved tap pu for GINE after _tap_pu_for_regulator_row."""
    reg_rows = catalog[catalog["edge_type"].astype(str).str.strip().str.lower() == "regulator"]
    n_zero = 0
    n_tot = 0
    lines: list[str] = []
    for _, row in reg_rows.iterrows():
        eid = int(row["edge_id"])
        rname = str(row["Regulator"]).strip()
        tcol = str(row.get("tap_column", "")).strip()
        v = float(reg_tap_map_step.get((0, eid), 0.0))
        n_tot += 1
        if abs(v) < 1e-12 or not np.isfinite(v):
            n_zero += 1
        if len(lines) < 8:
            exp = REGULATOR_TO_TAP_COL.get(rname, tcol)
            lines.append(f"edge_id={eid} Regulator={rname!r} -> tap_pu={v:.6f} (meta col {exp!r})")
    print(f"[reg_tap_diag] inference map to reg edge_attr: {n_zero}/{n_tot} edges resolved to ~0 (bad if all taps missing)")
    for ln in lines:
        print(f"[reg_tap_diag]   sample: {ln}")


def _print_reg_tap_series_summary(
    hist: dict[int, list[float]],
    catalog: pd.DataFrame,
    out_csv: Path | None,
) -> None:
    """After full day: variance of resolved tap per edge_id."""
    if not hist:
        print("[reg_tap_diag] no reg tap history collected")
        return
    reg_rows = catalog[catalog["edge_type"].astype(str).str.strip().str.lower() == "regulator"]
    eid_to_reg = {int(r["edge_id"]): str(r.get("Regulator", "")).strip() for _, r in reg_rows.iterrows()}
    rows_out: list[dict[str, float | int | str]] = []
    flat: list[int] = []
    varying: list[tuple[int, float]] = []
    for eid, series in sorted(hist.items()):
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            continue
        mn, mx = float(np.min(arr)), float(np.max(arr))
        rng = mx - mn
        sd = float(np.std(arr))
        rname = eid_to_reg.get(eid, "?")
        rows_out.append(
            {
                "edge_id": eid,
                "Regulator": rname,
                "n_steps": int(arr.size),
                "min_tap_pu": mn,
                "max_tap_pu": mx,
                "range_tap_pu": rng,
                "std_tap_pu": sd,
            }
        )
        if rng < 1e-9:
            flat.append(eid)
        else:
            varying.append((eid, rng))
    varying.sort(key=lambda x: -x[1])
    print(
        f"[reg_tap_diag] time series: {len(hist)} regulator edges with samples; "
        f"flat range <1e-9: {len(flat)}; varying: {len(varying)}"
    )
    if varying:
        print(f"[reg_tap_diag]   top 5 edges by tap range (pu): {[(e, f'{r:.6f}') for e, r in varying[:5]]}")
    if flat:
        print(f"[reg_tap_diag]   WARNING: flat taps (edge_id first 10): {flat[:10]}")
    if out_csv is not None and rows_out:
        pd.DataFrame(rows_out).sort_values("range_tap_pu", ascending=False).to_csv(out_csv, index=False)
        print(f"[reg_tap_diag] wrote per-edge tap stats -> {out_csv.resolve()}")


def _build_feature_meta(nodes_dir: Path, name_to_gidx: dict[str, int], global_type: dict[int, str], g2l: dict[str, dict[int, int]], counts: dict[str, int]) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, tuple[str, int]], dict[str, dict[str, int]]]:
    typed_name = {t: [""] * counts[t] for t in hm.NODE_TYPES}
    typed_dist = {t: np.zeros(counts[t], dtype=np.float32) for t in hm.NODE_TYPES}
    rep: dict[int, dict[str, float | str]] = {}

    use_cols = ["node", "node_idx", "electrical_distance_ohm", "p_load_kw", "q_load_kvar", "q_capacitor_bank"]
    for kind, rel in hm.NODE_FILES.items():
        path = nodes_dir / rel
        try:
            reader = pd.read_csv(path, usecols=lambda c: c in use_cols, chunksize=400_000) if kind == "load" else [pd.read_csv(path, usecols=lambda c: c in use_cols)]
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied opening CSV: {path}\n"
                "Close this file in Excel (and any preview panes), then rerun."
            ) from e
        try:
            iterator = reader
            for chunk in iterator:
                for r in chunk.itertuples(index=False):
                    g = int(float(r.node_idx)) if pd.notna(r.node_idx) else name_to_gidx.get(str(r.node).strip().lower())
                    if g is None:
                        continue
                    g = int(g)
                    if g not in rep:
                        rep[g] = {
                            "node": str(r.node).strip().lower(),
                            "electrical_distance_ohm": float(getattr(r, "electrical_distance_ohm")) if pd.notna(getattr(r, "electrical_distance_ohm")) else 0.0,
                        }
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while reading CSV: {path}\n"
                "Close this file in Excel (and any preview panes), then rerun."
            ) from e

    for g, t in global_type.items():
        li = g2l[t][g]
        typed_name[t][li] = str(rep.get(g, {}).get("node", ""))
        typed_dist[t][li] = float(rep.get(g, {}).get("electrical_distance_ohm", 0.0))

    dss_to_typed: dict[str, tuple[str, int]] = {}
    for t in hm.NODE_TYPES:
        for i, n in enumerate(typed_name[t]):
            if n:
                dss_to_typed[n] = (t, i)

    typed_to_dss_idx: dict[str, dict[str, int]] = {t: {} for t in hm.NODE_TYPES}
    return typed_name, typed_dist, dss_to_typed, typed_to_dss_idx


def run_compare(
    checkpoint: Path,
    dataset_dir: Path,
    node_index: Path,
    out_dir: Path,
    plot_nodes: list[str],
    npts: int,
    step_min: int,
    ymin: float,
    ymax: float,
    mv_sx_mapping: Path | None = None,
    daily_profile_csv: str | Path | None = None,
    reg_tap_diag: bool = True,
    device: str | None = None,
    show_plots: bool = True,
    monitoring_plots_subfolders: bool = False,
) -> None:
    device = resolve_inference_device(device)
    print(
        f"[compare_hetero_mv_daily] inference device: {device} "
        "(set device= or env GNN_COMPARE_DEVICE=auto|cpu|cuda)",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_dir = dataset_dir / "edges"
    nodes_dir = dataset_dir / "nodes"
    catalog = pd.read_csv(edges_dir / "hetero_mv_edge_catalog.csv")
    line_attr = pd.read_csv(edges_dir / "hetero_mv_line_edge_attr.csv")

    name_to_gidx = hm._read_node_idx_master(node_index)
    extra_names: set[str] = set()
    for fn in hm.NODE_FILES.values():
        df = pd.read_csv(nodes_dir / fn, usecols=["node"])
        extra_names.update(df["node"].astype(str).str.strip().tolist())

    g_list = hm._collect_global_node_indices(catalog, name_to_gidx, extra_names)
    membership = hm._membership_by_csv(nodes_dir)
    g2l, global_type, counts, edge_index_dict_cpu, line_ea_cpu = hm._build_typed_topology(catalog, line_attr, g_list, membership)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict_cpu.items()}
    line_ea = {k: v.to(device) for k, v in line_ea_cpu.items()}

    typed_name, typed_dist, dss_to_typed, _ = _build_feature_meta(nodes_dir, name_to_gidx, global_type, g2l, counts)

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_name = str(ck.get("cfg_name", checkpoint.stem))
    meta_ck = ck.get("meta") or {}
    ck_target_types = meta_ck.get("target_node_types")
    if ck_target_types is not None:
        ck_target_types = frozenset(str(x).lower() for x in ck_target_types)
    model, use_gine = _load_model(cfg_name, ck["state_dict"], edge_index_dict, device)
    model = maybe_torch_compile(model, label=cfg_name)
    print(f"[compare_hetero_mv_daily] model={cfg_name} device={device}")
    if ck_target_types is None:
        print("[compare_hetero_mv_daily] checkpoint meta: target_node_types=all (if meta missing, assume all storages supervised)")
    else:
        print(
            f"[compare_hetero_mv_daily] checkpoint meta: vmag supervised on storages {sorted(ck_target_types)}"
        )

    # Use the same baseline setup as run_daily_aggregate_dataset_8500.py:
    # compile daily entrypoint, detach Daily from loads, apply explicit mL[t] scaling.
    rd8500._compile_8500_daily_setup()
    reg_control_names: list[str] = rd8500._discover_reg_controls() if use_gine else []
    if use_gine:
        print(
            f"[compare_hetero_mv_daily] GINE: reg edge_attr from OpenDSS each step "
            f"({len(reg_control_names)} RegControls)"
        )
        if reg_tap_diag:
            _print_reg_edge_catalog_diag(catalog)
            _print_reg_training_csv_diag(edges_dir)
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    print(
        "[compare_hetero_mv_daily] OpenDSS solve timing: per-step snapshot mode (not daily warm-start).",
        flush=True,
    )
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    prof_path = rd8500._resolve_daily_profile_csv(daily_profile_csv)
    print(f"[compare_hetero_mv_daily] daily profile: {prof_path}", flush=True)
    mL = rd8500._daily_profile_5min(npts=npts, profile_csv=daily_profile_csv)

    all_nodes = []
    for n in dss.Circuit.AllNodeNames():
        s = str(n).strip().lower()
        if "." not in s:
            continue
        phs = s.rsplit(".", 1)[1]
        try:
            ph = int(phs)
        except ValueError:
            continue
        if ph in (1, 2, 3):
            all_nodes.append(s)
    all_nodes = list(dict.fromkeys(all_nodes))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    typed_to_dss_idx: dict[str, np.ndarray] = {t: np.full(counts[t], -1, dtype=np.int64) for t in hm.NODE_TYPES}
    for n, (t, li) in dss_to_typed.items():
        if n in node_to_idx:
            typed_to_dss_idx[t][li] = node_to_idx[n]

    repo_root = Path(__file__).resolve().parent
    mpath = mv_sx_mapping if mv_sx_mapping is not None else (repo_root / "8500-node" / "mv_x_sx_node_mapping_8500.csv")
    mv_sx_rules: list[dict[str, str]] = _load_mv_sx_mapping(mpath) if mpath.is_file() else []
    if mv_sx_rules:
        print(f"[compare_hetero_mv_daily] mv↔sx mapping: {len(mv_sx_rules)} rules from {mpath.resolve()}")
    else:
        print(
            f"[compare_hetero_mv_daily] WARNING: no MV↔sx mapping ({mpath} missing or empty). "
            "Hetero load rows use MV bus names (l…); OpenDSS loads are on sx… buses — "
            "P/Q in x['load'] will stay 0. Generate the CSV (IEEE8500_OpenDSS_timing.ipynb) or pass --mv-sx-mapping."
        )

    gset = set(g_list)

    def _tap_pu_for_regulator_row(reg_display: str, tap_raw: dict[str, float | int]) -> float:
        """Map catalog Regulator label → tap pu. Keys in tap_raw are reg_{RegControls.Name()}_tap_pu; must match gnn_sample_meta."""
        col = REGULATOR_TO_TAP_COL.get(str(reg_display).strip())
        if not col:
            return 0.0
        v = tap_raw.get(col)
        if v is not None and np.isfinite(float(v)):
            return float(v)
        col_l = col.lower()
        for k, val in tap_raw.items():
            if str(k).lower() == col_l and np.isfinite(float(val)):
                return float(val)
        # Underscore-insensitive match (handles reg_feeder_rega vs reg_feeder_reg_a style drift)
        cn = col_l.replace("_", "")
        for k, val in tap_raw.items():
            if str(k).lower().replace("_", "") == cn and np.isfinite(float(val)):
                return float(val)
        return 0.0

    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)
    v_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_gnn = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)

    def _make_x_dict() -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for t in hm.NODE_TYPES:
            out[t] = np.zeros((counts[t], hm.IN_DIMS[t]), dtype=np.float32)
            if "electrical_distance_ohm" in hm.TYPE_FEAT_COLS[t]:
                ci = hm.TYPE_FEAT_COLS[t].index("electrical_distance_ohm")
                out[t][:, ci] = typed_dist[t]
        return out

    def _make_edge_attr_dict_gine(reg_tap_map_step: dict[tuple[int, int], float]) -> dict[tuple[str, str, str], torch.Tensor]:
        """Line R/X from dataset + reg tap pu per edge, matching training (`_reg_attr_dict_per_keys`)."""
        out = dict(line_ea)
        reg_part = hm._reg_attr_dict_per_keys(
            0, catalog, gset, global_type, g2l, reg_tap_map_step, edge_index_dict, device
        )
        out.update(reg_part)
        return out

    n_nonconv = 0
    scenario_scale = 1.0
    first_feature_diag = True
    first_reg_diag = True
    reg_tap_hist: dict[int, list[float]] = {}
    open_apply_s_total = 0.0
    open_reassert_s_total = 0.0
    open_solve_only_s_total = 0.0
    open_get_s_total = 0.0
    feature_build_s_total = 0.0
    gnn_infer_s_total = 0.0
    gnn_forward_only_s_total = 0.0
    for i in range(npts):
        hr = int(i // 12)
        sec = int((i % 12) * (step_min * 60))
        m_t = float(mL[i])
        total_scale_t = scenario_scale * m_t
        kw_set = base_kw * total_scale_t
        kvar_set = base_kvar * total_scale_t
        t_apply0 = time.perf_counter()
        dss.Text.Command(f"set hour={hr} sec={sec}")
        for j, name in enumerate(base_names):
            dss.Loads.Name(name)
            dss.Loads.kW(float(kw_set[j]))
            dss.Loads.kvar(float(kvar_set[j]))
        t_apply1 = time.perf_counter()
        open_apply_s_total += t_apply1 - t_apply0

        t_reassert0 = time.perf_counter()
        reassert_snapshot_before_each_solve()
        t_reassert1 = time.perf_counter()
        open_reassert_s_total += t_reassert1 - t_reassert0

        t_solve0 = time.perf_counter()
        dss.Solution.Solve()
        t_solve1 = time.perf_counter()
        open_solve_only_s_total += t_solve1 - t_solve0

        if not dss.Solution.Converged():
            n_nonconv += 1
            continue

        t_get0 = time.perf_counter()
        vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(all_nodes)
        v_dss[i, :] = np.asarray(vmag, dtype=np.float32)
        t_get1 = time.perf_counter()
        open_get_s_total += t_get1 - t_get0

        t_fb0 = time.perf_counter()
        busphP_load: dict[tuple[str, int], float] = {}
        busphQ_load: dict[tuple[str, int], float] = {}
        for j, name in enumerate(base_names):
            for (bus, ph, w) in load_to_busph[name]:
                bk = str(bus).strip().lower()
                busphP_load[(bk, int(ph))] = busphP_load.get((bk, int(ph)), 0.0) + float(kw_set[j]) * float(w)
                busphQ_load[(bk, int(ph))] = busphQ_load.get((bk, int(ph)), 0.0) + float(kvar_set[j]) * float(w)
        x_np = _make_x_dict()

        node_P: dict[str, float] = {}
        node_Q: dict[str, float] = {}
        for (bus, ph), pval in busphP_load.items():
            nk = f"{str(bus).strip().lower()}.{int(ph)}"
            node_P[nk] = float(pval)
        for (bus, ph), qval in busphQ_load.items():
            nk = f"{str(bus).strip().lower()}.{int(ph)}"
            node_Q[nk] = float(qval)

        if mv_sx_rules:
            for rec in mv_sx_rules:
                mv = rec["mv_key"]
                tp = dss_to_typed.get(mv)
                if tp is None or tp[0] != "load":
                    continue
                pa = float(node_P.get(rec["load_a"], 0.0) + node_P.get(rec["load_b"], 0.0))
                qa = float(node_Q.get(rec["load_a"], 0.0) + node_Q.get(rec["load_b"], 0.0))
                x_np["load"][tp[1], 0] = pa
                x_np["load"][tp[1], 1] = qa
        else:
            for (bus, ph), pval in busphP_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                tp = dss_to_typed.get(node)
                if tp is not None and tp[0] == "load":
                    x_np["load"][tp[1], 0] = float(pval)
            for (bus, ph), qval in busphQ_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                tp = dss_to_typed.get(node)
                if tp is not None and tp[0] == "load":
                    x_np["load"][tp[1], 1] = float(qval)

        dss.Capacitors.First()
        while True:
            cn = dss.Capacitors.Name()
            dss.Circuit.SetActiveElement(f"Capacitor.{cn}")
            buses = dss.CktElement.BusNames()
            if buses and len(buses) > 0:
                b = str(buses[0]).split(".")[0].strip().lower()
                try:
                    qnom = float(dss.Capacitors.kvar())
                    st = dss.Capacitors.States()
                    if isinstance(st, (list, tuple, np.ndarray)):
                        on = bool(np.any(np.asarray(st, dtype=float) > 0.5))
                    else:
                        on = float(st) > 0.5
                    q_now = qnom if on else 0.0
                except Exception:
                    q_now = 0.0
                for ph in (1, 2, 3):
                    node = f"{b}.{ph}"
                    tp = dss_to_typed.get(node)
                    if tp is not None and tp[0] == "capacitor":
                        li = tp[1]
                        x_np["capacitor"][li, 0] += q_now / 3.0
            if not dss.Capacitors.Next():
                break

        if first_feature_diag:
            first_feature_diag = False
            pl = x_np["load"][:, 0]
            ql = x_np["load"][:, 1]
            nz = int(np.sum(np.abs(pl) + np.abs(ql) > 1e-3))
            print(
                f"[compare_hetero_mv_daily] feature diag (first converged step): "
                f"hetero load slots with |P|+|Q|>1e-3 kW/kvar: {nz}/{counts['load']}"
            )
            if nz < max(8, counts["load"] // 50):
                if not mv_sx_rules:
                    print(
                        "[compare_hetero_mv_daily] hint: MV↔sx mapping was not loaded — OpenDSS P/Q are aggregated "
                        "on sx….phase buses, but hetero load nodes are l….phase; without the mapping CSV they are "
                        "never rolled up onto x['load']. Place 8500-node/mv_x_sx_node_mapping_8500.csv or use "
                        "--mv-sx-mapping."
                    )
                else:
                    print(
                        "[compare_hetero_mv_daily] hint: most load P/Q stayed 0 after MV rollup — check mapping "
                        "leaf names vs node_P keys (sx….phase), or taps mis-keyed (see GINE tap line below)."
                    )
            if use_gine and reg_tap_diag and first_reg_diag:
                tr = rd8500._read_reg_control_state(reg_control_names)
                n_tap_nz = sum(
                    1 for v in tr.values() if np.isfinite(float(v)) and abs(float(v)) > 1e-9
                )
                print(
                    f"  OpenDSS reg tap dict: {len(tr)} keys, {n_tap_nz} nonzero  "
                    f"(sample: {sorted(tr.keys())[:3]})"
                )
                exp0 = REGULATOR_TO_TAP_COL.get("FEEDER_REGA", "")
                if exp0 and exp0 not in tr and not any(
                    str(k).lower().replace("_", "") == exp0.lower().replace("_", "") for k in tr
                ):
                    print(
                        f"  warning: expected meta column {exp0!r} not in tap_raw — "
                        "reg edge_attr may be all zeros (name mismatch vs RegControls.Name())."
                    )

        reg_tap_map_step: dict[tuple[int, int], float] | None = None
        if use_gine:
            tap_raw = rd8500._read_reg_control_state(reg_control_names)
            reg_tap_map_step = {}
            for _, row in catalog.iterrows():
                if str(row["edge_type"]).strip().lower() != "regulator":
                    continue
                eid = int(row["edge_id"])
                rname = str(row["Regulator"]).strip()
                reg_tap_map_step[(0, eid)] = _tap_pu_for_regulator_row(rname, tap_raw)
            if reg_tap_diag:
                _accumulate_reg_tap_history(reg_tap_hist, reg_tap_map_step)
                if first_reg_diag:
                    _print_reg_tap_open_dss_key_coverage(tap_raw, reg_control_names, catalog)
                    _print_reg_tap_inference_first_step(catalog, reg_tap_map_step, tap_raw)
                    first_reg_diag = False

        t_fb1 = time.perf_counter()
        feature_build_s_total += t_fb1 - t_fb0

        t_gnn0 = time.perf_counter()
        x_dict = {t: torch.from_numpy(x_np[t]).to(device) for t in hm.NODE_TYPES}
        with torch.no_grad():
            t_fwd0 = time.perf_counter()
            if use_gine:
                assert reg_tap_map_step is not None
                pred = model(x_dict, edge_index_dict, _make_edge_attr_dict_gine(reg_tap_map_step))
            else:
                pred = model(x_dict, edge_index_dict)
            t_fwd1 = time.perf_counter()
        gnn_forward_only_s_total += t_fwd1 - t_fwd0

        for t in hm.NODE_TYPES:
            arr = pred[t].detach().cpu().numpy()
            idxs = typed_to_dss_idx[t]
            good = idxs >= 0
            v_gnn[i, idxs[good]] = arr[good]

        sync_inference_device(device)
        t_gnn1 = time.perf_counter()
        gnn_infer_s_total += t_gnn1 - t_gnn0

        if (i + 1) % max(1, npts // 12) == 0:
            print(
                f"[{i + 1}/{npts}] timing so far — OpenDSS apply={open_apply_s_total:.2f}s | "
                f"reassert={open_reassert_s_total:.2f}s solve_only={open_solve_only_s_total:.2f}s | "
                f"get V={open_get_s_total:.2f}s | "
                f"feature build={feature_build_s_total:.2f}s | "
                f"GNN bucket={gnn_infer_s_total:.2f}s fwd-only={gnn_forward_only_s_total:.2f}s",
                flush=True,
            )

    n_ok = int(npts - n_nonconv)

    print_mv_daily_timing_summary(
        n_ok=n_ok,
        npts=npts,
        n_nonconv=n_nonconv,
        open_apply_s_total=open_apply_s_total,
        open_reassert_s_total=open_reassert_s_total,
        open_solve_only_s_total=open_solve_only_s_total,
        open_get_s_total=open_get_s_total,
        feature_build_s_total=feature_build_s_total,
        gnn_forward_only_s_total=gnn_forward_only_s_total,
        gnn_bucket_s_total=gnn_infer_s_total,
        device=str(device),
        title="Daily Timing Summary (hetero MV vs OpenDSS)",
        feature_label="Hetero feature build",
        log_prefix="[compare_hetero_mv_daily]",
    )

    mask = np.isfinite(v_dss) & np.isfinite(v_gnn)
    mae = float(np.mean(np.abs(v_dss[mask] - v_gnn[mask])))
    rmse = float(np.sqrt(np.mean((v_dss[mask] - v_gnn[mask]) ** 2)))
    print(f"\nOverall: MAE={mae:.6f} pu  RMSE={rmse:.6f} pu  n_points={int(mask.sum())} nonconv={n_nonconv}")

    if use_gine and reg_tap_diag and reg_tap_hist:
        _print_reg_tap_series_summary(
            reg_tap_hist,
            catalog,
            out_dir / f"daily_reg_tap_timeseries_{_safe_cfg_stem(cfg_name)}.csv",
        )

    node_rows = []
    for i, n in enumerate(all_nodes):
        m = np.isfinite(v_dss[:, i]) & np.isfinite(v_gnn[:, i])
        if m.any():
            node_rows.append((n, float(np.mean(np.abs(v_dss[m, i] - v_gnn[m, i])))))
    df_mae = pd.DataFrame(node_rows, columns=["node", "mae"]).sort_values("mae", ascending=False)
    df_mae.to_csv(out_dir / f"daily_mae_per_node_{cfg_name}.csv", index=False)

    # Hetero **load** nodes only: how much GNN |V| varies over the day.
    # Use float64 for stats; sort by **range** (max−min) first — std can show ~1e-7 float noise when flat.
    _var_eps = 1e-6
    var_rows: list[tuple[str, float, float, int]] = []
    for n, (t, _li) in dss_to_typed.items():
        if t != "load":
            continue
        if n not in node_to_idx:
            continue
        j = node_to_idx[n]
        col = v_gnn[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 2:
            continue
        w = np.asarray(col[fin], dtype=np.float64)
        std = float(np.std(w))
        rng = float(np.max(w) - np.min(w))
        var_rows.append((n, std, rng, int(fin.sum())))
    if var_rows:
        df_var = pd.DataFrame(var_rows, columns=["node", "std_pu", "range_pu", "n_finite_pts"])
        df_var = df_var.sort_values(["range_pu", "std_pu"], ascending=[False, False]).reset_index(drop=True)
        df_var.to_csv(out_dir / f"daily_gnn_variation_load_nodes_{cfg_name}.csv", index=False)
        max_rng = float(df_var["range_pu"].max())
        if max_rng < _var_eps:
            print(
                "\n[compare_hetero_mv_daily] GNN load predictions are **flat** over the day: "
                f"max (max−min) |V| ≈ {max_rng:.3e} pu (< {_var_eps:g}). "
                "Any ranking by std is float noise; ignore 'top' row."
            )
        top = df_var.iloc[0]
        print(
            f"\n[compare_hetero_mv_daily] load node with largest GNN |V| **range** (day): "
            f"{str(top['node'])!r}  range={float(top['range_pu']):.6f} pu  std={float(top['std_pu']):.6f} pu"
        )
        print("[compare_hetero_mv_daily] top 10 load nodes by range (then std), pu:")
        print(df_var.head(10).to_string(index=False))
    else:
        print("[compare_hetero_mv_daily] warning: no load nodes with >=2 finite GNN points for variation table.")

    for n in [str(x).strip().lower() for x in plot_nodes if str(x).strip().lower() in node_to_idx]:
        i = node_to_idx[n]
        tp = dss_to_typed.get(n)
        if tp is not None and ck_target_types is not None and tp[0] not in ck_target_types:
            print(
                f"[compare_hetero_mv_daily] warning: node {n!r} is hetero type {tp[0]!r} but checkpoint "
                f"only supervised {sorted(ck_target_types)} — GNN curve is not meaningful."
            )
        m = np.isfinite(v_dss[:, i]) & np.isfinite(v_gnn[:, i])
        n_mae = float(np.mean(np.abs(v_dss[m, i] - v_gnn[m, i]))) if m.any() else np.nan
        fig = plt.figure(figsize=(10, 4.2))
        plt.plot(t_hours, v_dss[:, i], linewidth=2.0, label="OpenDSS baseline")
        plt.plot(t_hours, v_gnn[:, i], "--", linewidth=1.6, label=f"{cfg_name} (MAE={n_mae:.4f})")
        plt.xlabel("Hour of day")
        plt.ylabel("Voltage magnitude (pu)")
        plt.title(f"24h voltage @ {n}")
        plt.ylim(ymin, ymax)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if monitoring_plots_subfolders:
            plot_dir = out_dir / "monitoring_plots" / n.replace(".", "_")
            plot_dir.mkdir(parents=True, exist_ok=True)
            png_path = plot_dir / f"daily_compare_{cfg_name}_{n.replace('.', '_')}.png"
        else:
            png_path = out_dir / f"daily_compare_{cfg_name}_{n.replace('.', '_')}.png"
        plt.savefig(png_path, dpi=160)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    err = np.abs(v_dss[mask] - v_gnn[mask])
    fig_h = plt.figure(figsize=(8.2, 4.2))
    plt.hist(err, bins=120, alpha=0.9)
    plt.xlabel("|V_gnn - V_dss| (pu)")
    plt.ylabel("Count")
    plt.title(f"Error distribution: {cfg_name} vs OpenDSS")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"daily_error_hist_{cfg_name}.png", dpi=170)
    if show_plots:
        plt.show()
    else:
        plt.close(fig_h)

    print("\nSaved:", out_dir.resolve())
    print(df_mae.head(10).to_string(index=False))


def _safe_cfg_stem(s: str) -> str:
    t = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s).strip())[:96]
    return t or "model"


def run_compare_juxtapose(
    checkpoint_a: Path,
    checkpoint_b: Path,
    dataset_dir: Path,
    node_index: Path,
    out_dir: Path,
    npts: int,
    step_min: int,
    ymin: float,
    ymax: float,
    mv_sx_mapping: Path | None,
    daily_profile_csv: str | Path | None = None,
    top_disagree: int = 10,
    disagree_scope: str = "load",
    also_plot_nodes: list[str] | None = None,
    reg_tap_diag: bool = True,
    juxtapose_mode: Literal["disagree", "both_fail_dss", "lowest_min_v_dss"] = "disagree",
    device: str | None = None,
    show_plots: bool = True,
    monitoring_plots_subfolders: bool = False,
) -> None:
    """
    One daily OpenDSS run; two GNN forwards per step.

    - ``disagree`` (default): rank nodes by mean |V_a − V_b| (pu); save CSV; plot top‑K.
    - ``both_fail_dss``: rank by how badly **both** models match OpenDSS — primary key is
      min(MAE_a, MAE_b) vs DSS per node (high ⇒ even the better model is far from DSS).
    - ``lowest_min_v_dss``: rank loads (or ``disagree_scope``) by **lowest** daily minimum
      OpenDSS |V| (pu); plot top‑K most voltage-stressed nodes with OpenDSS + both GNNs.
    """
    device = resolve_inference_device(device)
    print(
        f"[compare_hetero_mv_daily] inference device: {device} "
        "(set device= or env GNN_COMPARE_DEVICE=auto|cpu|cuda)",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    also_plot_nodes = also_plot_nodes or []

    edges_dir = dataset_dir / "edges"
    nodes_dir = dataset_dir / "nodes"
    catalog = pd.read_csv(edges_dir / "hetero_mv_edge_catalog.csv")
    line_attr = pd.read_csv(edges_dir / "hetero_mv_line_edge_attr.csv")

    name_to_gidx = hm._read_node_idx_master(node_index)
    extra_names: set[str] = set()
    for fn in hm.NODE_FILES.values():
        df = pd.read_csv(nodes_dir / fn, usecols=["node"])
        extra_names.update(df["node"].astype(str).str.strip().tolist())

    g_list = hm._collect_global_node_indices(catalog, name_to_gidx, extra_names)
    membership = hm._membership_by_csv(nodes_dir)
    g2l, global_type, counts, edge_index_dict_cpu, line_ea_cpu = hm._build_typed_topology(catalog, line_attr, g_list, membership)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict_cpu.items()}
    line_ea = {k: v.to(device) for k, v in line_ea_cpu.items()}

    typed_name, typed_dist, dss_to_typed, _ = _build_feature_meta(nodes_dir, name_to_gidx, global_type, g2l, counts)

    cka = torch.load(checkpoint_a, map_location="cpu", weights_only=False)
    ckb = torch.load(checkpoint_b, map_location="cpu", weights_only=False)
    cfg_a = str(cka.get("cfg_name", checkpoint_a.stem))
    cfg_b = str(ckb.get("cfg_name", checkpoint_b.stem))
    model_a, use_gine_a = _load_model(cfg_a, cka["state_dict"], edge_index_dict, device)
    model_b, use_gine_b = _load_model(cfg_b, ckb["state_dict"], edge_index_dict, device)
    model_a = maybe_torch_compile(model_a, label=f"juxtapose A {cfg_a}")
    model_b = maybe_torch_compile(model_b, label=f"juxtapose B {cfg_b}")
    stem_a = _safe_cfg_stem(cfg_a)
    stem_b = _safe_cfg_stem(cfg_b)
    print(
        f"[compare_hetero_mv_daily] juxtapose: mode={juxtapose_mode}  A={cfg_a} (gine={use_gine_a})  "
        f"B={cfg_b} (gine={use_gine_b})  device={device}"
    )

    rd8500._compile_8500_daily_setup()
    need_reg = use_gine_a or use_gine_b
    reg_control_names: list[str] = rd8500._discover_reg_controls() if need_reg else []
    if need_reg:
        print(f"[compare_hetero_mv_daily] juxtapose: regulator taps from OpenDSS ({len(reg_control_names)} RegControls)")
        if reg_tap_diag:
            _print_reg_edge_catalog_diag(catalog)
            _print_reg_training_csv_diag(edges_dir)
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    print(
        "[compare_hetero_mv_daily] juxtapose: OpenDSS solve uses per-step snapshot mode (not daily warm-start).",
        flush=True,
    )
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    prof_path_j = rd8500._resolve_daily_profile_csv(daily_profile_csv)
    print(f"[compare_hetero_mv_daily] juxtapose daily profile: {prof_path_j}", flush=True)
    mL = rd8500._daily_profile_5min(npts=npts, profile_csv=daily_profile_csv)

    all_nodes: list[str] = []
    for n in dss.Circuit.AllNodeNames():
        s = str(n).strip().lower()
        if "." not in s:
            continue
        phs = s.rsplit(".", 1)[1]
        try:
            ph = int(phs)
        except ValueError:
            continue
        if ph in (1, 2, 3):
            all_nodes.append(s)
    all_nodes = list(dict.fromkeys(all_nodes))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    typed_to_dss_idx: dict[str, np.ndarray] = {t: np.full(counts[t], -1, dtype=np.int64) for t in hm.NODE_TYPES}
    for n, (t, li) in dss_to_typed.items():
        if n in node_to_idx:
            typed_to_dss_idx[t][li] = node_to_idx[n]

    repo_root = Path(__file__).resolve().parent
    mpath = mv_sx_mapping if mv_sx_mapping is not None else (repo_root / "8500-node" / "mv_x_sx_node_mapping_8500.csv")
    mv_sx_rules: list[dict[str, str]] = _load_mv_sx_mapping(mpath) if mpath.is_file() else []
    if mv_sx_rules:
        print(f"[compare_hetero_mv_daily] mv↔sx mapping: {len(mv_sx_rules)} rules from {mpath.resolve()}")
    else:
        print(
            f"[compare_hetero_mv_daily] WARNING: no MV↔sx mapping ({mpath} missing or empty). "
            "P/Q in x['load'] will stay 0 for both models."
        )

    gset = set(g_list)

    def _tap_pu_for_regulator_row(reg_display: str, tap_raw: dict[str, float | int]) -> float:
        col = REGULATOR_TO_TAP_COL.get(str(reg_display).strip())
        if not col:
            return 0.0
        v = tap_raw.get(col)
        if v is not None and np.isfinite(float(v)):
            return float(v)
        col_l = col.lower()
        for k, val in tap_raw.items():
            if str(k).lower() == col_l and np.isfinite(float(val)):
                return float(val)
        cn = col_l.replace("_", "")
        for k, val in tap_raw.items():
            if str(k).lower().replace("_", "") == cn and np.isfinite(float(val)):
                return float(val)
        return 0.0

    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)
    v_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_a = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_b = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)

    def _make_x_dict() -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for t in hm.NODE_TYPES:
            out[t] = np.zeros((counts[t], hm.IN_DIMS[t]), dtype=np.float32)
            if "electrical_distance_ohm" in hm.TYPE_FEAT_COLS[t]:
                ci = hm.TYPE_FEAT_COLS[t].index("electrical_distance_ohm")
                out[t][:, ci] = typed_dist[t]
        return out

    def _make_edge_attr_dict_gine(reg_tap_map_step: dict[tuple[int, int], float]) -> dict[tuple[str, str, str], torch.Tensor]:
        out = dict(line_ea)
        reg_part = hm._reg_attr_dict_per_keys(
            0, catalog, gset, global_type, g2l, reg_tap_map_step, edge_index_dict, device
        )
        out.update(reg_part)
        return out

    def _forward_one(
        model: torch.nn.Module,
        use_gine: bool,
        x_d: dict[str, torch.Tensor],
        reg_map: dict[tuple[int, int], float] | None,
    ) -> dict[str, torch.Tensor]:
        if use_gine:
            assert reg_map is not None
            return model(x_d, edge_index_dict, _make_edge_attr_dict_gine(reg_map))
        return model(x_d, edge_index_dict)

    n_nonconv = 0
    first_feature_diag = True
    first_reg_diag = True
    reg_tap_hist: dict[int, list[float]] = {}
    open_apply_s_total = 0.0
    open_reassert_s_total = 0.0
    open_solve_only_s_total = 0.0
    open_get_s_total = 0.0
    feature_build_s_total = 0.0
    gnn_infer_s_total = 0.0
    gnn_forward_only_a_s_total = 0.0
    gnn_forward_only_b_s_total = 0.0
    for i in range(npts):
        hr = int(i // 12)
        sec = int((i % 12) * (step_min * 60))
        m_t = float(mL[i])
        total_scale_t = m_t
        kw_set = base_kw * total_scale_t
        kvar_set = base_kvar * total_scale_t

        t_apply0 = time.perf_counter()
        dss.Text.Command(f"set hour={hr} sec={sec}")
        for j, name in enumerate(base_names):
            dss.Loads.Name(name)
            dss.Loads.kW(float(kw_set[j]))
            dss.Loads.kvar(float(kvar_set[j]))
        t_apply1 = time.perf_counter()
        open_apply_s_total += t_apply1 - t_apply0

        t_reassert0 = time.perf_counter()
        reassert_snapshot_before_each_solve()
        t_reassert1 = time.perf_counter()
        open_reassert_s_total += t_reassert1 - t_reassert0

        t_solve0 = time.perf_counter()
        dss.Solution.Solve()
        t_solve1 = time.perf_counter()
        open_solve_only_s_total += t_solve1 - t_solve0

        if not dss.Solution.Converged():
            n_nonconv += 1
            continue

        t_get0 = time.perf_counter()
        vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(all_nodes)
        v_dss[i, :] = np.asarray(vmag, dtype=np.float32)
        t_get1 = time.perf_counter()
        open_get_s_total += t_get1 - t_get0

        t_fb0 = time.perf_counter()
        busphP_load: dict[tuple[str, int], float] = {}
        busphQ_load: dict[tuple[str, int], float] = {}
        for j, name in enumerate(base_names):
            for (bus, ph, w) in load_to_busph[name]:
                bk = str(bus).strip().lower()
                busphP_load[(bk, int(ph))] = busphP_load.get((bk, int(ph)), 0.0) + float(kw_set[j]) * float(w)
                busphQ_load[(bk, int(ph))] = busphQ_load.get((bk, int(ph)), 0.0) + float(kvar_set[j]) * float(w)

        x_np = _make_x_dict()
        node_P: dict[str, float] = {}
        node_Q: dict[str, float] = {}
        for (bus, ph), pval in busphP_load.items():
            nk = f"{str(bus).strip().lower()}.{int(ph)}"
            node_P[nk] = float(pval)
        for (bus, ph), qval in busphQ_load.items():
            nk = f"{str(bus).strip().lower()}.{int(ph)}"
            node_Q[nk] = float(qval)

        if mv_sx_rules:
            for rec in mv_sx_rules:
                mv = rec["mv_key"]
                tp = dss_to_typed.get(mv)
                if tp is None or tp[0] != "load":
                    continue
                pa = float(node_P.get(rec["load_a"], 0.0) + node_P.get(rec["load_b"], 0.0))
                qa = float(node_Q.get(rec["load_a"], 0.0) + node_Q.get(rec["load_b"], 0.0))
                x_np["load"][tp[1], 0] = pa
                x_np["load"][tp[1], 1] = qa
        else:
            for (bus, ph), pval in busphP_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                tp = dss_to_typed.get(node)
                if tp is not None and tp[0] == "load":
                    x_np["load"][tp[1], 0] = float(pval)
            for (bus, ph), qval in busphQ_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                tp = dss_to_typed.get(node)
                if tp is not None and tp[0] == "load":
                    x_np["load"][tp[1], 1] = float(qval)

        dss.Capacitors.First()
        while True:
            cn = dss.Capacitors.Name()
            dss.Circuit.SetActiveElement(f"Capacitor.{cn}")
            buses = dss.CktElement.BusNames()
            if buses and len(buses) > 0:
                b = str(buses[0]).split(".")[0].strip().lower()
                try:
                    qnom = float(dss.Capacitors.kvar())
                    st = dss.Capacitors.States()
                    if isinstance(st, (list, tuple, np.ndarray)):
                        on = bool(np.any(np.asarray(st, dtype=float) > 0.5))
                    else:
                        on = float(st) > 0.5
                    q_now = qnom if on else 0.0
                except Exception:
                    q_now = 0.0
                for ph in (1, 2, 3):
                    node = f"{b}.{ph}"
                    tp = dss_to_typed.get(node)
                    if tp is not None and tp[0] == "capacitor":
                        li = tp[1]
                        x_np["capacitor"][li, 0] += q_now / 3.0
            if not dss.Capacitors.Next():
                break

        if first_feature_diag:
            first_feature_diag = False
            pl = x_np["load"][:, 0]
            ql = x_np["load"][:, 1]
            nz = int(np.sum(np.abs(pl) + np.abs(ql) > 1e-3))
            print(
                f"[compare_hetero_mv_daily] juxtapose feature diag: hetero load slots with |P|+|Q|>1e-3: {nz}/{counts['load']}"
            )

        reg_tap_map_shared: dict[tuple[int, int], float] | None = None
        if need_reg:
            tap_raw_step = rd8500._read_reg_control_state(reg_control_names)
            reg_tap_map_shared = {}
            for _, row in catalog.iterrows():
                if str(row["edge_type"]).strip().lower() != "regulator":
                    continue
                eid = int(row["edge_id"])
                rname = str(row["Regulator"]).strip()
                reg_tap_map_shared[(0, eid)] = _tap_pu_for_regulator_row(rname, tap_raw_step)
            if reg_tap_diag:
                _accumulate_reg_tap_history(reg_tap_hist, reg_tap_map_shared)
                if first_reg_diag:
                    _print_reg_tap_open_dss_key_coverage(tap_raw_step, reg_control_names, catalog)
                    _print_reg_tap_inference_first_step(catalog, reg_tap_map_shared, tap_raw_step)
                    first_reg_diag = False

        t_fb1 = time.perf_counter()
        feature_build_s_total += t_fb1 - t_fb0

        t_gnn0 = time.perf_counter()
        x_dict = {t: torch.from_numpy(x_np[t]).to(device) for t in hm.NODE_TYPES}
        with torch.no_grad():
            t_fa0 = time.perf_counter()
            pred_a = _forward_one(model_a, use_gine_a, x_dict, reg_tap_map_shared if use_gine_a else None)
            t_fa1 = time.perf_counter()
            gnn_forward_only_a_s_total += t_fa1 - t_fa0
            t_fb0_ = time.perf_counter()
            pred_b = _forward_one(model_b, use_gine_b, x_dict, reg_tap_map_shared if use_gine_b else None)
            t_fb1_ = time.perf_counter()
            gnn_forward_only_b_s_total += t_fb1_ - t_fb0_

        for t in hm.NODE_TYPES:
            arr_a = pred_a[t].detach().cpu().numpy()
            arr_b = pred_b[t].detach().cpu().numpy()
            idxs = typed_to_dss_idx[t]
            good = idxs >= 0
            v_a[i, idxs[good]] = arr_a[good]
            v_b[i, idxs[good]] = arr_b[good]

        sync_inference_device(device)
        t_gnn1 = time.perf_counter()
        gnn_infer_s_total += t_gnn1 - t_gnn0

        if (i + 1) % 24 == 0:
            print(
                f"[{i + 1}/{npts}] juxtapose — apply={open_apply_s_total:.2f}s reassert={open_reassert_s_total:.2f}s "
                f"solve_only={open_solve_only_s_total:.2f}s get V={open_get_s_total:.2f}s | "
                f"feature={feature_build_s_total:.2f}s | GNN bucket={gnn_infer_s_total:.2f}s "
                f"fwd-only A={gnn_forward_only_a_s_total:.2f}s B={gnn_forward_only_b_s_total:.2f}s",
                flush=True,
            )

    n_ok = int(npts - n_nonconv)
    gnn_forward_only_s_total = gnn_forward_only_a_s_total + gnn_forward_only_b_s_total

    print(f"\n[compare_hetero_mv_daily] juxtapose: nonconv={n_nonconv}")

    print_mv_daily_timing_summary(
        n_ok=n_ok,
        npts=npts,
        n_nonconv=n_nonconv,
        open_apply_s_total=open_apply_s_total,
        open_reassert_s_total=open_reassert_s_total,
        open_solve_only_s_total=open_solve_only_s_total,
        open_get_s_total=open_get_s_total,
        feature_build_s_total=feature_build_s_total,
        gnn_forward_only_s_total=gnn_forward_only_s_total,
        gnn_bucket_s_total=gnn_infer_s_total,
        device=str(device),
        title=f"Daily Timing Summary (juxtapose: {cfg_a} vs {cfg_b})",
        feature_label="Hetero feature build (shared)",
        log_prefix="[compare_hetero_mv_daily]",
        gnn_forward_only_parts=(gnn_forward_only_a_s_total, gnn_forward_only_b_s_total),
        gnn_forward_only_part_labels=(f"A ({stem_a})", f"B ({stem_b})"),
    )

    if need_reg and reg_tap_diag and reg_tap_hist:
        _print_reg_tap_series_summary(
            reg_tap_hist,
            catalog,
            out_dir / f"daily_reg_tap_timeseries_{stem_a}_vs_{stem_b}.csv",
        )

    disagree_rows: list[tuple] = []
    for n, j in node_to_idx.items():
        ht = ""
        tp = dss_to_typed.get(n)
        if tp is not None:
            ht = str(tp[0])
        if disagree_scope == "load":
            if tp is None or tp[0] != "load":
                continue
        else:
            if tp is None:
                continue

        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_a[:, j]) & np.isfinite(v_b[:, j])
        if m.sum() < 2:
            continue
        vd = v_dss[m, j].astype(np.float64)
        da = v_a[m, j].astype(np.float64)
        db = v_b[m, j].astype(np.float64)
        err_a = np.abs(vd - da)
        err_b = np.abs(vd - db)
        mae_a = float(np.mean(err_a))
        mae_b = float(np.mean(err_b))
        mad_ab = float(np.mean(np.abs(da - db)))

        if juxtapose_mode == "disagree":
            disagree_rows.append((n, ht, mad_ab, float(np.max(np.abs(da - db)))))
        elif juxtapose_mode == "both_fail_dss":
            min_mae = min(mae_a, mae_b)
            max_mae = max(mae_a, mae_b)
            mean_comb = 0.5 * (mae_a + mae_b)
            disagree_rows.append((n, ht, mae_a, mae_b, min_mae, max_mae, mean_comb, mad_ab))
        else:
            min_v_dss = float(np.min(vd))
            max_v_dss = float(np.max(vd))
            disagree_rows.append((n, ht, min_v_dss, max_v_dss, mae_a, mae_b, mad_ab))

    if juxtapose_mode == "disagree":
        df_dis = pd.DataFrame(disagree_rows, columns=["node", "hetero_type", "mean_abs_diff_pu", "max_abs_diff_pu"])
        df_dis = df_dis.sort_values("mean_abs_diff_pu", ascending=False).reset_index(drop=True)
        out_csv = out_dir / f"daily_juxtapose_disagreement_{stem_a}_vs_{stem_b}.csv"
        rank_desc = "mean |A−B|"
    elif juxtapose_mode == "both_fail_dss":
        df_dis = pd.DataFrame(
            disagree_rows,
            columns=[
                "node",
                "hetero_type",
                "mae_a_vs_dss_pu",
                "mae_b_vs_dss_pu",
                "min_mae_vs_dss_pu",
                "max_mae_vs_dss_pu",
                "mean_combined_err_vs_dss_pu",
                "mean_abs_diff_ab_pu",
            ],
        )
        df_dis = df_dis.sort_values(
            ["min_mae_vs_dss_pu", "max_mae_vs_dss_pu", "mean_combined_err_vs_dss_pu"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out_csv = out_dir / f"daily_juxtapose_both_fail_vs_dss_{stem_a}_vs_{stem_b}.csv"
        rank_desc = "min MAE vs OpenDSS (both models bad when high)"
    else:
        df_dis = pd.DataFrame(
            disagree_rows,
            columns=[
                "node",
                "hetero_type",
                "min_v_open_dss_pu",
                "max_v_open_dss_pu",
                "mae_a_vs_dss_pu",
                "mae_b_vs_dss_pu",
                "mean_abs_diff_ab_pu",
            ],
        )
        df_dis = df_dis.sort_values(
            ["min_v_open_dss_pu", "max_v_open_dss_pu", "node"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        out_csv = out_dir / f"daily_juxtapose_lowest_min_v_dss_{stem_a}_vs_{stem_b}.csv"
        rank_desc = "lowest daily min OpenDSS |V| (most stressed)"

    df_dis.to_csv(out_csv, index=False)
    print(f"[compare_hetero_mv_daily] wrote ranking ({rank_desc}) -> {out_csv.resolve()}")
    if len(df_dis) == 0:
        print("[compare_hetero_mv_daily] juxtapose: no nodes matched disagree_scope with finite A/B/DSS — nothing to plot.")
        return

    top_n = min(top_disagree, len(df_dis))
    top_nodes = df_dis.head(top_n)["node"].astype(str).tolist()
    plot_set: list[str] = []
    for x in top_nodes + [str(s).strip().lower() for s in also_plot_nodes]:
        if x and x in node_to_idx and x not in plot_set:
            plot_set.append(x)

    print(f"[compare_hetero_mv_daily] plotting {len(plot_set)} nodes (top {top_n} by {rank_desc}, plus also-nodes).")
    for n in plot_set:
        j = node_to_idx[n]
        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_a[:, j]) & np.isfinite(v_b[:, j])
        mad = float(np.mean(np.abs(v_a[m, j] - v_b[m, j]))) if m.any() else np.nan
        mae_a = float(np.mean(np.abs(v_dss[m, j] - v_a[m, j]))) if m.any() else np.nan
        mae_b = float(np.mean(np.abs(v_dss[m, j] - v_b[m, j]))) if m.any() else np.nan
        fig = plt.figure(figsize=(10, 4.5))
        plt.plot(t_hours, v_dss[:, j], linewidth=2.0, label="OpenDSS")
        plt.plot(t_hours, v_a[:, j], "--", linewidth=1.6, label=f"{cfg_a} (MAE vs DSS {mae_a:.4f})")
        plt.plot(t_hours, v_b[:, j], "-.", linewidth=1.6, label=f"{cfg_b} (MAE vs DSS {mae_b:.4f})")
        plt.xlabel("Hour of day")
        plt.ylabel("Voltage magnitude (pu)")
        if juxtapose_mode == "disagree":
            plt.title(f"24h @ {n}   mean|A−B|={mad:.5f} pu")
            fn = out_dir / f"daily_juxtapose_{stem_a}_vs_{stem_b}_{n.replace('.', '_')}.png"
        elif juxtapose_mode == "both_fail_dss":
            mm = min(mae_a, mae_b)
            plt.title(f"24h @ {n}   min(MAE vs DSS)={mm:.5f}   mean|A−B|={mad:.5f} pu")
            fn = out_dir / f"daily_juxtapose_both_fail_vs_dss_{stem_a}_vs_{stem_b}_{n.replace('.', '_')}.png"
        else:
            min_v = float(np.min(v_dss[m, j])) if m.any() else float("nan")
            plt.title(
                f"24h @ {n}   min OpenDSS |V|={min_v:.5f} pu   MAE_A={mae_a:.4f} MAE_B={mae_b:.4f}   mean|A−B|={mad:.5f} pu"
            )
            fn = out_dir / f"daily_juxtapose_lowest_min_v_dss_{stem_a}_vs_{stem_b}_{n.replace('.', '_')}.png"
        plt.ylim(ymin, ymax)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if monitoring_plots_subfolders:
            plot_dir = out_dir / "monitoring_plots" / str(n).replace(".", "_")
            plot_dir.mkdir(parents=True, exist_ok=True)
            fn = plot_dir / fn.name
        plt.savefig(fn, dpi=160)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    print("\nSaved:", out_dir.resolve())
    print(df_dis.head(min(15, len(df_dis))).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="8500 daily OpenDSS vs hetero checkpoint comparison")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to hetero *_best.pt checkpoint")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"),
    )
    p.add_argument(
        "--node-index",
        type=Path,
        default=Path("datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("gnn2_daily_compare_8500_output"))
    p.add_argument("--nodes", type=str, default="m1026891.1,m1026891.2,m1026891.3", help="Comma-separated nodes to plot")
    p.add_argument("--npts", type=int, default=288)
    p.add_argument(
        "--daily-profile",
        type=str,
        default="5minDayShape.csv",
        metavar="CSV",
        help="Load-shape file under 8500-node/ (e.g. 5minDayShape.csv, 5minDayShape2.csv, 5minDayShape3.csv) or absolute path.",
    )
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--ymin", type=float, default=0.85)
    p.add_argument("--ymax", type=float, default=1.10)
    p.add_argument(
        "--mv-sx-mapping",
        type=Path,
        default=None,
        help="MV↔sx leaf pairing CSV (default: 8500-node/mv_x_sx_node_mapping_8500.csv). Required for nonzero load P/Q.",
    )
    p.add_argument(
        "--vs-checkpoint",
        type=Path,
        default=None,
        help="Second checkpoint: run one OpenDSS trajectory and juxtapose both GNNs; rank nodes by mean |V_a−V_b| and plot top-K.",
    )
    p.add_argument(
        "--top-disagree",
        type=int,
        default=10,
        help="With --vs-checkpoint: how many highest-disagreement nodes to plot (default 10).",
    )
    p.add_argument(
        "--disagree-scope",
        choices=("load", "hetero_all"),
        default="load",
        help="With --vs-checkpoint: rank disagreement only among hetero load nodes (default) or all hetero node types.",
    )
    p.add_argument(
        "--also-nodes",
        type=str,
        default="",
        help="With --vs-checkpoint: comma-separated extra nodes to plot in addition to top-disagree.",
    )
    p.add_argument(
        "--no-reg-tap-diag",
        action="store_true",
        help="Disable regulator tap diagnostics (catalog vs map, training CSV ranges, OpenDSS key coverage, per-edge time series).",
    )
    p.add_argument(
        "--juxtapose-mode",
        choices=("disagree", "both-fail-dss", "lowest-min-v-dss"),
        default="disagree",
        help="With --vs-checkpoint: disagree = |V_a−V_b|; both-fail-dss = min(MAE vs DSS); lowest-min-v-dss = lowest daily min OpenDSS |V| (stressed nodes).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="STR",
        help="GNN inference device: auto, cpu, or cuda (default: use env GNN_COMPARE_DEVICE or auto).",
    )
    p.set_defaults(show_plots=True)
    p.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="Display matplotlib figures (default).",
    )
    p.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Save PNGs only; do not call plt.show().",
    )
    p.add_argument(
        "--monitoring-plots-subfolders",
        action="store_true",
        default=False,
        help="Save per-node 24h plots under out_dir/monitoring_plots/<node>/ (default: False).",
    )
    args = p.parse_args()

    mv_path = args.mv_sx_mapping.resolve() if args.mv_sx_mapping else None
    reg_diag = not bool(args.no_reg_tap_diag)

    if args.vs_checkpoint is not None:
        _jm = str(args.juxtapose_mode)
        if _jm == "both-fail-dss":
            jmode: Literal["disagree", "both_fail_dss", "lowest_min_v_dss"] = "both_fail_dss"
        elif _jm == "lowest-min-v-dss":
            jmode = "lowest_min_v_dss"
        else:
            jmode = "disagree"
        run_compare_juxtapose(
            checkpoint_a=args.checkpoint.resolve(),
            checkpoint_b=args.vs_checkpoint.resolve(),
            dataset_dir=args.dataset_dir.resolve(),
            node_index=args.node_index.resolve(),
            out_dir=args.out_dir.resolve(),
            npts=int(args.npts),
            step_min=int(args.step_min),
            ymin=float(args.ymin),
            ymax=float(args.ymax),
            mv_sx_mapping=mv_path,
            daily_profile_csv=str(args.daily_profile),
            top_disagree=int(args.top_disagree),
            disagree_scope=str(args.disagree_scope),
            also_plot_nodes=[x.strip() for x in str(args.also_nodes).split(",") if x.strip()],
            reg_tap_diag=reg_diag,
            juxtapose_mode=jmode,
            device=args.device,
            show_plots=args.show_plots,
            monitoring_plots_subfolders=args.monitoring_plots_subfolders,
        )
    else:
        run_compare(
            checkpoint=args.checkpoint.resolve(),
            dataset_dir=args.dataset_dir.resolve(),
            node_index=args.node_index.resolve(),
            out_dir=args.out_dir.resolve(),
            plot_nodes=[x.strip() for x in args.nodes.split(",") if x.strip()],
            npts=int(args.npts),
            step_min=int(args.step_min),
            ymin=float(args.ymin),
            ymax=float(args.ymax),
            mv_sx_mapping=mv_path,
            daily_profile_csv=str(args.daily_profile),
            reg_tap_diag=reg_diag,
            device=args.device,
            show_plots=args.show_plots,
            monitoring_plots_subfolders=args.monitoring_plots_subfolders,
        )


if __name__ == "__main__":
    main()

