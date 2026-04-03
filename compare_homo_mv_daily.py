"""
Daily OpenDSS vs **homogeneous** MV GNN (train_homo_gine_csv HomoGINE / HomoGCNRes).

Mirrors compare_hetero_mv_daily.run_compare outputs:
  - Timing summary (OpenDSS apply / solve / get V / homo feature build / GNN forward).
    OpenDSS solve uses per-step snapshot mode after circuit compile; optional ``torch.compile``
    on the GNN is controlled by ``GNN_TORCH_COMPILE`` (default off on Windows, on elsewhere; set ``0`` to disable).
  - Global MAE / RMSE vs OpenDSS
  - daily_mae_per_node_*.csv, daily_gnn_variation_load_nodes_*.csv
  - Per-node 24h plots, error histogram

Requires:
  - Checkpoint: ``homo_*_h{H}_L{L}_best.pt`` (state dict only) from train_homo_gine_csv.py
  - ``train_metrics.json`` in the same folder (hidden, layers, model, n_features)
  - ``feature_norm.pt`` from training (unless trained with --no_normalize; then pass feature_norm_path=None and raw features are used — not recommended)

Node features at inference match training: P/Q, q_cap, 12× (tap_pu × downstream mask from electrical distance CSV).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss

import run_injection_dataset as inj
import run_daily_aggregate_dataset_8500 as rd8500
from build_hetero_mv_edge_dataset import REGULATOR_TO_TAP_COL
from compare_gnn_inference_utils import maybe_torch_compile
from compare_mv_daily_timing import print_mv_daily_timing_summary, resolve_inference_device, sync_inference_device
from compare_opendss_snapshot_helpers import (
    force_snapshot_mode_for_compare_timing,
    reassert_snapshot_before_each_solve,
)

from train_homo_gine_csv import (
    HomoGCNRes,
    HomoGINE,
    NODE_FEAT_COLS,
    TAP_FEAT_COLS,
    _build_old_to_new,
    _load_line_edges_supervised,
)

REG_ORDER: tuple[str, ...] = tuple(TAP_FEAT_COLS)


def _resolve_dist_csv(dataset_dir: Path) -> Path:
    bundle = dataset_dir
    p_edges = bundle / "edges" / "load_electrical_distance_to_each_regulator.csv"
    p_root = bundle / "load_electrical_distance_to_each_regulator.csv"
    if p_edges.is_file():
        return p_edges
    if p_root.is_file():
        return p_root
    raise FileNotFoundError(
        f"Missing load_electrical_distance_to_each_regulator.csv under {bundle} (edges/ or root)."
    )


def _load_homo_node_order(nodes_csv: Path) -> tuple[list[str], dict[int, int]]:
    """Same ordering as train_homo_gine_csv._load_and_stack_nodes (first sample, sort by node_idx)."""
    usecols = ["sample_id", "node_idx", "node"] + list(NODE_FEAT_COLS)
    df = pd.read_csv(nodes_csv, usecols=usecols)
    sample_ids = sorted(df["sample_id"].unique().tolist())
    first = df[df["sample_id"] == sample_ids[0]].sort_values("node_idx")
    names = first["node"].astype(str).str.strip().tolist()
    node_order = first["node_idx"].to_numpy()
    old_to_new = _build_old_to_new(node_order)
    return names, old_to_new


def _load_dist_matrix(names: list[str], dist_csv: Path) -> np.ndarray:
    hdr = pd.read_csv(dist_csv, nrows=0).columns
    usecols = ["node"] + [c for c in REG_ORDER if c in hdr]
    if len(usecols) < 13:
        raise ValueError(f"Distance CSV missing regulator columns: {dist_csv}")
    df = pd.read_csv(dist_csv, usecols=usecols)
    df["_nk"] = df["node"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["_nk"], keep="first")
    idx = df.set_index("_nk")
    N = len(names)
    out = np.zeros((N, 12), dtype=np.float32)
    for i, n in enumerate(names):
        nk = str(n).strip().lower()
        if nk not in idx.index:
            continue
        row = idx.loc[nk]
        for k, reg in enumerate(REG_ORDER):
            out[i, k] = float(row[reg]) if reg in row.index else 0.0
    return out


def _tap_pu_for_meta_col(tap_raw: dict, col: str) -> float:
    if col not in tap_raw:
        col_l = col.lower()
        for k, val in tap_raw.items():
            if str(k).lower() == col_l and np.isfinite(float(val)):
                return float(val)
        cn = col_l.replace("_", "")
        for k, val in tap_raw.items():
            if str(k).lower().replace("_", "") == cn and np.isfinite(float(val)):
                return float(val)
        return 0.0
    v = tap_raw.get(col)
    return float(v) if v is not None and np.isfinite(float(v)) else 0.0


def _load_mv_sx_mapping(path: Path) -> list[dict[str, str]]:
    import csv

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
            la, lb = (sx1, sx2) if sx1 and sx2 else (lv1, lv2)
            rules.append({"mv_key": mv.lower(), "load_a": la.lower(), "load_b": lb.lower()})
    return rules


def _safe_stem(s: str) -> str:
    t = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s).strip())[:96]
    return t or "homo"


def _resolve_checkpoint_path(checkpoint: Path) -> Path:
    """
    Resolve ``checkpoint`` if moved under ``GCN/`` or ``GINE/``, or newer run folders
    (``GINE-128-4``, ``GCN-128-4``, ``GCN-64-3``) next to ``homo_mv_8500``.

    Also handles stale paths (e.g. ``homo_gine_h128_L4_best.pt`` at ``homo_mv_8500/`` root)
    when the real file lives only under ``GINE-128-4/``.
    """
    p = Path(checkpoint).resolve()
    if p.is_file():
        return p
    parent = p.parent
    name = p.name
    for sub in ("GCN", "GINE", "GINE-128-4", "GINE-64-2", "GINE-64-3", "GCN-128-4", "GCN-64-3"):
        alt = parent / sub / name
        if alt.is_file():
            return alt
    stem_l = name.lower()
    # Any homo_*_best.pt in the matching subfolder (filename / hyperparams may differ from an old default)
    if "homo_gcn" in stem_l:
        for sub in ("GCN", "GCN-128-4", "GCN-64-3"):
            d = parent / sub
            if d.is_dir():
                cands = list(d.glob("homo_gcn*_best.pt"))
                if cands:
                    return max(cands, key=lambda x: x.stat().st_mtime)
    if "homo_gine" in stem_l:
        for sub in ("GINE", "GINE-128-4", "GINE-64-2", "GINE-64-3"):
            d = parent / sub
            if d.is_dir():
                cands = list(d.glob("homo_gine*_best.pt"))
                if cands:
                    return max(cands, key=lambda x: x.stat().st_mtime)
    return p


def _resolve_meta_and_feature_norm(
    ckpt_path: Path,
    train_metrics_path: Path | None,
    feature_norm_path: Path | None,
) -> tuple[Path, Path]:
    """
    Prefer ``train_metrics.json`` / ``feature_norm.pt`` next to the checkpoint.

    If the checkpoint sits directly under ``homo_mv_8500/`` but training outputs are in
    ``GCN/``, ``GINE/``, or ``GCN-128-4`` / ``GINE-128-4`` / ``GCN-64-3``, fall back to those subfolders.
    """
    ckpt_dir = ckpt_path.parent

    if train_metrics_path:
        meta = Path(train_metrics_path).resolve()
        if not meta.is_file():
            raise FileNotFoundError(f"train_metrics_path not found: {meta}")
        fn = Path(feature_norm_path).resolve() if feature_norm_path else meta.parent / "feature_norm.pt"
        if not fn.is_file() and (meta.parent / "feature_norm.pt").is_file():
            fn = meta.parent / "feature_norm.pt"
        return meta, fn

    meta = ckpt_dir / "train_metrics.json"
    fn = Path(feature_norm_path).resolve() if feature_norm_path else ckpt_dir / "feature_norm.pt"

    if meta.is_file():
        if not fn.is_file() and (meta.parent / "feature_norm.pt").is_file():
            fn = meta.parent / "feature_norm.pt"
        return meta, fn

    grand = ckpt_dir
    name = ckpt_path.name.lower()
    if "gcn" in name:
        for sub in ("GCN-128-4", "GCN-64-3", "GCN"):
            if (grand / sub / "train_metrics.json").is_file():
                m = grand / sub / "train_metrics.json"
                f = grand / sub / "feature_norm.pt"
                return m, f
    if "gine" in name:
        for sub in ("GINE-128-4", "GINE-64-2", "GINE-64-3", "GINE"):
            if (grand / sub / "train_metrics.json").is_file():
                m = grand / sub / "train_metrics.json"
                f = grand / sub / "feature_norm.pt"
                return m, f

    raise FileNotFoundError(
        f"Need train_metrics.json next to checkpoint or under GCN/ / GINE/ / GCN-128-4/ / GINE-128-4/:\n"
        f"  tried: {meta}\n"
        f"  also: {grand / 'GCN' / 'train_metrics.json'}, {grand / 'GINE' / 'train_metrics.json'}, …"
    )


def run_compare_homo(
    checkpoint: Path,
    dataset_dir: Path,
    out_dir: Path,
    plot_nodes: list[str],
    npts: int,
    step_min: int,
    ymin: float,
    ymax: float,
    mv_sx_mapping: Path | None = None,
    nodes_csv: Path | None = None,
    train_metrics_path: Path | None = None,
    feature_norm_path: Path | None = None,
    dropout: float = 0.15,
    device: str | None = None,
    show_plots: bool = True,
    monitoring_plots_subfolders: bool = False,
) -> None:
    """
    Args:
        checkpoint: Path to ``homo_gine_h*_L*_best.pt`` (or homo_gcn_...) state dict.
        dataset_dir: ``.../Heterogenous GNN dataset`` (contains edges/, nodes/).
        nodes_csv: Defaults to ``.../nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv``.
        train_metrics_path: Defaults to ``checkpoint.parent / train_metrics.json``.
        feature_norm_path: Defaults to ``checkpoint.parent / feature_norm.pt``; if missing, inference uses raw features (warns).
        device: ``None`` uses env ``GNN_COMPARE_DEVICE`` or ``auto`` (CUDA if available else CPU). Pass ``cpu`` or ``cuda`` to force.
        show_plots: If False, save PNGs only (no ``plt.show()``).
        monitoring_plots_subfolders: If True, save per-node 24h plots under ``out_dir/monitoring_plots/<node>/``.
    """
    device = resolve_inference_device(device)
    print(
        f"[compare_homo_mv_daily] inference device: {device} "
        "(set device= or env GNN_COMPARE_DEVICE=auto|cpu|cuda)",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_requested = Path(checkpoint).resolve()
    ckpt_path = _resolve_checkpoint_path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Also tried GCN/ and GINE/ subfolders next to parent. Resolved: {ckpt_path}"
        )
    if ckpt_path != ckpt_requested:
        print(f"[compare_homo_mv_daily] resolved checkpoint -> {ckpt_path}", flush=True)

    meta_path, fn_path = _resolve_meta_and_feature_norm(ckpt_path, train_metrics_path, feature_norm_path)
    if meta_path.parent != ckpt_path.parent:
        print(f"[compare_homo_mv_daily] using train_metrics.json -> {meta_path}", flush=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_kind = str(meta.get("model", "gine")).lower()
    hidden = int(meta["hidden"])
    n_layers = int(meta["layers"])
    in_dim = int(meta["n_features"])

    use_norm = fn_path.is_file()
    if use_norm:
        fn_pack = torch.load(fn_path, map_location="cpu", weights_only=False)
        if isinstance(fn_pack, dict) and "mean" in fn_pack:
            feat_mean = torch.as_tensor(fn_pack["mean"], dtype=torch.float32).view(1, 1, -1)
            feat_std = torch.as_tensor(fn_pack["std"], dtype=torch.float32).clamp_min(1e-8).view(1, 1, -1)
        else:
            raise ValueError(f"Unexpected feature_norm.pt format: {fn_path}")
    else:
        print(
            f"[compare_homo_mv_daily] WARNING: no {fn_path.name} — running without z-score "
            "(training likely used --no_normalize)."
        )
        feat_mean = torch.zeros(1, 1, in_dim)
        feat_std = torch.ones(1, 1, in_dim)

    node_emb_dim = int(meta.get("node_emb_dim", 0))
    edge_emb_dim = int(meta.get("edge_emb_dim", 0))

    ds = Path(dataset_dir).resolve()
    edges_dir = ds / "edges"
    nodes_dir = ds / "nodes"
    if nodes_csv is None:
        nodes_csv = nodes_dir / "hetero_mv_nodes_load_transformer_reg_tap_only.csv"
    nodes_csv = Path(nodes_csv).resolve()
    if not nodes_csv.is_file():
        raise FileNotFoundError(nodes_csv)

    node_names, old_to_new = _load_homo_node_order(nodes_csv)
    N = len(node_names)
    name_to_homo = {str(n).strip().lower(): i for i, n in enumerate(node_names)}

    dist_csv = _resolve_dist_csv(ds)
    dist_mat = _load_dist_matrix(node_names, dist_csv)

    catalog_path = edges_dir / "hetero_mv_edge_catalog.csv"
    edge_index, edge_attr = _load_line_edges_supervised(catalog_path, old_to_new)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_kind == "gine":
        model = HomoGINE(
            in_dim=in_dim,
            edge_dim=2,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            num_nodes=N,
            num_edges=int(edge_index.shape[1]),
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
        ).to(device)
    else:
        model = HomoGCNRes(
            in_dim=in_dim,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            num_nodes=N,
            node_emb_dim=node_emb_dim,
        ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = maybe_torch_compile(model, label="compare_homo_mv_daily")

    cfg_stem = _safe_stem(ckpt_path.stem)
    print(f"[compare_homo_mv_daily] model={model_kind} hidden={hidden} L={n_layers} N={N} device={device}")

    rd8500._compile_8500_daily_setup()
    reg_control_names: list[str] = rd8500._discover_reg_controls()
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    print(
        "[compare_homo_mv_daily] OpenDSS: snapshot mode for each Solve() (not daily marching).",
        flush=True,
    )
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    mL = rd8500._daily_profile_5min(npts=npts)

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

    repo_root = Path(__file__).resolve().parent
    mpath = mv_sx_mapping if mv_sx_mapping is not None else (repo_root / "8500-node" / "mv_x_sx_node_mapping_8500.csv")
    mv_sx_rules: list[dict[str, str]] = _load_mv_sx_mapping(mpath) if mpath.is_file() else []
    if mv_sx_rules:
        print(f"[compare_homo_mv_daily] mv↔sx mapping: {len(mv_sx_rules)} rules from {mpath.resolve()}")
    else:
        print(f"[compare_homo_mv_daily] WARNING: no MV↔sx mapping at {mpath}")

    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)
    v_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_gnn = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)

    n_nonconv = 0
    scenario_scale = 1.0
    first_feature_diag = True
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
        x_homo = np.zeros((N, 15), dtype=np.float32)
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
                hi = name_to_homo.get(mv)
                if hi is None:
                    continue
                pa = float(node_P.get(rec["load_a"], 0.0) + node_P.get(rec["load_b"], 0.0))
                qa = float(node_Q.get(rec["load_a"], 0.0) + node_Q.get(rec["load_b"], 0.0))
                x_homo[hi, 0] = pa
                x_homo[hi, 1] = qa
        else:
            for (bus, ph), pval in busphP_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                hi = name_to_homo.get(node)
                if hi is not None:
                    x_homo[hi, 0] = float(pval)
            for (bus, ph), qval in busphQ_load.items():
                node = f"{str(bus).strip().lower()}.{int(ph)}"
                hi = name_to_homo.get(node)
                if hi is not None:
                    x_homo[hi, 1] = float(qval)

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
                    hi = name_to_homo.get(node)
                    if hi is not None:
                        x_homo[hi, 2] += q_now / 3.0
            if not dss.Capacitors.Next():
                break

        tap_raw = rd8500._read_reg_control_state(reg_control_names)
        for k, reg_name in enumerate(REG_ORDER):
            col = REGULATOR_TO_TAP_COL[reg_name]
            tv = _tap_pu_for_meta_col(tap_raw, col)
            mask = dist_mat[:, k] > 0.0
            x_homo[mask, 3 + k] = tv

        if first_feature_diag:
            first_feature_diag = False
            nz = int(np.sum(np.abs(x_homo[:, 0]) + np.abs(x_homo[:, 1]) > 1e-3))
            print(f"[compare_homo_mv_daily] feature diag (first step): nodes with |P|+|Q|>1e-3: {nz}/{N}")

        t_fb1 = time.perf_counter()
        feature_build_s_total += t_fb1 - t_fb0

        t_gnn0 = time.perf_counter()
        x_t = torch.from_numpy(x_homo).unsqueeze(0).to(device)
        if use_norm:
            x_t = (x_t - feat_mean.to(device)) / feat_std.to(device)
        x_in = x_t.squeeze(0)
        with torch.no_grad():
            t_fwd0 = time.perf_counter()
            if model_kind == "gine":
                pred = model(x_in, edge_index, edge_attr)
            else:
                pred = model(x_in, edge_index, edge_attr)
            t_fwd1 = time.perf_counter()
        gnn_forward_only_s_total += t_fwd1 - t_fwd0
        pred_np = pred.squeeze(-1).detach().cpu().numpy()

        for hi, name in enumerate(node_names):
            nk = str(name).strip().lower()
            j = node_to_idx.get(nk)
            if j is not None:
                v_gnn[i, j] = float(pred_np[hi])

        sync_inference_device(device)
        t_gnn1 = time.perf_counter()
        gnn_infer_s_total += t_gnn1 - t_gnn0

        if (i + 1) % max(1, npts // 12) == 0:
            print(
                f"[{i + 1}/{npts}] timing — apply={open_apply_s_total:.2f}s "
                f"reassert={open_reassert_s_total:.2f}s solve_only={open_solve_only_s_total:.2f}s "
                f"| get V={open_get_s_total:.2f}s | homo feat={feature_build_s_total:.2f}s "
                f"| GNN bucket={gnn_infer_s_total:.2f}s fwd-only={gnn_forward_only_s_total:.2f}s",
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
        title="Daily Timing Summary (homo MV vs OpenDSS)",
        feature_label="Homo feature build",
        log_prefix="[compare_homo_mv_daily]",
    )

    mask = np.isfinite(v_dss) & np.isfinite(v_gnn)
    mae = float(np.mean(np.abs(v_dss[mask] - v_gnn[mask])))
    rmse = float(np.sqrt(np.mean((v_dss[mask] - v_gnn[mask]) ** 2)))
    print(f"\nOverall: MAE={mae:.6f} pu  RMSE={rmse:.6f} pu  n_points={int(mask.sum())} nonconv={n_nonconv}", flush=True)

    node_rows = []
    for j, n in enumerate(all_nodes):
        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_gnn[:, j])
        if m.any():
            node_rows.append((n, float(np.mean(np.abs(v_dss[m, j] - v_gnn[m, j])))))
    df_mae = pd.DataFrame(node_rows, columns=["node", "mae"]).sort_values("mae", ascending=False)
    df_mae.to_csv(out_dir / f"daily_mae_per_node_{cfg_stem}.csv", index=False)

    _var_eps = 1e-6
    var_rows: list[tuple[str, float, float, int]] = []
    for n in node_names:
        nk = str(n).strip().lower()
        if nk not in node_to_idx:
            continue
        j = node_to_idx[nk]
        col = v_gnn[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 2:
            continue
        w = np.asarray(col[fin], dtype=np.float64)
        std = float(np.std(w))
        rng = float(np.max(w) - np.min(w))
        var_rows.append((nk, std, rng, int(fin.sum())))
    if var_rows:
        df_var = pd.DataFrame(var_rows, columns=["node", "std_pu", "range_pu", "n_finite_pts"])
        df_var = df_var.sort_values(["range_pu", "std_pu"], ascending=[False, False]).reset_index(drop=True)
        df_var.to_csv(out_dir / f"daily_gnn_variation_load_nodes_{cfg_stem}.csv", index=False)
        max_rng = float(df_var["range_pu"].max())
        if max_rng < _var_eps:
            print(
                f"\n[compare_homo_mv_daily] GNN predictions flat over day: max range ≈ {max_rng:.3e} pu",
                flush=True,
            )
        print(f"\n[compare_homo_mv_daily] top node by GNN |V| range: {df_var.iloc[0].to_dict()}", flush=True)

    for n in [str(x).strip().lower() for x in plot_nodes if str(x).strip().lower() in node_to_idx]:
        j = node_to_idx[n]
        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_gnn[:, j])
        n_mae = float(np.mean(np.abs(v_dss[m, j] - v_gnn[m, j]))) if m.any() else np.nan
        fig = plt.figure(figsize=(10, 4.2))
        plt.plot(t_hours, v_dss[:, j], linewidth=2.0, label="OpenDSS baseline")
        plt.plot(t_hours, v_gnn[:, j], "--", linewidth=1.6, label=f"homo {cfg_stem} (MAE={n_mae:.4f})")
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
            png_path = plot_dir / f"daily_compare_{cfg_stem}_{n.replace('.', '_')}.png"
        else:
            png_path = out_dir / f"daily_compare_{cfg_stem}_{n.replace('.', '_')}.png"
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
    plt.title(f"Error distribution: homo {cfg_stem} vs OpenDSS")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"daily_error_hist_{cfg_stem}.png", dpi=170)
    if show_plots:
        plt.show()
    else:
        plt.close(fig_h)

    print("\nSaved:", out_dir.resolve(), flush=True)
    print(df_mae.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    raise SystemExit("Import compare_homo_mv_daily.run_compare_homo or use compare_hetero_mv_daily_wrappers.")
