"""
Daily comparison for IEEE 8500-node feeder:
  OpenDSS vs MLP vs GNN on chosen bus-phase nodes, with timing.

This script is optimized for fast inference on the 8500 pipeline and includes:
  - pre-pinned host input buffers
  - overlapped host->device transfer + GNN compute via two CUDA streams
  - vectorized feature construction / tensor path (no per-node Python loops)

Usage (repo root):
  python compare_daily_8500_mlp_gnn.py <gnn_ckpt.pt> <mlp_ckpt.pt> --nodes 840.1 816.1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import opendssdirect as dss

import run_injection_dataset as inj

# 8500 dataset helpers (baseline element collection + bucket mapping)
import run_loadtype_dataset_8500 as loadtype8500

from train_gnn_8500 import ResidualGCN8500
from train_mlp_8500 import MLP8500


def _compile_8500_from_master_dir() -> None:
    """
    Compile 8500 master from its own directory so relative Redirect paths
    inside Master.dss (e.g., LineCodes2.dss) always resolve on Colab/local.
    """
    master_path = Path(loadtype8500.MASTER_8500).resolve()
    if not master_path.is_file():
        raise FileNotFoundError(f"Missing IEEE 8500 master: {master_path}")
    master_dir = master_path.parent

    # Linux/Colab is case-sensitive. Resolve Redirect target filename casing up front.
    master_to_run = _build_case_resolved_master_copy(master_path)

    prev_cwd = Path.cwd()
    try:
        os.chdir(master_dir)
        dss.Basic.ClearAll()
        # Use relative filename after cwd switch so nested Redirects are stable.
        dss.Text.Command(f'redirect "{master_to_run.name}"')
        dss.Solution.Mode(1)
        inj._apply_voltage_bases()
    finally:
        os.chdir(prev_cwd)


def _build_case_resolved_master_copy(master_path: Path) -> Path:
    """
    Build a temp master file with Redirect filenames rewritten to exact on-disk case.
    Returns the path to the rewritten file.
    """
    text = master_path.read_text(encoding="utf-8", errors="ignore")
    parent = master_path.parent
    name_lut = {p.name.lower(): p for p in parent.iterdir() if p.is_file()}
    out_lines: list[str] = []

    # Match full redirect line while preserving whitespace/comments around it.
    line_pat = re.compile(r"(?im)^(\s*redirect\s+)(\"?)([^\r\n\"!]+)(\"?)(.*)$")
    for line in text.splitlines(keepends=True):
        m = line_pat.match(line)
        if not m:
            out_lines.append(line)
            continue

        prefix, q1, ref, q2, suffix = m.groups()
        ref_clean = ref.strip()
        ref_path = (parent / ref_clean).resolve()
        if ref_path.is_file():
            out_lines.append(line)
            continue

        cand = name_lut.get(Path(ref_clean).name.lower())
        if cand is None:
            out_lines.append(line)
            continue

        rewritten = f"{prefix}{q1}{cand.name}{q2}{suffix}"
        # Preserve original newline style if line ended with newline.
        if line.endswith("\n") and not rewritten.endswith("\n"):
            rewritten += "\n"
        out_lines.append(rewritten)

    tmp_master = parent / "_Master_case_resolved_tmp.dss"
    tmp_master.write_text("".join(out_lines), encoding="utf-8")
    return tmp_master


def _infer_dataset_dir_from_ckpt(ckpt_path: str | os.PathLike) -> Path:
    p = Path(ckpt_path).resolve()
    for anc in [p] + list(p.parents):
        if anc.name == "loadtype_8500":
            return anc
    raise FileNotFoundError(
        f"Could not infer `datasets_gnn2/loadtype_8500` from checkpoint path: {p}\n"
        "Expected a path containing a folder named `loadtype_8500`."
    )


def _load_node_names_master(dataset_dir: Path) -> list[str]:
    node_csv = dataset_dir / "gnn_node_index_master.csv"
    if not node_csv.is_file():
        raise FileNotFoundError(f"Missing node index CSV: {node_csv}")
    df = pd.read_csv(node_csv)
    if "node_idx" not in df.columns or "node" not in df.columns:
        raise ValueError(f"Unexpected columns in {node_csv}. Need {['node_idx','node']} headers.")
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="raise").astype(int)
    df = df.sort_values("node_idx")
    nodes = df["node"].astype(str).tolist()
    return nodes


def _parse_bus_phase(node_name: str) -> tuple[str, int]:
    bus, phs = node_name.split(".")
    ph = int(phs)
    if ph not in (1, 2, 3):
        raise ValueError(f"Expected phase in {node_name!r} to be 1/2/3.")
    return bus, ph


def _read_profile_csv(csv_path: Path, npts: int) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing profile CSV: {csv_path}")
    return inj.read_profile_csv_two_col_noheader(str(csv_path), npts=npts, debug=False).astype(np.float32)


def _fill_onehot_node_type(
    *,
    X_raw: np.ndarray,
    out_onehot: np.ndarray,
    load_col_idx: np.ndarray,
    i_qcap: int,
    i_pv: int,
    i_pv_q: int,
    source_mask: np.ndarray,
    threshold: float = 1e-8,
) -> None:
    """Fill out_onehot [N,4] in-place: [load_bus, reg_bus, cap_bus, source_bus]."""
    X_abs = np.abs(X_raw)
    pv_activity = X_abs[:, i_pv] + X_abs[:, i_pv_q]
    cap_activity = X_abs[:, i_qcap]
    load_activity = X_abs[:, load_col_idx].mean(axis=1) if load_col_idx.size > 0 else np.zeros((X_raw.shape[0],), dtype=np.float32)

    is_pv = pv_activity > threshold
    is_cap = cap_activity > threshold
    is_load = load_activity > threshold
    is_reg = (~is_load) & (~is_pv) & (~is_cap) & (~source_mask)

    out_onehot.fill(0.0)
    out_onehot[:, 0] = is_load.astype(np.float32)
    out_onehot[:, 1] = is_reg.astype(np.float32)
    out_onehot[:, 2] = is_cap.astype(np.float32)
    out_onehot[:, 3] = source_mask.astype(np.float32)


def _instantiate_models(*, gnn_ckpt_path: str, mlp_ckpt_path: str, device: torch.device, dataset_dir: Path):
    # ---- Node + graph tensors (used for GNN) ----
    dataset_graph_dir = dataset_dir / "graph_tensors"
    edge_index_path = dataset_graph_dir / "edge_index.pt"
    edge_attr_path = dataset_graph_dir / "edge_attr.pt"
    if not edge_index_path.is_file():
        raise FileNotFoundError(f"Missing GNN edge_index: {edge_index_path}")
    if not edge_attr_path.is_file():
        raise FileNotFoundError(f"Missing GNN edge_attr: {edge_attr_path}")

    edge_index = torch.load(edge_index_path, map_location="cpu").long()
    edge_attr = torch.load(edge_attr_path, map_location="cpu").float()
    if edge_attr.ndim != 2 or edge_attr.shape[1] < 2:
        raise ValueError(f"Unexpected edge_attr shape: {tuple(edge_attr.shape)}. Need [E,>=2].")
    r = edge_attr[:, 0]
    x = edge_attr[:, 1]
    edge_weight = torch.sqrt(r * r + x * x).clamp_min(1e-9)

    edge_index = edge_index.to(device, non_blocking=False)
    edge_weight = edge_weight.to(device, non_blocking=False)

    # ---- GNN checkpoint ----
    gnn_ckpt = torch.load(gnn_ckpt_path, map_location="cpu", weights_only=False)
    gnn_state = gnn_ckpt.get("model_state")
    if gnn_state is None:
        raise ValueError(f"GNN checkpoint missing `model_state`: {gnn_ckpt_path}")

    gnn_in_dim = int(gnn_ckpt["in_dim"])
    gnn_hidden_dim = int(gnn_ckpt["hidden_dim"])
    gnn_num_layers = int(gnn_ckpt["num_layers"])
    gnn_dropout = float(gnn_ckpt["dropout"])
    gnn_mean = torch.as_tensor(gnn_ckpt["mean"], dtype=torch.float32)
    gnn_std = torch.as_tensor(gnn_ckpt["std"], dtype=torch.float32).clamp_min(1e-8)

    # Some checkpoints are mixed-format (e.g., model in_dim=18 but mean/std saved for 14 raw features).
    # Normalize stats to match model input dim by appending identity stats for extra onehot channels.
    gnn_stat_dim = int(gnn_mean.shape[-1])
    if gnn_stat_dim != gnn_in_dim:
        if gnn_stat_dim == 14 and gnn_in_dim == 18:
            z = torch.zeros((1, 4), dtype=torch.float32)
            o = torch.ones((1, 4), dtype=torch.float32)
            gnn_mean = torch.cat([gnn_mean.reshape(1, 14), z], dim=1)
            gnn_std = torch.cat([gnn_std.reshape(1, 14), o], dim=1)
        else:
            raise RuntimeError(
                f"Incompatible GNN checkpoint stats: in_dim={gnn_in_dim}, "
                f"mean/std dim={gnn_stat_dim}. Expected equal dims, or 14->18 expansion."
            )

    gnn_mean = gnn_mean.to(device, non_blocking=False)
    gnn_std = gnn_std.to(device, non_blocking=False)

    gnn_model = ResidualGCN8500(
        in_dim=gnn_in_dim,
        hidden_dim=gnn_hidden_dim,
        num_layers=gnn_num_layers,
        dropout=gnn_dropout,
    ).to(device)
    gnn_model.load_state_dict(gnn_state)
    gnn_model.eval()

    # ---- MLP checkpoint ----
    mlp_ckpt = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=False)
    mlp_state = mlp_ckpt.get("model_state")
    if mlp_state is None:
        raise ValueError(f"MLP checkpoint missing `model_state`: {mlp_ckpt_path}")

    mlp_in_dim = int(mlp_ckpt["in_dim"])
    mlp_out_dim = int(mlp_ckpt["out_dim"])
    mlp_hidden_dim = int(mlp_ckpt["hidden_dim"])
    mlp_num_hidden_layers = int(mlp_ckpt["num_hidden_layers"])
    mlp_mean = torch.as_tensor(mlp_ckpt["mean"], dtype=torch.float32)
    mlp_std = torch.as_tensor(mlp_ckpt["std"], dtype=torch.float32).clamp_min(1e-8)

    mlp_mean = mlp_mean.to(device, non_blocking=False)
    mlp_std = mlp_std.to(device, non_blocking=False)

    mlp_model = MLP8500(
        in_dim=mlp_in_dim,
        out_dim=mlp_out_dim,
        hidden=mlp_hidden_dim,
        num_hidden_layers=mlp_num_hidden_layers,
    ).to(device)
    mlp_model.load_state_dict(mlp_state)
    mlp_model.eval()

    return {
        "gnn_model": gnn_model,
        "gnn_edge_index": edge_index,
        "gnn_edge_weight": edge_weight,
        "gnn_mean": gnn_mean,
        "gnn_std": gnn_std,
        "mlp_model": mlp_model,
        "mlp_mean": mlp_mean,
        "mlp_std": mlp_std,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="IEEE 8500 daily comparison: OpenDSS vs MLP vs GNN (vmag only)."
    )
    ap.add_argument("gnn_ckpt_path", type=str, help="Path to gnn_8500 checkpoint (.pt).")
    ap.add_argument("mlp_ckpt_path", type=str, help="Path to mlp_8500 checkpoint (.pt).")
    ap.add_argument(
        "--nodes",
        nargs="+",
        default=["816.1"],
        help="List of bus.phase node names to plot (e.g. 840.1 848.2).",
    )
    ap.add_argument("--output-dir", type=str, default="gnn2_daily_compare_8500_output")
    ap.add_argument("--npts", type=int, default=None, help="Override number of steps (default: 288).")
    ap.add_argument("--step-min", type=int, default=5, help="Minutes per timestep for x-axis.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = _infer_dataset_dir_from_ckpt(args.gnn_ckpt_path)

    node_names_master = _load_node_names_master(dataset_dir)
    N = len(node_names_master)

    # ---- validate selected nodes exist ----
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    chosen_node_names = list(args.nodes)
    chosen_indices: list[int] = []
    for n in chosen_node_names:
        if n not in node_to_idx:
            raise ValueError(
                f"Node {n!r} not found in dataset node index ({dataset_dir}/gnn_node_index_master.csv)."
            )
        chosen_indices.append(node_to_idx[n])

    npts = int(args.npts) if args.npts is not None else int(inj.NPTS)

    # ---- load models + graph tensors ----
    models = _instantiate_models(
        gnn_ckpt_path=args.gnn_ckpt_path,
        mlp_ckpt_path=args.mlp_ckpt_path,
        device=device,
        dataset_dir=dataset_dir,
    )

    gnn_model: ResidualGCN8500 = models["gnn_model"]
    mlp_model: MLP8500 = models["mlp_model"]
    edge_index = models["gnn_edge_index"]
    edge_weight = models["gnn_edge_weight"]
    gnn_mean = models["gnn_mean"]  # [1, F_raw]
    gnn_std = models["gnn_std"]
    mlp_mean = models["mlp_mean"]  # [1, in_dim_flat]
    mlp_std = models["mlp_std"]

    # Feature column order must match `assemble_dataset_tensors_8500.DEFAULT_FEATURE_COLS`.
    # Keep it duplicated here to avoid importing the full module graph at runtime.
    feature_cols_8500: tuple[str, ...] = (
        "electrical_distance_ohm",
        "m1_p_kw",
        "m1_q_kvar",
        "m2_p_kw",
        "m2_q_kvar",
        "m4_p_kw",
        "m4_q_kvar",
        "m5_p_kw",
        "m5_q_kvar",
        "q_cap_kvar",
        "p_pv_kw",
        "q_pv_kvar",
        "p_sys_balance_kw",
        "q_sys_balance_kvar",
    )
    F_raw = len(feature_cols_8500)  # expected 14

    # ---- OpenDSS setup for IEEE 8500 ----
    _compile_8500_from_master_dir()
    base_loads, base_pvs = loadtype8500._collect_baselines()

    # Precompute load -> bucket -> bus-phase distribution.
    load_infos: list[dict] = []
    P_load_base_total = 0.0
    Q_load_base_total = 0.0
    for row in base_loads:
        name = row["name"]
        kw = float(row["kw"])
        kvar = float(row["kvar"])
        bkt = int(loadtype8500._model_bucket(row["model"]))
        busph = loadtype8500._busph_fracs_load(name)
        # Some models might end up with empty busph due to unusual definitions; skip safely.
        if not busph:
            continue
        load_infos.append({"name": name, "kw": kw, "kvar": kvar, "bkt": bkt, "busph": busph})
        P_load_base_total += kw
        Q_load_base_total += kvar

    # Precompute PV maps for setting Pmpp and reading actual P/Q by bus-phase.
    pv_to_dss, pv_to_busph = loadtype8500._build_pv_maps(base_pvs)
    pv_base_by_key = {str(pv["name"]).strip().lower(): float(pv["pmpp"]) for pv in base_pvs}

    # ---- profile multipliers (day) ----
    repo_root = Path(__file__).resolve().parent
    csv_dir = repo_root / "new dss from dr mirzaei"
    csvL = csv_dir / "5minDayShape.csv"
    csvPV = csv_dir / "5MinuteIrradiance.csv"
    mL = _read_profile_csv(csvL, npts=npts)
    mPV = _read_profile_csv(csvPV, npts=npts)

    # Pre-split node bus/phase and fast index mapping.
    node_bus = [None] * N
    node_phase = [0] * N
    busph_to_idx: dict[tuple[str, int], int] = {}
    for i, node in enumerate(node_names_master):
        bus, ph = _parse_bus_phase(node)
        node_bus[i] = bus
        node_phase[i] = ph
        busph_to_idx[(bus, ph)] = i

    # Precompute per-load sparse index arrays so per-step updates use np.add.at (vectorized).
    for li in load_infos:
        idx = np.asarray([busph_to_idx[(bus, ph)] for (bus, ph, _w) in li["busph"]], dtype=np.int64)
        w = np.asarray([float(_w) for (_bus, _ph, _w) in li["busph"]], dtype=np.float32)
        li["idx"] = idx
        li["w"] = w

    # Onehot precompute helpers.
    source_mask = np.array([str(n).lower().startswith("sourcebus") for n in node_names_master], dtype=bool)
    col_to_i = {c: i for i, c in enumerate(feature_cols_8500)}
    i_qcap = col_to_i["q_cap_kvar"]
    i_pv = col_to_i["p_pv_kw"]
    i_pv_q = col_to_i["q_pv_kvar"]
    load_col_idx = np.asarray(
        [col_to_i[c] for c in feature_cols_8500 if c.startswith("m") and (c.endswith("_kw") or c.endswith("_kvar"))],
        dtype=np.int64,
    )

    # Reusable numpy work buffers (vectorized feature path).
    x_raw_np = np.zeros((N, F_raw), dtype=np.float32)
    x_onehot_np = np.zeros((N, 4), dtype=np.float32)
    gnn_in_dim_effective = int(gnn_mean.shape[-1])
    x_in_np = np.zeros((N, gnn_in_dim_effective), dtype=np.float32)
    m1_p_np = np.zeros((N,), dtype=np.float32)
    m1_q_np = np.zeros((N,), dtype=np.float32)
    m2_p_np = np.zeros((N,), dtype=np.float32)
    m2_q_np = np.zeros((N,), dtype=np.float32)
    m4_p_np = np.zeros((N,), dtype=np.float32)
    m4_q_np = np.zeros((N,), dtype=np.float32)
    m5_p_np = np.zeros((N,), dtype=np.float32)
    m5_q_np = np.zeros((N,), dtype=np.float32)
    p_pv_np = np.zeros((N,), dtype=np.float32)
    q_pv_np = np.zeros((N,), dtype=np.float32)

    # Normalization arrays in CPU numpy for vectorized preprocessing.
    gnn_mean_np = gnn_mean.detach().cpu().numpy().reshape(1, -1).astype(np.float32, copy=False)
    gnn_std_np = gnn_std.detach().cpu().numpy().reshape(1, -1).astype(np.float32, copy=False)
    # Pre-pinned host input buffers + device ping-pong buffers for async H2D.
    gnn_in_dim = int(gnn_mean.shape[-1])
    use_cuda_pipeline = device.type == "cuda"
    if use_cuda_pipeline:
        host_x_gnn = torch.empty((N, gnn_in_dim), dtype=torch.float32, pin_memory=True)
        host_x_mlp = torch.empty((N * F_raw,), dtype=torch.float32, pin_memory=True)
        dev_x_gnn = [
            torch.empty((N, gnn_in_dim), dtype=torch.float32, device=device),
            torch.empty((N, gnn_in_dim), dtype=torch.float32, device=device),
        ]
        h2d_stream = torch.cuda.Stream(device=device)
        compute_stream = torch.cuda.Stream(device=device)
        copy_done_events = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]
    else:
        host_x_gnn = None
        host_x_mlp = None
        dev_x_gnn = None
        h2d_stream = None
        compute_stream = None
        copy_done_events = None

    # ---- run daily simulation ----
    os.makedirs(args.output_dir, exist_ok=True)
    t_hours = [t * float(args.step_min) / 60.0 for t in range(npts)]

    v_open = np.full((npts, len(chosen_indices)), np.nan, dtype=np.float32)
    v_mlp = np.full((npts, len(chosen_indices)), np.nan, dtype=np.float32)
    v_gnn = np.full((npts, len(chosen_indices)), np.nan, dtype=np.float32)

    open_apply_s_total = 0.0
    open_solve_s_total = 0.0
    open_get_s_total = 0.0

    mlp_infer_s_total = 0.0
    gnn_infer_s_total = 0.0
    feature_build_s_total = 0.0

    for t in range(npts):
        mL_t = float(mL[t])
        mPV_t = float(mPV[t])

        # ----- OpenDSS apply (loads + PV) + pre-solve feature context -----
        t_apply0 = time.perf_counter()

        busph_per_type = {m: ({}, {}) for m in (1, 2, 4, 5)}  # type -> (busphP, busphQ)
        P_load_total_kw = P_load_base_total * mL_t
        Q_load_total_kvar = Q_load_base_total * mL_t

        for li in load_infos:
            name = li["name"]
            kw = float(li["kw"]) * mL_t
            kvar = float(li["kvar"]) * mL_t
            dss.Loads.Name(name)
            dss.Loads.kW(kw)
            dss.Loads.kvar(kvar)

            bkt = int(li["bkt"])
            busphP_dict, busphQ_dict = busph_per_type[bkt]
            for (bus, ph, w) in li["busph"]:
                busphP_dict[(bus, ph)] = busphP_dict.get((bus, ph), 0.0) + kw * w
                busphQ_dict[(bus, ph)] = busphQ_dict.get((bus, ph), 0.0) + kvar * w

        # PV: only adjust Pmpp; actual Q comes from solve/VoltVar.
        for pv_key, dss_name in pv_to_dss.items():
            base_pmpp = float(pv_base_by_key[pv_key])
            dss.PVsystems.Name(dss_name)
            dss.PVsystems.Pmpp(base_pmpp * mPV_t)

        t_apply1 = time.perf_counter()
        open_apply_s_total += t_apply1 - t_apply0

        # ----- OpenDSS solve -----
        t_solve0 = time.perf_counter()
        dss.Solution.Solve()
        t_solve1 = time.perf_counter()
        open_solve_s_total += t_solve1 - t_solve0

        if not dss.Solution.Converged():
            continue

        # ----- collect OpenDSS voltages at chosen nodes + PV actual P/Q -----
        t_get0 = time.perf_counter()
        v_keep, _ = inj.get_all_node_voltage_pu_and_angle_filtered(chosen_node_names)
        # PV actual P/Q by bus-phase
        busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)

        v_open[t, :] = np.asarray(v_keep, dtype=np.float32)

        sum_p_pv_act = float(sum(busphP_pv_actual.values()))
        sum_q_pv_act = float(sum(busphQ_pv_actual.values()))

        p_sys_balance_kw = P_load_total_kw - sum_p_pv_act
        q_sys_balance_kvar = Q_load_total_kvar + sum_q_pv_act
        t_get1 = time.perf_counter()
        open_get_s_total += t_get1 - t_get0

        # ----- build X_raw for all N nodes (vectorized) -----
        t_fb0 = time.perf_counter()
        m1_p_np.fill(0.0)
        m1_q_np.fill(0.0)
        m2_p_np.fill(0.0)
        m2_q_np.fill(0.0)
        m4_p_np.fill(0.0)
        m4_q_np.fill(0.0)
        m5_p_np.fill(0.0)
        m5_q_np.fill(0.0)
        p_pv_np.fill(0.0)
        q_pv_np.fill(0.0)

        # Load features from sparse load->node contributions.
        for li in load_infos:
            idx = li["idx"]
            w = li["w"]
            kw = float(li["kw"]) * mL_t
            kvar = float(li["kvar"]) * mL_t
            bkt = int(li["bkt"])
            if bkt == 1:
                np.add.at(m1_p_np, idx, kw * w)
                np.add.at(m1_q_np, idx, kvar * w)
            elif bkt == 2:
                np.add.at(m2_p_np, idx, kw * w)
                np.add.at(m2_q_np, idx, kvar * w)
            elif bkt == 4:
                np.add.at(m4_p_np, idx, kw * w)
                np.add.at(m4_q_np, idx, kvar * w)
            elif bkt == 5:
                np.add.at(m5_p_np, idx, kw * w)
                np.add.at(m5_q_np, idx, kvar * w)

        # PV actual terms (post-solve).
        for (bus, ph), val in busphP_pv_actual.items():
            i = busph_to_idx.get((bus, ph))
            if i is not None:
                p_pv_np[i] += float(val)
        for (bus, ph), val in busphQ_pv_actual.items():
            i = busph_to_idx.get((bus, ph))
            if i is not None:
                q_pv_np[i] += float(val)

        # Compose X_raw in one vectorized pass.
        x_raw_np.fill(0.0)
        x_raw_np[:, 1] = m1_p_np
        x_raw_np[:, 2] = m1_q_np
        x_raw_np[:, 3] = m2_p_np
        x_raw_np[:, 4] = m2_q_np
        x_raw_np[:, 5] = m4_p_np
        x_raw_np[:, 6] = m4_q_np
        x_raw_np[:, 7] = m5_p_np
        x_raw_np[:, 8] = m5_q_np
        x_raw_np[:, 10] = p_pv_np
        x_raw_np[:, 11] = q_pv_np
        x_raw_np[:, 12] = np.float32(p_sys_balance_kw)
        x_raw_np[:, 13] = np.float32(q_sys_balance_kvar)

        t_fb1 = time.perf_counter()
        feature_build_s_total += t_fb1 - t_fb0

        # ----- MLP inference -----
        t_mlp0 = time.perf_counter()
        x_flat = x_raw_np.reshape(N * F_raw)
        if use_cuda_pipeline:
            host_x_mlp.copy_(torch.from_numpy(x_flat), non_blocking=False)
            x_flat_t = host_x_mlp.to(device, non_blocking=True)
            x_flat_n = (x_flat_t - mlp_mean.squeeze(0)) / mlp_std.squeeze(0)
        else:
            x_flat_t = torch.from_numpy(x_flat).to(device)
            x_flat_n = (x_flat_t - mlp_mean.squeeze(0)) / mlp_std.squeeze(0)
        with torch.no_grad():
            pred_flat = mlp_model(x_flat_n.unsqueeze(0)).squeeze(0)  # [N]
        pred_np = pred_flat.detach().float().cpu().numpy()
        v_mlp[t, :] = pred_np[chosen_indices].astype(np.float32, copy=False)
        t_mlp1 = time.perf_counter()
        mlp_infer_s_total += t_mlp1 - t_mlp0

        # ----- GNN inference (pinned + async H2D + stream overlap) -----
        t_gnn0 = time.perf_counter()
        _fill_onehot_node_type(
            X_raw=x_raw_np,
            out_onehot=x_onehot_np,
            load_col_idx=load_col_idx,
            i_qcap=i_qcap,
            i_pv=i_pv,
            i_pv_q=i_pv_q,
            source_mask=source_mask,
            threshold=1e-8,
        )

        if gnn_in_dim_effective == F_raw:
            # Older checkpoint style: model trained on the base 14 features only.
            x_in_np[:, :] = (x_raw_np - gnn_mean_np) / gnn_std_np
        elif gnn_in_dim_effective == F_raw + 4:
            # Newer checkpoint style: base 14 features + node-type onehot.
            x_in_np[:, :F_raw] = (x_raw_np - gnn_mean_np[:, :F_raw]) / gnn_std_np[:, :F_raw]
            x_in_np[:, F_raw:] = x_onehot_np
        else:
            raise RuntimeError(
                f"Unsupported GNN input dim from checkpoint: {gnn_in_dim_effective}. "
                f"Expected {F_raw} or {F_raw + 4} for this 8500 pipeline."
            )

        if use_cuda_pipeline:
            slot = t & 1
            host_x_gnn.copy_(torch.from_numpy(x_in_np), non_blocking=False)
            with torch.cuda.stream(h2d_stream):
                dev_x_gnn[slot].copy_(host_x_gnn, non_blocking=True)
                copy_done_events[slot].record(h2d_stream)

            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(copy_done_events[slot])
                with torch.no_grad():
                    pred_vmag = gnn_model(dev_x_gnn[slot], edge_index, edge_weight=edge_weight)
            compute_stream.synchronize()
        else:
            x_in_t = torch.from_numpy(x_in_np).to(device)
            with torch.no_grad():
                pred_vmag = gnn_model(x_in_t, edge_index, edge_weight=edge_weight)

        pred_np = pred_vmag.detach().float().cpu().numpy()
        v_gnn[t, :] = pred_np[chosen_indices].astype(np.float32, copy=False)
        t_gnn1 = time.perf_counter()
        gnn_infer_s_total += t_gnn1 - t_gnn0

        # progress
        if (t + 1) % max(1, npts // 12) == 0:
            print(f"[{t+1}/{npts}] OpenDSS apply={open_apply_s_total:.2f}s so far | "
                  f"MLP infer={mlp_infer_s_total:.2f}s so far | GNN infer={gnn_infer_s_total:.2f}s so far",
                  flush=True)

    # ---- timing summary ----
    n_ok = np.isfinite(v_open[:, 0]).sum()
    def _per_step(total_s: float) -> float:
        return total_s / max(n_ok, 1)

    print("\n=== Daily Timing Summary (IEEE 8500) ===", flush=True)
    print(f"Device: {device}")
    print(f"Timesteps converged: {int(n_ok)}/{npts}")
    print(f"OpenDSS apply+feature-ctx: total {open_apply_s_total:.4f}s | mean {1000*_per_step(open_apply_s_total):.3f} ms/ok-step")
    print(f"OpenDSS solve:             total {open_solve_s_total:.4f}s | mean {1000*_per_step(open_solve_s_total):.3f} ms/ok-step")
    print(f"OpenDSS collect PV/VMag: total {open_get_s_total:.4f}s | mean {1000*_per_step(open_get_s_total):.3f} ms/ok-step")
    print(f"Feature build (X_raw):    total {feature_build_s_total:.4f}s | mean {1000*_per_step(feature_build_s_total):.3f} ms/ok-step")
    print(f"MLP inference:             total {mlp_infer_s_total:.4f}s | mean {1000*_per_step(mlp_infer_s_total):.3f} ms/ok-step")
    print(f"GNN inference:            total {gnn_infer_s_total:.4f}s | mean {1000*_per_step(gnn_infer_s_total):.3f} ms/ok-step")

    # ---- plots ----
    time_hours = np.asarray(t_hours, dtype=np.float32)
    for j, node in enumerate(chosen_node_names):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        ax.plot(time_hours, v_open[:, j], label="OpenDSS", linewidth=2)
        ax.plot(time_hours, v_mlp[:, j], label="MLP", linewidth=1.6)
        ax.plot(time_hours, v_gnn[:, j], label="GNN", linewidth=1.6)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"IEEE 8500 Daily Voltage Profile at {node}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_path = Path(args.output_dir) / f"voltage_profile_{node.replace('.','_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    print(f"\nSaved plots to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

