"""Offline physics verification: PyTorch Y-bus vs OpenDSS snapshot (optional).

Used by ``nonunique.ipynb`` section 6 and ``tests/test_pf_physics_loss.py`` patterns.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent

S_BASE_KVA_DEFAULT = 5000.0
KV_BASE_DEFAULT = 12.47

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

NODE_FEATURE_COLS: tuple[str, ...] = ("p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar")

HETERO_MV_NODES_REL = (
    Path("Heterogenous GNN dataset") / "nodes" / "hetero_mv_nodes_load_transformer.csv"
)
NODES8500_REL = (
    Path("datasets_gnn2_from pc") / "loadtype_8500" / "gnn_node_features_and_targets.csv"
)


def z_base_ohm(*, s_base_kva: float = S_BASE_KVA_DEFAULT, kv_base: float = KV_BASE_DEFAULT) -> float:
    return (float(kv_base) * 1000.0) ** 2 / (float(s_base_kva) * 1000.0)


def opendss_available() -> bool:
    try:
        import opendssdirect  # noqa: F401

        return True
    except Exception:
        return False


def summarize_abs_kw(x: np.ndarray) -> dict[str, float]:
    """Robust stats on |ΔP| or |ΔQ| in kW/kvar (masked MV nodes)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0.0, "max": float("nan"), "mean": float("nan"), "p95": float("nan"), "median": float("nan")}
    return {
        "n": float(a.size),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "median": float(np.median(a)),
    }


def nodal_power_kw_from_yv(
    v_re: np.ndarray,
    v_im: np.ndarray,
    y_re: np.ndarray,
    y_im: np.ndarray,
    *,
    s_base_kva: float = S_BASE_KVA_DEFAULT,
    v_scale_volts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``S = V · conj(Y V)`` → (P_kW, Q_kvar).

    Physical: pass ``v_scale_volts`` to convert pu ``V`` to volts; ``Y`` in Siemens; divide by 1000.
    Legacy: per-unit ``V`` and ``Y``; multiply by ``s_base_kva``.
    """
    if v_scale_volts is not None:
        vs = np.asarray(v_scale_volts, dtype=np.float64).reshape(-1)
        v_re = v_re * vs
        v_im = v_im * vs
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    if v_scale_volts is not None:
        return s_re / 1000.0, s_im / 1000.0
    s_base = float(s_base_kva)
    return s_re * s_base, s_im * s_base


def balance_residual_kw(
    p_inj_kw: np.ndarray,
    q_inj_kvar: np.ndarray,
    p_yv_kw: np.ndarray,
    q_yv_kvar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return p_inj_kw - p_yv_kw, q_inj_kvar - q_yv_kvar


@dataclass
class SnapshotState:
    sample_id: int
    n_nodes: int
    node_to_local: dict[str, int]
    v_re: np.ndarray
    v_im: np.ndarray
    p_inj_kw: np.ndarray
    q_inj_kvar: np.ndarray
    p_load_kw: np.ndarray
    q_load_kvar: np.ndarray
    p_pv_kw: np.ndarray
    q_pv_kvar: np.ndarray
    tap_pu: np.ndarray
    cap_on: np.ndarray
    mask_mv: np.ndarray
    y_re_line: np.ndarray
    y_im_line: np.ndarray
    y_re_full: np.ndarray
    y_im_full: np.ndarray
    meta_row: dict[str, Any]
    pf_data_root: Path
    v_scale_volts: np.ndarray | None = None
    use_physical_units: bool = True


def resolve_verify_data_roots(
    *,
    repo: Path | None = None,
    pf_data_root: Path | None = None,
    chunk_parent: Path | None = None,
) -> tuple[Path, Path]:
    """Return (repo, pf_data_root) for dailyagg snapshot verification."""
    repo = (repo or REPO).resolve()
    if pf_data_root is not None and pf_data_root.is_dir():
        return repo, pf_data_root.resolve()
    from gnn2_pf_data_paths import resolve_pf_catalog_paths

    _, _, root = resolve_pf_catalog_paths(
        repo=repo,
        preferred_root=None,
        chunk_parent=chunk_parent,
    )
    return repo, root.resolve()


def load_snapshot_state(
    sample_id: int,
    *,
    repo: Path | None = None,
    pf_data_root: Path | None = None,
    chunk_parent: Path | None = None,
    s_base_kva: float = S_BASE_KVA_DEFAULT,
    kv_base: float = KV_BASE_DEFAULT,
    use_physical_units: bool = True,
    pf_bus_kv_base_csv: Path | None = None,
) -> SnapshotState:
    """Load one dailyagg snapshot into numpy arrays (label V, controls, feature injections)."""
    import pandas as pd
    import torch

    import train_da_gps_multitask_complex_voltage_gine as pfmod

    repo, data_root = resolve_verify_data_roots(
        repo=repo, pf_data_root=pf_data_root, chunk_parent=chunk_parent
    )
    from gnn2_pf_data_paths import resolve_nodes_pv_csv

    idx_path = data_root / "gnn_node_index_master.csv"
    edges_path = data_root / "gnn_edges_phase_static.csv"
    meta_path = data_root / "gnn_sample_meta.csv"
    het_path = data_root / HETERO_MV_NODES_REL
    nodes_pv_csv = resolve_nodes_pv_csv(
        repo=repo, data_root=data_root, chunk_parent=chunk_parent
    )
    for p in (idx_path, edges_path, meta_path, het_path, nodes_pv_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Missing snapshot CSV: {p}")

    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1
    z_base = z_base_ohm(s_base_kva=s_base_kva, kv_base=kv_base)

    v_scale: np.ndarray | None = None
    if use_physical_units:
        from gnn2_pf_bus_kv import load_or_build_bus_kv_tensors

        v_scale, _, _ = load_or_build_bus_kv_tensors(
            repo=repo,
            data_root=data_root,
            node_to_local=ntl,
            n_nodes=n_nodes,
            cache_csv=pf_bus_kv_base_csv,
        )

    z_for_reg = None if use_physical_units else z_base
    reg_cols = list(TARGET_REG_COLS)
    cap_cols = list(TARGET_CAP_COLS)
    reg_catalog = data_root / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
    reg_edges = pfmod._load_regulator_edges_for_pf(reg_catalog, ntl, reg_cols, z_for_reg)
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    if use_physical_units:
        y_re_b, y_im_b = pfmod._build_ybus_siemens_from_edge_csv(
            edges_path, ntl, n_nodes, skip_undirected=skip
        )
    else:
        y_re_b, y_im_b = pfmod._build_ybus_pu_from_edge_csv(
            edges_path, ntl, n_nodes, z_base, skip_undirected=skip
        )
    cap_banks = pfmod._resolve_cap_bus_nodes(
        cap_cols,
        ntl,
        cap_nodes_csv=data_root / "capacitor_involved_nodes.csv",
        meta_csv=meta_path,
        capacitors_dss=repo / "8500-node" / "Capacitors.dss",
    )

    meta = pd.read_csv(meta_path)
    mrow = meta[meta["sample_id"] == int(sample_id)].iloc[0]
    tap = np.array([float(mrow[c]) for c in reg_cols], dtype=np.float64)
    cap_on = np.array([float(mrow[c]) for c in cap_cols], dtype=np.float64)
    tap_t = torch.tensor(tap, dtype=torch.float32).unsqueeze(0)
    cap_t = torch.tensor(cap_on, dtype=torch.float32).unsqueeze(0)
    y_re_f, y_im_f = pfmod._ybus_with_predicted_controls(
        y_re_b,
        y_im_b,
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        tap_pu=tap_t,
        cap_on=cap_t,
        s_base_kva=float(s_base_kva),
        batch_size=1,
        use_physical_units=use_physical_units,
        v_scale_volts=torch.tensor(v_scale, dtype=torch.float32) if v_scale is not None else None,
    )

    het = pd.read_csv(het_path)
    het = het[het["sample_id"] == int(sample_id)]
    pv = pd.read_csv(nodes_pv_csv, usecols=["sample_id", "node", "p_pv_kw", "q_pv_kvar"])
    pv = pv[pv["sample_id"] == int(sample_id)]
    pv_map = {
        str(r["node"]).strip().lower(): (float(r["p_pv_kw"]), float(r["q_pv_kvar"]))
        for _, r in pv.iterrows()
    }

    p_load = np.zeros(n_nodes, dtype=np.float64)
    q_load = np.zeros(n_nodes, dtype=np.float64)
    p_pv = np.zeros(n_nodes, dtype=np.float64)
    q_pv = np.zeros(n_nodes, dtype=np.float64)
    v_re = np.zeros(n_nodes, dtype=np.float64)
    v_im = np.zeros(n_nodes, dtype=np.float64)
    for _, row in het.iterrows():
        ni = int(row["node_idx"])
        ang = np.deg2rad(float(row["vang_deg"]))
        mag = float(row["vmag_pu"])
        v_re[ni] = mag * np.cos(ang)
        v_im[ni] = mag * np.sin(ang)
        p_load[ni] = float(row["p_load_kw"])
        q_load[ni] = float(row["q_load_kvar"])
        key = str(row["node"]).strip().lower()
        if key in pv_map:
            p_pv[ni], q_pv[ni] = pv_map[key]

    p_inj = p_pv - p_load
    q_inj = -q_pv - q_load

    dist_path = data_root / "electrical_distance_from_substation.csv"
    mask = np.zeros(n_nodes, dtype=bool)
    dist = pd.read_csv(dist_path)
    for _, row in dist.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl or pfmod._is_pf_slack_source_node(node):
            continue
        if float(row["electrical_distance_ohm"]) > 1e-9:
            mask[int(ntl[node])] = True

    return SnapshotState(
        sample_id=int(sample_id),
        n_nodes=n_nodes,
        node_to_local=ntl,
        v_re=v_re,
        v_im=v_im,
        p_inj_kw=p_inj,
        q_inj_kvar=q_inj,
        p_load_kw=p_load,
        q_load_kvar=q_load,
        p_pv_kw=p_pv,
        q_pv_kvar=q_pv,
        tap_pu=tap,
        cap_on=cap_on,
        mask_mv=mask,
        y_re_line=y_re_b.numpy(),
        y_im_line=y_im_b.numpy(),
        y_re_full=y_re_f[0].numpy(),
        y_im_full=y_im_f[0].numpy(),
        v_scale_volts=v_scale,
        use_physical_units=use_physical_units,
        meta_row={str(k): mrow[k] for k in mrow.index},
        pf_data_root=data_root,
    )


def apply_random_perturbation(
    snap: SnapshotState,
    *,
    rng: np.random.Generator,
    sigma_v_ri: float = 0.0,
    sigma_tap: float = 0.0,
    flip_cap_prob: float = 0.0,
    s_base_kva: float = S_BASE_KVA_DEFAULT,
    kv_base: float = KV_BASE_DEFAULT,
) -> SnapshotState:
    """Return a shallow copy with optional random V / controls (for stress tests)."""
    import torch

    import train_da_gps_multitask_complex_voltage_gine as pfmod

    v_re = snap.v_re.copy()
    v_im = snap.v_im.copy()
    if sigma_v_ri > 0:
        v_re = v_re + rng.normal(0.0, sigma_v_ri, size=v_re.shape)
        v_im = v_im + rng.normal(0.0, sigma_v_ri, size=v_im.shape)

    tap = snap.tap_pu.copy()
    if sigma_tap > 0:
        tap = np.clip(tap + rng.normal(0.0, sigma_tap, size=tap.shape), 0.9, 1.1)

    cap_on = snap.cap_on.copy()
    if flip_cap_prob > 0:
        flip = rng.random(cap_on.shape) < float(flip_cap_prob)
        cap_on = np.where(flip, 1.0 - cap_on, cap_on)

    repo = REPO
    z_base = z_base_ohm(s_base_kva=s_base_kva, kv_base=kv_base)
    reg_cols = list(TARGET_REG_COLS)
    cap_cols = list(TARGET_CAP_COLS)
    reg_catalog = snap.pf_data_root / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
    z_for_reg = None if snap.use_physical_units else z_base
    reg_edges = pfmod._load_regulator_edges_for_pf(reg_catalog, snap.node_to_local, reg_cols, z_for_reg)
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    edges_path = snap.pf_data_root / "gnn_edges_phase_static.csv"
    if snap.use_physical_units:
        y_re_b, y_im_b = pfmod._build_ybus_siemens_from_edge_csv(
            edges_path, snap.node_to_local, snap.n_nodes, skip_undirected=skip
        )
    else:
        y_re_b, y_im_b = pfmod._build_ybus_pu_from_edge_csv(
            edges_path, snap.node_to_local, snap.n_nodes, z_base, skip_undirected=skip
        )
    cap_banks = pfmod._resolve_cap_bus_nodes(
        cap_cols,
        snap.node_to_local,
        cap_nodes_csv=snap.pf_data_root / "capacitor_involved_nodes.csv",
        meta_csv=snap.pf_data_root / "gnn_sample_meta.csv",
        capacitors_dss=repo / "8500-node" / "Capacitors.dss",
    )
    y_re_f, y_im_f = pfmod._ybus_with_predicted_controls(
        y_re_b,
        y_im_b,
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        tap_pu=torch.tensor(tap, dtype=torch.float32).unsqueeze(0),
        cap_on=torch.tensor(cap_on, dtype=torch.float32).unsqueeze(0),
        s_base_kva=float(s_base_kva),
        batch_size=1,
        use_physical_units=snap.use_physical_units,
        v_scale_volts=(
            torch.tensor(snap.v_scale_volts, dtype=torch.float32)
            if snap.v_scale_volts is not None
            else None
        ),
    )
    out = SnapshotState(**{**snap.__dict__})
    out.v_re = v_re
    out.v_im = v_im
    out.tap_pu = tap
    out.cap_on = cap_on
    out.y_re_full = y_re_f[0].numpy()
    out.y_im_full = y_im_f[0].numpy()
    return out


def pytorch_verify_at_state(
    snap: SnapshotState,
    *,
    use_full_y: bool = True,
    s_base_kva: float = S_BASE_KVA_DEFAULT,
) -> dict[str, Any]:
    """Run PyTorch-side YV and balance checks; return per-node residuals (MV mask)."""
    y_re = snap.y_re_full if use_full_y else snap.y_re_line
    y_im = snap.y_im_full if use_full_y else snap.y_im_line
    p_yv, q_yv = nodal_power_kw_from_yv(
        snap.v_re,
        snap.v_im,
        y_re,
        y_im,
        s_base_kva=s_base_kva,
        v_scale_volts=snap.v_scale_volts if snap.use_physical_units else None,
    )
    dp, dq = balance_residual_kw(snap.p_inj_kw, snap.q_inj_kvar, p_yv, q_yv)
    m = snap.mask_mv
    return {
        "p_yv_kw": p_yv,
        "q_yv_kvar": q_yv,
        "dp_kw": dp,
        "dq_kvar": dq,
        "abs_dp_mv": np.abs(dp[m]),
        "abs_dq_mv": np.abs(dq[m]),
        "stats_p": summarize_abs_kw(np.abs(dp[m])),
        "stats_q": summarize_abs_kw(np.abs(dq[m])),
        "use_full_y": use_full_y,
    }


def _norm_device_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _norm_reg_tap_stem(col: str) -> str | None:
    m = re.match(r"^reg_(.+)_tap_pu$", str(col).strip(), flags=re.IGNORECASE)
    return _norm_device_stem(m.group(1)) if m else None


def _norm_cap_stem(col: str) -> str | None:
    m = re.match(r"^cap_(.+)_n_steps_on$", str(col).strip(), flags=re.IGNORECASE)
    return _norm_device_stem(m.group(1)) if m else None


def _controls_dict_from_snapshot(snap: SnapshotState) -> tuple[dict[str, float], dict[str, float]]:
    reg_by_stem = {_norm_reg_tap_stem(c): float(snap.tap_pu[i]) for i, c in enumerate(TARGET_REG_COLS)}
    cap_by_stem = {_norm_cap_stem(c): float(snap.cap_on[i]) for i, c in enumerate(TARGET_CAP_COLS)}
    return reg_by_stem, cap_by_stem


def _pq_from_ckt_total_powers(pwr) -> tuple[float, float] | None:
    if pwr is None or len(pwr) < 2:
        return None
    if len(pwr) == 2:
        return (-float(pwr[0]), -float(pwr[1]))
    p_sum, q_sum = 0.0, 0.0
    n_pair = (len(pwr) // 2) * 2
    for i in range(0, n_pair, 2):
        p_sum += float(pwr[i])
        q_sum += float(pwr[i + 1])
    return (-p_sum, -q_sum)


def opendss_solve_snapshot(
    snap: SnapshotState,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """OpenDSS snapshot solve at meta load/PV scale; return nodal injections and voltages.

    Loads/PV are applied with the sample's ``effective_total_scale`` and irradiance from
    ``m_loadshape`` / ``t_index`` in meta (same recipe as dailyagg generation). Capacitor
  shunt is in OpenDSS elements and in PyTorch ``Y`` — not added to ``P_inj`` on either side.
    """
    if not opendss_available():
        raise RuntimeError("opendssdirect is not installed")

    import opendssdirect as dss

    from nonunique_opendss_daily import (
        DailySimConfig,
        apply_explicit_loads_pv,
        collect_load_bases,
        collect_pv_bases,
        compile_and_setup,
        detach_daily_loadshape,
        inject_controller_warmstart,
        neutralize_irrad_loadshape,
    )
    from run_da_gps_daily_opendss_compare import align_da_gps_trajectory_to_opendss_names
    from run_injection_dataset import build_load_device_maps, build_pv_device_maps, get_all_bus_phase_nodes

    repo = (repo or REPO).resolve()
    cfg = DailySimConfig(repo_root=repo)
    meta = snap.meta_row
    t_index = int(meta.get("t_index", 0))
    m_eff = float(meta.get("effective_total_scale", meta.get("m_loadshape", 1.0)))
    ir_t = float(meta.get("m_loadshape", 1.0))

    os.chdir(cfg.grid_dir)
    compile_and_setup(cfg, snapshot=True)
    detach_daily_loadshape()
    neutralize_irrad_loadshape(cfg)

    reg_names = sorted(dss.RegControls.AllNames())
    cap_names = sorted(dss.Capacitors.AllNames())
    pv_names = sorted(dss.PVsystems.AllNames())

    reg_traj = snap.tap_pu.reshape(1, -1)
    cap_traj = snap.cap_on.reshape(1, -1)
    reg_map = align_da_gps_trajectory_to_opendss_names(reg_names, list(TARGET_REG_COLS), reg_traj)
    cap_map = align_da_gps_trajectory_to_opendss_names(cap_names, list(TARGET_CAP_COLS), cap_traj)
    inject_controller_warmstart(0, reg_names, cap_names, reg_map, cap_map, cap_threshold=0.5)

    load_names, base_kw, base_kvar = collect_load_bases()
    _, pv_base = collect_pv_bases()
    apply_explicit_loads_pv(load_names, base_kw, base_kvar, pv_names, pv_base, m_eff, ir_t)

    dss.Text.Command("set controlmode=off")
    dss.Solution.Solve()
    converged = bool(dss.Solution.Converged())

    node_names, _, _, bus_to_phases = get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = build_load_device_maps(bus_to_phases)
    _, pv_to_dss, pv_to_busph = build_pv_device_maps()

    busph_p_load: dict[tuple[str, int], float] = {}
    busph_q_load: dict[tuple[str, int], float] = {}
    for nm in loads_dss:
        dss.Circuit.SetActiveElement(f"Load.{nm}")
        pq = _pq_from_ckt_total_powers(dss.CktElement.TotalPowers())
        if pq is None:
            continue
        p_tot, q_tot = pq
        dev_key = _norm_device_stem(nm)
        for bus, ph, w in dev_to_busph_load.get(dev_key, []):
            key = (str(bus).strip().lower(), int(ph))
            busph_p_load[key] = busph_p_load.get(key, 0.0) + p_tot * float(w)
            busph_q_load[key] = busph_q_load.get(key, 0.0) + q_tot * float(w)

    busph_p_pv: dict[tuple[str, int], float] = {}
    busph_q_pv: dict[tuple[str, int], float] = {}
    for raw in pv_names:
        dss.PVsystems.Name(raw)
        try:
            p_inj = float(dss.PVsystems.kW())
            q_inj = float(dss.PVsystems.kvar())
        except Exception:
            p_inj, q_inj = 0.0, 0.0
        if abs(p_inj) <= 1e-3:
            dss.Circuit.SetActiveElement(f"PVSystem.{raw}")
            got = _pq_from_ckt_total_powers(dss.CktElement.TotalPowers())
            if got is not None:
                p_inj, q_inj = got
        pv_key = _norm_device_stem(raw)
        for bus, ph, w in pv_to_busph.get(pv_key, []):
            key = (str(bus).strip().lower(), int(ph))
            busph_p_pv[key] = busph_p_pv.get(key, 0.0) + p_inj * float(w)
            busph_q_pv[key] = busph_q_pv.get(key, 0.0) + q_inj * float(w)

    p_inj_dss = np.full(snap.n_nodes, np.nan, dtype=np.float64)
    q_inj_dss = np.full(snap.n_nodes, np.nan, dtype=np.float64)
    v_re_dss = np.full(snap.n_nodes, np.nan, dtype=np.float64)
    v_im_dss = np.full(snap.n_nodes, np.nan, dtype=np.float64)

    for node, li in snap.node_to_local.items():
        parts = str(node).rsplit(".", 1)
        if len(parts) != 2:
            continue
        bus, phs = parts[0], parts[1]
        try:
            ph = int(phs)
        except ValueError:
            continue
        key = (bus, ph)
        p_load = busph_p_load.get(key, 0.0)
        q_load = busph_q_load.get(key, 0.0)
        p_pv = busph_p_pv.get(key, 0.0)
        q_pv = busph_q_pv.get(key, 0.0)
        p_inj_dss[li] = p_pv - p_load
        q_inj_dss[li] = -q_pv - q_load
        try:
            dss.Circuit.SetActiveBus(bus)
            vmag = float(dss.Bus.puVmagAngle()[0])
            vang = float(dss.Bus.puVmagAngle()[1])
            ang = np.deg2rad(vang)
            v_re_dss[li] = vmag * np.cos(ang)
            v_im_dss[li] = vmag * np.sin(ang)
        except Exception:
            pass

    return {
        "converged": converged,
        "p_inj_kw": p_inj_dss,
        "q_inj_kvar": q_inj_dss,
        "v_re": v_re_dss,
        "v_im": v_im_dss,
        "m_eff": m_eff,
        "ir_t": ir_t,
        "t_index": t_index,
        "n_node_names": len(node_names),
    }


def compare_pytorch_opendss(
    snap: SnapshotState,
    dss_out: dict[str, Any],
    *,
    s_base_kva: float = S_BASE_KVA_DEFAULT,
) -> dict[str, Any]:
    """Compare feature vs OpenDSS injections and YV power balance on MV nodes."""
    m = snap.mask_mv
    fin = np.isfinite(dss_out["p_inj_kw"][m]) & np.isfinite(dss_out["q_inj_kvar"][m])
    m_eff = m.copy()
    m_eff[m] = m[m] & fin

    d_p_inj = snap.p_inj_kw - dss_out["p_inj_kw"]
    d_q_inj = snap.q_inj_kvar - dss_out["q_inj_kvar"]

    py_label = pytorch_verify_at_state(snap, use_full_y=True, s_base_kva=s_base_kva)
    p_yv_dss_v, q_yv_dss_v = nodal_power_kw_from_yv(
        np.nan_to_num(dss_out["v_re"], nan=0.0),
        np.nan_to_num(dss_out["v_im"], nan=0.0),
        snap.y_re_full,
        snap.y_im_full,
        s_base_kva=s_base_kva,
        v_scale_volts=snap.v_scale_volts if snap.use_physical_units else None,
    )
    dp_dss_v, dq_dss_v = balance_residual_kw(dss_out["p_inj_kw"], dss_out["q_inj_kvar"], p_yv_dss_v, q_yv_dss_v)

    d_r_p = py_label["dp_kw"] - dp_dss_v
    d_r_q = py_label["dq_kvar"] - dq_dss_v

    return {
        "inj_dp_mv": np.abs(d_p_inj[m_eff]),
        "inj_dq_mv": np.abs(d_q_inj[m_eff]),
        "inj_stats_p": summarize_abs_kw(np.abs(d_p_inj[m_eff])),
        "inj_stats_q": summarize_abs_kw(np.abs(d_q_inj[m_eff])),
        "residual_gap_p_mv": np.abs(d_r_p[m_eff]),
        "residual_gap_q_mv": np.abs(d_r_q[m_eff]),
        "residual_gap_stats_p": summarize_abs_kw(np.abs(d_r_p[m_eff])),
        "residual_gap_stats_q": summarize_abs_kw(np.abs(d_r_q[m_eff])),
        "py_stats_at_label_v": py_label,
        "dss_converged": bool(dss_out.get("converged", False)),
    }
