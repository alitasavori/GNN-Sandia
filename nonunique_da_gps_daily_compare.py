"""DA-GPS vs OpenDSS **native daily QSTS** compare (notebook cell 2).

OpenDSS truth path: compile once, ``mode=daily``, sequential ``Solve()`` with warm-start
carry-forward — **not** snapshot-by-snapshot independent solves.

DA-GPS path: GNN-only inference (``skip_opendss=True``), native 288@5 min then resampled
onto the notebook ``step_min`` grid.

Default load/PV profiles: ``load_day_004.csv`` / ``irr_day_004.csv`` (reference driver).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import opendssdirect as dss
import torch

import run_daily_aggregate_dataset_8500 as rd8500
import run_injection_dataset as inj
from compare_opendss_snapshot_helpers import (
    apply_daily_march_solver_knobs,
    apply_scenario_scale_to_load_nameplates,
    compile_and_bind_parity_daily_opendss,
    prepare_parity_profiles,
)
from nonunique_da_gps import _restore_matplotlib_inline
from nonunique_opendss_daily import (
    NATIVE_NPTS,
    NATIVE_STEP_MIN,
    DA_GPS_REF_LOAD_PROFILE,
    DA_GPS_REF_PV_PROFILE,
    MONITOR_CANDIDATES,
    DailySimConfig,
    log_da_gps_device,
    make_run_result,
    read_solve_iterations,
    resample_daily_profile_2d,
    resolve_da_gps_device,
    resolve_monitor_nodes,
    tap_pu_to_tap_number,
)
from compare_mv_daily_timing import amortize_gnn_timing_to_display_grid, format_gnn_grid_log
from nonunique_plots import plot_all

DAILY_COMPARE_STYLE = [("OpenDSS daily", "-")]

def _resolve_repo_root() -> Path:
    """Prefer ``GNN2_REPO_ROOT`` (Colab), else this module's directory (local Windows)."""
    env = os.environ.get("GNN2_REPO_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parent


REPO_ROOT = _resolve_repo_root()


def load_cache_node_order(cache_pt: Path) -> list[str]:
    z = torch.load(Path(cache_pt).resolve(), map_location="cpu", weights_only=False)
    if "node_to_local" not in z:
        raise KeyError(f"{cache_pt} missing node_to_local")
    ntl = z["node_to_local"]
    node_to_local = {str(k).strip().lower(): int(v) for k, v in ntl.items()}
    return sorted(node_to_local.keys(), key=lambda k: node_to_local[k])


def _circuit_node_lut() -> dict[str, str]:
    return {str(n).strip().lower(): str(n).strip() for n in dss.Circuit.AllNodeNames()}


def filter_cache_nodes_on_circuit(cache_nodes: list[str]) -> list[str]:
    lut = _circuit_node_lut()
    out = [lut[nk] for nk in cache_nodes if nk in lut]
    miss = [nk for nk in cache_nodes if nk not in lut]
    if miss:
        print(
            f"[da_gps_daily_compare] {len(miss)} cache nodes absent from compiled circuit "
            f"(showing up to 5): {miss[:5]}",
            flush=True,
        )
    return out


def run_opendss_daily_truth(
    cfg: DailySimConfig,
    *,
    load_csv: Path,
    irr_csv: Path,
    plot_nodes: list[str],
    scenario_scale: float = 1.0,
    daily_stress: float = 0.0,
) -> dict[str, Any]:
    """Native daily march: one compile, ``npts`` sequential ``Solve()`` calls."""
    profiles = prepare_parity_profiles(
        load_csv,
        irr_csv,
        npts=cfg.npts,
        step_min=float(cfg.step_min),
        daily_stress=daily_stress,
    )
    print(
        f"[da_gps_daily_compare] OpenDSS daily: compile once -> {cfg.npts} sequential Solve() "
        f"(step_min={cfg.step_min} min, mode=daily, warm-start carry-forward)",
        flush=True,
    )
    print(
        f"[da_gps_daily_compare] profiles: load={load_csv.name} irr={irr_csv.name} "
        f"scenario_scale={scenario_scale:g}",
        flush=True,
    )
    compile_and_bind_parity_daily_opendss(
        profiles,
        npts=cfg.npts,
        step_min=float(cfg.step_min),
    )
    apply_scenario_scale_to_load_nameplates(scenario_scale)
    apply_daily_march_solver_knobs(step_min=float(cfg.step_min))

    reg_names = rd8500._discover_reg_controls()
    cap_names = rd8500._discover_capacitors()
    n_reg, n_cap = len(reg_names), len(cap_names)
    n_nodes = len(plot_nodes)

    v_dss = np.full((cfg.npts, n_nodes), np.nan, dtype=np.float64)
    va_dss = np.full((cfg.npts, n_nodes), np.nan, dtype=np.float64)
    reg_tap = np.full((cfg.npts, n_reg), np.nan, dtype=np.float64)
    cap_on = np.full((cfg.npts, n_cap), np.nan, dtype=np.float64)
    converged = np.zeros(cfg.npts, dtype=bool)
    solve_s = np.zeros(cfg.npts, dtype=np.float64)
    collect_s = np.zeros(cfg.npts, dtype=np.float64)
    control_iters = np.zeros(cfg.npts, dtype=np.int32)
    pf_iters_total = np.zeros(cfg.npts, dtype=np.int32)

    t_wall0 = time.perf_counter()
    for i in range(cfg.npts):
        t0 = time.perf_counter()
        dss.Solution.Solve()
        solve_s[i] = time.perf_counter() - t0
        converged[i] = bool(dss.Solution.Converged())
        ctrl, pf_total, _ = read_solve_iterations()
        control_iters[i] = ctrl
        pf_iters_total[i] = pf_total
        if not converged[i]:
            continue
        t1 = time.perf_counter()
        vmag, vang = inj.get_all_node_voltage_pu_and_angle_filtered(plot_nodes)
        v_dss[i, :] = np.asarray(vmag, dtype=np.float64)
        va_dss[i, :] = np.asarray(vang, dtype=np.float64)
        tap_raw = rd8500._read_reg_control_state(reg_names)
        for j, nm in enumerate(reg_names):
            reg_tap[i, j] = float(tap_raw.get(f"reg_{nm}_tap_pu", np.nan))
        cap_raw = rd8500._read_capacitor_sample_fields(cap_names)
        for j, nm in enumerate(cap_names):
            cap_on[i, j] = float(cap_raw.get(f"cap_{nm}_n_steps_on", np.nan))
        collect_s[i] = time.perf_counter() - t1

    total_wall = time.perf_counter() - t_wall0
    mode_after = ""
    try:
        mode_after = str(dss.Solution.Mode()).strip().lower()
    except Exception:
        pass
    print(
        f"[da_gps_daily_compare] OpenDSS daily finished: {int(converged.sum())}/{cfg.npts} converged, "
        f"wall={total_wall:.2f}s, Solution.Mode() after last step={mode_after!r}",
        flush=True,
    )
    return {
        "plot_nodes": plot_nodes,
        "v_dss": v_dss,
        "va_dss": va_dss,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "reg_tap": reg_tap,
        "cap_on": cap_on,
        "converged": converged,
        "solve_s": solve_s,
        "collect_s": collect_s,
        "control_iters": control_iters,
        "pf_iters_total": pf_iters_total,
        "total_wall_s": total_wall,
        "hours": cfg.hours.copy(),
    }


def run_da_gps_predictions(
    cfg: DailySimConfig,
    plot_nodes: list[str],
    *,
    device: str | None = None,
    ref_sample_index: int = 0,
    scenario_scale: float = 1.0,
    inline_backend: str | None = None,
) -> dict[str, Any]:
    """GNN-only DA-GPS inference; resample native 288@5 min onto ``cfg`` grid."""
    cwd_before = os.getcwd()
    t0 = time.perf_counter()
    try:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        repo_root = str(cfg.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        os.chdir(cfg.repo_root)
        dev = resolve_da_gps_device(device)
        log_da_gps_device(dev)
        print(
            format_gnn_grid_log(
                amortize_gnn_timing_to_display_grid(
                    display_npts=cfg.npts,
                    display_step_min=cfg.step_min,
                    internal_npts=NATIVE_NPTS,
                    internal_step_min=NATIVE_STEP_MIN,
                    gnn_setup_once_s=None,
                    gnn_per_step_s=None,
                    gnn_total_wall_s=None,
                    gnn_n_ok=None,
                )
            ),
            flush=True,
        )
        sys.modules.pop("run_da_gps_daily_opendss_compare", None)
        from run_da_gps_daily_opendss_compare import (
            align_da_gps_trajectory_to_opendss_names,
            run_da_gps_daily_voltages,
        )

        bundle = run_da_gps_daily_voltages(
            run_dir=cfg.da_gps_run_dir,
            cache_pt=cfg.da_gps_cache_pt,
            checkpoint=cfg.da_gps_checkpoint,
            node_names=plot_nodes,
            load_profile_path=str(cfg.da_gps_load_profile),
            pv_irradiance_profile_path=str(cfg.da_gps_pv_profile),
            der_max_kw=float(cfg.der_nominal_kw if cfg.include_der else 0.0),
            der_buses=str(cfg.der_bus if cfg.include_der else ""),
            der_profile_path=str(cfg.der_profile_csv if cfg.include_der else "") or None,
            npts=int(NATIVE_NPTS),
            step_min=int(NATIVE_STEP_MIN),
            scenario_scale=float(scenario_scale),
            ref_sample_index=int(ref_sample_index),
            skip_opendss=True,
            return_device_states=True,
            device=dev,
        )
        voltages_native = bundle["voltages"]
        reg_native = np.asarray(bundle["reg_tap_pu"], dtype=float)
        cap_native = np.asarray(bundle["cap_sigmoid"], dtype=float)
        reg_cols = list(bundle["reg_cols"])
        cap_cols = list(bundle["cap_cols"])

        da_hours = cfg.hours
        da_src_h = np.arange(int(NATIVE_NPTS), dtype=float) * (NATIVE_STEP_MIN / 60.0)
        if cfg.step_min == NATIVE_STEP_MIN and cfg.npts == NATIVE_NPTS:
            voltages = voltages_native
            reg_rs = reg_native
            cap_rs = cap_native
        else:
            voltages = {
                k: np.interp(da_hours, da_src_h, np.asarray(v, dtype=float))
                for k, v in voltages_native.items()
            }
            reg_rs = resample_daily_profile_2d(
                reg_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=NATIVE_NPTS,
                method="nearest",
            )
            cap_rs = resample_daily_profile_2d(
                cap_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=NATIVE_NPTS,
                method="nearest",
            )

        gnn_wall = time.perf_counter() - t0
        timing = amortize_gnn_timing_to_display_grid(
            display_npts=cfg.npts,
            display_step_min=cfg.step_min,
            internal_npts=NATIVE_NPTS,
            internal_step_min=NATIVE_STEP_MIN,
            gnn_setup_once_s=bundle.get("gnn_setup_once_s"),
            gnn_per_step_s=bundle.get("gnn_per_step_s"),
            gnn_total_wall_s=bundle.get("gnn_total_wall_s"),
            gnn_n_ok=bundle.get("n_ok"),
        )
        return {
            "voltages": voltages,
            "reg_cols": reg_cols,
            "cap_cols": cap_cols,
            "reg_native": reg_rs,
            "cap_native": cap_rs,
            "align_fn": align_da_gps_trajectory_to_opendss_names,
            "hours": da_hours,
            "gnn_wall_s": gnn_wall,
            "gnn_setup_once_s": timing["gnn_setup_once_s"],
            "gnn_per_step_s": timing["gnn_per_step_s"],
            "gnn_total_wall_s": timing["gnn_total_wall_s"],
            "n_ok": timing["n_ok"],
            "gnn_grid": timing,
        }
    finally:
        try:
            os.chdir(cwd_before)
        except Exception:
            os.chdir(cfg.grid_dir)
        if inline_backend is not None:
            _restore_matplotlib_inline(inline_backend)


def _align_gnn_devices(
    gnn: dict[str, Any],
    reg_names: list[str],
    cap_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    align = gnn["align_fn"]
    reg_by = align(reg_names, gnn["reg_cols"], gnn["reg_native"])
    cap_by = align(cap_names, gnn["cap_cols"], gnn["cap_native"])
    return reg_by, cap_by


def _monitor_volts_matrix(
    v_dss: np.ndarray,
    collect_nodes: list[str],
    monitor_nodes: list[str],
) -> np.ndarray:
    """Extract monitor-node |V| columns from a collect_nodes-aligned voltage matrix."""
    npts = v_dss.shape[0]
    out = np.full((npts, len(monitor_nodes)), np.nan, dtype=np.float64)
    for j, nk in enumerate(monitor_nodes):
        if nk in collect_nodes:
            out[:, j] = v_dss[:, collect_nodes.index(nk)]
    return out


def _monitor_voltage_mae_rmse(
    v_dss: np.ndarray,
    gnn_voltages: dict[str, np.ndarray],
    monitor_nodes: list[str],
    *,
    npts: int,
) -> tuple[float, float, dict[str, float]]:
    errs: list[float] = []
    per_node: dict[str, float] = {}
    for j, nk in enumerate(monitor_nodes):
        vg = gnn_voltages.get(nk)
        if vg is None:
            vg = gnn_voltages.get(str(nk).lower())
        if vg is None:
            continue
        vg = np.asarray(vg, dtype=np.float64)[:npts]
        vd = v_dss[:, j]
        m = np.isfinite(vd) & np.isfinite(vg)
        if not m.any():
            continue
        node_mae = float(np.mean(np.abs(vd[m] - vg[m])))
        per_node[nk] = node_mae
        errs.extend(np.abs(vd[m] - vg[m]).tolist())
    if not errs:
        return float("nan"), float("nan"), per_node
    arr = np.asarray(errs, dtype=np.float64)
    return float(np.mean(arr)), float(np.sqrt(np.mean(arr**2))), per_node


def _build_opendss_run_for_plots(
    cfg: DailySimConfig,
    dss: dict[str, Any],
    monitor_nodes: list[str],
    collect_nodes: list[str],
) -> dict[str, Any]:
    """Shape OpenDSS daily-truth arrays into the ``runs[0]`` dict expected by ``nonunique_plots``."""
    taps = np.column_stack(
        [tap_pu_to_tap_number(dss["reg_tap"][:, j]) for j in range(len(dss["reg_names"]))]
    )
    arrays = {
        "taps": taps,
        "cap_on": dss["cap_on"],
        "volts": _monitor_volts_matrix(dss["v_dss"], collect_nodes, monitor_nodes),
        "converged": dss["converged"],
        "control_iters": dss["control_iters"],
        "pf_iters_total": dss["pf_iters_total"],
        "pf_iters_most": np.zeros(cfg.npts, dtype=np.int32),
    }
    return make_run_result(
        "OpenDSS daily",
        dss["reg_names"],
        dss["cap_names"],
        monitor_nodes,
        arrays,
        wall_s=float(dss["total_wall_s"]),
        mode="daily",
    )


def print_timing_summary(
    *,
    n_ok: int,
    npts: int,
    step_min: int,
    dss_wall_s: float,
    dss_solve_s: np.ndarray,
    dss_collect_s: np.ndarray,
    gnn_wall_s: float,
    gnn_setup_once_s: float | None = None,
    gnn_per_step_s: float | None = None,
    gnn_total_wall_s: float | None = None,
    gnn_n_ok: int | None = None,
    gnn_grid: dict[str, float | int | bool] | None = None,
) -> None:
    n_ok = max(1, int(n_ok))
    npts = max(1, int(npts))
    mean_solve_ms = 1000.0 * float(np.sum(dss_solve_s[:npts])) / n_ok
    mean_collect_ms = 1000.0 * float(np.sum(dss_collect_s[:npts])) / n_ok
    dss_solve_total = float(np.sum(dss_solve_s[:npts]))
    dss_solve_per = dss_solve_total / n_ok
    gnn_ms_per = 1000.0 * gnn_wall_s / npts
    ratio = dss_wall_s / gnn_wall_s if gnn_wall_s > 1e-9 else float("nan")
    print("\n[da_gps_daily_compare] === Timing summary ===", flush=True)
    print(f"  step_min={step_min} min, npts={npts}", flush=True)
    if gnn_grid is not None:
        print(f"  {format_gnn_grid_log(gnn_grid, prefix='').strip()}", flush=True)
    print(
        f"  OpenDSS Solve() wall:         {n_ok} × {dss_solve_per:.4f}s = {dss_solve_total:.4f}s  "
        f"(compile-once not timed; loop wall incl. collect = {dss_wall_s:.2f} s)",
        flush=True,
    )
    print(f"  OpenDSS mean Solve() / step:  {mean_solve_ms:.2f} ms  (converged steps only in denominator)", flush=True)
    print(f"  OpenDSS mean collect V/step:  {mean_collect_ms:.2f} ms", flush=True)
    if (
        gnn_setup_once_s is not None
        and gnn_per_step_s is not None
        and gnn_total_wall_s is not None
        and gnn_n_ok is not None
        and gnn_n_ok > 0
    ):
        print(
            f"  DA-GPS deployment wall:       {float(gnn_setup_once_s):.4f}s + {int(gnn_n_ok)} × "
            f"{float(gnn_per_step_s):.4f}s = {float(gnn_total_wall_s):.4f}s  "
            f"(gnn_setup_once_s + npts × gnn_per_step_s = gnn_total_wall_s; per displayed step)",
            flush=True,
        )
    print(f"  DA-GPS GNN total wall:        {gnn_wall_s:.2f} s", flush=True)
    print(f"  DA-GPS mean wall / step:      {gnn_ms_per:.2f} ms  (at step_min={step_min}, npts={npts})", flush=True)
    if np.isfinite(ratio):
        print(f"  Wall speedup (OpenDSS/GNN):   {ratio:.2f}x", flush=True)


def _final_devices_one_liner(
    reg_names: list[str],
    cap_names: list[str],
    reg_tap: np.ndarray,
    cap_on: np.ndarray,
) -> str:
    taps = ", ".join(f"{nm}={int(tap_pu_to_tap_number(reg_tap[-1, j]))}" for j, nm in enumerate(reg_names))
    caps = ", ".join(f"{nm}={int(cap_on[-1, j])}" for j, nm in enumerate(cap_names))
    return f"final taps: {taps}; final cap steps ON: {caps}"


def _da_gps_compare_summary(
    *,
    cfg: DailySimConfig,
    load_csv: Path,
    irr_csv: Path,
    monitor_nodes: list[str],
    per_node_mae: dict[str, float],
    mae: float,
    rmse: float,
    dss_wall_s: float,
    gnn: dict[str, Any],
    dss: dict[str, Any],
    n_voltage_figures: int,
    n_reg_figures: int,
    n_cap_figures: int,
    n_control_iter_figures: int,
    n_pf_iter_figures: int,
) -> dict[str, Any]:
    gnn_grid = gnn.get("gnn_grid") or {}
    speedup = dss_wall_s / gnn["gnn_wall_s"] if gnn["gnn_wall_s"] > 1e-9 else float("nan")
    return {
        "mode": "da_gps_daily_compare",
        "step_min": cfg.step_min,
        "npts": cfg.npts,
        "gnn_internal_npts": int(gnn_grid.get("internal_npts", NATIVE_NPTS)),
        "gnn_internal_step_min": int(gnn_grid.get("internal_step_min", NATIVE_STEP_MIN)),
        "gnn_resampled": bool(gnn_grid.get("resampled", False)),
        "load_profile": str(load_csv),
        "pv_profile": str(irr_csv),
        "monitor_nodes": monitor_nodes,
        "overall_mae_pu": mae,
        "overall_rmse_pu": rmse,
        "per_node_mae_pu": per_node_mae,
        "dss_wall_s": dss_wall_s,
        "gnn_wall_s": gnn["gnn_wall_s"],
        "gnn_setup_once_s": gnn.get("gnn_setup_once_s"),
        "gnn_per_step_s": gnn.get("gnn_per_step_s"),
        "gnn_total_wall_s": gnn.get("gnn_total_wall_s"),
        "wall_speedup": speedup,
        "final_devices": _final_devices_one_liner(
            dss["reg_names"], dss["cap_names"], dss["reg_tap"], dss["cap_on"]
        ),
        "n_figures_total": n_voltage_figures + n_reg_figures + n_cap_figures
        + n_control_iter_figures + n_pf_iter_figures,
    }


def run_da_gps_daily_compare_and_plot(
    cfg: DailySimConfig | None = None,
    *,
    show: bool = True,
    plot_all_cache_nodes: bool = False,
    plot_all_max_nodes: int = 0,
    out_dir: Path | str | None = None,
    load_profile_path: Path | str | None = None,
    pv_profile_path: Path | str | None = None,
    ref_sample_index: int = 0,
    scenario_scale: float = 1.0,
    daily_stress: float = 0.0,
    device: str | None = None,
) -> dict[str, Any]:
    """Compare DA-GPS GNN voltages vs OpenDSS native daily march truth.

    By default, OpenDSS collection, GNN inference, MAE, and plots use the four
    ``MONITOR_CANDIDATES`` monitor nodes (same stacked-voltage / per-device suite as
    cells 0/1 via ``nonunique_plots.plot_all``). Set ``plot_all_cache_nodes=True`` to
    also run inference on all cache∩circuit nodes (full-node MAE printed; voltage plots
    remain monitor-only).
    """
    cfg = cfg or DailySimConfig()
    load_csv = Path(load_profile_path or cfg.da_gps_load_profile).resolve()
    irr_csv = Path(pv_profile_path or cfg.da_gps_pv_profile).resolve()
    out_path = Path(out_dir).resolve() if out_dir is not None else None

    inline_backend = plt.get_backend()
    cache_nodes = load_cache_node_order(cfg.da_gps_cache_pt)
    print("=" * 72)
    print("DA-GPS daily compare (OpenDSS native daily QSTS truth)")
    print(f"  step_min={cfg.step_min} min, npts={cfg.npts}")
    print(f"  load/PV profiles: {load_csv} / {irr_csv}")
    print(f"  cache .pt: {cfg.da_gps_cache_pt}")
    print(f"  checkpoint: {cfg.da_gps_checkpoint}")
    print("=" * 72)

    # Throwaway compile so monitor nodes and cache∩circuit can be resolved.
    profiles_probe = prepare_parity_profiles(
        load_csv, irr_csv, npts=cfg.npts, step_min=float(cfg.step_min), daily_stress=daily_stress
    )
    compile_and_bind_parity_daily_opendss(
        profiles_probe, npts=cfg.npts, step_min=float(cfg.step_min)
    )
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    cache_on_circuit = filter_cache_nodes_on_circuit(cache_nodes)
    print(
        f"[da_gps_daily_compare] monitor nodes ({len(monitor_nodes)}): {monitor_nodes}",
        flush=True,
    )
    print(
        f"[da_gps_daily_compare] cache x circuit nodes: {len(cache_on_circuit)} / {len(cache_nodes)}",
        flush=True,
    )
    if plot_all_cache_nodes:
        collect_nodes = cache_on_circuit
        print(
            "[da_gps_daily_compare] plot_all_cache_nodes=True: collecting/plotting cache∩circuit nodes",
            flush=True,
        )
    else:
        collect_nodes = monitor_nodes
        print(
            f"[da_gps_daily_compare] default: collect/plot monitor nodes only "
            f"(candidates={MONITOR_CANDIDATES})",
            flush=True,
        )

    dss = run_opendss_daily_truth(
        cfg,
        load_csv=load_csv,
        irr_csv=irr_csv,
        plot_nodes=collect_nodes,
        scenario_scale=scenario_scale,
        daily_stress=daily_stress,
    )

    gnn = run_da_gps_predictions(
        cfg,
        collect_nodes,
        device=device,
        ref_sample_index=ref_sample_index,
        scenario_scale=scenario_scale,
        inline_backend=inline_backend,
    )
    gnn_reg, gnn_cap = _align_gnn_devices(gnn, dss["reg_names"], dss["cap_names"])
    gnn_voltages = gnn["voltages"]

    monitor_volts = _monitor_volts_matrix(dss["v_dss"], collect_nodes, monitor_nodes)
    mae, rmse, per_node_mae = _monitor_voltage_mae_rmse(
        monitor_volts,
        gnn_voltages,
        monitor_nodes,
        npts=cfg.npts,
    )
    if np.isfinite(mae):
        print(
            f"\n[da_gps_daily_compare] Monitor-node |V| vs DA-GPS: "
            f"MAE={mae:.6f} pu  RMSE={rmse:.6f} pu",
            flush=True,
        )
        for nk, node_mae in per_node_mae.items():
            print(f"    {nk}: MAE={node_mae:.6f} pu", flush=True)
    else:
        print("\n[da_gps_daily_compare] WARNING: no overlapping finite monitor |V| points", flush=True)

    if plot_all_cache_nodes:
        mask = np.isfinite(dss["v_dss"])
        v_gnn_all = np.full_like(dss["v_dss"], np.nan)
        for j, nk in enumerate(collect_nodes):
            v = gnn_voltages.get(nk) or gnn_voltages.get(str(nk).lower())
            if v is not None:
                v_gnn_all[:, j] = np.asarray(v, dtype=np.float64)[: cfg.npts]
        full_mask = mask & np.isfinite(v_gnn_all)
        if full_mask.any():
            full_mae = float(np.mean(np.abs(dss["v_dss"][full_mask] - v_gnn_all[full_mask])))
            print(
                f"[da_gps_daily_compare] Full cache∩circuit |V| MAE={full_mae:.6f} pu "
                f"(text only; voltage plots still use monitor nodes)",
                flush=True,
            )

    dss_run = _build_opendss_run_for_plots(cfg, dss, monitor_nodes, collect_nodes)
    runs = [dss_run]
    style_lut = {lbl: (ls, lbl) for lbl, ls in DAILY_COMPARE_STYLE}

    plot_all(
        cfg,
        runs,
        style_lut,
        da_gps_voltages=gnn_voltages,
        da_gps_reg_by_name=gnn_reg,
        da_gps_cap_by_name=gnn_cap,
        da_gps_hours=gnn["hours"],
        voltage_suptitle="OpenDSS native daily QSTS vs DA-GPS (monitor |V|)",
        show=show,
    )

    print(
        f"\n[da_gps_daily_compare] OpenDSS daily truth: "
        f"{int(dss['converged'].sum())}/{cfg.npts} converged; "
        f"{_final_devices_one_liner(dss['reg_names'], dss['cap_names'], dss['reg_tap'], dss['cap_on'])}",
        flush=True,
    )

    n_reg = len(dss["reg_names"])
    n_cap = len(dss["cap_names"])
    n_voltage_figures = 1
    n_reg_figures = n_reg
    n_cap_figures = n_cap
    n_control_iter_figures = 1
    n_pf_iter_figures = 1

    n_ok = int(dss["converged"].sum())
    print_timing_summary(
        n_ok=n_ok,
        npts=cfg.npts,
        step_min=cfg.step_min,
        dss_wall_s=float(dss["total_wall_s"]),
        dss_solve_s=dss["solve_s"],
        dss_collect_s=dss["collect_s"],
        gnn_wall_s=float(gnn["gnn_wall_s"]),
        gnn_setup_once_s=gnn.get("gnn_setup_once_s"),
        gnn_per_step_s=gnn.get("gnn_per_step_s"),
        gnn_total_wall_s=gnn.get("gnn_total_wall_s"),
        gnn_n_ok=gnn.get("n_ok"),
        gnn_grid=gnn.get("gnn_grid"),
    )

    return _da_gps_compare_summary(
        cfg=cfg,
        load_csv=load_csv,
        irr_csv=irr_csv,
        monitor_nodes=monitor_nodes,
        per_node_mae=per_node_mae,
        mae=mae,
        rmse=rmse,
        dss_wall_s=float(dss["total_wall_s"]),
        gnn=gnn,
        dss=dss,
        n_voltage_figures=n_voltage_figures,
        n_reg_figures=n_reg_figures,
        n_cap_figures=n_cap_figures,
        n_control_iter_figures=n_control_iter_figures,
        n_pf_iter_figures=n_pf_iter_figures,
    )


__all__ = [
    "load_cache_node_order",
    "run_da_gps_daily_compare_and_plot",
    "run_opendss_daily_truth",
    "run_da_gps_predictions",
]
