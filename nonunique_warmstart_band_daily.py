"""Daily snapshot band from N independent random controller warm-starts per timestep."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import opendssdirect as dss

from nonunique_da_gps import align_da_gps_devices, load_da_gps_overlay
from nonunique_opendss_daily import (
    DailySimConfig,
    apply_explicit_loads_pv,
    compile_and_setup,
    detach_daily_loadshape,
    load_der_multiplier_profile,
    neutralize_irrad_loadshape,
    randomize_controllers,
    read_loadshape_mult,
    read_solve_iterations,
    resolve_monitor_nodes,
    safe_solve,
    set_der_injection,
    collect_load_bases,
    collect_pv_bases,
    node_voltages_pu,
)
from nonunique_plots import plot_warmstart_band_all


def _snapshot_step_context(cfg: DailySimConfig):
    """Compile snapshot circuit and return static handles for explicit per-step loads."""
    compile_and_setup(cfg, snapshot=True)
    detach_daily_loadshape()
    irr_mult = read_loadshape_mult("IrradDay001", cfg=cfg)
    neutralize_irrad_loadshape(cfg)
    load_names, base_kw, base_kvar = collect_load_bases()
    pv_names, pv_base = collect_pv_bases()
    load_mult = read_loadshape_mult("Day5min", cfg=cfg)
    reg_names = list(dss.RegControls.AllNames())
    cap_names = list(dss.Capacitors.AllNames())
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    return {
        "irr_mult": irr_mult,
        "load_names": load_names,
        "base_kw": base_kw,
        "base_kvar": base_kvar,
        "pv_names": pv_names,
        "pv_base": pv_base,
        "load_mult": load_mult,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "monitor_nodes": monitor_nodes,
    }


def _apply_snapshot_timestep(
    ctx: dict[str, Any],
    cfg: DailySimConfig,
    t: int,
    *,
    der_mult,
) -> None:
    apply_explicit_loads_pv(
        ctx["load_names"],
        ctx["base_kw"],
        ctx["base_kvar"],
        ctx["pv_names"],
        ctx["pv_base"],
        ctx["load_mult"][t],
        ctx["irr_mult"][t],
    )
    if cfg.include_der and der_mult is not None:
        set_der_injection(cfg, t, der_mult)
    dss.Text.Command(f"set hour={t // cfg.steps_per_hour} sec={(t % cfg.steps_per_hour) * cfg.step_sec}")
    dss.Text.Command("set mode=snapshot")


def run_warmstart_band_daily(
    cfg: DailySimConfig | None = None,
    *,
    n_warm_starts: int = 10,
    monitor_nodes: list[str] | None = None,
    include_da_gps: bool | None = None,
    plot_reg_cap: bool = True,
    plot_warmstart_lines: bool = True,
    seed: int | None = 42,
    show: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Run N independent random controller warm-starts per displayed timestep."""
    cfg = cfg or DailySimConfig()
    if include_da_gps is not None:
        cfg.include_da_gps = bool(include_da_gps)
    if monitor_nodes is not None:
        cfg.monitor_candidates = list(monitor_nodes)

    inline_backend = plt.get_backend()
    rng = np.random.default_rng(seed)
    der_mult = (
        load_der_multiplier_profile(cfg.der_profile_csv, cfg=cfg) if cfg.include_der else None
    )
    npts = int(cfg.npts)
    n_ws = int(n_warm_starts)
    if n_ws < 1:
        raise ValueError(f"n_warm_starts must be >= 1, got {n_warm_starts}")

    print(
        f"Warm-start band daily: step_min={cfg.step_min} min, npts={npts}, "
        f"n_warm_starts={n_ws} (independent random reg/cap init per solve)"
    )

    ctx = _snapshot_step_context(cfg)
    reg_names = ctx["reg_names"]
    cap_names = ctx["cap_names"]
    nodes = ctx["monitor_nodes"]
    n_mon = len(nodes)
    n_reg = len(reg_names)
    n_cap = len(cap_names)

    volts = np.full((npts, n_ws, n_mon), np.nan, dtype=float)
    taps = np.full((npts, n_ws, n_reg), np.nan, dtype=float)
    cap_on = np.full((npts, n_ws, n_cap), np.nan, dtype=float)
    converged = np.zeros((npts, n_ws), dtype=bool)
    control_iters = np.zeros((npts, n_ws), dtype=int)
    pf_iters = np.zeros((npts, n_ws), dtype=int)

    t_wall0 = time.perf_counter()
    for t in range(npts):
        dss.Basic.ClearAll()
        ctx = _snapshot_step_context(cfg)
        _apply_snapshot_timestep(ctx, cfg, t, der_mult=der_mult)
        for k in range(n_ws):
            randomize_controllers(rng)
            try:
                dss.Solution.InitSnap()
            except Exception:
                pass
            safe_solve()
            converged[t, k] = bool(dss.Solution.Converged())
            for i, nm in enumerate(reg_names):
                dss.RegControls.Name(nm)
                taps[t, k, i] = dss.RegControls.TapNumber()
            for j, nm in enumerate(cap_names):
                dss.Capacitors.Name(nm)
                cap_on[t, k, j] = sum(dss.Capacitors.States())
            volts[t, k] = node_voltages_pu(nodes)
            ctrl, pf_total, _ = read_solve_iterations()
            control_iters[t, k] = ctrl
            pf_iters[t, k] = pf_total
        if (t + 1) % max(1, npts // 6) == 0 or t == npts - 1:
            print(f"  timestep {t + 1}/{npts} done")

    wall_s = time.perf_counter() - t_wall0
    print(f"Warm-start band: wall time {wall_s:.1f}s")

    volts_min = np.nanmin(volts, axis=1)
    volts_max = np.nanmax(volts, axis=1)
    volts_mean = np.nanmean(volts, axis=1)
    taps_min = np.nanmin(taps, axis=1)
    taps_max = np.nanmax(taps, axis=1)
    cap_min = np.nanmin(cap_on, axis=1)
    cap_max = np.nanmax(cap_on, axis=1)

    overlay = load_da_gps_overlay(cfg, nodes, device=device, inline_backend=inline_backend)
    da_v, da_reg, da_cap, da_hours = align_da_gps_devices(overlay, reg_names, cap_names)

    inside_frac: dict[str, float] = {}
    if da_v is not None:
        for j, node in enumerate(nodes):
            v_da = da_v.get(node) or da_v.get(str(node).lower())
            if v_da is None:
                continue
            v_da = np.asarray(v_da, dtype=float).ravel()[:npts]
            inside = (v_da >= volts_min[:, j]) & (v_da <= volts_max[:, j])
            inside_frac[node] = float(np.mean(inside))
        if inside_frac:
            print("\nDA-GPS |V| inside OpenDSS warm-start band [min,max] per monitor:")
            for node, frac in inside_frac.items():
                print(f"  {node}: {100.0 * frac:.1f}% of timesteps inside band")

    plot_warmstart_band_all(
        cfg,
        monitor_nodes=nodes,
        reg_names=reg_names,
        cap_names=cap_names,
        volts=volts,
        volts_min=volts_min,
        volts_max=volts_max,
        taps=taps,
        taps_min=taps_min,
        taps_max=taps_max,
        cap_on=cap_on,
        cap_min=cap_min,
        cap_max=cap_max,
        da_gps_voltages=da_v,
        da_gps_reg_by_name=da_reg,
        da_gps_cap_by_name=da_cap,
        da_gps_hours=da_hours,
        n_warm_starts=n_ws,
        plot_reg_cap=plot_reg_cap,
        plot_warmstart_lines=plot_warmstart_lines,
        show=show,
    )

    n_conv = int(converged.sum())
    print(
        f"\nConvergence: {n_conv}/{npts * n_ws} solves converged "
        f"(mean control iters={control_iters.mean():.2f}, PF iters={pf_iters.mean():.2f})"
    )

    return {
        "mode": "warmstart_band_daily",
        "cfg": cfg,
        "n_warm_starts": n_ws,
        "monitor_nodes": nodes,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "volts": volts,
        "volts_min": volts_min,
        "volts_max": volts_max,
        "volts_mean": volts_mean,
        "taps": taps,
        "taps_min": taps_min,
        "taps_max": taps_max,
        "cap_on": cap_on,
        "cap_min": cap_min,
        "cap_max": cap_max,
        "converged": converged,
        "control_iters": control_iters,
        "pf_iters": pf_iters,
        "wall_s": wall_s,
        "da_gps_voltages": da_v,
        "da_gps_reg_by_name": da_reg,
        "da_gps_cap_by_name": da_cap,
        "da_gps_hours": da_hours,
        "da_gps_inside_band_frac": inside_frac,
    }
