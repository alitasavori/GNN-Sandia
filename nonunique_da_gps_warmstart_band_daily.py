"""DA-GPS vs OpenDSS warm-start **band** (N random reg/cap inits per step) + inside-band metrics.

Uses the same load/PV profiles as ``run_da_gps_daily_opendss_compare`` (e.g. load_day_004 /
irr_day_004). OpenDSS runs snapshot solves with independent controller warm-starts; DA-GPS is
GNN-only on the same grid. Plots shaded min–max bands for |V|, regulators, capacitors, and
meta-aux scalars, with DA-GPS trajectories overlaid.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import opendssdirect as dss
import pandas as pd

from compare_gnn_inference_utils import (
    read_gnn_batch_steps,
    read_gnn_cuda_graphs,
    read_gnn_defer_d2h,
)
from compare_mv_daily_timing import amortize_gnn_timing_to_display_grid, format_gnn_grid_log
from compare_opendss_snapshot_helpers import prepare_parity_profiles
from nonunique_da_gps import align_da_gps_devices
from nonunique_da_gps_daily_compare import (
    filter_cache_nodes_on_circuit,
    load_cache_node_order,
    print_timing_summary,
)
from nonunique_opendss_daily import (
    DailySimConfig,
    apply_explicit_loads_pv,
    build_der_injection_record,
    collect_load_bases,
    collect_pv_bases,
    compile_and_setup,
    da_gps_inference_grid,
    detach_daily_loadshape,
    display_hours_array,
    load_der_multiplier_profile,
    log_da_gps_device,
    neutralize_irrad_loadshape,
    node_voltages_pu,
    randomize_controllers,
    resample_daily_profile_2d,
    resolve_da_gps_device,
    resolve_monitor_nodes,
    safe_solve,
    set_der_injection,
    tap_pu_to_tap_number,
)
from nonunique_plots import plot_warmstart_device_band, plot_warmstart_voltage_band
from run_da_gps_daily_opendss_compare import (
    _circuit_losses_kw_kvar,
    _dss_scalar_for_meta_aux_col,
    _path_str_for_png_write,
    _read_pv_totals_post_solve_kw_kvar,
    _resolve_voltage_png_dir,
    _safe_stem,
    align_da_gps_trajectory_to_opendss_names,
    run_da_gps_daily_voltages,
)


def _inside_band_fraction(y: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
    if not m.any():
        return float("nan")
    return float(np.mean((y[m] >= ymin[m]) & (y[m] <= ymax[m])))


# CRPS / interval scores need a predictive distribution or calibrated quantile fan; warm-start
# clouds are empirical min-max envelopes from N controller inits, not a full ensemble CDF.
# Band proximity gives graded credit near the cloud via smooth decay outside [lo, hi].
_EPS_BAND = 1e-9


def _band_proximity_continuous_per_step(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    *,
    min_scale: float = 1e-4,
) -> np.ndarray:
    """Per-timestep cloud proximity in [0, 1].

    Inside [ymin, ymax]: 1.  Outside: exp(-d / scale) with
    d = distance to nearest band edge and scale = max(0.5 * (ymax - ymin), min_scale).
    """
    y = np.asarray(y, dtype=np.float64)
    ymin = np.asarray(ymin, dtype=np.float64)
    ymax = np.asarray(ymax, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
    scores = np.full(y.shape, np.nan, dtype=np.float64)
    if not m.any():
        return scores
    lo = ymin[m]
    hi = ymax[m]
    yy = y[m]
    scale = np.maximum(0.5 * (hi - lo), float(min_scale))
    inside = (yy >= lo) & (yy <= hi)
    d = np.where(yy > hi, yy - hi, np.where(yy < lo, lo - yy, 0.0))
    scores[m] = np.where(inside, 1.0, np.exp(-d / scale))
    return scores


def _band_proximity_discrete_per_step(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    *,
    step_scale: float = 1.0,
) -> np.ndarray:
    """Discrete cloud proximity: 1 inside integer band; exp(-d / step_scale) outside.

    ``y`` is already discretized (tap # or cap steps ON). ``step_scale`` is one tap step
    or the capacitor bank step count for normalization.
    """
    y = np.asarray(y, dtype=np.float64)
    ymin = np.asarray(ymin, dtype=np.float64)
    ymax = np.asarray(ymax, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
    scores = np.full(y.shape, np.nan, dtype=np.float64)
    if not m.any():
        return scores
    lo = ymin[m]
    hi = ymax[m]
    yy = y[m]
    scale = max(float(step_scale), 1.0)
    inside = (yy >= lo) & (yy <= hi)
    d = np.where(yy > hi, yy - hi, np.where(yy < lo, lo - yy, 0.0))
    scores[m] = np.where(inside, 1.0, np.exp(-d / scale))
    return scores


def _mean_band_proximity(scores: np.ndarray) -> float:
    m = np.isfinite(scores)
    if not m.any():
        return float("nan")
    return float(np.mean(scores[m]))


def _band_outside_distance_per_step(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """Per-timestep distance to nearest band edge when outside; 0 when inside.

    Uses the same edge distance ``d`` as cloud proximity (before ``exp(-d/scale)``).
    Units follow ``y``: pu for voltage/meta aux, tap steps for regulators, cap
    steps ON for capacitors.
    """
    y = np.asarray(y, dtype=np.float64)
    ymin = np.asarray(ymin, dtype=np.float64)
    ymax = np.asarray(ymax, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
    dist = np.full(y.shape, np.nan, dtype=np.float64)
    if not m.any():
        return dist
    lo = ymin[m]
    hi = ymax[m]
    yy = y[m]
    d = np.where(yy > hi, yy - hi, np.where(yy < lo, lo - yy, 0.0))
    dist[m] = d
    return dist


def _mean_finite_dict(items: dict[str, float]) -> float:
    """Mean over finite values in a per-device metric dict."""
    vals = [float(v) for v in items.values() if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _aggregate_group_band_metrics(
    inside: dict[str, float],
    proximity: dict[str, float],
    outside_dist: dict[str, float],
    set_dist: dict[str, float],
) -> dict[str, float | int]:
    """Mean inside-band frac, cloud proximity, set distance, and outside distance."""
    return {
        "n_devices": len(inside),
        "mean_inside_band_frac": _mean_finite_dict(inside),
        "mean_cloud_proximity": _mean_finite_dict(proximity),
        "mean_set_distance": _mean_finite_dict(set_dist),
        "mean_outside_distance": _mean_finite_dict(outside_dist),
    }


def _build_aggregated_band_metrics(
    inside: dict[str, dict[str, float]],
    proximity: dict[str, dict[str, float]],
    outside_dist: dict[str, dict[str, float]],
    set_dist: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    """Aggregated means for voltage, regulator, capacitor, and meta_aux groups."""
    groups = ("voltage", "regulator", "capacitor", "meta_aux")
    out: dict[str, dict[str, float | int]] = {}
    for group in groups:
        ib = inside.get(group) or {}
        if not ib:
            continue
        out[group] = _aggregate_group_band_metrics(
            ib,
            proximity.get(group) or {},
            outside_dist.get(group) or {},
            set_dist.get(group) or {},
        )
    return out


_GROUP_UNITS = {
    "voltage": "pu",
    "regulator": "tap steps",
    "capacitor": "cap steps ON",
    "meta_aux": "pu/kW/kvar",
}


def _print_aggregated_band_summary(aggregated: dict[str, dict[str, float | int]]) -> None:
    """Print group-level means before per-device breakdown."""
    if not aggregated:
        return
    print("\n=== DA-GPS aggregated warm-start band metrics (all devices) ===")
    for group, stats in aggregated.items():
        n = int(stats.get("n_devices", 0))
        frac = float(stats.get("mean_inside_band_frac", float("nan")))
        prox = float(stats.get("mean_cloud_proximity", float("nan")))
        sdist = float(stats.get("mean_set_distance", float("nan")))
        odist = float(stats.get("mean_outside_distance", float("nan")))
        unit = _GROUP_UNITS.get(group, "")
        frac_s = f"{100.0 * frac:.1f}%" if np.isfinite(frac) else "n/a"
        prox_s = f"{prox:.3f}" if np.isfinite(prox) else "n/a"
        sdist_s = f"{sdist:.4g}" if np.isfinite(sdist) else "n/a"
        odist_s = f"{odist:.4g}" if np.isfinite(odist) else "n/a"
        print(
            f"  [{group}] n={n}  inside={frac_s}  "
            f"cloud_proximity={prox_s}  set_distance={sdist_s}  "
            f"mean_outside_distance={odist_s} {unit}",
            flush=True,
        )


def _mean_set_distance(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
) -> float:
    """Mean distance to nearest point in the valid set [ymin, ymax].

    Per timestep: 0 inside the band; otherwise distance to the nearest edge.
    Averaged over **all** valid timesteps (inside steps contribute 0). This is
    the set-distance / nearest-valid-solution error when the warm-start cloud
    defines the feasible interval.
    """
    dist = _band_outside_distance_per_step(y, ymin, ymax)
    m = np.isfinite(dist)
    if not m.any():
        return float("nan")
    return float(np.mean(dist[m]))


def _mean_outside_distance(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
) -> float:
    """Mean distance to nearest band edge over **outside-only** timesteps.

    Returns 0.0 when every valid timestep is inside the band; ``nan`` when no
    valid timesteps exist. ``frac_outside`` is ``1 - inside_band_frac``.
    """
    y = np.asarray(y, dtype=np.float64)
    ymin = np.asarray(ymin, dtype=np.float64)
    ymax = np.asarray(ymax, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
    if not m.any():
        return float("nan")
    lo = ymin[m]
    hi = ymax[m]
    yy = y[m]
    inside = (yy >= lo) & (yy <= hi)
    if not np.any(~inside):
        return 0.0
    d = np.where(yy > hi, yy - hi, np.where(yy < lo, lo - yy, 0.0))
    return float(np.mean(d[~inside]))


def _band_proximity_continuous(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    *,
    min_scale: float = 1e-4,
) -> float:
    return _mean_band_proximity(
        _band_proximity_continuous_per_step(y, ymin, ymax, min_scale=min_scale)
    )


def _band_proximity_discrete(
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    *,
    step_scale: float = 1.0,
) -> float:
    return _mean_band_proximity(
        _band_proximity_discrete_per_step(y, ymin, ymax, step_scale=step_scale)
    )


CAP_ON_THRESHOLD = 0.5


def _discretize_reg_tap_pu(y: np.ndarray) -> np.ndarray:
    """Map reg-head tap (pu) to integer OpenDSS TapNumber before inside-band checks."""
    return np.rint(tap_pu_to_tap_number(np.asarray(y, dtype=np.float64))).astype(np.float64)


def _discretize_cap_sigmoid(
    y: np.ndarray,
    *,
    n_steps: int,
    threshold: float = CAP_ON_THRESHOLD,
) -> np.ndarray:
    """Threshold cap sigmoid to bank steps ON (0 or ``n_steps``), matching OpenDSS ``sum(States())``."""
    y = np.asarray(y, dtype=np.float64)
    n = max(0, int(n_steps))
    if n <= 0:
        return np.zeros_like(y, dtype=np.float64)
    return np.where(y >= float(threshold), float(n), 0.0).astype(np.float64)


def _cap_num_steps_by_name(cap_names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for nm in cap_names:
        try:
            dss.Capacitors.Name(nm)
            n = max(0, int(dss.Capacitors.NumSteps()))
        except Exception:
            n = 1
        out[str(nm)] = n
        out[str(nm).lower()] = n
    return out


def _dict_lookup(d: dict[str, Any] | None, key: str) -> Any | None:
    """Case-insensitive dict get without ``or`` (numpy arrays break boolean ``or``)."""
    if not d:
        return None
    for k in (key, str(key).lower(), str(key)):
        if k in d:
            return d[k]
    return None


def _log_gnn_inference_opts(*, device: str, gnn_batch_steps: int | None) -> None:
    """Log the same CUDA inference knobs used by ``run_da_gps_daily_opendss_compare``."""
    import torch

    dev = torch.device(device)
    batch_k = read_gnn_batch_steps(gnn_batch_steps)
    print(
        "[da_gps_warmstart_band] GNN inference opts (same as daily compare): "
        f"device={device}  batch_steps={batch_k}  "
        f"defer_d2h={read_gnn_defer_d2h()}  cuda_graphs={read_gnn_cuda_graphs(device=dev)}  "
        f"(env: GNN_BATCH_STEPS, GNN_DEFER_D2H, GNN_CUDA_GRAPHS, GNN_TORCH_COMPILE)",
        flush=True,
    )


def _print_warmstart_envelope_timing(
    *,
    npts: int,
    n_warm_starts: int,
    dss_step_wall_s: np.ndarray,
    dss_solve_only_s: np.ndarray,
) -> None:
    """Extra OpenDSS lines beyond the daily-compare-style snapshot-Solve summary."""
    n_ws = int(n_warm_starts)
    step_wall = np.asarray(dss_step_wall_s, dtype=np.float64).ravel()[:npts]
    solve_only = np.asarray(dss_solve_only_s, dtype=np.float64).reshape(npts, n_ws)
    n_ok_steps = int(np.sum(np.isfinite(step_wall) & (step_wall > 0)))
    n_ok_steps = max(1, n_ok_steps)
    mean_step_wall_ms = 1000.0 * float(np.nansum(step_wall)) / n_ok_steps
    conv = np.isfinite(solve_only) & (solve_only > 0)
    n_ok_solves = int(conv.sum())
    mean_snap_solve_ms = 1000.0 * float(np.nansum(solve_only[conv])) / max(1, n_ok_solves)
    print("\n[da_gps_warmstart_band] === Warm-start envelope overhead ===", flush=True)
    print(
        f"  OpenDSS display-step wall (compile + {n_ws} warm-starts/step): "
        f"mean {mean_step_wall_ms:.1f} ms/step",
        flush=True,
    )
    print(
        f"  OpenDSS snapshot solves in band: {n_ok_solves} total, "
        f"mean {mean_snap_solve_ms:.1f} ms/solve",
        flush=True,
    )
    print(
        "  Note: Colab 3–4× speedup used **daily QSTS** Solve(); snapshot solves above are "
        "a different OpenDSS mode but comparable GNN deployment timers.",
        flush=True,
    )


def _checkpoint_stem(cfg: DailySimConfig) -> str:
    ck = Path(cfg.da_gps_checkpoint)
    return _safe_stem(ck.parent.name or ck.stem)


def _stem_short(cfg_stem: str, *, n: int = 20) -> str:
    s = _safe_stem(cfg_stem)
    return (s[:n] if s else "ckpt")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_path_str_for_png_write(path), index=False)


def _write_json_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path_str_for_png_write(path), "w", encoding="utf-8") as f:
        f.write(text)


def _warmstart_voltage_png_basename(
    *,
    stem_safe: str,
    rk: str,
    tot: str,
    inside_frac: float,
    node_fn: str,
) -> str:
    pct = f"{100.0 * inside_frac:.1f}" if np.isfinite(inside_frac) else "nan"
    st = stem_safe[:24] if stem_safe else "ckpt"
    return f"vws_{st}_r{rk}_o{tot}_in{pct}pct_{node_fn}.png"


def _save_voltage_band_png(
    path: Path,
    cfg: DailySimConfig,
    *,
    node: str,
    j: int,
    volts: np.ndarray,
    volts_min: np.ndarray,
    volts_max: np.ndarray,
    n_warm_starts: int,
    inside_frac: float,
    rank_human: int,
    n_plot_ranked: int,
    da_gps_voltages: dict[str, np.ndarray] | None,
    da_gps_hours: np.ndarray | None,
    plot_warmstart_lines: bool,
    dpi: int,
    fig_w: float,
    fig_h: float,
    show: bool,
) -> None:
    from nonunique_plots import _plot_npts, _trim_1d, _trim_step_matrix

    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    v_lo = _trim_step_matrix(volts_min, n)[:, j]
    v_hi = _trim_step_matrix(volts_max, n)[:, j]
    ax.fill_between(hours, v_lo, v_hi, alpha=0.25, color="steelblue", label="OpenDSS band [min,max]")
    if plot_warmstart_lines:
        for k in range(int(volts.shape[1])):
            ax.plot(
                hours,
                _trim_step_matrix(volts, n)[:, k, j],
                color="steelblue",
                alpha=0.12,
                linewidth=0.8,
            )
    if da_gps_voltages is not None and da_gps_hours is not None:
        v_da = _dict_lookup(da_gps_voltages, node)
        if v_da is not None:
            lab = f"DA-GPS inside={100.0 * inside_frac:.1f}%" if np.isfinite(inside_frac) else "DA-GPS"
            ax.plot(
                _trim_1d(da_gps_hours, n),
                _trim_1d(v_da, n),
                label=lab,
                linestyle="-.",
                color="magenta",
                linewidth=1.5,
            )
    pct = f"{100.0 * inside_frac:.1f}%" if np.isfinite(inside_frac) else "n/a"
    ax.set_title(
        f"24h @ {node}  |  inside-band rank {rank_human}/{n_plot_ranked} "
        f"(worst→best; {pct} inside warm-start band)"
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("|V| (pu)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_path_str_for_png_write(path), dpi=int(dpi))
    if show:
        plt.show()
    plt.close(fig)


def _save_device_band_png(
    path: Path,
    cfg: DailySimConfig,
    *,
    device_name: str,
    values: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
    idx: int,
    ylabel: str,
    title_prefix: str,
    n_warm_starts: int,
    inside_frac: float,
    da_gps_by_name: dict[str, np.ndarray] | None,
    da_gps_hours: np.ndarray | None,
    da_transform=None,
    plot_warmstart_lines: bool,
    dpi: int,
    show: bool,
) -> None:
    from nonunique_plots import _plot_npts, _trim_1d, _trim_step_matrix

    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    lo = _trim_step_matrix(vmin, n)[:, idx]
    hi = _trim_step_matrix(vmax, n)[:, idx]
    ax.fill_between(hours, lo, hi, alpha=0.25, color="steelblue", label="OpenDSS band [min,max]")
    if plot_warmstart_lines:
        for k in range(int(values.shape[1])):
            ax.plot(
                hours,
                _trim_step_matrix(values, n)[:, k, idx],
                color="steelblue",
                alpha=0.12,
                linewidth=0.8,
            )
    if da_gps_by_name is not None and da_gps_hours is not None:
        y_da = _dict_lookup(da_gps_by_name, device_name)
        if y_da is not None and np.isfinite(y_da).any():
            y_plot = _trim_1d(y_da, n)
            if da_transform is not None:
                y_plot = da_transform(y_plot)
            lab = f"DA-GPS inside={100.0 * inside_frac:.1f}%" if np.isfinite(inside_frac) else "DA-GPS"
            ax.plot(
                _trim_1d(da_gps_hours, n),
                y_plot,
                label=lab,
                linestyle="-.",
                color="magenta",
                linewidth=1.5,
            )
    pct = f"{100.0 * inside_frac:.1f}%" if np.isfinite(inside_frac) else "n/a"
    ax.set_title(f"{title_prefix} {device_name} ({n_warm_starts} warm starts/step; inside {pct})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_path_str_for_png_write(path), dpi=int(dpi))
    if show:
        plt.show()
    plt.close(fig)


def _meta_dss_row(meta_cols: list[str], pv_names: list[str]) -> np.ndarray:
    p_loss_kw, q_loss_kvar = _circuit_losses_kw_kvar()
    pv_tot = _read_pv_totals_post_solve_kw_kvar(pv_names)
    out = np.full(len(meta_cols), np.nan, dtype=np.float64)
    for j, col in enumerate(meta_cols):
        v = _dss_scalar_for_meta_aux_col(
            col, pv_totals=pv_tot, p_loss_kw=p_loss_kw, q_loss_kvar=q_loss_kvar
        )
        if v is not None and np.isfinite(v):
            out[j] = float(v)
    return out


def _snapshot_parity_context(
    cfg: DailySimConfig,
    *,
    load_csv: Path,
    irr_csv: Path,
    daily_stress: float,
):
    profiles = prepare_parity_profiles(
        load_csv,
        irr_csv,
        npts=cfg.npts,
        step_min=float(cfg.step_min),
        daily_stress=float(daily_stress),
    )
    compile_and_setup(cfg, snapshot=True)
    detach_daily_loadshape()
    neutralize_irrad_loadshape(cfg)
    load_names, base_kw, base_kvar = collect_load_bases()
    pv_names, pv_base = collect_pv_bases()
    reg_names = list(dss.RegControls.AllNames())
    cap_names = list(dss.Capacitors.AllNames())
    cap_num_steps = _cap_num_steps_by_name(cap_names)
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    return {
        "m_load": np.asarray(profiles.m_eff, dtype=np.float64),
        "m_irr": np.asarray(profiles.m_irr, dtype=np.float64),
        "load_names": load_names,
        "base_kw": base_kw,
        "base_kvar": base_kvar,
        "pv_names": pv_names,
        "pv_base": pv_base,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "cap_num_steps": cap_num_steps,
        "monitor_nodes": monitor_nodes,
    }


def _apply_snapshot_timestep(ctx: dict[str, Any], cfg: DailySimConfig, t: int, *, der_mult) -> None:
    apply_explicit_loads_pv(
        ctx["load_names"],
        ctx["base_kw"],
        ctx["base_kvar"],
        ctx["pv_names"],
        ctx["pv_base"],
        float(ctx["m_load"][t]),
        float(ctx["m_irr"][t]),
    )
    if cfg.include_der and der_mult is not None:
        set_der_injection(cfg, t, der_mult)
    dss.Text.Command(f"set hour={t // cfg.steps_per_hour} sec={(t % cfg.steps_per_hour) * cfg.step_sec}")
    dss.Text.Command("set mode=snapshot")


def _load_da_gps_bundle(
    cfg: DailySimConfig,
    collect_nodes: list[str],
    *,
    load_csv: Path,
    irr_csv: Path,
    ref_sample_index: int,
    scenario_scale: float,
    daily_stress: float,
    device: str | None,
    gnn_batch_steps: int | None = None,
) -> dict[str, Any]:
    cwd_before = os.getcwd()
    try:
        repo_root = str(cfg.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        os.chdir(cfg.repo_root)
        dev = resolve_da_gps_device(device)
        log_da_gps_device(dev)
        inf_npts, inf_step_min = da_gps_inference_grid(cfg)
        _log_gnn_inference_opts(device=str(dev), gnn_batch_steps=gnn_batch_steps)
        print(
            format_gnn_grid_log(
                amortize_gnn_timing_to_display_grid(
                    display_npts=cfg.npts,
                    display_step_min=cfg.step_min,
                    internal_npts=inf_npts,
                    internal_step_min=inf_step_min,
                    gnn_setup_once_s=None,
                    gnn_per_step_s=None,
                    gnn_total_wall_s=None,
                    gnn_n_ok=None,
                ),
                prefix="[da_gps_warmstart_band]",
            ),
            flush=True,
        )
        t_gnn_wrap0 = time.perf_counter()
        sys.modules.pop("run_da_gps_daily_opendss_compare", None)
        from run_da_gps_daily_opendss_compare import run_da_gps_daily_voltages

        der_max_kw = float(cfg.der_nominal_kw if cfg.include_der else 0.0)
        der_max_kvar = float(cfg.der_nominal_kvar if cfg.include_der else 0.0)
        # Match OpenDSS Q/P; default 0.1 only when kW is unset (legacy GNN default).
        der_q_frac_p = (der_max_kvar / der_max_kw) if der_max_kw > 0.0 else 0.1

        bundle = run_da_gps_daily_voltages(
            run_dir=cfg.da_gps_run_dir,
            cache_pt=cfg.da_gps_cache_pt,
            checkpoint=cfg.da_gps_checkpoint,
            node_names=collect_nodes,
            load_profile_path=str(load_csv),
            pv_irradiance_profile_path=str(irr_csv),
            der_max_kw=der_max_kw,
            der_buses=str(cfg.der_bus if cfg.include_der else ""),
            der_q_frac_p=float(der_q_frac_p),
            der_profile_path=str(cfg.der_profile_csv if cfg.include_der else "") or None,
            npts=int(inf_npts),
            step_min=int(inf_step_min),
            daily_stress=float(daily_stress),
            scenario_scale=float(scenario_scale),
            ref_sample_index=int(ref_sample_index),
            skip_opendss=True,
            return_device_states=True,
            device=dev,
            gnn_batch_steps=gnn_batch_steps,
        )
        gnn_wall_s = time.perf_counter() - t_gnn_wrap0
        da_hours = display_hours_array(cfg)
        voltages_native = bundle["voltages"]
        reg_native = np.asarray(bundle["reg_tap_pu"], dtype=float)
        cap_native = np.asarray(bundle["cap_sigmoid"], dtype=float)
        meta_native = np.asarray(bundle.get("meta_aux_gnn", np.zeros((inf_npts, 0))), dtype=float)
        reg_cols = list(bundle["reg_cols"])
        cap_cols = list(bundle["cap_cols"])
        meta_cols = list(bundle.get("meta_aux_cols") or [])

        if int(inf_npts) == int(cfg.npts) and int(inf_step_min) == int(cfg.step_min):
            voltages = voltages_native
            reg_rs = reg_native
            cap_rs = cap_native
            meta_rs = meta_native
        else:
            da_src_h = np.arange(int(inf_npts), dtype=float) * (float(inf_step_min) / 60.0)
            voltages = {
                k: np.interp(da_hours, da_src_h, np.asarray(v, dtype=float))
                for k, v in voltages_native.items()
            }
            reg_rs = resample_daily_profile_2d(
                reg_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=int(inf_npts),
                native_step_min=int(inf_step_min),
                method="nearest",
            )
            cap_rs = resample_daily_profile_2d(
                cap_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=int(inf_npts),
                native_step_min=int(inf_step_min),
                method="nearest",
            )
            meta_rs = (
                resample_daily_profile_2d(
                    meta_native,
                    npts=cfg.npts,
                    step_min=cfg.step_min,
                    native_npts=int(inf_npts),
                    native_step_min=int(inf_step_min),
                    method="linear",
                )
                if meta_native.size
                else meta_native
            )
        timing = amortize_gnn_timing_to_display_grid(
            display_npts=cfg.npts,
            display_step_min=cfg.step_min,
            internal_npts=int(bundle.get("npts", inf_npts)),
            internal_step_min=inf_step_min,
            gnn_setup_once_s=bundle.get("gnn_setup_once_s"),
            gnn_per_step_s=bundle.get("gnn_per_step_s"),
            gnn_total_wall_s=bundle.get("gnn_total_wall_s"),
            gnn_n_ok=bundle.get("n_ok"),
        )
        return {
            "voltages": voltages,
            "hours": da_hours,
            "reg_native": reg_rs,
            "cap_native": cap_rs,
            "meta_native": meta_rs,
            "reg_cols": reg_cols,
            "cap_cols": cap_cols,
            "meta_cols": meta_cols,
            "align_fn": align_da_gps_trajectory_to_opendss_names,
            "gnn_wall_s": float(gnn_wall_s),
            "gnn_setup_once_s": timing["gnn_setup_once_s"],
            "gnn_per_step_s": timing["gnn_per_step_s"],
            "gnn_total_wall_s": timing["gnn_total_wall_s"],
            "gnn_n_ok": timing["n_ok"],
            "gnn_grid": timing,
        }
    finally:
        try:
            os.chdir(cwd_before)
        except Exception:
            os.chdir(cfg.grid_dir)


def run_da_gps_warmstart_band_daily(
    cfg: DailySimConfig | None = None,
    *,
    n_warm_starts: int = 5,
    warm_start_mode: str = "uniform",
    warm_start_randomize_static_caps: bool = False,
    load_profile_path: Path | str | None = None,
    pv_profile_path: Path | str | None = None,
    monitor_nodes: list[str] | None = None,
    plot_all_cache_nodes: bool = True,
    plot_all_max_nodes: int = 0,
    out_dir: Path | str | None = None,
    voltage_png_subdir: str = "",
    voltage_plot_dpi: int = 0,
    voltage_plot_fig_w: float = 0.0,
    voltage_plot_fig_h: float = 0.0,
    ref_sample_index: int = 0,
    scenario_scale: float = 1.0,
    daily_stress: float = 0.0,
    plot_reg_cap: bool = True,
    plot_meta_aux: bool = True,
    plot_warmstart_lines: bool = True,
    write_voltage_pngs: bool = True,
    seed: int | None = 42,
    show: bool = True,
    device: str | None = None,
    gnn_batch_steps: int | None = None,
) -> dict[str, Any]:
    """Warm-start cloud vs DA-GPS with inside-band metrics (voltage, reg, cap, meta aux).

    When ``plot_all_cache_nodes=True`` (default), scope matches ``run_da_gps_daily_opendss_compare``
    with ``--plot-all-cache-nodes``: all cache∩circuit nodes, PNGs under ``out_dir/daily_voltage/``,
    reg/cap/meta PNGs + CSV/JSON in ``out_dir``. Only OpenDSS truth differs (warm-start band vs single QSTS).

    Set ``write_voltage_pngs=False`` (with ``plot_reg_cap`` / ``plot_meta_aux`` False) for metrics-only
    multi-scenario sweeps: still computes aggregated band metrics and writes CSV/JSON when ``out_dir``
    is set, but skips per-node voltage PNGs.
    """
    from nonunique_opendss_daily import DA_GPS_REF_LOAD_PROFILE, DA_GPS_REF_PV_PROFILE

    cfg = cfg or DailySimConfig()
    load_csv = Path(load_profile_path or cfg.da_gps_load_profile or DA_GPS_REF_LOAD_PROFILE).resolve()
    irr_csv = Path(pv_profile_path or cfg.da_gps_pv_profile or DA_GPS_REF_PV_PROFILE).resolve()
    if monitor_nodes is not None:
        cfg.monitor_candidates = list(monitor_nodes)

    inline_backend = plt.get_backend()
    _ = inline_backend  # preserve notebook inline backend across Agg plots
    rng = np.random.default_rng(seed)
    der_mult = (
        load_der_multiplier_profile(cfg.der_profile_csv, cfg=cfg) if cfg.include_der else None
    )
    npts = int(cfg.npts)
    n_ws = int(n_warm_starts)
    if n_ws < 1:
        raise ValueError(f"n_warm_starts must be >= 1, got {n_warm_starts}")

    cfg_stem = _checkpoint_stem(cfg)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is not None:
        out_path = Path(out_dir).expanduser().resolve()
    elif plot_all_cache_nodes:
        # Shallow path under repo root — deep checkpoint folders exceed Windows MAX_PATH (~260).
        out_path = Path(cfg.repo_root).expanduser().resolve() / "warmstart_band_runs" / tag
    else:
        out_path = None

    cache_nodes = load_cache_node_order(cfg.da_gps_cache_pt)
    print("=" * 72)
    print("DA-GPS warm-start band daily (OpenDSS snapshot + N random controller inits/step)")
    print(f"  step_min={cfg.step_min} min, npts={npts}, n_warm_starts={n_ws}")
    print(
        f"  warm_start_mode={warm_start_mode!r}  "
        f"randomize_static_caps={bool(warm_start_randomize_static_caps)}",
        flush=True,
    )
    print(f"  load/PV profiles: {load_csv} / {irr_csv}")
    print(f"  cache .pt: {cfg.da_gps_cache_pt}")
    print(f"  checkpoint: {cfg.da_gps_checkpoint}")
    if out_path is not None:
        print(f"  out_dir: {out_path}")
    print("=" * 72)

    ctx0 = _snapshot_parity_context(cfg, load_csv=load_csv, irr_csv=irr_csv, daily_stress=daily_stress)
    monitor_resolved = ctx0["monitor_nodes"]
    cache_on_circuit = filter_cache_nodes_on_circuit(cache_nodes)
    print(
        f"[da_gps_warmstart_band] monitor nodes ({len(monitor_resolved)}): {monitor_resolved}",
        flush=True,
    )
    print(
        f"[da_gps_warmstart_band] cache x circuit nodes: {len(cache_on_circuit)} / {len(cache_nodes)}",
        flush=True,
    )
    if plot_all_cache_nodes:
        collect_nodes = cache_on_circuit
        print(
            "[da_gps_warmstart_band] plot_all_cache_nodes=True: collecting/plotting cache∩circuit nodes",
            flush=True,
        )
    else:
        collect_nodes = list(monitor_nodes) if monitor_nodes is not None else monitor_resolved
        print(
            f"[da_gps_warmstart_band] collect/plot {len(collect_nodes)} monitor nodes only",
            flush=True,
        )

    reg_names = ctx0["reg_names"]
    cap_names = ctx0["cap_names"]
    cap_num_steps = dict(ctx0.get("cap_num_steps") or {})
    pv_names = ctx0["pv_names"]
    n_nodes, n_reg, n_cap = len(collect_nodes), len(reg_names), len(cap_names)

    gnn_bundle = _load_da_gps_bundle(
        cfg,
        collect_nodes,
        load_csv=load_csv,
        irr_csv=irr_csv,
        ref_sample_index=ref_sample_index,
        scenario_scale=scenario_scale,
        daily_stress=daily_stress,
        device=device,
        gnn_batch_steps=gnn_batch_steps,
    )
    meta_cols = list(gnn_bundle.get("meta_cols") or [])
    n_meta = len(meta_cols)
    da_v, da_reg, da_cap, da_hours = align_da_gps_devices(gnn_bundle, reg_names, cap_names)
    da_reg_disc: dict[str, np.ndarray] = {}
    if da_reg is not None:
        for nm in reg_names:
            y_raw = _dict_lookup(da_reg, nm)
            if y_raw is None:
                continue
            da_reg_disc[str(nm)] = _discretize_reg_tap_pu(np.asarray(y_raw, dtype=float).ravel()[:npts])
    da_cap_disc: dict[str, np.ndarray] = {}
    if da_cap is not None:
        for nm in cap_names:
            y_raw = _dict_lookup(da_cap, nm)
            if y_raw is None:
                continue
            n_steps = int(_dict_lookup(cap_num_steps, nm) or 1)
            da_cap_disc[str(nm)] = _discretize_cap_sigmoid(
                np.asarray(y_raw, dtype=float).ravel()[:npts],
                n_steps=n_steps,
            )
    da_meta: dict[str, np.ndarray] = {}
    if n_meta > 0:
        for j, col in enumerate(meta_cols):
            da_meta[str(col)] = np.asarray(gnn_bundle["meta_native"][:, j], dtype=np.float64)

    volts = np.full((npts, n_ws, n_nodes), np.nan, dtype=float)
    taps = np.full((npts, n_ws, n_reg), np.nan, dtype=float)
    cap_on = np.full((npts, n_ws, n_cap), np.nan, dtype=float)
    meta_dss = np.full((npts, n_ws, n_meta), np.nan, dtype=float)
    converged = np.zeros((npts, n_ws), dtype=bool)
    dss_step_wall_s = np.zeros(npts, dtype=np.float64)
    dss_solve_only_s = np.full((npts, n_ws), np.nan, dtype=np.float64)
    dss_collect_s = np.zeros(npts, dtype=np.float64)

    t_wall0 = time.perf_counter()
    for t in range(npts):
        t_step0 = time.perf_counter()
        dss.Basic.ClearAll()
        ctx = _snapshot_parity_context(cfg, load_csv=load_csv, irr_csv=irr_csv, daily_stress=daily_stress)
        _apply_snapshot_timestep(ctx, cfg, t, der_mult=der_mult)
        for k in range(n_ws):
            randomize_controllers(
                rng,
                dynamic_only=not bool(warm_start_randomize_static_caps),
                mode=str(warm_start_mode),
                warm_start_index=int(k),
            )
            try:
                dss.Solution.InitSnap()
            except Exception:
                pass
            t_solve0 = time.perf_counter()
            safe_solve()
            dss_solve_only_s[t, k] = time.perf_counter() - t_solve0
            converged[t, k] = bool(dss.Solution.Converged())
            if not converged[t, k]:
                continue
            t_collect0 = time.perf_counter()
            for i, nm in enumerate(reg_names):
                dss.RegControls.Name(nm)
                taps[t, k, i] = dss.RegControls.TapNumber()
            for j, nm in enumerate(cap_names):
                dss.Capacitors.Name(nm)
                cap_on[t, k, j] = sum(dss.Capacitors.States())
            volts[t, k] = node_voltages_pu(collect_nodes)
            if n_meta > 0:
                meta_dss[t, k] = _meta_dss_row(meta_cols, pv_names)
            dss_collect_s[t] += time.perf_counter() - t_collect0
        dss_step_wall_s[t] = time.perf_counter() - t_step0
        if (t + 1) % max(1, npts // 6) == 0 or t == npts - 1:
            print(f"  timestep {t + 1}/{npts} done")

    wall_s = time.perf_counter() - t_wall0
    print(f"Warm-start band: wall time {wall_s:.1f}s")

    # Per-display-step mean snapshot Solve() (comparable GNN denominator to daily compare).
    dss_solve_per_display = np.full(npts, np.nan, dtype=np.float64)
    for t in range(npts):
        m = np.isfinite(dss_solve_only_s[t]) & (dss_solve_only_s[t] > 0)
        if m.any():
            dss_solve_per_display[t] = float(np.mean(dss_solve_only_s[t, m]))
    n_dss_ok = int(np.sum(np.isfinite(dss_solve_per_display)))
    print_timing_summary(
        n_ok=max(1, n_dss_ok),
        npts=npts,
        step_min=int(cfg.step_min),
        dss_wall_s=float(wall_s),
        dss_solve_s=dss_solve_per_display,
        dss_collect_s=dss_collect_s,
        gnn_wall_s=float(gnn_bundle.get("gnn_wall_s", 0.0)),
        gnn_setup_once_s=gnn_bundle.get("gnn_setup_once_s"),
        gnn_per_step_s=gnn_bundle.get("gnn_per_step_s"),
        gnn_total_wall_s=gnn_bundle.get("gnn_total_wall_s"),
        gnn_n_ok=gnn_bundle.get("gnn_n_ok"),
        gnn_grid=gnn_bundle.get("gnn_grid"),
    )
    _print_warmstart_envelope_timing(
        npts=npts,
        n_warm_starts=n_ws,
        dss_step_wall_s=dss_step_wall_s,
        dss_solve_only_s=dss_solve_only_s,
    )

    volts_min = np.nanmin(volts, axis=1)
    volts_max = np.nanmax(volts, axis=1)
    taps_min = np.nanmin(taps, axis=1)
    taps_max = np.nanmax(taps, axis=1)
    cap_min = np.nanmin(cap_on, axis=1)
    cap_max = np.nanmax(cap_on, axis=1)
    meta_min = np.nanmin(meta_dss, axis=1) if n_meta > 0 else None
    meta_max = np.nanmax(meta_dss, axis=1) if n_meta > 0 else None

    inside: dict[str, dict[str, float]] = {"voltage": {}, "regulator": {}, "capacitor": {}, "meta_aux": {}}
    proximity: dict[str, dict[str, float]] = {"voltage": {}, "regulator": {}, "capacitor": {}, "meta_aux": {}}
    outside_dist: dict[str, dict[str, float]] = {
        "voltage": {},
        "regulator": {},
        "capacitor": {},
        "meta_aux": {},
    }
    set_dist: dict[str, dict[str, float]] = {
        "voltage": {},
        "regulator": {},
        "capacitor": {},
        "meta_aux": {},
    }
    cap_steps_by_name = _cap_num_steps_by_name(cap_names)

    if da_v is not None:
        for j, node in enumerate(collect_nodes):
            v_da = _dict_lookup(da_v, node)
            if v_da is None:
                continue
            v_da = np.asarray(v_da, dtype=float).ravel()[:npts]
            v_lo = volts_min[:, j]
            v_hi = volts_max[:, j]
            frac = _inside_band_fraction(v_da, v_lo, v_hi)
            inside["voltage"][node] = frac
            proximity["voltage"][node] = _band_proximity_continuous(
                v_da, v_lo, v_hi, min_scale=1e-4
            )
            outside_dist["voltage"][node] = _mean_outside_distance(v_da, v_lo, v_hi)
            set_dist["voltage"][node] = _mean_set_distance(v_da, v_lo, v_hi)

    if da_reg_disc:
        for nm in reg_names:
            y_da = _dict_lookup(da_reg_disc, nm)
            if y_da is None:
                continue
            j = reg_names.index(nm)
            lo = taps_min[:, j]
            hi = taps_max[:, j]
            inside["regulator"][nm] = _inside_band_fraction(y_da, lo, hi)
            proximity["regulator"][nm] = _band_proximity_discrete(
                y_da, lo, hi, step_scale=1.0
            )
            outside_dist["regulator"][nm] = _mean_outside_distance(y_da, lo, hi)
            set_dist["regulator"][nm] = _mean_set_distance(y_da, lo, hi)

    if da_cap_disc:
        for nm in cap_names:
            y_da = _dict_lookup(da_cap_disc, nm)
            if y_da is None:
                continue
            j = cap_names.index(nm)
            lo = cap_min[:, j]
            hi = cap_max[:, j]
            inside["capacitor"][nm] = _inside_band_fraction(y_da, lo, hi)
            cap_scale = float(_dict_lookup(cap_steps_by_name, nm) or 1.0)
            proximity["capacitor"][nm] = _band_proximity_discrete(
                y_da, lo, hi, step_scale=max(1.0, cap_scale)
            )
            outside_dist["capacitor"][nm] = _mean_outside_distance(y_da, lo, hi)
            set_dist["capacitor"][nm] = _mean_set_distance(y_da, lo, hi)

    if da_meta and meta_min is not None and meta_max is not None:
        for col in meta_cols:
            y_da = _dict_lookup(da_meta, col)
            if y_da is None:
                continue
            y_da = np.asarray(y_da, dtype=float).ravel()[:npts]
            j = meta_cols.index(col)
            lo = meta_min[:, j]
            hi = meta_max[:, j]
            inside["meta_aux"][col] = _inside_band_fraction(y_da, lo, hi)
            proximity["meta_aux"][col] = _band_proximity_continuous(
                y_da, lo, hi, min_scale=1e-3
            )
            outside_dist["meta_aux"][col] = _mean_outside_distance(y_da, lo, hi)
            set_dist["meta_aux"][col] = _mean_set_distance(y_da, lo, hi)

    aggregated = _build_aggregated_band_metrics(inside, proximity, outside_dist, set_dist)
    _print_aggregated_band_summary(aggregated)

    print("\n=== DA-GPS inside OpenDSS warm-start band [min,max] (per device) ===")
    print(
        "  (regulator/capacitor: DA-GPS rounded to discrete tap # / steps ON before inside-band check)",
        flush=True,
    )
    for group, items in inside.items():
        if not items:
            continue
        print(f"  [{group}]")
        shown = 0
        for name, frac in sorted(items.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else 1.0)):
            if shown < 8 or group != "voltage" or not plot_all_cache_nodes:
                print(f"    {name}: {100.0 * frac:.1f}% of timesteps inside band")
                shown += 1
        if group == "voltage" and plot_all_cache_nodes and len(items) > 8:
            print(f"    ... ({len(items)} nodes total; see per-node CSV in out_dir)")

    print("\n=== DA-GPS cloud proximity to OpenDSS warm-start band (per device) ===")
    print(
        "  score in [0,1]: 1 inside band; outside decays as exp(-d/scale) "
        "(continuous: scale=max(half-width,1e-4); discrete: tap/cap step scale)",
        flush=True,
    )
    for group, items in proximity.items():
        if not items:
            continue
        print(f"  [{group}]")
        shown = 0
        for name, prox in sorted(items.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1.0)):
            if shown < 8 or group != "voltage" or not plot_all_cache_nodes:
                print(f"    {name}: cloud proximity {prox:.3f}")
                shown += 1
        if group == "voltage" and plot_all_cache_nodes and len(items) > 8:
            print(f"    ... ({len(items)} nodes total; see per-node CSV in out_dir)")

    print("\n=== DA-GPS set distance to OpenDSS warm-start band (per device) ===")
    print(
        "  mean distance to nearest valid point in [lo, hi] over all timesteps "
        "(0 inside band); units: pu (|V|/meta), tap steps (reg), cap steps ON (cap)",
        flush=True,
    )
    for group, items in set_dist.items():
        if not items:
            continue
        print(f"  [{group}]")
        shown = 0
        for name, dist in sorted(items.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1.0), reverse=True):
            if shown < 8 or group != "voltage" or not plot_all_cache_nodes:
                print(f"    {name}: set distance {dist:.4g}")
                shown += 1
        if group == "voltage" and plot_all_cache_nodes and len(items) > 8:
            print(f"    ... ({len(items)} nodes total; see per-node CSV in out_dir)")

    print("\n=== DA-GPS mean outside distance to OpenDSS warm-start band (per device) ===")
    print(
        "  mean edge distance over outside timesteps only (0 when always inside); "
        "units: pu (|V|/meta), tap steps (reg), cap steps ON (cap)",
        flush=True,
    )
    for group, items in outside_dist.items():
        if not items:
            continue
        print(f"  [{group}]")
        shown = 0
        for name, dist in sorted(items.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1.0), reverse=True):
            if shown < 8 or group != "voltage" or not plot_all_cache_nodes:
                print(f"    {name}: mean outside distance {dist:.4g}")
                shown += 1
        if group == "voltage" and plot_all_cache_nodes and len(items) > 8:
            print(f"    ... ({len(items)} nodes total; see per-node CSV in out_dir)")

    volt_png_dir: Path | None = None
    vdpi = int(voltage_plot_dpi) if int(voltage_plot_dpi) > 0 else (96 if plot_all_cache_nodes else 160)
    vf_w = float(voltage_plot_fig_w) if float(voltage_plot_fig_w) > 0 else (7.5 if plot_all_cache_nodes else 10.0)
    vf_h = float(voltage_plot_fig_h) if float(voltage_plot_fig_h) > 0 else (3.2 if plot_all_cache_nodes else 4.2)
    stem_safe = _safe_stem(cfg_stem)
    stem_short = _stem_short(cfg_stem)
    aux_outputs: dict[str, Any] = {}
    t_hours = display_hours_array(cfg)
    n_plot_ranked = len(collect_nodes)
    der_record = build_der_injection_record(cfg, der_mult)

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        volt_png_dir = _resolve_voltage_png_dir(
            out_path,
            plot_all_cache_nodes=plot_all_cache_nodes,
            voltage_png_subdir=voltage_png_subdir,
        )
        print(
            f"[da_gps_warmstart_band] daily |V| band PNG directory: {volt_png_dir} "
            f"(dpi={vdpi}, figsize=({vf_w:.3g} x {vf_h:.3g}) in)",
            flush=True,
        )

        node_rows = []
        for j, node in enumerate(collect_nodes):
            frac = inside["voltage"].get(node, float("nan"))
            prox = proximity["voltage"].get(node, float("nan"))
            odist = outside_dist["voltage"].get(node, float("nan"))
            sdist = set_dist["voltage"].get(node, float("nan"))
            node_rows.append((str(node).strip().lower(), frac, prox, sdist, odist))
        df_inside = pd.DataFrame(
            node_rows,
            columns=[
                "node",
                "inside_frac_dagps",
                "cloud_proximity_dagps",
                "set_distance_dagps",
                "mean_outside_distance_dagps",
            ],
        ).sort_values("inside_frac_dagps", ascending=True)
        inside_csv = out_path / f"inside_band_per_node_{stem_short}.csv"
        _write_csv(df_inside, inside_csv)
        aux_outputs["warmstart_inside_band_per_node_csv"] = str(inside_csv)
        print(f"[da_gps_warmstart_band] wrote {inside_csv}", flush=True)

        plot_list = [str(nk).strip().lower() for nk in collect_nodes]
        node_to_j = {str(nk).strip().lower(): j for j, nk in enumerate(collect_nodes)}
        plot_rows: list[tuple[str, float]] = []
        for n in plot_list:
            j = node_to_j.get(n)
            if j is None:
                continue
            plot_rows.append((n, inside["voltage"].get(collect_nodes[j], float("nan"))))

        def _rank_sort_key(row: tuple[str, float]) -> tuple[int, float, str]:
            _nk, frac = row
            if np.isfinite(frac):
                return (0, float(frac), _nk)
            return (1, 0.0, _nk)

        plot_rows.sort(key=_rank_sort_key)
        n_all_for_rank = len(plot_rows)
        if plot_all_cache_nodes and int(plot_all_max_nodes) > 0 and n_all_for_rank > int(plot_all_max_nodes):
            cap_m = int(plot_all_max_nodes)
            plot_rows = plot_rows[:cap_m]
            print(
                f"[da_gps_warmstart_band] plot_all_max_nodes={cap_m}: PNGs for worst "
                f"{len(plot_rows)}/{n_all_for_rank} nodes by inside-band %.",
                flush=True,
            )
        if len(plot_rows) > 500 and int(plot_all_max_nodes) <= 0:
            print(
                "[da_gps_warmstart_band] NOTE: many |V| band PNGs under daily_voltage/. "
                "Use plot_all_max_nodes=N for worst-N only.",
                flush=True,
            )
        n_plot_ranked = len(plot_rows)
        rank_w = max(4, len(str(max(1, n_plot_ranked))))
        if write_voltage_pngs:
            volt_png_dir.mkdir(parents=True, exist_ok=True)
            for rank_idx, (n, frac) in enumerate(plot_rows):
                j = node_to_j[n]
                node_disp = collect_nodes[j]
                rk = str(rank_idx + 1).zfill(rank_w)
                tot = str(n_plot_ranked).zfill(rank_w)
                png_path = volt_png_dir / _warmstart_voltage_png_basename(
                    stem_safe=stem_safe,
                    rk=rk,
                    tot=tot,
                    inside_frac=frac,
                    node_fn=n.replace(".", "_"),
                )
                _save_voltage_band_png(
                    png_path,
                    cfg,
                    node=node_disp,
                    j=j,
                    volts=volts,
                    volts_min=volts_min,
                    volts_max=volts_max,
                    n_warm_starts=n_ws,
                    inside_frac=frac,
                    rank_human=rank_idx + 1,
                    n_plot_ranked=n_plot_ranked,
                    da_gps_voltages=da_v,
                    da_gps_hours=da_hours,
                    plot_warmstart_lines=plot_warmstart_lines,
                    dpi=vdpi,
                    fig_w=vf_w,
                    fig_h=vf_h,
                    show=False,
                )
        else:
            print(
                "[da_gps_warmstart_band] write_voltage_pngs=False: skipping |V| PNGs "
                "(metrics / CSV / JSON still written).",
                flush=True,
            )

        if plot_reg_cap and n_reg > 0:
            reg_rows: dict[str, object] = {
                "step_idx": list(range(npts)),
                "hour": t_hours.astype(np.float64).tolist(),
            }
            for jr, nm in enumerate(reg_names):
                reg_rows[f"{nm}__band_min"] = taps_min[:, jr].astype(np.float64).tolist()
                reg_rows[f"{nm}__band_max"] = taps_max[:, jr].astype(np.float64).tolist()
                y_da = _dict_lookup(da_reg_disc, nm) if da_reg_disc else None
                if y_da is not None:
                    reg_rows[f"{nm}__dagps_tap"] = np.asarray(y_da, dtype=float).ravel()[:npts].tolist()
            reg_csv = out_path / f"regulator_tap_{stem_short}.csv"
            _write_csv(pd.DataFrame(reg_rows), reg_csv)
            aux_outputs["warmstart_regulator_tap_csv"] = str(reg_csv)
            for jr, nm in enumerate(reg_names):
                frac = inside["regulator"].get(nm, float("nan"))
                ppath = out_path / f"regulator_tap_{stem_short}_{_safe_stem(nm)}.png"
                _save_device_band_png(
                    ppath,
                    cfg,
                    device_name=nm,
                    values=taps,
                    vmin=taps_min,
                    vmax=taps_max,
                    idx=jr,
                    ylabel="tap #",
                    title_prefix="Regulator",
                    n_warm_starts=n_ws,
                    inside_frac=frac,
                    da_gps_by_name=da_reg_disc,
                    da_gps_hours=da_hours,
                    da_transform=None,
                    plot_warmstart_lines=plot_warmstart_lines,
                    dpi=vdpi,
                    show=show and not plot_all_cache_nodes,
                )

        if plot_reg_cap and n_cap > 0:
            cap_rows: dict[str, object] = {
                "step_idx": list(range(npts)),
                "hour": t_hours.astype(np.float64).tolist(),
            }
            for jc, nm in enumerate(cap_names):
                cap_rows[f"{nm}__band_min"] = cap_min[:, jc].astype(np.float64).tolist()
                cap_rows[f"{nm}__band_max"] = cap_max[:, jc].astype(np.float64).tolist()
                y_da = _dict_lookup(da_cap_disc, nm) if da_cap_disc else None
                if y_da is not None:
                    cap_rows[f"{nm}__dagps_steps"] = np.asarray(y_da, dtype=float).ravel()[:npts].tolist()
            cap_csv = out_path / f"cap_bank_status_{stem_short}.csv"
            _write_csv(pd.DataFrame(cap_rows), cap_csv)
            aux_outputs["warmstart_cap_bank_status_csv"] = str(cap_csv)
            for jc, nm in enumerate(cap_names):
                frac = inside["capacitor"].get(nm, float("nan"))
                ppath = out_path / f"cap_bank_{stem_short}_{_safe_stem(nm)}.png"
                _save_device_band_png(
                    ppath,
                    cfg,
                    device_name=nm,
                    values=cap_on,
                    vmin=cap_min,
                    vmax=cap_max,
                    idx=jc,
                    ylabel="steps ON",
                    title_prefix="Capacitor",
                    n_warm_starts=n_ws,
                    inside_frac=frac,
                    da_gps_by_name=da_cap_disc,
                    da_gps_hours=da_hours,
                    da_transform=None,
                    plot_warmstart_lines=plot_warmstart_lines,
                    dpi=vdpi,
                    show=show and not plot_all_cache_nodes,
                )

        if plot_meta_aux and n_meta > 0 and meta_min is not None and meta_max is not None:
            meta_rows: dict[str, object] = {
                "step_idx": list(range(npts)),
                "hour": t_hours.astype(np.float64).tolist(),
            }
            for jm, col in enumerate(meta_cols):
                meta_rows[f"{col}__band_min"] = meta_min[:, jm].astype(np.float64).tolist()
                meta_rows[f"{col}__band_max"] = meta_max[:, jm].astype(np.float64).tolist()
                y_da = _dict_lookup(da_meta, col)
                if y_da is not None:
                    meta_rows[f"{col}__dagps"] = np.asarray(y_da, dtype=float).ravel()[:npts].tolist()
            meta_csv = out_path / f"meta_aux_{stem_short}.csv"
            _write_csv(pd.DataFrame(meta_rows), meta_csv)
            aux_outputs["warmstart_meta_aux_csv"] = str(meta_csv)
            for jm, col in enumerate(meta_cols):
                cl = str(col).lower()
                if "kvar" in cl:
                    ylab = "kvar"
                elif "kw" in cl or "_p_" in cl or "loss" in cl:
                    ylab = "kW"
                else:
                    ylab = "value"
                frac = inside["meta_aux"].get(col, float("nan"))
                ppath = out_path / f"meta_aux_{stem_short}_{_safe_stem(col)}.png"
                _save_device_band_png(
                    ppath,
                    cfg,
                    device_name=col,
                    values=meta_dss[:, :, jm : jm + 1],
                    vmin=meta_min[:, jm : jm + 1],
                    vmax=meta_max[:, jm : jm + 1],
                    idx=0,
                    ylabel=ylab,
                    title_prefix="Meta aux",
                    n_warm_starts=n_ws,
                    inside_frac=frac,
                    da_gps_by_name=da_meta,
                    da_gps_hours=da_hours,
                    da_transform=None,
                    plot_warmstart_lines=plot_warmstart_lines,
                    dpi=vdpi,
                    show=show and not plot_all_cache_nodes,
                )

        summary = {
            "mode": "da_gps_warmstart_band_daily",
            "run_dir": str(Path(cfg.da_gps_run_dir).resolve()),
            "checkpoint": str(Path(cfg.da_gps_checkpoint).resolve()),
            "cache_pt": str(Path(cfg.da_gps_cache_pt).resolve()),
            "out_dir": str(out_path),
            "n_warm_starts": n_ws,
            "warm_start_mode": str(warm_start_mode),
            "warm_start_randomize_static_caps": bool(warm_start_randomize_static_caps),
            "npts": npts,
            "step_min": int(cfg.step_min),
            "ref_sample_index": int(ref_sample_index),
            "scenario_scale": float(scenario_scale),
            "daily_stress": float(daily_stress),
            "plot_all_cache_nodes": bool(plot_all_cache_nodes),
            "n_collect_nodes": n_nodes,
            "n_voltage_pngs": n_plot_ranked if plot_all_cache_nodes else len(collect_nodes),
            "der": der_record,
            "aggregated": aggregated,
            "inside_band_frac": inside,
            "cloud_proximity": proximity,
            "set_distance": set_dist,
            "mean_outside_distance": outside_dist,
            "aux_outputs": aux_outputs,
            "wall_s_opendss_warmstart": wall_s,
            "timing": {
                "gnn_wall_s": float(gnn_bundle.get("gnn_wall_s", 0.0)),
                "gnn_setup_once_s": gnn_bundle.get("gnn_setup_once_s"),
                "gnn_per_step_s": gnn_bundle.get("gnn_per_step_s"),
                "gnn_total_wall_s": gnn_bundle.get("gnn_total_wall_s"),
                "gnn_n_ok": gnn_bundle.get("gnn_n_ok"),
                "opendss_warmstart_wall_s": float(wall_s),
                "opendss_display_step_wall_s_mean": float(np.nanmean(dss_step_wall_s)),
                "opendss_snapshot_solve_ms_mean": float(
                    1000.0
                    * np.nanmean(dss_solve_only_s[np.isfinite(dss_solve_only_s) & (dss_solve_only_s > 0)])
                )
                if np.isfinite(dss_solve_only_s).any()
                else float("nan"),
            },
        }
        for group, stats in aggregated.items():
            summary[f"mean_inside_frac_{group}_dagps"] = float(
                stats.get("mean_inside_band_frac", float("nan"))
            )
            summary[f"mean_cloud_proximity_{group}_dagps"] = float(
                stats.get("mean_cloud_proximity", float("nan"))
            )
            summary[f"mean_set_distance_{group}_dagps"] = float(
                stats.get("mean_set_distance", float("nan"))
            )
            summary[f"mean_outside_distance_{group}_dagps"] = float(
                stats.get("mean_outside_distance", float("nan"))
            )
        summary_path = out_path / "run_summary.json"
        _write_json_text(summary_path, json.dumps(summary, indent=2))
        print(f"[da_gps_warmstart_band] wrote {summary_path}", flush=True)

    elif not plot_all_cache_nodes:
        plot_warmstart_voltage_band(
            cfg,
            monitor_nodes=collect_nodes,
            volts=volts,
            volts_min=volts_min,
            volts_max=volts_max,
            n_warm_starts=n_ws,
            da_gps_voltages=da_v,
            da_gps_hours=da_hours,
            plot_warmstart_lines=plot_warmstart_lines,
            show=show,
        )
        if plot_reg_cap:
            plot_warmstart_device_band(
                cfg,
                device_names=reg_names,
                values=taps,
                vmin=taps_min,
                vmax=taps_max,
                ylabel="tap #",
                title_prefix="Regulator",
                da_gps_by_name=da_reg_disc,
                da_gps_hours=da_hours,
                da_transform=None,
                n_warm_starts=n_ws,
                plot_warmstart_lines=plot_warmstart_lines,
                show=show,
            )
            plot_warmstart_device_band(
                cfg,
                device_names=cap_names,
                values=cap_on,
                vmin=cap_min,
                vmax=cap_max,
                ylabel="steps ON",
                title_prefix="Capacitor",
                da_gps_by_name=da_cap_disc,
                da_gps_hours=da_hours,
                da_transform=None,
                n_warm_starts=n_ws,
                plot_warmstart_lines=plot_warmstart_lines,
                show=show,
            )
        if plot_meta_aux and n_meta > 0 and meta_min is not None and meta_max is not None:
            for j, col in enumerate(meta_cols):
                cl = str(col).lower()
                if "kvar" in cl:
                    ylab = "kvar"
                elif "kw" in cl or "_p_" in cl or "loss" in cl:
                    ylab = "kW"
                else:
                    ylab = "value"
                plot_warmstart_device_band(
                    cfg,
                    device_names=[col],
                    values=meta_dss[:, :, j : j + 1],
                    vmin=meta_min[:, j : j + 1],
                    vmax=meta_max[:, j : j + 1],
                    ylabel=ylab,
                    title_prefix="Meta aux",
                    da_gps_by_name=da_meta,
                    da_gps_hours=da_hours,
                    da_transform=None,
                    n_warm_starts=n_ws,
                    plot_warmstart_lines=plot_warmstart_lines,
                    show=show,
                )

    return {
        "mode": "da_gps_warmstart_band_daily",
        "cfg": cfg,
        "out_dir": str(out_path) if out_path is not None else None,
        "n_warm_starts": n_ws,
        "warm_start_mode": str(warm_start_mode),
        "warm_start_randomize_static_caps": bool(warm_start_randomize_static_caps),
        "load_profile": str(load_csv),
        "pv_profile": str(irr_csv),
        "der_injection": der_record,
        "collect_nodes": collect_nodes,
        "monitor_nodes": monitor_resolved,
        "plot_all_cache_nodes": plot_all_cache_nodes,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "meta_aux_cols": meta_cols,
        "volts": volts,
        "volts_min": volts_min,
        "volts_max": volts_max,
        "taps": taps,
        "taps_min": taps_min,
        "taps_max": taps_max,
        "cap_on": cap_on,
        "cap_min": cap_min,
        "cap_max": cap_max,
        "meta_dss": meta_dss,
        "meta_min": meta_min,
        "meta_max": meta_max,
        "da_gps_voltages": da_v,
        "da_gps_reg_by_name": da_reg_disc or da_reg,
        "da_gps_cap_by_name": da_cap_disc or da_cap,
        "da_gps_meta_by_name": da_meta,
        "da_gps_hours": da_hours,
        "da_gps_inside_band_frac": inside,
        "da_gps_cloud_proximity": proximity,
        "da_gps_set_distance": set_dist,
        "da_gps_mean_outside_distance": outside_dist,
        "da_gps_aggregated": aggregated,
        "converged": converged,
        "wall_s": wall_s,
        "gnn_timing": {
            "gnn_wall_s": gnn_bundle.get("gnn_wall_s"),
            "gnn_setup_once_s": gnn_bundle.get("gnn_setup_once_s"),
            "gnn_per_step_s": gnn_bundle.get("gnn_per_step_s"),
            "gnn_total_wall_s": gnn_bundle.get("gnn_total_wall_s"),
            "gnn_grid": gnn_bundle.get("gnn_grid"),
        },
        "opendss_timing": {
            "wall_s": wall_s,
            "display_step_wall_s": dss_step_wall_s,
            "solve_only_s": dss_solve_only_s,
        },
    }


__all__ = ["run_da_gps_warmstart_band_daily"]
