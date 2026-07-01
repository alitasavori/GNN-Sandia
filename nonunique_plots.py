"""Plotting helpers for nonunique OpenDSS experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nonunique_opendss_daily import DailySimConfig, display_hours_array, tap_pu_to_tap_number


def _plot_npts(cfg: DailySimConfig) -> int:
    return max(1, int(cfg.npts))


def _trim_1d(series, n: int) -> np.ndarray:
    return np.asarray(series, dtype=float).ravel()[:n]


def _trim_step_matrix(matrix, n: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        return arr[:n]
    return arr[:n, ...]


def plot_voltage_monitors(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    style_lut: dict[str, tuple[str, str]],
    *,
    da_gps_voltages: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    suptitle: str,
    show: bool = True,
):
    monitor_nodes = runs[0]["monitor_nodes"]
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    n_nodes = len(monitor_nodes)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 3.5 * n_nodes), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(suptitle, fontsize=11)
    for ax, node, j in zip(axes, monitor_nodes, range(n_nodes)):
        for run in runs:
            ls, lbl = style_lut[run["label"]]
            ax.plot(hours, _trim_step_matrix(run["volts"], n)[:, j], label=lbl, linestyle=ls)
        if da_gps_voltages is not None and da_gps_hours is not None:
            v_da = da_gps_voltages.get(node)
            if v_da is None:
                v_da = da_gps_voltages.get(str(node).lower())
            if v_da is not None:
                ax.plot(
                    _trim_1d(da_gps_hours, n),
                    _trim_1d(v_da, n),
                    label="DA-GPS",
                    linestyle="-.",
                    color="magenta",
                    linewidth=1.5,
                )
        ax.set_title(node)
        ax.set_ylabel("|V| (pu)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Hour of day")
    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def plot_regulator_taps(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    style_lut: dict[str, tuple[str, str]],
    *,
    da_gps_reg_by_name: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    show: bool = True,
):
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    reg_names = runs[0]["reg_names"]
    for idx, rname in enumerate(reg_names):
        fig, ax = plt.subplots(figsize=(16, 4))
        for run in runs:
            ls, lbl = style_lut[run["label"]]
            ax.plot(hours, _trim_step_matrix(run["taps"], n)[:, idx], label=lbl, linestyle=ls)
        if da_gps_reg_by_name is not None and da_gps_hours is not None:
            y_reg = da_gps_reg_by_name.get(rname)
            if y_reg is None:
                y_reg = da_gps_reg_by_name.get(str(rname).lower())
            if y_reg is not None and np.isfinite(y_reg).any():
                ax.plot(
                    _trim_1d(da_gps_hours, n),
                    tap_pu_to_tap_number(_trim_1d(y_reg, n)),
                    label="DA-GPS",
                    linestyle="-.",
                    color="magenta",
                    linewidth=1.5,
                )
        ax.set_title(f"Regulator {rname}")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("tap #")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        plt.close(fig)


def plot_capacitor_steps(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    style_lut: dict[str, tuple[str, str]],
    *,
    da_gps_cap_by_name: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    show: bool = True,
):
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    cap_names = runs[0]["cap_names"]
    for j, cname in enumerate(cap_names):
        fig, ax = plt.subplots(figsize=(16, 4))
        for run in runs:
            ls, lbl = style_lut[run["label"]]
            ax.plot(hours, _trim_step_matrix(run["cap_on"], n)[:, j], label=lbl, linestyle=ls)
        if da_gps_cap_by_name is not None and da_gps_hours is not None:
            y_cap = da_gps_cap_by_name.get(cname)
            if y_cap is None:
                y_cap = da_gps_cap_by_name.get(str(cname).lower())
            if y_cap is not None and np.isfinite(y_cap).any():
                ax.plot(
                    _trim_1d(da_gps_hours, n),
                    _trim_1d(y_cap, n),
                    label="DA-GPS",
                    linestyle="-.",
                    color="magenta",
                    linewidth=1.5,
                )
        ax.set_title(f"Capacitor {cname}")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("steps ON")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        plt.close(fig)


def plot_iteration_series(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    style_lut: dict[str, tuple[str, str]],
    *,
    field: str,
    ylabel: str,
    title: str,
    show: bool = True,
):
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    fig, ax = plt.subplots(figsize=(16, 4))
    for run in runs:
        ls, lbl = style_lut[run["label"]]
        ax.plot(hours, _trim_1d(run[field], n), label=lbl, linestyle=ls)
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def plot_all(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    style_lut: dict[str, tuple[str, str]],
    *,
    da_gps_voltages: dict[str, np.ndarray] | None = None,
    da_gps_reg_by_name: dict[str, np.ndarray] | None = None,
    da_gps_cap_by_name: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    voltage_suptitle: str,
    show: bool = True,
):
    plot_voltage_monitors(
        cfg,
        runs,
        style_lut,
        da_gps_voltages=da_gps_voltages,
        da_gps_hours=da_gps_hours,
        suptitle=voltage_suptitle,
        show=show,
    )
    plot_regulator_taps(
        cfg,
        runs,
        style_lut,
        da_gps_reg_by_name=da_gps_reg_by_name,
        da_gps_hours=da_gps_hours,
        show=show,
    )
    plot_capacitor_steps(
        cfg,
        runs,
        style_lut,
        da_gps_cap_by_name=da_gps_cap_by_name,
        da_gps_hours=da_gps_hours,
        show=show,
    )
    plot_iteration_series(
        cfg,
        runs,
        style_lut,
        field="control_iters",
        ylabel="Control iterations",
        title=f"Control iterations per {cfg.step_min}-min step (Solution.ControlIterations)",
        show=show,
    )
    plot_iteration_series(
        cfg,
        runs,
        style_lut,
        field="pf_iters_total",
        ylabel="PF iterations (total)",
        title=f"Power-flow iterations per {cfg.step_min}-min step (Solution.TotalIterations)",
        show=show,
    )


def plot_warmstart_voltage_band(
    cfg: DailySimConfig,
    *,
    monitor_nodes: list[str],
    volts: np.ndarray,
    volts_min: np.ndarray,
    volts_max: np.ndarray,
    n_warm_starts: int,
    da_gps_voltages: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    plot_warmstart_lines: bool = True,
    show: bool = True,
):
    """Shaded min–max band across independent warm starts with optional DA-GPS overlay."""
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    n_nodes = len(monitor_nodes)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 3.5 * n_nodes), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"OpenDSS |V| band from {n_warm_starts} independent random controller warm-starts per step",
        fontsize=11,
    )
    for ax, node, j in zip(axes, monitor_nodes, range(n_nodes)):
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
            v_da = da_gps_voltages.get(node)
            if v_da is None:
                v_da = da_gps_voltages.get(str(node).lower())
            if v_da is not None:
                ax.plot(
                    _trim_1d(da_gps_hours, n),
                    _trim_1d(v_da, n),
                    label="DA-GPS",
                    linestyle="-.",
                    color="magenta",
                    linewidth=1.5,
                )
        ax.set_title(node)
        ax.set_ylabel("|V| (pu)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Hour of day")
    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def plot_warmstart_device_band(
    cfg: DailySimConfig,
    *,
    device_names: list[str],
    values: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
    ylabel: str,
    title_prefix: str,
    da_gps_by_name: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    da_transform=None,
    n_warm_starts: int,
    plot_warmstart_lines: bool = True,
    show: bool = True,
):
    n = _plot_npts(cfg)
    hours = display_hours_array(cfg)
    for idx, dname in enumerate(device_names):
        fig, ax = plt.subplots(figsize=(16, 4))
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
            y_da = da_gps_by_name.get(dname)
            if y_da is None:
                y_da = da_gps_by_name.get(str(dname).lower())
            if y_da is not None and np.isfinite(y_da).any():
                y_plot = _trim_1d(y_da, n)
                if da_transform is not None:
                    y_plot = da_transform(y_plot)
                ax.plot(
                    _trim_1d(da_gps_hours, n),
                    y_plot,
                    label="DA-GPS",
                    linestyle="-.",
                    color="magenta",
                    linewidth=1.5,
                )
        ax.set_title(f"{title_prefix} {dname} ({n_warm_starts} warm starts/step)")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        plt.close(fig)


def plot_warmstart_band_all(
    cfg: DailySimConfig,
    *,
    monitor_nodes: list[str],
    reg_names: list[str],
    cap_names: list[str],
    volts: np.ndarray,
    volts_min: np.ndarray,
    volts_max: np.ndarray,
    taps: np.ndarray,
    taps_min: np.ndarray,
    taps_max: np.ndarray,
    cap_on: np.ndarray,
    cap_min: np.ndarray,
    cap_max: np.ndarray,
    da_gps_voltages: dict[str, np.ndarray] | None = None,
    da_gps_reg_by_name: dict[str, np.ndarray] | None = None,
    da_gps_cap_by_name: dict[str, np.ndarray] | None = None,
    da_gps_hours: np.ndarray | None = None,
    n_warm_starts: int,
    plot_reg_cap: bool = True,
    plot_warmstart_lines: bool = True,
    show: bool = True,
):
    plot_warmstart_voltage_band(
        cfg,
        monitor_nodes=monitor_nodes,
        volts=volts,
        volts_min=volts_min,
        volts_max=volts_max,
        n_warm_starts=n_warm_starts,
        da_gps_voltages=da_gps_voltages,
        da_gps_hours=da_gps_hours,
        plot_warmstart_lines=plot_warmstart_lines,
        show=show,
    )
    if not plot_reg_cap:
        return
    plot_warmstart_device_band(
        cfg,
        device_names=reg_names,
        values=taps,
        vmin=taps_min,
        vmax=taps_max,
        ylabel="tap #",
        title_prefix="Regulator",
        da_gps_by_name=da_gps_reg_by_name,
        da_gps_hours=da_gps_hours,
        da_transform=tap_pu_to_tap_number,
        n_warm_starts=n_warm_starts,
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
        da_gps_by_name=da_gps_cap_by_name,
        da_gps_hours=da_gps_hours,
        da_transform=None,
        n_warm_starts=n_warm_starts,
        plot_warmstart_lines=plot_warmstart_lines,
        show=show,
    )


def print_run_summary(
    cfg: DailySimConfig,
    runs: list[dict[str, Any]],
    *,
    header: str | None = None,
    compare_final: bool = True,
):
    if header:
        print(header)
    for run in runs:
        n_conv = int(run["converged"].sum())
        print(f"\n{run['label']}: {n_conv}/{cfg.npts} steps converged")
        print(
            f"  iterations: control mean={run['control_iters'].mean():.2f} "
            f"(max {run['control_iters'].max()}), "
            f"PF total mean={run['pf_iters_total'].mean():.2f} "
            f"(max {run['pf_iters_total'].max()})"
        )
        if "wall_s" in run:
            print(f"  wall time: {run['wall_s']:.1f}s; mode={run.get('mode', '')}")
        print("  final regulator taps:")
        for nm, tap in zip(run["reg_names"], run["taps"][-1]):
            print(f"    {nm}: {int(tap)}")
        print("  final cap steps ON:")
        for nm, st in zip(run["cap_names"], run["cap_on"][-1]):
            print(f"    {nm}: {int(st)}")

    if not compare_final or len(runs) < 2:
        return

    ref = runs[0]
    print("\nDifferences at final step:")
    diff_regs = [
        (nm, *(int(r["taps"][-1, i]) for r in runs))
        for i, nm in enumerate(ref["reg_names"])
        if len({int(r["taps"][-1, i]) for r in runs}) > 1
    ]
    diff_caps = [
        (nm, *(int(r["cap_on"][-1, j]) for r in runs))
        for j, nm in enumerate(ref["cap_names"])
        if len({int(r["cap_on"][-1, j]) for r in runs}) > 1
    ]
    print(f"  regulators with differing final taps: {len(diff_regs)}")
    for row in diff_regs:
        print(f"    {row[0]}: " + ", ".join(f"{runs[k]['label']}={row[k+1]}" for k in range(len(runs))))
    print(f"  capacitors with differing final steps ON: {len(diff_caps)}")
    for row in diff_caps:
        print(f"    {row[0]}: " + ", ".join(f"{runs[k]['label']}={row[k+1]}" for k in range(len(runs))))
