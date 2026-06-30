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
