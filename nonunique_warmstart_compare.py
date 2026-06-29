"""Hybrid DA-GPS warm-start vs plain daily march (two OpenDSS trajectories)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import opendssdirect as dss

from nonunique_da_gps import align_da_gps_devices, load_da_gps_overlay
from nonunique_opendss_daily import (
    DailySimConfig,
    compile_and_setup,
    empty_run_arrays,
    inject_controller_warmstart,
    load_der_multiplier_profile,
    make_run_result,
    record_step,
    resolve_monitor_nodes,
    set_der_injection,
)
from nonunique_plots import plot_all, print_run_summary

WARMSTART_STYLES = [
    ("daily_no_warmstart", "-"),
    ("daily_da_gps_warmstart", "--"),
]

# OpenDSS does not expose a supported API to inject complex bus voltages as an initial
# guess before Solve() in daily or snapshot mode (verified 2026-06-29 via _tmp_voltage_seed_probe.py):
#   - Bus.puVmagAngle / PuVoltage / VMagAngle are read-only in opendssdirect
#   - Solution.InitSnap() only runs internal Y-matrix flat-start, not user V injection
#   - Daily QSTS already carries prior-step V as the PF initial guess automatically
# Warm-start here uses regulator TapNumber and capacitor States only (controller-state memory).


def run_daily_march(
    cfg: DailySimConfig,
    label: str,
    *,
    der_mult,
    warmstart: bool,
    reg_tap_pu_by_name: dict | None = None,
    cap_sigmoid_by_name: dict | None = None,
):
    compile_and_setup(cfg)
    reg_names = list(dss.RegControls.AllNames())
    cap_names = list(dss.Capacitors.AllNames())
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    arrays = empty_run_arrays(cfg, reg_names, cap_names, monitor_nodes)

    if warmstart:
        print(f"{label}: DA-GPS controller warm-start before each Solve() (incl. t=0)")
    else:
        print(f"{label}: plain daily march, OpenDSS defaults at t=0, no GNN injection")

    dss.RegControls.Name(reg_names[0])
    init_tap = dss.RegControls.TapNumber()
    dss.Capacitors.Name(cap_names[0])
    init_cap_on = sum(dss.Capacitors.States())
    print(f"  post-compile defaults tap[{reg_names[0]}]={init_tap}, cap_on[{cap_names[0]}]={init_cap_on}")

    for t in range(cfg.npts):
        if warmstart and reg_tap_pu_by_name and cap_sigmoid_by_name:
            inject_controller_warmstart(
                t,
                reg_names,
                cap_names,
                reg_tap_pu_by_name,
                cap_sigmoid_by_name,
            )
        if cfg.include_der and der_mult is not None:
            set_der_injection(cfg, t, der_mult)
        dss.Solution.Solve()
        record_step(
            t,
            reg_names,
            cap_names,
            monitor_nodes,
            arrays["taps"],
            arrays["cap_on"],
            arrays["volts"],
            arrays["converged"],
            arrays["control_iters"],
            arrays["pf_iters_total"],
            arrays["pf_iters_most"],
        )

    return make_run_result(label, reg_names, cap_names, monitor_nodes, arrays)


def run_warmstart_compare(
    cfg: DailySimConfig | None = None,
    *,
    show: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or DailySimConfig()
    inline_backend = plt.get_backend()
    der_mult = (
        load_der_multiplier_profile(cfg.der_profile_csv, cfg=cfg) if cfg.include_der else None
    )

    print(f"Warm-start compare: step_min={cfg.step_min} min, npts={cfg.npts}")
    print(
        "Voltage warm-start: NOT applied (OpenDSS has no supported pre-Solve V injection in daily mode); "
        "using regulator taps + capacitor states only."
    )

    # Probe monitor nodes from a throwaway compile to run DA-GPS once upfront.
    compile_and_setup(cfg)
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    reg_names = list(dss.RegControls.AllNames())
    cap_names = list(dss.Capacitors.AllNames())

    overlay = load_da_gps_overlay(cfg, monitor_nodes, device=device, inline_backend=inline_backend)
    if overlay is None:
        raise RuntimeError(
            "DA-GPS overlay is required for warm-start compare (set include_da_gps=True and check paths)"
        )
    da_v, da_reg, da_cap, da_hours = align_da_gps_devices(overlay, reg_names, cap_names)
    if not da_reg or not da_cap:
        raise RuntimeError("DA-GPS device heads did not align to OpenDSS reg/cap names")

    print("Running daily_no_warmstart...")
    run_no = run_daily_march(
        cfg,
        "daily_no_warmstart",
        der_mult=der_mult,
        warmstart=False,
    )
    print("Running daily_da_gps_warmstart...")
    run_ws = run_daily_march(
        cfg,
        "daily_da_gps_warmstart",
        der_mult=der_mult,
        warmstart=True,
        reg_tap_pu_by_name=da_reg,
        cap_sigmoid_by_name=da_cap,
    )
    runs = [run_no, run_ws]
    style_lut = {lbl: (ls, lbl) for lbl, ls in WARMSTART_STYLES}

    plot_all(
        cfg,
        runs,
        style_lut,
        da_gps_voltages=da_v,
        da_gps_reg_by_name=da_reg,
        da_gps_cap_by_name=da_cap,
        da_gps_hours=da_hours,
        voltage_suptitle="DA-GPS warm-start vs plain daily march (monitor |V|)",
        show=show,
    )

    print_run_summary(
        cfg,
        runs,
        header=(
            "\nDA-GPS overlay (magenta): reference prediction for |V|, taps, caps.\n"
            "Iteration comparison (warm-start should reduce or match control/PF iterations when "
            "injection lands near the converged manifold):"
        ),
    )

    ctrl_no = float(run_no["control_iters"].mean())
    ctrl_ws = float(run_ws["control_iters"].mean())
    pf_no = float(run_no["pf_iters_total"].mean())
    pf_ws = float(run_ws["pf_iters_total"].mean())
    print(
        f"\nMean control iterations: no_warmstart={ctrl_no:.2f}, da_gps_warmstart={ctrl_ws:.2f} "
        f"(delta {ctrl_ws - ctrl_no:+.2f})"
    )
    print(
        f"Mean PF iterations:      no_warmstart={pf_no:.2f}, da_gps_warmstart={pf_ws:.2f} "
        f"(delta {pf_ws - pf_no:+.2f})"
    )

    return {
        "cfg": cfg,
        "runs": runs,
        "da_gps_voltages": da_v,
        "da_gps_reg_by_name": da_reg,
        "da_gps_cap_by_name": da_cap,
        "da_gps_hours": da_hours,
        "mean_control_iters": {"daily_no_warmstart": ctrl_no, "daily_da_gps_warmstart": ctrl_ws},
        "mean_pf_iters": {"daily_no_warmstart": pf_no, "daily_da_gps_warmstart": pf_ws},
    }
