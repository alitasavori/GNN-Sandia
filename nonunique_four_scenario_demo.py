"""Four-scenario non-uniqueness demo (init_low / init_high / init_default / snapshot_cold)."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import opendssdirect as dss

from nonunique_da_gps import align_da_gps_devices, load_da_gps_overlay
from nonunique_opendss_daily import (
    DailySimConfig,
    apply_explicit_loads_pv,
    compile_and_setup,
    detach_daily_loadshape,
    empty_run_arrays,
    load_der_multiplier_profile,
    make_run_result,
    neutralize_irrad_loadshape,
    read_loadshape_mult,
    record_step,
    resolve_monitor_nodes,
    set_controllers,
    set_der_injection,
    collect_load_bases,
    collect_pv_bases,
)
from nonunique_plots import plot_all, print_run_summary

RUN_STYLES = [
    ("init_low", "-"),
    ("init_high", "--"),
    ("init_default", ":"),
    ("snapshot_cold", "dotted"),
]


def run_march(cfg: DailySimConfig, label: str, *, low_init: bool | None, der_mult):
    compile_and_setup(cfg)
    if low_init is not None:
        set_controllers(low_init)
    reg_names = list(dss.RegControls.AllNames())
    cap_names = list(dss.Capacitors.AllNames())
    monitor_nodes = resolve_monitor_nodes(cfg.monitor_candidates)
    arrays = empty_run_arrays(cfg, reg_names, cap_names, monitor_nodes)

    dss.RegControls.Name(reg_names[0])
    init_tap = dss.RegControls.TapNumber()
    dss.Capacitors.Name(cap_names[0])
    init_cap_on = sum(dss.Capacitors.States())
    if low_init is None:
        print(
            f"{label}: OpenDSS post-compile defaults "
            f"tap[{reg_names[0]}]={init_tap}, cap_on[{cap_names[0]}]={init_cap_on}"
        )
    else:
        print(f"{label}: pre-solve tap[{reg_names[0]}]={init_tap}, cap_on[{cap_names[0]}]={init_cap_on}")

    for t in range(cfg.npts):
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


def run_snapshot_cold(cfg: DailySimConfig, label: str = "snapshot_cold", *, der_mult):
    print(f"Running {label} ({cfg.npts} cold snapshot solves, step_min={cfg.step_min} min)...")
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
    arrays = empty_run_arrays(cfg, reg_names, cap_names, monitor_nodes)

    t_wall0 = time.perf_counter()
    for t in range(cfg.npts):
        dss.Basic.ClearAll()
        compile_and_setup(cfg, snapshot=True)
        detach_daily_loadshape()
        neutralize_irrad_loadshape(cfg)
        apply_explicit_loads_pv(
            load_names, base_kw, base_kvar, pv_names, pv_base, load_mult[t], irr_mult[t]
        )
        if cfg.include_der and der_mult is not None:
            set_der_injection(cfg, t, der_mult)
        dss.Text.Command(f"set hour={t // cfg.steps_per_hour} sec={(t % cfg.steps_per_hour) * cfg.step_sec}")
        dss.Text.Command("set mode=snapshot")
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

    wall_s = time.perf_counter() - t_wall0
    mode_id = int(dss.Solution.Mode())
    mode_txt = "snapshot" if mode_id == 0 else f"mode={mode_id}"
    print(f"{label}: wall time {wall_s:.1f}s; Solution.Mode() after last step = {mode_txt}")
    return make_run_result(
        label,
        reg_names,
        cap_names,
        monitor_nodes,
        arrays,
        wall_s=wall_s,
        mode=mode_txt,
    )


def run_four_scenario_demo(
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

    print(f"Simulation timestep: step_min={cfg.step_min} min, npts={cfg.npts} ({cfg.day_hours}h day)")
    if cfg.include_der:
        print("=" * 72)
        print(
            f"DER injection ON: bus={cfg.der_bus}, nominal P={cfg.der_nominal_kw} kW, "
            f"Q={cfg.der_nominal_kvar} kvar"
        )
        print(f"  profile: {cfg.der_profile_csv}")
        if der_mult is not None:
            print(f"  multiplier range [{der_mult.min():.4f}, {der_mult.max():.4f}]")
        print("=" * 72)

    print("Running init_low (low taps, caps OFF)...")
    run_a = run_march(cfg, "init_low", low_init=True, der_mult=der_mult)
    print("Running init_high (high taps, caps ON)...")
    run_b = run_march(cfg, "init_high", low_init=False, der_mult=der_mult)
    print("Running init_default (OpenDSS post-compile defaults)...")
    run_c = run_march(cfg, "init_default", low_init=None, der_mult=der_mult)
    run_d = run_snapshot_cold(cfg, der_mult=der_mult)
    all_runs = [run_a, run_b, run_c, run_d]
    style_lut = {lbl: (ls, lbl) for lbl, ls in RUN_STYLES}

    overlay = load_da_gps_overlay(cfg, run_a["monitor_nodes"], device=device, inline_backend=inline_backend)
    da_v, da_reg, da_cap, da_hours = align_da_gps_devices(
        overlay, run_a["reg_names"], run_a["cap_names"]
    )

    plot_all(
        cfg,
        all_runs,
        style_lut,
        da_gps_voltages=da_v,
        da_gps_reg_by_name=da_reg,
        da_gps_cap_by_name=da_cap,
        da_gps_hours=da_hours,
        voltage_suptitle=(
            "Control hysteresis / non-uniqueness: daily marches vs cold snapshot (no path dependence)"
        ),
        show=show,
    )

    if da_v is not None:
        print(
            "\nDA-GPS overlay (magenta dash-dot): monitor |V|, regulator tap, "
            "capacitor P(bank on) vs OpenDSS runs."
        )
    print_run_summary(cfg, all_runs, compare_final=True)

    return {
        "cfg": cfg,
        "runs": all_runs,
        "da_gps_voltages": da_v,
        "da_gps_reg_by_name": da_reg,
        "da_gps_cap_by_name": da_cap,
        "da_gps_hours": da_hours,
    }
