"""OpenDSS daily march with fixed controller initialization each step.

Same load/PV profile application as Method A / parity helpers, but before every
``Solve()`` restores:
  - all RegControl TapNumber = 0
  - CapControl banks OFF
  - fixed (no CapControl) banks ON (e.g. CAPBank3)

Writes / merges ``__dss_fixed_*`` columns into an existing Method A day folder
(``daily_voltage_monitor.csv``, regulator/cap/meta CSVs).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

import compare_opendss_snapshot_helpers as parity
import run_injection_dataset as inj
import run_original_style_dataset_8500_unbalanced as rd8500


STEP_MIN = 5


def _read_cap_n_on(cap_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for nm in cap_names:
        try:
            dss.Capacitors.Name(str(nm))
            st = dss.Capacitors.States()
            if st is None:
                steps: list[int] = []
            elif isinstance(st, (list, tuple, np.ndarray)):
                steps = [int(x) for x in st]
            else:
                steps = [int(st)]
            out[str(nm)] = float(sum(1 for x in steps if int(x) > 0))
        except Exception:
            out[str(nm)] = float("nan")
    return out


def _read_reg_taps(reg_names: list[str]) -> dict[str, float]:
    return rd8500._read_reg_control_state(reg_names)


def _read_pv_pq(pv_names: list[str]) -> dict[str, tuple[float, float]]:
    """(+P,+Q) kW/kvar **injected** into the feeder — same sign as Method A / dataset."""
    return rd8500._read_pv_totals_post_solve_kw_kvar(pv_names)


def _node_vmag_vang(nodes: list[str]) -> dict[str, tuple[float, float]]:
    """Return {node: (vmag_pu, vang_deg)} for requested bus.phase nodes."""
    names, _, _, _ = inj.get_all_bus_phase_nodes()
    vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(names)
    idx = {str(n).lower(): i for i, n in enumerate(names)}
    out: dict[str, tuple[float, float]] = {}
    for n in nodes:
        i = idx.get(str(n).lower())
        if i is None:
            out[str(n)] = (float("nan"), float("nan"))
        else:
            out[str(n)] = (float(vmag_m[i]), float(vang_m[i]))
    return out


def run_fixedctrl_opendss_day(
    *,
    load_csv: Path,
    irr_csv: Path,
    npts: int = 288,
    step_min: int = STEP_MIN,
    scenario_scale: float = 1.0,
    daily_stress: float = 0.0,
    stress_clip_lo: float = 0.1,
    stress_clip_hi: float = 3.0,
    monitor_nodes: list[str] | None = None,
    reset_mode: str = "per_step",
) -> dict[str, object]:
    """Solve one representative day with fixed controller initialization.

    ``reset_mode``:
      - ``\"per_step\"`` (default): reset taps/caps before **every** Solve — same rule as
        training fixed-controller-init (independent Static settles each step).
      - ``\"per_day\"``: reset once at step 0, then carry (optional; not used for paper).

    Reset rule (unchanged): RegControl TapNumber=0; CapControl banks OFF; fixed banks
    (no CapControl, e.g. CAPBank3) ON.

    Returns dict with arrays / name lists used by ``merge_fixedctrl_into_day_dir``.
    """
    mode = str(reset_mode).strip().lower()
    if mode not in ("per_day", "per_step"):
        raise ValueError(f"reset_mode must be 'per_day' or 'per_step', got {reset_mode!r}")

    load_csv = Path(load_csv)
    irr_csv = Path(irr_csv)
    profiles = parity.prepare_parity_profiles(
        load_csv,
        irr_csv,
        npts=int(npts),
        step_min=float(step_min),
        daily_stress=float(daily_stress),
        stress_clip_lo=float(stress_clip_lo),
        stress_clip_hi=float(stress_clip_hi),
    )
    parity.compile_and_bind_parity_daily_opendss(
        profiles,
        npts=int(npts),
        step_min=float(step_min),
    )
    rd8500._detach_daily_loadshape_from_loads()
    # Same as Method A snapshot path: neutralize PV Daily= irradiance shapes so
    # explicit ``Pmpp = Pmpp0 × m_irr`` is not double-counted (→ fake "curtailment").
    parity.neutralize_pv_irrad_loadshape_for_snapshot(
        npts=int(npts), step_min=float(step_min)
    )
    try:
        dss.Text.Command("set mode=snapshot")
        dss.Text.Command("set controlmode=static")
        # Per-step reset starts from TapNumber=0 / CapControl OFF every sample.
        # Settling that climb often needs ~60–80 control iterations; too low a cap
        # stops mid-hunt. Unfinished control → skip (NaN), never save/plot as data.
        _max_ctrl = 120 if mode == "per_step" else 30
        dss.Text.Command(f"set maxcontroliter={int(_max_ctrl)}")
        dss.Text.Command("set number=1")
    except Exception:
        _max_ctrl = 120 if mode == "per_step" else 30
        pass

    base_names, base_kw, base_kvar = parity.collect_unscaled_load_bases()
    pv_names = sorted(dss.PVsystems.AllNames() or [])
    pv_base: dict[str, float] = {}
    for nm in pv_names:
        try:
            dss.PVsystems.Name(str(nm))
            pv_base[str(nm)] = float(dss.PVsystems.Pmpp())
        except Exception:
            pv_base[str(nm)] = 0.0
    # restore nameplate after any prior Pmpp scaling
    for nm, p0 in pv_base.items():
        try:
            dss.PVsystems.Name(nm)
            dss.PVsystems.Pmpp(float(p0))
        except Exception:
            pass

    reg_names = rd8500._discover_reg_controls()
    cap_names = rd8500._discover_capacitors()
    controlled_caps = sorted(rd8500._controlled_capacitor_names())
    if not monitor_nodes:
        monitor_nodes = ["l3216370.1", "l3216370.2", "l3216370.3"]

    hours = np.arange(int(npts), dtype=np.float64) * (float(step_min) / 60.0)
    reg_tap = {f"reg_{nm}_tap_pu": np.full(npts, np.nan) for nm in reg_names}
    # OpenDSS CapControl names may differ in case; store by lowercase capacitor name
    cap_on = {f"cap_{nm}_n_steps_on": np.full(npts, np.nan) for nm in cap_names}
    vmag = {n: np.full(npts, np.nan) for n in monitor_nodes}
    vang = {n: np.full(npts, np.nan) for n in monitor_nodes}
    pv2_p = np.full(npts, np.nan)
    pv2_q = np.full(npts, np.nan)
    p_loss = np.full(npts, np.nan)
    q_loss = np.full(npts, np.nan)
    converged = np.zeros(npts, dtype=bool)
    skip_reasons: list[str] = []

    # identify pv2 by name
    pv2_name = None
    for nm in pv_names:
        if "pv2" in str(nm).lower():
            pv2_name = str(nm)
            break
    if pv2_name is None and pv_names:
        pv2_name = str(pv_names[0])

    print(
        f"[fixedctrl] reset_mode={mode}  maxcontroliter={_max_ctrl}  regs={len(reg_names)}  "
        f"CapControl={controlled_caps}  "
        f"fixed_ON={[c for c in cap_names if str(c).lower() not in set(controlled_caps)]}",
        flush=True,
    )

    t0 = time.perf_counter()
    for i in range(int(npts)):
        m_t = parity.step_load_multiplier(profiles.m_eff, i, float(scenario_scale))
        ir_t = parity.step_irradiance_multiplier(profiles.m_irr, i)
        # Reset Pmpp bases then scale (apply_explicit sets Pmpp = base * ir)
        for nm, p0 in pv_base.items():
            try:
                dss.PVsystems.Name(nm)
                dss.PVsystems.Pmpp(float(p0))
            except Exception:
                pass
        parity.apply_explicit_loads_and_pv_pmpp(
            base_names=base_names,
            base_kw=base_kw,
            base_kvar=base_kvar,
            m_t=float(m_t),
            pv_names=pv_names,
            pv_base_pmpp_kw=pv_base,
            ir_t=float(ir_t),
        )
        hr = int(i // 12)
        sec = int((i % 12) * (int(step_min) * 60))
        dss.Text.Command(f"set hour={hr} sec={sec}")

        if mode == "per_step" or i == 0:
            rd8500._reset_controllers_to_compile_defaults(reg_names, cap_names)

        solve_err: str | None = None
        try:
            dss.Solution.Solve()
        except Exception as exc:  # noqa: BLE001
            solve_err = str(exc)
        pf_ok = bool(dss.Solution.Converged())
        try:
            n_ctrl = int(dss.Solution.ControlIterations())
        except Exception:
            n_ctrl = -1
        ctrl_hit_cap = bool(n_ctrl >= int(_max_ctrl))
        ctrl_err = bool(solve_err and "control" in solve_err.lower())
        ok = bool(pf_ok) and not ctrl_hit_cap and not ctrl_err and solve_err is None
        # If DSS only complained about control iters, still treat as control failure.
        if ctrl_err or ctrl_hit_cap:
            ok = False
        elif solve_err is not None and not pf_ok:
            ok = False
        elif solve_err is not None and pf_ok and not ctrl_err:
            # Unexpected non-control error with PF converged — skip to be safe.
            ok = False
        converged[i] = ok
        if not ok:
            reason = (
                f"step={i} hour={hours[i]:.3f} pf_ok={pf_ok} "
                f"ctrl_iters={n_ctrl}/{_max_ctrl} err={solve_err!r}"
            )
            skip_reasons.append(reason)
            print(f"[fixedctrl] SKIP (no save/plot) {reason}", flush=True)
            continue

        taps = rd8500._read_reg_control_state(reg_names)
        for k, v in taps.items():
            if k in reg_tap:
                reg_tap[k][i] = float(v)
        caps = _read_cap_n_on(cap_names)
        for nm, v in caps.items():
            key = f"cap_{nm}_n_steps_on"
            # match case-insensitive to discovered keys
            for kk in list(cap_on.keys()):
                if kk.lower() == key.lower():
                    cap_on[kk][i] = float(v)
                    break
        vv = _node_vmag_vang(monitor_nodes)
        for n, (vm, va) in vv.items():
            vmag[n][i] = vm
            vang[n][i] = va
        if pv2_name:
            pq = _read_pv_pq([pv2_name]).get(pv2_name, (np.nan, np.nan))
            pv2_p[i] = float(pq[0])
            pv2_q[i] = float(pq[1])
        pl, ql = rd8500._circuit_losses_kw_kvar()
        p_loss[i] = float(pl)
        q_loss[i] = float(ql)

        if (i + 1) % 48 == 0 or (i + 1) == npts:
            print(
                f"[fixedctrl] step {i+1}/{npts}  ok={int(converged.sum())}  "
                f"skipped={len(skip_reasons)}  elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    n_ok = int(np.sum(converged))
    n_skip = int(npts) - n_ok
    print(
        f"[fixedctrl] DONE reset_mode={mode} maxcontroliter={_max_ctrl}  "
        f"ok={n_ok}/{npts} skipped={n_skip}  wall_s={time.perf_counter()-t0:.1f}",
        flush=True,
    )
    if skip_reasons:
        print(
            f"[fixedctrl] WARNING: {n_skip} steps left as NaN (controller/PF did not fully settle). "
            f"They will NOT be plotted as real device values.",
            flush=True,
        )

    return {
        "hours": hours,
        "converged": converged,
        "reg_tap": reg_tap,
        "cap_on": cap_on,
        "vmag": vmag,
        "vang": vang,
        "pv2_p": pv2_p,
        "pv2_q": pv2_q,
        "p_loss": p_loss,
        "q_loss": q_loss,
        "pv2_name": pv2_name,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "monitor_nodes": list(monitor_nodes),
        "wall_s": float(time.perf_counter() - t0),
        "reset_mode": mode,
        "controlled_caps": controlled_caps,
        "maxcontroliter": int(_max_ctrl),
        "n_skipped": n_skip,
        "skip_reasons": skip_reasons,
    }


def _merge_col(df: pd.DataFrame, col: str, values: np.ndarray) -> None:
    n = len(df)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < n:
        pad = np.full(n - len(arr), np.nan)
        arr = np.concatenate([arr, pad])
    elif len(arr) > n:
        arr = arr[:n]
    df[col] = arr


def merge_fixedctrl_into_day_dir(day_dir: Path, result: dict[str, object]) -> dict[str, Path]:
    """Add ``__dss_fixed_*`` columns into Method A CSVs under ``day_dir``."""
    day_dir = Path(day_dir)
    written: dict[str, Path] = {}
    hours = np.asarray(result["hours"], dtype=np.float64)

    # ----- voltage -----
    volt_path = day_dir / "daily_voltage_monitor.csv"
    if volt_path.is_file():
        vdf = pd.read_csv(volt_path)
    else:
        vdf = pd.DataFrame(
            {
                "step_idx": np.arange(len(hours), dtype=int),
                "hour": hours,
            }
        )
    for n, arr in result["vmag"].items():  # type: ignore[union-attr]
        stem = str(n)
        alt = stem.replace(".", "_")
        _merge_col(vdf, f"{stem}__dss_fixed_vmag_pu", arr)  # type: ignore[arg-type]
        _merge_col(vdf, f"{alt}__dss_fixed_vmag_pu", arr)  # type: ignore[arg-type]
    for n, arr in result["vang"].items():  # type: ignore[union-attr]
        stem = str(n)
        alt = stem.replace(".", "_")
        _merge_col(vdf, f"{stem}__dss_fixed_vang_deg", arr)  # type: ignore[arg-type]
        _merge_col(vdf, f"{alt}__dss_fixed_vang_deg", arr)  # type: ignore[arg-type]
    vdf.to_csv(volt_path, index=False)
    written["voltage"] = volt_path

    # ----- regulators -----
    reg_hits = sorted(day_dir.glob("daily_regulator_tap_*.csv"))
    if reg_hits:
        rdf = pd.read_csv(reg_hits[0])
        reg_path = reg_hits[0]
    else:
        reg_path = day_dir / "daily_regulator_tap_fixedctrl.csv"
        rdf = pd.DataFrame({"step_idx": np.arange(len(hours), dtype=int), "hour": hours})
    for stem, arr in result["reg_tap"].items():  # type: ignore[union-attr]
        _merge_col(rdf, f"{stem}__dss_fixed_tap_pu", arr)  # type: ignore[arg-type]
    rdf.to_csv(reg_path, index=False)
    written["reg"] = reg_path

    # ----- capacitors -----
    cap_hits = sorted(day_dir.glob("daily_cap_bank_status_*.csv"))
    if cap_hits:
        cdf = pd.read_csv(cap_hits[0])
        cap_path = cap_hits[0]
    else:
        cap_path = day_dir / "daily_cap_bank_status_fixedctrl.csv"
        cdf = pd.DataFrame({"step_idx": np.arange(len(hours), dtype=int), "hour": hours})
    for stem, arr in result["cap_on"].items():  # type: ignore[union-attr]
        _merge_col(cdf, f"{stem}__dss_fixed_n_steps_on", arr)  # type: ignore[arg-type]
    cdf.to_csv(cap_path, index=False)
    written["cap"] = cap_path

    # ----- meta aux -----
    meta_hits = sorted(day_dir.glob("daily_meta_aux_*.csv"))
    if meta_hits:
        mdf = pd.read_csv(meta_hits[0])
        meta_path = meta_hits[0]
    else:
        meta_path = day_dir / "daily_meta_aux_fixedctrl.csv"
        mdf = pd.DataFrame({"step_idx": np.arange(len(hours), dtype=int), "hour": hours})
    _merge_col(mdf, "pv_pv2_p_post_kw__dss_fixed", result["pv2_p"])  # type: ignore[arg-type]
    _merge_col(mdf, "pv_pv2_q_post_kvar__dss_fixed", result["pv2_q"])  # type: ignore[arg-type]
    _merge_col(mdf, "p_loss_total_post_kw__dss_fixed", result["p_loss"])  # type: ignore[arg-type]
    _merge_col(mdf, "q_loss_total_post_kvar__dss_fixed", result["q_loss"])  # type: ignore[arg-type]
    mdf.to_csv(meta_path, index=False)
    written["meta"] = meta_path

    # summary sidecar
    side = day_dir / "fixedctrl_opendss_summary.json"
    import json

    side.write_text(
        json.dumps(
            {
                "wall_s": result["wall_s"],
                "n_converged": int(np.sum(result["converged"])),  # type: ignore[arg-type]
                "npts": int(len(hours)),
                "monitor_nodes": result["monitor_nodes"],
                "pv2_name": result["pv2_name"],
                "reset_mode": result.get("reset_mode"),
                "controlled_caps": result.get("controlled_caps"),
                "maxcontroliter": result.get("maxcontroliter"),
                "n_skipped": result.get("n_skipped"),
                "n_ok": int(np.sum(result["converged"])) if result.get("converged") is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["summary"] = side
    return written
