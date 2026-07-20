"""Daily OpenDSS vs Log(v) 3LPF for local ieee34 / 906 / 8500 masters.

Drives a full day (default 288 x 5 min) from feeder daily profiles:
- **8500:** ``a representativ days/load_day_*.csv`` + ``irr_day_*.csv``
- **ieee34:** native DSS ``daily=`` loadshapes (Mode=Daily)
- **906:** LVTestCase ``Yearly=`` shapes copied to ``Daily=`` then Mode=Daily

Reports per-step and day-summary accuracy; plots |V|, regulator taps, and
capacitor on/off/kvar evolution. Optional sequential FastLogv vs OpenDSS
wall-clock over the same daily samples.

Fair speed+accuracy protocol (``control_mode=synced`` + ``time_speed=True``):
  OD Static settles → sync loads/PV/caps/**taps** into Log(v) → rebuild ``A`` when
  taps change (analytical Tau; regulator ``Ỹ`` on A, ``ỹ`` on RHS) → FastLogv
  RHS+solve. Metrics/plots use Fast voltages (same control state as OD).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_PLOT_NODES = {
    # Distal / spot-load + regulated buses (P/Z/I models + LTC secondary).
    "ieee34": ["890.1", "844.1", "840.1", "814r.1", "852r.1"],
    # LVTestCase shapes are Yearly= (not Daily=); bus 1 is near the stiff source.
    "906": ["817.1", "860.1", "896.1", "906.1"],
    "8500": ["l2841632.1", "190-8593.1"],
}
DEFAULT_REG_COLS = {
    "ieee34": [
        "reg_rega1_tap_pu",
        "reg_rega2_tap_pu",
        "reg_rega3_tap_pu",
        "reg_regb1_tap_pu",
        "reg_regb2_tap_pu",
        "reg_regb3_tap_pu",
    ],
    "906": [],
    "8500": ["reg_vreg3_a_tap_pu", "reg_feeder_rega_tap_pu", "reg_vreg2_a_tap_pu"],
}
DEFAULT_CAP_NAMES = {
    "ieee34": ["c844", "c848"],
    "906": [],
    "8500": ["capbank0a", "capbank1a", "capbank2a", "capbank3"],
}


def _ensure_paths(repo: Path) -> None:
    repo = Path(repo).resolve()
    pkg = str(repo / "Log-v-3LPF")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)


def _bus_key(case, bus: str) -> str:
    b = str(bus)
    if b in case.bus_phases:
        return b
    bl = b.lower()
    for k in case.bus_phases:
        if str(k).lower() == bl:
            return str(k)
    raise KeyError(bus)


def _node_vm_va(case, node: str, algo: str) -> tuple[float, float]:
    s = str(node).strip().lower()
    if "." not in s:
        raise ValueError(f"expected bus.phase, got {node!r}")
    bus_s, ph_s = s.rsplit(".", 1)
    ph = int(ph_s)
    bus = _bus_key(case, bus_s)
    phases = list(case.bus_phases[bus])
    if ph not in phases:
        raise KeyError(f"phase {ph} not on bus {bus} (have {phases})")
    i = phases.index(ph)
    vm = float(np.asarray(case.results[algo]["vm"][bus], dtype=float)[i])
    va = float(np.asarray(case.results[algo]["va"][bus], dtype=float)[i])
    return vm, va


def _pick_plot_nodes(case, requested: list[str]) -> list[str]:
    out = []
    for n in requested:
        try:
            _bus_key(case, n.rsplit(".", 1)[0])
            out.append(str(n).strip().lower())
        except Exception:
            continue
    if out:
        return out
    for bus, phases in case.bus_phases.items():
        if phases:
            return [f"{str(bus).lower()}.{int(phases[0])}"]
    raise RuntimeError("no plot nodes available")


def _ensure_daily_shapes_from_yearly() -> int:
    """LVTestCase (906) attaches shapes as ``Yearly=``, not ``Daily=``.

    ``Mode=Daily`` only samples the Daily property, so without this copy the
    day stays at base kW and |V| looks flat. Returns number of loads updated.
    """
    import opendssdirect as dss

    n = 0
    try:
        names = [x for x in dss.Loads.AllNames() if x and x != "NONE"]
    except Exception:
        return 0
    for name in names:
        try:
            dss.Loads.Name(name)
            yearly = str(dss.Loads.Yearly() or "").strip()
            daily = str(dss.Loads.Daily() or "").strip()
            if not yearly or yearly.upper() == "NONE":
                continue
            if daily and daily.upper() != "NONE":
                continue
            dss.Text.Command(f"Edit Load.{name} Daily={yearly}")
            n += 1
        except Exception:
            continue
    return n


def _detach_daily_from_loads() -> None:
    """Clear ``Daily=`` on loads so explicit kW/kvar are not scaled twice."""
    import opendssdirect as dss

    try:
        if not dss.Loads.First():
            return
    except Exception:
        return
    while True:
        nm = dss.Loads.Name()
        dss.Loads.Name(nm)
        try:
            dss.Loads.Daily("")
        except Exception:
            pass
        if not dss.Loads.Next():
            break


def _loadshape_pmult_array(shape_name: str, *, npts: int, step_min: float) -> np.ndarray:
    """Read (and resample) a loadshape PMult vector for snapshot explicit loads."""
    import opendssdirect as dss

    from nonunique_opendss_daily import resample_daily_profile

    name = str(shape_name or "").strip()
    if not name or name.upper() == "NONE":
        return np.ones(int(npts), dtype=np.float64)
    mult = None
    for api in ("LoadShapes", "LoadShape"):
        try:
            getattr(dss, api).Name(name)
            mult = np.asarray(getattr(dss, api).PMult(), dtype=np.float64).ravel()
            break
        except Exception:
            continue
    if mult is None or mult.size == 0:
        return np.ones(int(npts), dtype=np.float64)
    if mult.size != int(npts):
        mult = resample_daily_profile(
            mult, npts=int(npts), step_min=int(round(float(step_min)))
        )
    return mult[: int(npts)]


def _collect_native_snapshot_loads(
    *, npts: int, step_min: float
) -> tuple[list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    """Nameplate kW/kvar + per-step mult arrays from native DSS loadshapes."""
    import opendssdirect as dss

    entries: list[tuple[str, float, float, str]] = []
    try:
        if dss.Loads.First():
            while True:
                nm = str(dss.Loads.Name())
                dss.Loads.Name(nm)
                if not nm.lower().startswith("pv_"):
                    daily = str(dss.Loads.Daily() or "").strip()
                    if not daily or daily.upper() == "NONE":
                        yearly = str(dss.Loads.Yearly() or "").strip()
                        daily = yearly if yearly and yearly.upper() != "NONE" else ""
                    entries.append(
                        (nm, float(dss.Loads.kW()), float(dss.Loads.kvar()), daily)
                    )
                if not dss.Loads.Next():
                    break
    except Exception:
        pass
    _detach_daily_from_loads()
    names = [e[0] for e in entries]
    base_kw = np.array([e[1] for e in entries], dtype=np.float64)
    base_kvar = np.array([e[2] for e in entries], dtype=np.float64)
    mult_arrays = [
        _loadshape_pmult_array(e[3], npts=int(npts), step_min=float(step_min))
        for e in entries
    ]
    return names, base_kw, base_kvar, mult_arrays


def _collect_native_pv_irradiance(
    *, npts: int, step_min: float
) -> tuple[list[str], dict[str, float], dict[str, np.ndarray]]:
    """PV nameplate Pmpp and per-step irradiance mult (explicit snapshot Pmpp)."""
    import opendssdirect as dss

    from compare_opendss_snapshot_helpers import discover_pv_system_names, read_pv_base_pmpp_kw

    pv_names = discover_pv_system_names()
    base_pmpp = read_pv_base_pmpp_kw(pv_names)
    irr_mults: dict[str, np.ndarray] = {}
    for pv in pv_names:
        shape = ""
        try:
            dss.PVsystems.Name(pv)
            shape = str(dss.PVsystems.Daily() or "").strip()
        except Exception:
            pass
        if not shape or shape.upper() == "NONE":
            shape = "IrradDay001"
        irr_mults[pv] = _loadshape_pmult_array(
            shape, npts=int(npts), step_min=float(step_min)
        )
    return pv_names, base_pmpp, irr_mults


def _apply_native_loads_at_step(
    step_i: int,
    base_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    load_mult_arrays: list[np.ndarray],
    *,
    scenario_scale: float = 1.0,
) -> None:
    import opendssdirect as dss

    sc = float(scenario_scale)
    i = int(step_i)
    for j, name in enumerate(base_names):
        m = float(load_mult_arrays[j][i]) * sc
        dss.Loads.Name(name)
        dss.Loads.kW(float(base_kw[j] * m))
        dss.Loads.kvar(float(base_kvar[j] * m))


def _apply_native_pv_pmpp_at_step(
    step_i: int,
    pv_names: list[str],
    pv_base_pmpp: dict[str, float],
    pv_irr_mults: dict[str, np.ndarray],
) -> None:
    import opendssdirect as dss

    i = int(step_i)
    for pv_nm in pv_names:
        b0 = float(pv_base_pmpp.get(pv_nm, 0.0))
        if b0 <= 0.0:
            continue
        ir = float(pv_irr_mults[pv_nm][i])
        try:
            dss.PVsystems.Name(pv_nm)
            dss.PVsystems.Pmpp(b0 * ir)
        except Exception:
            pass


def _import_logv_daily_timing():
    """Import timing helpers; reload if a stale kernel module lacks Log(v) symbols."""
    import importlib
    import sys

    mod = sys.modules.get("compare_mv_daily_timing")
    if mod is not None and not hasattr(mod, "print_logv_daily_timing_summary"):
        importlib.reload(mod)
    from compare_mv_daily_timing import (  # noqa: PLC0415
        print_logv_daily_timing_summary,
        print_logv_extra_od_probes,
    )

    return print_logv_daily_timing_summary, print_logv_extra_od_probes


def print_logv_daily_speed_summary(series: dict[str, Any], *, feeder: str = "") -> None:
    """Notebook-friendly Method A-style speed summary from ``run_daily_opendss_vs_logv3lpf`` output."""
    import numpy as np

    print_logv_daily_timing_summary, print_logv_extra_od_probes = _import_logv_daily_timing()

    summ = series.get("summary") or {}
    n_ok = int(summ.get("n_ok") or np.isfinite(series.get("mae_vm_all_nodes", [])).sum())
    npts = int(series.get("npts") or 0)
    n_bad = int(summ.get("n_bad") or series.get("n_bad") or 0)
    od = series.get("od_timing") or {}
    print_logv_daily_timing_summary(
        n_ok=n_ok,
        npts=npts,
        n_nonconv=n_bad,
        open_apply_s_total=float(od.get("apply_s", 0.0)),
        open_reassert_s_total=float(od.get("reassert_s", 0.0)),
        open_solve_only_s_total=float(od.get("solve_s", 0.0)),
        open_get_s_total=float(od.get("collect_s", 0.0)),
        logv_stock_s_total=float(np.nansum(series.get("t_logv_s", []))),
        fast_forward_s_total=float(np.nansum(series.get("t_fast_s", []))),
        fast_refresh_s_total=float(np.nansum(series.get("t_fast_refresh_s", []))),
        n_fast_refresh=int(summ.get("n_fast_refresh") or 0),
        feeder=str(feeder or series.get("feeder") or ""),
        control_mode=str(series.get("control_mode") or ""),
    )
    warm = float(od.get("warm_daily_probe_s", 0.0) or 0.0)
    cold = float(od.get("cold_snapshot_probe_s", 0.0) or 0.0)
    if warm > 0.0 or cold > 0.0:
        print_logv_extra_od_probes(
            n_ok=n_ok,
            warm_daily_s_total=warm if warm > 0.0 else None,
            cold_snapshot_s_total=cold if cold > 0.0 else None,
        )
    mae = np.asarray(series.get("mae_vm_all_nodes", []), float)
    m_mae = np.isfinite(mae)
    if m_mae.any():
        print(
            f"  |V| MAE (Log(v) vs OD): mean={float(np.mean(mae[m_mae])):.6f} pu  "
            f"logv_source={summ.get('logv_source', '?')}  fair_fast={summ.get('fair_fast')}",
            flush=True,
        )


_DSS_MODEL_LABEL = {
    1: "M1 constP",
    2: "M2 constZ",
    4: "M4 exp",
    5: "M5 constI",
    8: "M8 ZIP",
}


def _load_model_node_groups(case) -> dict[str, list[str]]:
    """Map DSS load-model label -> phase-node names served by those loads."""
    groups: dict[str, list[str]] = {}
    loads = getattr(case, "loads", None)
    if loads is None or len(loads) == 0:
        return groups
    for i in range(len(loads)):
        name = str(loads.name[i])
        if name.lower().startswith("pv_"):
            continue
        if "dss_model" in loads.columns:
            m = int(loads.dss_model[i])
        else:
            m = int(loads.model[i])
        label = _DSS_MODEL_LABEL.get(m, f"M{m}")
        bus = str(loads.bus[i]).split(".")[0]
        phases = loads.phases[i]
        try:
            ph_list = list(phases) if not isinstance(phases, (str, bytes)) else [phases]
        except Exception:
            ph_list = [1, 2, 3]
        for ph in ph_list:
            try:
                p = int(ph)
            except Exception:
                continue
            if p <= 0:
                continue
            node = f"{bus.lower()}.{p}"
            groups.setdefault(label, []).append(node)
    for k, nodes in list(groups.items()):
        groups[k] = list(dict.fromkeys(nodes))
    return groups


def _tap_fingerprint(case) -> tuple:
    """Hashable transformer tap state (analytical Tau uses these values)."""
    out = []
    xfm = getattr(case, "transformers", None)
    if xfm is None or len(xfm) == 0:
        return tuple(out)
    for i in range(len(xfm)):
        taps = tuple(float(t) for t in xfm.taps[i])
        out.append((str(xfm.name[i]).lower(), taps))
    return tuple(out)


def _cap_fingerprint(case) -> tuple:
    caps = getattr(case, "capacitors", None)
    if caps is None or len(caps) == 0:
        return tuple()
    return tuple(
        (str(caps.name[i]).lower(), float(caps.kvar[i])) for i in range(len(caps))
    )


def _rebuild_logv_A_from_case_taps(case) -> None:
    """Rebuild Log(v) base ``A`` from ``case.transformers.taps`` (analytical Tau).

    Does not pull OpenDSS YPrim (that path was unstable on large feeders).
    """
    import logv3lpf.linpf as linpf

    linpf.get_transformer_matrices(case)
    linpf.calculate_base_matrices(case)


def _apply_fast_solution_to_case(case, vm, va) -> None:
    """Write FastLogv reduced (vm, va) into ``case.results['logv3lpf']``."""
    import logv3lpf.linpf as linpf

    case.vm = np.asarray(vm, dtype=float).reshape(-1, 1)
    case.va = np.asarray(va, dtype=float).reshape(-1, 1)
    linpf.process_logv3lpf_solution(case)


def _sync_loads_pv_caps_taps_from_opendss(case, *, sync_taps: bool = True) -> None:
    """Copy OpenDSS settled injections / caps / (optional) taps into Log(v).

    Tap values are written into ``case.transformers.taps``. Callers that need
    those taps in the linear model must rebuild ``A`` via
    ``_rebuild_logv_A_from_case_taps`` (analytical Tau — not OpenDSS YPrim).
    """
    import opendssdirect as dss
    import logv3lpf.linpf as linpf

    for i in range(len(case.loads)):
        name = str(case.loads.name[i])
        if name.lower().startswith("pv_"):
            continue
        try:
            dss.Loads.Name(name)
            # Loads.kW()/kvar() are nominal; daily/yearly multipliers are not
            # reflected there. Use terminal powers after Solve for the actual
            # operating-point injection (needed for ieee34/906 Mode=Daily).
            powers = np.asarray(dss.CktElement.Powers(), dtype=float)
            if powers.size >= 2:
                case.loads.at[i, "kW"] = float(powers[0::2].sum())
                case.loads.at[i, "kvar"] = float(powers[1::2].sum())
            else:
                case.loads.at[i, "kW"] = float(dss.Loads.kW())
                case.loads.at[i, "kvar"] = float(dss.Loads.kvar())
        except Exception:
            continue

    try:
        pv_names = [n for n in dss.PVsystems.AllNames() if n and n != "NONE"]
    except Exception:
        pv_names = []
    for pv in pv_names:
        try:
            dss.PVsystems.Name(pv)
            powers = np.asarray(dss.CktElement.Powers(), dtype=float)
            p_kw = float(powers[0::2].sum())
            q_kvar = float(powers[1::2].sum())
        except Exception:
            continue
        row = f"pv_{pv}"
        hit = case.loads.index[case.loads.name.astype(str) == row]
        if len(hit) == 0:
            hit = case.loads.index[
                case.loads.name.astype(str).str.lower() == row.lower()
            ]
        if len(hit) == 0:
            continue
        j = int(hit[0])
        case.loads.at[j, "kW"] = p_kw
        case.loads.at[j, "kvar"] = q_kvar

    if getattr(case, "capacitors", None) is not None and len(case.capacitors):
        try:
            cap_names = [n for n in dss.Capacitors.AllNames() if n and n != "NONE"]
        except Exception:
            cap_names = []
        for cn in cap_names:
            try:
                dss.Capacitors.Name(cn)
                st = np.asarray(dss.Capacitors.States(), dtype=float)
                on = bool(np.sum(st > 0) > 0.5)
                rated = float(dss.Capacitors.kvar()) if hasattr(dss.Capacitors, "kvar") else None
            except Exception:
                continue
            hit = case.capacitors.index[
                case.capacitors.name.astype(str).str.lower() == str(cn).lower()
            ]
            if len(hit) == 0:
                continue
            j = int(hit[0])
            if rated is None:
                rated = float(case.capacitors.kvar[j])
            new_kvar = float(rated) if on else 0.0
            if abs(float(case.capacitors.kvar[j]) - new_kvar) > 1e-6:
                case.capacitors.at[j, "kvar"] = new_kvar

    if sync_taps:
        for i in range(len(case.transformers)):
            name = str(case.transformers.name[i])
            try:
                dss.Transformers.Name(name)
                nwind = int(dss.Transformers.NumWindings())
                taps = []
                for w in range(1, nwind + 1):
                    dss.Transformers.Wdg(w)
                    taps.append(float(dss.Transformers.Tap()))
                case.transformers.at[i, "taps"] = taps
            except Exception:
                continue


def _read_cap_states_opendss() -> dict[str, dict[str, float]]:
    import opendssdirect as dss

    out: dict[str, dict[str, float]] = {}
    try:
        names = [n for n in dss.Capacitors.AllNames() if n and n != "NONE"]
    except Exception:
        return out
    for cn in names:
        try:
            dss.Capacitors.Name(cn)
            st = np.asarray(dss.Capacitors.States(), dtype=float)
            on = float(1.0 if np.sum(st > 0) > 0.5 else 0.0)
            kvar = float(dss.Capacitors.kvar()) if hasattr(dss.Capacitors, "kvar") else float("nan")
            out[str(cn).lower()] = {"on": on, "kvar": kvar if on > 0.5 else 0.0}
        except Exception:
            continue
    return out


def _read_cap_states_logv(case) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if getattr(case, "capacitors", None) is None:
        return out
    for i in range(len(case.capacitors)):
        name = str(case.capacitors.name[i]).lower()
        kvar = float(case.capacitors.kvar[i])
        out[name] = {"on": float(1.0 if abs(kvar) > 1e-6 else 0.0), "kvar": kvar}
    return out


def _show(fig, *, save_plots: bool, out_path: Path | None, show_plots: bool) -> None:
    import matplotlib.pyplot as plt

    if save_plots and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _print_sample_table(series: dict[str, Any], *, max_rows: int = 24) -> None:
    """Print a compact per-step accuracy table (subsample if many steps)."""
    t = np.asarray(series["t_hours"], float)
    mae = np.asarray(series["mae_vm_all_nodes"], float)
    n = len(t)
    if n == 0:
        return
    if n <= max_rows:
        idxs = list(range(n))
    else:
        idxs = sorted(set(np.linspace(0, n - 1, max_rows, dtype=int).tolist()))
    print("\n  Per-step |V| MAE (subsample of day; full arrays in `series`)")
    print(f"  {'step':>5} {'hour':>7} {'|V| MAE':>10}  status")
    for i in idxs:
        ok = "ok" if np.isfinite(mae[i]) else "FAIL"
        print(f"  {i:5d} {t[i]:7.2f} {mae[i]:10.5f}  {ok}")
    print(
        f"  day summary: mean={np.nanmean(mae):.5f}  "
        f"p95={np.nanpercentile(mae[np.isfinite(mae)], 95) if np.isfinite(mae).any() else float('nan'):.5f}  "
        f"max={np.nanmax(mae):.5f}  n_ok={int(np.isfinite(mae).sum())}/{n}"
    )


def run_daily_opendss_vs_logv3lpf(
    repo: Path | None = None,
    *,
    feeder: str = "8500",
    day: int = 4,
    npts: int = 288,
    step_min: float = 5.0,
    control_mode: str = "synced",
    plot_nodes: list[str] | tuple[str, ...] | None = None,
    reg_cols: list[str] | tuple[str, ...] | None = None,
    cap_names: list[str] | tuple[str, ...] | None = None,
    daily_stress: float = 0.0,
    scenario_scale: float = 1.0,
    out_dir: Path | None = None,
    show_plots: bool = True,
    save_plots: bool = False,
    save_csv: bool = True,
    time_speed: bool = False,
    time_od_cold_snapshot: bool = True,
    report_every: int | None = None,
) -> dict[str, Any]:
    """Daily OpenDSS vs Log(v) for ``feeder`` in {ieee34, 906, 8500}.

    ``control_mode``:
      - ``synced`` (default for daily): OD Static; copy loads/caps/PV/taps into
        Log(v); rebuild ``A`` when OD taps change (fair same-state compare).
      - ``off``: OD ControlMode=OFF; Log(v) uses frozen DSS taps (rebuild once).
      - ``static``: OD Static; Log(v) runs own RegControl (caps/PV still synced from OD).

    ``time_speed``: time FastLogv vs OpenDSS on each daily sample.
      With ``synced``/``off``, Fast tracks the baked ``A`` and its voltages drive
      metrics/plots. With ``static``, Fast is an init-``A`` probe only.

    ``time_od_cold_snapshot``: when ``time_speed``, optionally time a **cold**
      InitSnap+Solve probe each step (reported under "Extra OD probes", not primary).
      Primary OpenDSS metric is **Solve() only** in **snapshot** mode (Method A style).
    """
    import matplotlib.pyplot as plt
    import opendssdirect as dss

    from logv3lpf.DSSParser import DSScase
    from logv3lpf_daily_demo import (
        FEEDERS,
        compile_cmd,
        ensure_logv3lpf,
        resolve_feeder,
        run_logv_autonomous,
        _apply_opendss_cap_pv_to_logv,
        _reset_reg_taps_opendss,
        _store_logv_regtaps,
    )
    import logv3lpf
    import logv3lpf.linpf as linpf
    import run_injection_dataset as inj

    repo = Path(repo or REPO_ROOT).resolve()
    _ensure_paths(repo)
    ensure_logv3lpf(repo)

    key = resolve_feeder(feeder)
    mode = str(control_mode).strip().lower()
    if mode not in ("off", "synced", "static"):
        raise ValueError("control_mode must be off|synced|static")
    dss_path = Path(FEEDERS[key]["dss"](repo)).resolve()
    if plot_nodes is None:
        plot_nodes = list(DEFAULT_PLOT_NODES.get(key, ["1.1"]))
    if reg_cols is None:
        reg_cols = list(DEFAULT_REG_COLS.get(key, []))
    if cap_names is None:
        cap_names = list(DEFAULT_CAP_NAMES.get(key, []))

    out_dir = Path(out_dir or (repo / "outputs" / "logv3lpf_daily_compare")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case = logv3lpf.case(
        compile_cmd(dss_path),
        FEEDERS[key]["sourcebus"],
        float(FEEDERS[key]["refvm"]),
        0,
    )
    od_ctrl = "OFF" if mode == "off" else "Static"

    plot_nodes = _pick_plot_nodes(case, [str(n) for n in plot_nodes])
    reg_cols_l = [str(c).strip().lower() for c in reg_cols]

    # Discover caps if none requested
    try:
        all_caps = [str(n).lower() for n in dss.Capacitors.AllNames() if n and n != "NONE"]
    except Exception:
        all_caps = []
    cap_names_l = [str(c).lower() for c in cap_names]
    if not cap_names_l:
        cap_names_l = all_caps[:6]

    # ---- feeder-specific OpenDSS daily preparation ----
    use_rep_days = key == "8500"
    base_names: list[str] = []
    base_kw = base_kvar = None
    m_eff = None
    pv_names: list[str] = []
    pv_base_pmpp: dict[str, float] = {}
    reg_names: list[str] = []
    n_daily_from_yearly = 0

    if use_rep_days:
        import run_daily_aggregate_dataset_8500 as rd8500
        from compare_opendss_snapshot_helpers import (
            apply_explicit_loads_and_pv_pmpp,
            reassert_snapshot_and_set_clock,
            setup_da_gps_snapshot_opendss,
            step_irradiance_multiplier,
            step_load_multiplier,
        )
        from run_da_gps_daily_opendss_compare import (
            _lookup_reg_tap_pu,
            _rebind_irradiance_loadshape_irradday001,
        )

        profiles = repo / "a representativ days"
        load_csv = profiles / f"load_day_{int(day):03d}.csv"
        irr_csv = profiles / f"irr_day_{int(day):03d}.csv"
        if not load_csv.is_file() or not irr_csv.is_file():
            raise FileNotFoundError(f"Need {load_csv} and {irr_csv}")

        _rebind_irradiance_loadshape_irradday001(
            irr_csv, npts=int(npts), step_min=float(step_min)
        )
        setup_da_gps_snapshot_opendss(npts=int(npts), step_min=float(step_min))
        pv_names = [str(n).strip() for n in dss.PVsystems.AllNames() if n and n != "NONE"]
        for pv_nm in pv_names:
            try:
                dss.PVsystems.Name(pv_nm)
                pv_base_pmpp[pv_nm] = float(dss.PVsystems.Pmpp())
            except Exception:
                pv_base_pmpp[pv_nm] = 0.0
        loads, _ = rd8500._collect_loads_and_maps()
        base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
        base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
        base_names = [str(d["name"]) for d in loads]
        m_raw = np.asarray(
            inj.read_profile_csv_two_col_noheader(str(load_csv), npts=int(npts), debug=False),
            dtype=np.float64,
        )
        if abs(float(daily_stress)) > 1e-12:
            m_eff = np.clip(m_raw * (1.0 + float(daily_stress)), 0.0, 10.0)
        else:
            m_eff = np.clip(m_raw, 0.0, 10.0)
        m_irr = np.asarray(
            inj.read_profile_csv_two_col_noheader(str(irr_csv), npts=int(npts), debug=False),
            dtype=np.float64,
        )
        reg_names = rd8500._discover_reg_controls()
        lookup_tap = _lookup_reg_tap_pu
        read_reg = rd8500._read_reg_control_state

        def apply_od_loads(step_i: int) -> None:
            m_t = step_load_multiplier(m_eff, step_i, scenario_scale)
            ir_t = step_irradiance_multiplier(m_irr, step_i)
            apply_explicit_loads_and_pv_pmpp(
                base_names=base_names,
                base_kw=base_kw,
                base_kvar=base_kvar,
                m_t=m_t,
                pv_names=pv_names,
                pv_base_pmpp_kw=pv_base_pmpp,
                ir_t=ir_t,
            )

        reassert = lambda i: reassert_snapshot_and_set_clock(i, step_min=float(step_min))  # noqa: E731
    else:
        from compare_opendss_snapshot_helpers import (
            force_snapshot_mode_for_compare_timing,
            neutralize_pv_irrad_loadshape_for_snapshot,
            reassert_snapshot_and_set_clock,
        )

        try:
            pv_names = [str(n).strip() for n in dss.PVsystems.AllNames() if n and n != "NONE"]
        except Exception:
            pv_names = []
        try:
            reg_names = [str(n) for n in dss.RegControls.AllNames() if n and n != "NONE"]
        except Exception:
            reg_names = []
        if not reg_cols_l and reg_names:
            reg_cols_l = [f"reg_{n.lower()}_tap_pu" for n in reg_names[:8]]

        n_daily_from_yearly = _ensure_daily_shapes_from_yearly()
        base_names, base_kw, base_kvar, load_mult_arrays = _collect_native_snapshot_loads(
            npts=int(npts), step_min=float(step_min)
        )
        pv_names, pv_base_pmpp, pv_irr_mults = _collect_native_pv_irradiance(
            npts=int(npts), step_min=float(step_min)
        )
        neutralize_pv_irrad_loadshape_for_snapshot(npts=int(npts), step_min=float(step_min))
        force_snapshot_mode_for_compare_timing()

        def apply_od_loads(step_i: int) -> None:
            _apply_native_loads_at_step(
                step_i,
                base_names,
                base_kw,
                base_kvar,
                load_mult_arrays,
                scenario_scale=float(scenario_scale),
            )
            _apply_native_pv_pmpp_at_step(step_i, pv_names, pv_base_pmpp, pv_irr_mults)

        def reassert(i: int) -> None:
            reassert_snapshot_and_set_clock(i, step_min=float(step_min))

        def lookup_tap(col, tap_raw):
            want = "".join(ch for ch in str(col).lower() if ch.isalnum())
            for k, v in tap_raw.items():
                if "".join(ch for ch in str(k).lower() if ch.isalnum()) == want:
                    return float(v)
            return None

        def read_reg(names):
            out = {}
            for rn in names:
                try:
                    dss.RegControls.Name(rn)
                    out[f"reg_{rn}_tap_pu"] = 1.0 + float(dss.RegControls.TapNumber()) * 0.00625
                except Exception:
                    pass
            return out

    # Snapshot + explicit loads for all feeders (Method A). ControlMode after shape wiring.
    try:
        dss.Text.Command(f"Set ControlMode={od_ctrl}")
        dss.Text.Command("Set MaxControlIter=100")
    except Exception:
        pass

    # Fast path for speed timing. Reload so a long-lived notebook kernel cannot
    # keep a stale FastLogvSolver that re-ran get_loads / dropped dense H.
    fast_solver = None
    if time_speed:
        import importlib

        import logv_fast_solver as _lfs

        importlib.reload(_lfs)
        FastLogvSolver = _lfs.FastLogvSolver
        fast_solver = FastLogvSolver(case)
        # Warm LU / BLAS outside the per-step timer (first call can be 5–10× cold).
        try:
            fast_solver.solve(fast_solver.P0, fast_solver.Q0)
        except Exception:
            pass

    # Synced/off: bake OD taps into A; Fast voltages drive metrics when timed.
    fair_fast = bool(time_speed) and mode in ("synced", "off")
    prev_tap_fp: tuple | None = None
    prev_cap_fp: tuple | None = None
    n_A_rebuilds = 0
    n_fast_refresh = 0

    load_model_nodes = _load_model_node_groups(case)

    t_hours = np.arange(int(npts), dtype=np.float64) * (float(step_min) / 60.0)
    series: dict[str, Any] = {
        "feeder": key,
        "control_mode": mode,
        "day": int(day),
        "npts": int(npts),
        "step_min": float(step_min),
        "t_hours": t_hours,
        "nodes": {
            n: {
                "opendss_vm": np.full(npts, np.nan),
                "opendss_va": np.full(npts, np.nan),
                "logv_vm": np.full(npts, np.nan),
                "logv_va": np.full(npts, np.nan),
            }
            for n in plot_nodes
        },
        "regs": {
            c: {"opendss": np.full(npts, np.nan), "logv": np.full(npts, np.nan)}
            for c in reg_cols_l
        },
        "caps": {
            c: {
                "opendss_on": np.full(npts, np.nan),
                "logv_on": np.full(npts, np.nan),
                "opendss_kvar": np.full(npts, np.nan),
                "logv_kvar": np.full(npts, np.nan),
            }
            for c in cap_names_l
        },
        "load_model_mae": {
            lab: np.full(npts, np.nan) for lab in load_model_nodes
        },
        "mae_vm_all_nodes": np.full(npts, np.nan),
        "rmse_vm_all_nodes": np.full(npts, np.nan),
        "t_opendss_apply_s": np.full(npts, np.nan),
        "t_opendss_reassert_s": np.full(npts, np.nan),
        "t_opendss_solve_s": np.full(npts, np.nan),
        "t_opendss_collect_s": np.full(npts, np.nan),
        "t_opendss_s": np.full(npts, np.nan),
        "t_opendss_cold_s": np.full(npts, np.nan),
        "t_opendss_warm_daily_s": np.full(npts, np.nan),
        "t_logv_s": np.full(npts, np.nan),
        "t_fast_s": np.full(npts, np.nan),
        "t_fast_refresh_s": np.full(npts, np.nan),
    }

    n_bad = 0
    open_apply_s_total = 0.0
    open_reassert_s_total = 0.0
    open_solve_only_s_total = 0.0
    open_get_s_total = 0.0
    warm_daily_probe_s_total = 0.0
    cold_snapshot_probe_s_total = 0.0
    t0 = time.perf_counter()
    steps_per_hour = max(int(round(60.0 / float(step_min))), 1)
    every = int(report_every) if report_every is not None else max(int(npts) // 8, 1)

    print("=" * 72)
    print(
        f"DAILY OpenDSS vs Log(v)  feeder={key}  mode={mode}  "
        f"npts={npts}  step={step_min}min  day={day}"
    )
    print(f"  DSS: {dss_path}")
    print(f"  plot nodes: {plot_nodes}")
    print(f"  regs: {reg_cols_l or '(none)'}")
    print(f"  caps: {cap_names_l or '(none)'}")
    if load_model_nodes:
        print(
            "  load models: "
            + ", ".join(f"{k}×{len(v)}" for k, v in sorted(load_model_nodes.items()))
        )
    print(f"  time_speed (FastLogv): {time_speed}")
    do_od_cold = bool(time_speed) and bool(time_od_cold_snapshot)
    if do_od_cold:
        print(
            "  OD cold snapshot probe: InitSnap+Solve after metrics "
            f"(ControlMode={od_ctrl}; extra probe only — not primary OD metric)"
        )
    print("  OpenDSS primary metric: snapshot Solve() only (Method A buckets)")
    if fair_fast:
        print(
            "  fair Fast protocol: bake OD taps into A on change; "
            "Fast voltages -> metrics/plots (synced/off)"
        )
    elif time_speed and mode == "static":
        print(
            "  note: static Fast is init-A probe only; plots/metrics use stock autonomous Log(v)"
        )
    if use_rep_days:
        print("  profile: 8500 representative-day CSV -> explicit kW/kvar (snapshot)")
    else:
        extra = (
            f"; copied Yearly->Daily on {n_daily_from_yearly} loads"
            if n_daily_from_yearly
            else ""
        )
        print(
            "  profile: native DSS loadshapes -> explicit kW/kvar each step (snapshot)"
            f"{extra}; plot distal LV buses (source-end bus 1 is nearly flat)"
        )
    print("=" * 72)

    for i in range(int(npts)):
        t_apply0 = time.perf_counter()
        apply_od_loads(i)
        t_apply1 = time.perf_counter()
        open_apply_s_total += t_apply1 - t_apply0
        series["t_opendss_apply_s"][i] = t_apply1 - t_apply0

        t_reassert0 = time.perf_counter()
        reassert(i)
        t_reassert1 = time.perf_counter()
        open_reassert_s_total += t_reassert1 - t_reassert0
        series["t_opendss_reassert_s"][i] = t_reassert1 - t_reassert0

        t_solve0 = time.perf_counter()
        dss.Solution.Solve()
        t_solve1 = time.perf_counter()
        open_solve_only_s_total += t_solve1 - t_solve0
        series["t_opendss_solve_s"][i] = t_solve1 - t_solve0
        series["t_opendss_s"][i] = t_solve1 - t_solve0
        if not dss.Solution.Converged():
            n_bad += 1
            continue

        t_get0 = time.perf_counter()
        all_nodes = []
        for n in dss.Circuit.AllNodeNames():
            s = str(n).strip().lower()
            if "." not in s:
                continue
            try:
                ph = int(s.rsplit(".", 1)[1])
            except ValueError:
                continue
            if ph in (1, 2, 3):
                all_nodes.append(s)
        all_nodes = list(dict.fromkeys(all_nodes))
        vm_od, va_od = inj.get_all_node_voltage_pu_and_angle_filtered(all_nodes)
        t_get1 = time.perf_counter()
        open_get_s_total += t_get1 - t_get0
        series["t_opendss_collect_s"][i] = t_get1 - t_get0
        name_to_i = {n: k for k, n in enumerate(all_nodes)}

        tap_raw = read_reg(reg_names) if reg_names else {}
        for col in reg_cols_l:
            vv = lookup_tap(col, tap_raw)
            series["regs"][col]["opendss"][i] = (
                float(vv) if vv is not None and np.isfinite(vv) else np.nan
            )

        cap_od = _read_cap_states_opendss()
        for cn in cap_names_l:
            st = cap_od.get(cn) or {}
            series["caps"][cn]["opendss_on"][i] = float(st.get("on", np.nan))
            series["caps"][cn]["opendss_kvar"][i] = float(st.get("kvar", np.nan))

        # Log(v) path
        sync_taps = mode in ("synced", "off")
        _sync_loads_pv_caps_taps_from_opendss(case, sync_taps=sync_taps)
        if mode == "static":
            _apply_opendss_cap_pv_to_logv(case)

        # Synced/off: bake OD taps into A (analytical Tau; Ỹ in A, ỹ on RHS).
        # Fast refreshes factorization only — never re-run FD init inside t_fast.
        need_fast_refresh = False
        if mode in ("synced", "off"):
            tap_fp = _tap_fingerprint(case)
            if tap_fp != prev_tap_fp:
                _rebuild_logv_A_from_case_taps(case)
                n_A_rebuilds += 1
                prev_tap_fp = tap_fp
                need_fast_refresh = True
            cap_fp = _cap_fingerprint(case)
            if cap_fp != prev_cap_fp:
                prev_cap_fp = cap_fp
                need_fast_refresh = True

        if time_speed and fast_solver is not None and need_fast_refresh:
            try:
                t_r0 = time.perf_counter()
                fast_solver.refresh_zip_cap_and_factor(case)
                series["t_fast_refresh_s"][i] = time.perf_counter() - t_r0
                n_fast_refresh += 1
                # Warm new factor/H outside the online timer (cold BLAS spike).
                try:
                    fast_solver.solve(fast_solver.P0, fast_solver.Q0)
                except Exception:
                    pass
            except Exception:
                series["t_fast_refresh_s"][i] = float("nan")
                # Last resort: full rebuild (still outside t_fast)
                try:
                    import importlib

                    import logv_fast_solver as _lfs

                    importlib.reload(_lfs)
                    fast_solver = _lfs.FastLogvSolver(case)
                    try:
                        fast_solver.solve(fast_solver.P0, fast_solver.Q0)
                    except Exception:
                        pass
                except Exception:
                    fast_solver = None

        # Fast online solve FIRST (isolated timer: ONLY solve(P,Q) — no P/Q
        # gather, no refresh, no process_logv3lpf_solution, no stock run_case).
        vm_f = va_f = None
        if time_speed and fast_solver is not None:
            try:
                P = np.asarray(
                    [float(case.loads.kW[int(j)]) for j in fast_solver.cp_idx],
                    dtype=float,
                )
                Q = np.asarray(
                    [float(case.loads.kvar[int(j)]) for j in fast_solver.cp_idx],
                    dtype=float,
                )
                t_f0 = time.perf_counter()
                vm_f, va_f = fast_solver.solve(P, Q)
                series["t_fast_s"][i] = time.perf_counter() - t_f0
            except Exception:
                series["t_fast_s"][i] = float("nan")
                vm_f = va_f = None

        t_lv0 = time.perf_counter()
        if mode == "static":
            run_logv_autonomous(case, max_iter=40)
        else:
            case.run_case("logv3lpf")
        series["t_logv_s"][i] = time.perf_counter() - t_lv0
        _store_logv_regtaps(case)
        DSScase.process_openDSS_solution(case)

        # Fair protocol: metrics/plots use Fast voltages (stock timed above only).
        if fair_fast and vm_f is not None and va_f is not None:
            _apply_fast_solution_to_case(case, vm_f, va_f)

        # Cold OD snapshot timing (after warm metrics recorded). Reset taps to
        # nominal so Static must re-settle controls from a true flat start —
        # unlike InitSnap alone, which kept warm-settled taps and stayed fast.
        # Not the four_scenario ClearAll+recompile path.
        if do_od_cold:
            try:
                if od_ctrl.upper() == "STATIC":
                    _reset_reg_taps_opendss()
                dss.Text.Command(f"Set ControlMode={od_ctrl}")
                t_c0 = time.perf_counter()
                dss.Solution.InitSnap()
                dss.Solution.Solve()
                dt_cold = time.perf_counter() - t_c0
                series["t_opendss_cold_s"][i] = dt_cold
                if dss.Solution.Converged():
                    cold_snapshot_probe_s_total += dt_cold
            except Exception:
                series["t_opendss_cold_s"][i] = float("nan")

        for node in plot_nodes:
            if node in name_to_i:
                series["nodes"][node]["opendss_vm"][i] = float(vm_od[name_to_i[node]])
                series["nodes"][node]["opendss_va"][i] = float(va_od[name_to_i[node]])
            try:
                lvm, lva = _node_vm_va(case, node, "logv3lpf")
                series["nodes"][node]["logv_vm"][i] = lvm
                series["nodes"][node]["logv_va"][i] = lva
            except Exception:
                pass

        lv_taps = (case.results.get("logv3lpf") or {}).get("regtap") or {}
        for col in reg_cols_l:
            stem = col
            if stem.startswith("reg_") and stem.endswith("_tap_pu"):
                stem = stem[len("reg_") : -len("_tap_pu")]
            stem_alnum = "".join(ch for ch in stem.lower() if ch.isalnum())
            tap_v = np.nan
            for rn, tv in lv_taps.items():
                nm = "".join(ch for ch in str(rn).lower() if ch.isalnum())
                if nm == stem_alnum or stem_alnum in nm or nm in stem_alnum:
                    tap_v = float(tv)
                    break
            if not np.isfinite(tap_v):
                for ti in range(len(case.transformers)):
                    nm = "".join(
                        ch for ch in str(case.transformers.name[ti]).lower() if ch.isalnum()
                    )
                    if nm == stem_alnum or stem_alnum in nm or nm in stem_alnum:
                        taps = case.transformers.taps[ti]
                        tap_v = float(taps[1]) if len(taps) > 1 else float(taps[0])
                        break
            series["regs"][col]["logv"][i] = tap_v

        cap_lv = _read_cap_states_logv(case)
        for cn in cap_names_l:
            st = cap_lv.get(cn) or {}
            series["caps"][cn]["logv_on"][i] = float(st.get("on", np.nan))
            series["caps"][cn]["logv_kvar"][i] = float(st.get("kvar", np.nan))

        errs = []
        vm_od_map = (case.results.get("openDSS") or {}).get("vm") or {}
        # Case-insensitive bus lookup (OpenDSS vs Log(v) key casing can differ).
        od_by_lower = {str(b).lower(): v for b, v in vm_od_map.items()}
        for bus, vm_l in case.results.get("logv3lpf", {}).get("vm", {}).items():
            vm_o = vm_od_map.get(bus)
            if vm_o is None:
                vm_o = od_by_lower.get(str(bus).lower())
            if vm_o is None:
                continue
            a = np.asarray(vm_l, float)
            b = np.asarray(vm_o, float)
            n = min(len(a), len(b))
            if n <= 0:
                continue
            d = np.abs(a[:n] - b[:n])
            d = d[np.isfinite(d)]
            if d.size:
                errs.append(d)
        if errs:
            e = np.concatenate(errs)
            series["mae_vm_all_nodes"][i] = float(np.mean(e))
            series["rmse_vm_all_nodes"][i] = float(np.sqrt(np.mean(e ** 2)))

        # |V| MAE at buses served by each DSS load model (P / Z / I / exp / ZIP)
        for lab, nodes_g in load_model_nodes.items():
            g_errs = []
            for node in nodes_g:
                if node not in name_to_i:
                    continue
                try:
                    lvm, _ = _node_vm_va(case, node, "logv3lpf")
                except Exception:
                    continue
                vod = float(vm_od[name_to_i[node]])
                if np.isfinite(vod) and np.isfinite(lvm):
                    g_errs.append(abs(vod - float(lvm)))
            if g_errs:
                series["load_model_mae"][lab][i] = float(np.mean(g_errs))

        if (i + 1) % every == 0 or i == 0 or i == npts - 1:
            print(
                f"  [{i+1}/{npts}] apply={open_apply_s_total:.2f}s "
                f"reassert={open_reassert_s_total:.2f}s "
                f"solve={open_solve_only_s_total:.2f}s getV={open_get_s_total:.2f}s  "
                f"|V| MAE={series['mae_vm_all_nodes'][i]:.5f}  "
                f"LV={series['t_logv_s'][i]*1e3:.1f}ms"
                + (
                    f"  Fast={series['t_fast_s'][i]*1e3:.1f}ms"
                    if time_speed and np.isfinite(series["t_fast_s"][i])
                    else ""
                ),
                flush=True,
            )

    wall = time.perf_counter() - t0

    # ---- end-of-run audit ----
    print("=" * 72)
    print("DAILY AUDIT — accuracy over all samples")
    print("=" * 72)
    print(f"  feeder / mode : {key} / {mode}")
    print(f"  steps         : {npts - n_bad}/{npts} converged  wall={wall:.1f}s")
    print(f"  mean |V| MAE  : {np.nanmean(series['mae_vm_all_nodes']):.6f} pu")
    print(f"  mean |V| RMSE : {np.nanmean(series['rmse_vm_all_nodes']):.6f} pu")
    print(f"  max  |V| MAE  : {np.nanmax(series['mae_vm_all_nodes']):.6f} pu")
    for node in plot_nodes:
        nd = series["nodes"][node]
        m = np.isfinite(nd["opendss_vm"]) & np.isfinite(nd["logv_vm"])
        if m.any():
            mae = float(np.mean(np.abs(nd["opendss_vm"][m] - nd["logv_vm"][m])))
            print(
                f"  node {node:16s}  |V| MAE={mae:.5f}  "
                f"OD[{np.nanmin(nd['opendss_vm']):.4f},{np.nanmax(nd['opendss_vm']):.4f}]  "
                f"LV[{np.nanmin(nd['logv_vm']):.4f},{np.nanmax(nd['logv_vm']):.4f}]"
            )
    for col in reg_cols_l:
        rg = series["regs"][col]
        m = np.isfinite(rg["opendss"]) & np.isfinite(rg["logv"])
        if m.any():
            print(
                f"  tap  {col:28s}  MAE={float(np.mean(np.abs(rg['opendss'][m]-rg['logv'][m]))):.5f}"
            )
    for cn in cap_names_l:
        cg = series["caps"][cn]
        m = np.isfinite(cg["opendss_on"]) & np.isfinite(cg["logv_on"])
        if m.any():
            disagree = int(np.sum(np.abs(cg["opendss_on"][m] - cg["logv_on"][m]) > 0.5))
            print(
                f"  cap  {cn:28s}  on/off disagree steps={disagree}/{int(m.sum())}"
            )
    if series.get("load_model_mae"):
        print("  Load-model |V| MAE (nodes of loads with that DSS model)")
        for lab, arr in series["load_model_mae"].items():
            m = np.isfinite(arr)
            n_nodes = len(load_model_nodes.get(lab, []))
            if m.any():
                print(
                    f"    {lab:16s}  n_nodes={n_nodes:4d}  "
                    f"mean MAE={float(np.mean(arr[m])):.5f}  "
                    f"max MAE={float(np.max(arr[m])):.5f}"
                )
    if time_speed:
        print_logv_daily_timing_summary, print_logv_extra_od_probes = _import_logv_daily_timing()

        n_ok = int(npts - n_bad)
        logv_stock_s_total = float(np.nansum(series["t_logv_s"]))
        fast_forward_s_total = float(np.nansum(series["t_fast_s"]))
        fast_refresh_s_total = float(np.nansum(series["t_fast_refresh_s"]))
        print_logv_daily_timing_summary(
            n_ok=n_ok,
            npts=int(npts),
            n_nonconv=int(n_bad),
            open_apply_s_total=open_apply_s_total,
            open_reassert_s_total=open_reassert_s_total,
            open_solve_only_s_total=open_solve_only_s_total,
            open_get_s_total=open_get_s_total,
            logv_stock_s_total=logv_stock_s_total,
            fast_forward_s_total=fast_forward_s_total,
            fast_refresh_s_total=fast_refresh_s_total,
            n_fast_refresh=int(n_fast_refresh),
            feeder=key,
            control_mode=mode,
        )
        if do_od_cold and cold_snapshot_probe_s_total > 0.0:
            print_logv_extra_od_probes(
                n_ok=n_ok,
                cold_snapshot_s_total=cold_snapshot_probe_s_total,
            )
    if mode in ("synced", "off") and n_A_rebuilds:
        print(f"  Log(v) A rebuilds (tap changes): {n_A_rebuilds}")
    _print_sample_table(series)
    print("=" * 72)

    # ---- CSV of every sample ----
    if save_csv:
        import csv

        csv_path = out_dir / f"daily_{key}_mode_{mode}_npts{npts}_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "step",
                "hour",
                "mae_vm",
                "rmse_vm",
                "t_opendss_apply_s",
                "t_opendss_reassert_s",
                "t_opendss_solve_s",
                "t_opendss_collect_s",
                "t_opendss_s",
                "t_opendss_cold_s",
                "t_logv_s",
                "t_fast_s",
                "t_fast_refresh_s",
            ]
            for node in plot_nodes:
                tag = node.replace(".", "_")
                header += [
                    f"od_vm_{tag}",
                    f"lv_vm_{tag}",
                    f"od_va_{tag}",
                    f"lv_va_{tag}",
                ]
            for col in reg_cols_l:
                header += [f"od_{col}", f"lv_{col}"]
            for cn in cap_names_l:
                header += [
                    f"od_on_{cn}",
                    f"lv_on_{cn}",
                    f"od_kvar_{cn}",
                    f"lv_kvar_{cn}",
                ]
            w.writerow(header)
            for i in range(int(npts)):
                row = [
                    i,
                    float(t_hours[i]),
                    float(series["mae_vm_all_nodes"][i]),
                    float(series["rmse_vm_all_nodes"][i]),
                    float(series["t_opendss_apply_s"][i]),
                    float(series["t_opendss_reassert_s"][i]),
                    float(series["t_opendss_solve_s"][i]),
                    float(series["t_opendss_collect_s"][i]),
                    float(series["t_opendss_s"][i]),
                    float(series["t_opendss_cold_s"][i]),
                    float(series["t_logv_s"][i]),
                    float(series["t_fast_s"][i]),
                    float(series["t_fast_refresh_s"][i]),
                ]
                for node in plot_nodes:
                    nd = series["nodes"][node]
                    row += [
                        float(nd["opendss_vm"][i]),
                        float(nd["logv_vm"][i]),
                        float(nd["opendss_va"][i]),
                        float(nd["logv_va"][i]),
                    ]
                for col in reg_cols_l:
                    rg = series["regs"][col]
                    row += [float(rg["opendss"][i]), float(rg["logv"][i])]
                for cn in cap_names_l:
                    cg = series["caps"][cn]
                    row += [
                        float(cg["opendss_on"][i]),
                        float(cg["logv_on"][i]),
                        float(cg["opendss_kvar"][i]),
                        float(cg["logv_kvar"][i]),
                    ]
                w.writerow(row)
        series["csv_path"] = str(csv_path)
        print(f"  wrote {csv_path}")

    # ---- plots: voltage ----
    for node in plot_nodes:
        nd = series["nodes"][node]
        fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        axs[0].plot(t_hours, nd["opendss_vm"], color="C0", lw=2.0, label="OpenDSS |V|")
        axs[0].plot(t_hours, nd["logv_vm"], color="C1", ls="--", lw=1.6, label="Log(v) |V|")
        m = np.isfinite(nd["opendss_vm"]) & np.isfinite(nd["logv_vm"])
        mae = (
            float(np.mean(np.abs(nd["opendss_vm"][m] - nd["logv_vm"][m])))
            if m.any()
            else float("nan")
        )
        axs[0].set_ylabel("|V| (pu)")
        axs[0].set_title(f"{key} day @ {node} — OpenDSS vs Log(v)  (MAE={mae:.4f} pu)")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")
        axs[1].plot(t_hours, nd["opendss_va"], color="C0", lw=2.0, label="OpenDSS angle")
        axs[1].plot(t_hours, nd["logv_va"], color="C1", ls="--", lw=1.6, label="Log(v) angle")
        axs[1].set_ylabel("V angle (deg)")
        axs[1].set_xlabel("Hour of day")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")
        fig.tight_layout()
        _show(
            fig,
            save_plots=save_plots,
            out_path=out_dir / f"daily_voltage_{key}_{node.replace('.', '_')}.png",
            show_plots=show_plots,
        )

    # ---- plots: regulators ----
    for col in reg_cols_l:
        rg = series["regs"][col]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(
            t_hours,
            rg["opendss"],
            color="C0",
            lw=2.0,
            drawstyle="steps-post",
            label="OpenDSS tap",
        )
        ax.plot(
            t_hours,
            rg["logv"],
            color="C1",
            ls="--",
            lw=1.6,
            drawstyle="steps-post",
            label="Log(v) tap",
        )
        m = np.isfinite(rg["opendss"]) & np.isfinite(rg["logv"])
        mae = (
            float(np.mean(np.abs(rg["opendss"][m] - rg["logv"][m])))
            if m.any()
            else float("nan")
        )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Tap (pu)")
        ax.set_title(f"{key} regulator tap: {col}  (MAE={mae:.5f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _show(
            fig,
            save_plots=save_plots,
            out_path=out_dir / f"daily_{key}_{col}.png",
            show_plots=show_plots,
        )

    # ---- plots: capacitors ----
    for cn in cap_names_l:
        cg = series["caps"][cn]
        fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
        axs[0].plot(
            t_hours,
            cg["opendss_on"],
            color="C0",
            lw=2.0,
            drawstyle="steps-post",
            label="OpenDSS on",
        )
        axs[0].plot(
            t_hours,
            cg["logv_on"],
            color="C1",
            ls="--",
            lw=1.6,
            drawstyle="steps-post",
            label="Log(v) on",
        )
        axs[0].set_ylabel("on (1) / off (0)")
        axs[0].set_title(f"{key} capacitor state: {cn}")
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")
        axs[1].plot(
            t_hours,
            cg["opendss_kvar"],
            color="C0",
            lw=2.0,
            drawstyle="steps-post",
            label="OpenDSS kvar",
        )
        axs[1].plot(
            t_hours,
            cg["logv_kvar"],
            color="C1",
            ls="--",
            lw=1.6,
            drawstyle="steps-post",
            label="Log(v) kvar",
        )
        axs[1].set_ylabel("kvar")
        axs[1].set_xlabel("Hour of day")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")
        fig.tight_layout()
        _show(
            fig,
            save_plots=save_plots,
            out_path=out_dir / f"daily_{key}_cap_{cn}.png",
            show_plots=show_plots,
        )

    # ---- plots: load-model |V| MAE (P / Z / I / exp) ----
    lm = series.get("load_model_mae") or {}
    if lm:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for j, (lab, arr) in enumerate(sorted(lm.items())):
            ax.plot(
                t_hours,
                arr,
                lw=1.8,
                label=f"{lab} (n={len(load_model_nodes.get(lab, []))})",
            )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("MAE |V| (pu)")
        ax.set_title(
            f"{key} |V| MAE by DSS load model  "
            f"(Log(v) supports M1/M2/M5 natively; M4/M8 remapped→constP)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        _show(
            fig,
            save_plots=save_plots,
            out_path=out_dir / f"daily_{key}_load_model_mae.png",
            show_plots=show_plots,
        )

    # ---- MAE over day ----
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_hours, series["mae_vm_all_nodes"], color="C3", lw=1.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE |V| (pu)")
    ax.set_title(
        f"{key} all-node |V| MAE over day  "
        f"(mean={np.nanmean(series['mae_vm_all_nodes']):.4f} pu)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _show(
        fig,
        save_plots=save_plots,
        out_path=out_dir / f"daily_{key}_mae_vm.png",
        show_plots=show_plots,
    )

    if time_speed and (
        np.isfinite(series["t_fast_s"]).any()
        or np.isfinite(series["t_opendss_cold_s"]).any()
    ):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(
            t_hours,
            series["t_opendss_solve_s"] * 1e3,
            color="C0",
            lw=1.6,
            label="OD Solve() only",
        )
        if np.isfinite(series["t_opendss_cold_s"]).any():
            ax.plot(
                t_hours,
                series["t_opendss_cold_s"] * 1e3,
                color="C0",
                lw=1.4,
                ls="--",
                label="OD cold probe",
            )
        if np.isfinite(series["t_fast_s"]).any():
            ax.plot(
                t_hours,
                series["t_fast_s"] * 1e3,
                color="C2",
                lw=1.6,
                label="FastLogv",
            )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Solve time (ms)")
        ax.set_title(f"{key} per-sample solve time (daily profile)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _show(
            fig,
            save_plots=save_plots,
            out_path=out_dir / f"daily_{key}_speed_ms.png",
            show_plots=show_plots,
        )

    n_ok_steps = int(npts - n_bad)
    series["od_timing"] = {
        "apply_s": float(open_apply_s_total),
        "reassert_s": float(open_reassert_s_total),
        "solve_s": float(open_solve_only_s_total),
        "collect_s": float(open_get_s_total),
        "cold_snapshot_probe_s": float(cold_snapshot_probe_s_total),
        "warm_daily_probe_s": float(warm_daily_probe_s_total),
    }
    series["out_dir"] = out_dir
    series["wall_s"] = wall
    series["n_bad"] = n_bad
    series["summary"] = {
        "mean_mae_vm": float(np.nanmean(series["mae_vm_all_nodes"])),
        "mean_rmse_vm": float(np.nanmean(series["rmse_vm_all_nodes"])),
        "max_mae_vm": float(np.nanmax(series["mae_vm_all_nodes"])),
        "n_ok": n_ok_steps,
        "n_bad": int(n_bad),
        "wall_s": float(wall),
        "n_A_rebuilds": int(n_A_rebuilds),
        "n_fast_refresh": int(n_fast_refresh),
        "fair_fast": bool(fair_fast),
        "logv_source": "fast" if fair_fast else "stock",
        "mean_t_opendss_apply_ms": float(np.nanmean(series["t_opendss_apply_s"]) * 1e3)
        if np.isfinite(series["t_opendss_apply_s"]).any()
        else float("nan"),
        "mean_t_opendss_reassert_ms": float(np.nanmean(series["t_opendss_reassert_s"]) * 1e3)
        if np.isfinite(series["t_opendss_reassert_s"]).any()
        else float("nan"),
        "mean_t_opendss_solve_ms": float(np.nanmean(series["t_opendss_solve_s"]) * 1e3)
        if np.isfinite(series["t_opendss_solve_s"]).any()
        else float("nan"),
        "mean_t_opendss_collect_ms": float(np.nanmean(series["t_opendss_collect_s"]) * 1e3)
        if np.isfinite(series["t_opendss_collect_s"]).any()
        else float("nan"),
        "mean_t_opendss_ms": float(np.nanmean(series["t_opendss_solve_s"]) * 1e3)
        if np.isfinite(series["t_opendss_solve_s"]).any()
        else float("nan"),
        "mean_t_opendss_cold_ms": float(np.nanmean(series["t_opendss_cold_s"]) * 1e3)
        if np.isfinite(series["t_opendss_cold_s"]).any()
        else float("nan"),
        "mean_t_fast_ms": float(np.nanmean(series["t_fast_s"]) * 1e3)
        if np.isfinite(series["t_fast_s"]).any()
        else float("nan"),
        "mean_t_fast_refresh_ms": float(np.nanmean(series["t_fast_refresh_s"]) * 1e3)
        if np.isfinite(series["t_fast_refresh_s"]).any()
        else float("nan"),
        "mean_t_logv_ms": float(np.nanmean(series["t_logv_s"]) * 1e3)
        if np.isfinite(series["t_logv_s"]).any()
        else float("nan"),
    }
    return series
