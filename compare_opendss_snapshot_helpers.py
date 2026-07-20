"""
OpenDSS snapshot-mode helpers for daily compare scripts.

Shared by ``compare_daily_vs_snapshot_autonomous_devices.run_snapshot_series`` and
``run_da_gps_daily_opendss_compare`` so the per-step OpenDSS snapshot path stays identical.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opendssdirect as dss

# ``run_daily_aggregate_dataset_8500`` imports this module — do not import rd8500 at top level.
# A previous compare default of 10 matched an old MLP script but causes **(#485) Max Control
# Iterations Exceeded** on IEEE 8500 with regulators/caps. Default **20000** matches compile.
# For timing experiments with a tighter cap, set env ``GNN_COMPARE_MAXCONTROLITER`` (e.g. 200).
_DEFAULT_MAX_CONTROLITER = 20000
# Shared across daily march and snapshot reassert (compile uses 30; parity paths override to 20).
_PARITY_MAXITERATIONS = 20
_DEFAULT_BASE_KVA_KW = 10000.0


@dataclass(frozen=True)
class ParityProfiles:
    """Per-step multipliers and CSV paths bound into OpenDSS for daily-mode parity."""

    m_raw: np.ndarray
    m_eff: np.ndarray
    m_irr: np.ndarray
    load_bind_csv: Path
    irr_bind_csv: Path


def _max_control_iter() -> int:
    return int(os.environ.get("GNN_COMPARE_MAXCONTROLITER", str(_DEFAULT_MAX_CONTROLITER)))


def force_snapshot_mode_for_compare_timing() -> None:
    """
    After compiling the daily 8500 circuit, switch to snapshot solves for fair timing vs
    warm-started daily marching.

    - ``maxiterations=20`` matches ``8500-node/Run_8500Node_Daily_5min.dss`` and
      ``compare_daily_8500_mlp_gnn.py`` (ResidualGCN path).
    - ``maxcontroliter`` defaults to **20000** (same order of magnitude as compile). Override with
      env ``GNN_COMPARE_MAXCONTROLITER`` (e.g. lower for ablations; too low ⇒ DSS #485).
    """
    dss.Text.Command("set mode=snapshot")
    dss.Text.Command("set maxiterations=20")
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    # ``set mode=snapshot`` for text bookkeeping; ``Solution.Mode(1)`` (Daily) matches
    # ``compare_daily_8500_mlp_gnn`` and the historical DA-GPS / snapshot compare path.
    dss.Solution.Mode(1)


def reassert_snapshot_before_each_solve() -> None:
    """Re-apply snapshot mode and solver caps before ``Solution.Solve()`` each timestep."""
    dss.Text.Command("set mode=snapshot")
    dss.Text.Command("set maxiterations=20")
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    dss.Solution.Mode(1)


def neutralize_pv_irrad_loadshape_for_snapshot(*, npts: int, step_min: float = 5.0) -> None:
    """Point ``Loadshape.IrradDay001`` (PV ``Daily=``) at unity mults for snapshot solves.

    OpenDSS ``mode=snapshot`` does not apply PV ``Daily=`` irradiance the same way as native
    ``mode=daily`` even when ``hour``/``sec`` are set after ``reassert_snapshot``. Explicit
    ``Pmpp = Pmpp0 × m_irr[t]`` per step (``compare_daily_8500_mlp_gnn`` style) is required;
    leaving the real profile on ``IrradDay001`` would double-count irradiance.

    No-op when ``IrradDay001`` is absent (ieee34 / 906 and other non-8500 masters).
    """
    try:
        names = {str(x).strip().lower() for x in (dss.LoadShapes.AllNames() or [])}
    except Exception:
        names = set()
    if "irradday001" not in names:
        return
    n = int(max(1, npts))
    interval_h = float(step_min) / 60.0
    ones = ",".join("1" for _ in range(n))
    try:
        dss.Text.Command(
            f"Edit Loadshape.IrradDay001 npts={n} interval={interval_h} mult=({ones})"
        )
    except Exception:
        return


def rebind_irradiance_loadshape_irradday001(
    irr_csv: Path,
    *,
    npts: int,
    step_min: float,
) -> None:
    """Point ``Loadshape.IrradDay001`` at ``irr_csv`` column 2 (before neutralize for snapshot)."""
    fp = irr_csv.expanduser().resolve()
    if not fp.is_file():
        raise FileNotFoundError(f"Irradiance CSV for IrradDay001 rebind: {fp}")
    n = int(max(1, npts))
    interval_h = float(step_min) / 60.0
    cmd = (
        f"Edit Loadshape.IrradDay001 npts={n} interval={interval_h} "
        f'mult=(file="{fp.as_posix()}", col=2, header=no)'
    )
    try:
        dss.Text.Command(cmd)
    except Exception as e:
        raise RuntimeError(f"OpenDSS failed to rebind IrradDay001: {cmd!r} ({e})") from e


def setup_da_gps_snapshot_opendss(
    *,
    npts: int,
    irr_csv: Path | None = None,
    step_min: float = 5.0,
) -> None:
    """Post-compile wiring shared by DA-GPS daily compare and ``run_snapshot_series``.

    Snapshot solves use explicit ``Pmpp = Pmpp0 × m_irr[t]``; ``IrradDay001`` is neutralized so
    irradiance is never read from OpenDSS loadshapes (avoids double-count vs daily ``Daily=``).
    ``irr_csv`` is accepted for API compatibility but ignored here.
    """
    import run_daily_aggregate_dataset_8500 as rd8500

    _ = irr_csv
    rd8500._detach_daily_loadshape_from_loads()
    neutralize_pv_irrad_loadshape_for_snapshot(npts=int(npts), step_min=float(step_min))
    force_snapshot_mode_for_compare_timing()


def discover_pv_system_names() -> list[str]:
    """PVSystem names (First/Next walk union AllNames)."""
    seen: dict[str, None] = {}
    try:
        if dss.PVsystems.First():
            while True:
                nm = str(dss.PVsystems.Name()).strip()
                if nm:
                    seen.setdefault(nm, None)
                if not dss.PVsystems.Next():
                    break
    except Exception:
        pass
    try:
        alln = getattr(dss.PVsystems, "AllNames", None)
        if callable(alln):
            for x in alln():
                nm = str(x).strip()
                if nm:
                    seen.setdefault(nm, None)
    except Exception:
        pass
    return sorted(seen.keys())


def read_pv_base_pmpp_kw(pv_names: list[str]) -> dict[str, float]:
    """Nameplate ``Pmpp`` (kW) per PVSystem after compile, before per-step scaling."""
    out: dict[str, float] = {}
    for raw in pv_names:
        name = str(raw).strip()
        if not name:
            continue
        try:
            dss.PVsystems.Name(name)
            out[name] = float(dss.PVsystems.Pmpp())
        except Exception:
            out[name] = 0.0
    return out


def snapshot_step_hr_sec(step_index: int, *, step_min: float = 5.0) -> tuple[int, int]:
    """5-min step index ``i`` → ``(hour, sec)`` for OpenDSS clock."""
    hr = int(step_index // 12)
    sec = int((step_index % 12) * (float(step_min) * 60))
    return hr, sec


def apply_explicit_loads_and_pv_pmpp(
    *,
    base_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    m_t: float,
    pv_names: list[str],
    pv_base_pmpp_kw: dict[str, float],
    ir_t: float,
) -> None:
    """Set explicit load kW/kvar and ``Pmpp = Pmpp0 × m_irr`` (DA-GPS apply block)."""
    kw_set = base_kw * float(m_t)
    kvar_set = base_kvar * float(m_t)
    for j, name in enumerate(base_names):
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw_set[j]))
        dss.Loads.kvar(float(kvar_set[j]))
    ir = float(ir_t)
    for pv_nm in pv_names:
        b0 = float(pv_base_pmpp_kw.get(str(pv_nm).strip(), 0.0))
        if b0 <= 0.0:
            continue
        try:
            dss.PVsystems.Name(str(pv_nm).strip())
            dss.PVsystems.Pmpp(float(b0) * ir)
        except Exception:
            pass


def reassert_snapshot_and_set_clock(step_index: int, *, step_min: float = 5.0) -> None:
    """``reassert_snapshot_before_each_solve()`` then ``set hour/sec`` (clock after reassert)."""
    reassert_snapshot_before_each_solve()
    hr, sec = snapshot_step_hr_sec(step_index, step_min=float(step_min))
    dss.Text.Command(f"set hour={hr} sec={sec}")


def stress_profile(
    m_raw: np.ndarray,
    *,
    daily_stress: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """``m_eff = clip(1 + (m_raw - 1) * (1 + stress), lo, hi)`` — shared with DA-GPS daily compare."""
    m = np.asarray(m_raw, dtype=np.float64)
    dev = m - 1.0
    m_eff = 1.0 + dev * (1.0 + float(daily_stress))
    return np.clip(m_eff, float(lo), float(hi))


def write_two_col_profile_csv(path: Path, mult: np.ndarray, *, step_min: float) -> None:
    n = int(len(mult))
    t = np.arange(n, dtype=np.float64) * (float(step_min) / 60.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([t, mult]), delimiter=",", fmt="%.8g")


def _profile_bind_csv(
    source_csv: Path,
    mult: np.ndarray,
    *,
    prefix: str,
    step_min: float,
    npts: int,
) -> Path:
    """Return ``source_csv`` when ``mult`` matches col-2 of file; else write a temp CSV."""
    import run_injection_dataset as inj

    fp = source_csv.expanduser().resolve()
    if not fp.is_file():
        raise FileNotFoundError(fp)
    m_file = np.asarray(
        inj.read_profile_csv_two_col_noheader(str(fp), npts=int(npts), debug=False),
        dtype=np.float64,
    )
    m_use = np.asarray(mult[: int(npts)], dtype=np.float64)
    if m_file.shape[0] == m_use.shape[0] and np.allclose(m_file, m_use, rtol=0.0, atol=1e-9):
        return fp
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(fd)
    out = Path(tmp_name)
    write_two_col_profile_csv(out, m_use, step_min=float(step_min))
    return out


def _profile_mult_on_grid(
    csv_path: Path,
    *,
    npts: int,
    step_min: float,
    native_npts: int = 288,
    native_step_min: float = 5.0,
) -> np.ndarray:
    """Read a daily profile CSV and map it onto ``(npts, step_min)``.

    Native training/driver CSVs are usually 288 rows at 5-min spacing. Coarser
    OpenDSS / GNN grids use linear resampling over the 24 h day, not truncation.
    """
    import run_injection_dataset as inj
    from nonunique_opendss_daily import resample_daily_profile

    fp = csv_path.expanduser().resolve()
    m_full = np.asarray(
        inj.read_profile_csv_two_col_noheader(str(fp), npts=int(native_npts), debug=False),
        dtype=np.float64,
    )
    if int(npts) == int(m_full.shape[0]) and float(step_min) == float(native_step_min):
        return m_full
    return resample_daily_profile(
        m_full,
        npts=int(npts),
        step_min=int(step_min),
        native_npts=int(m_full.shape[0]),
    )


def prepare_parity_profiles(
    load_csv: Path,
    irr_csv: Path,
    *,
    npts: int,
    step_min: float,
    daily_stress: float = 0.0,
    stress_clip_lo: float = 0.1,
    stress_clip_hi: float = 3.0,
) -> ParityProfiles:
    """Single source for load / irradiance multipliers used by daily march and snapshot paths."""
    m_raw = _profile_mult_on_grid(load_csv, npts=int(npts), step_min=float(step_min))
    m_eff = stress_profile(
        m_raw,
        daily_stress=float(daily_stress),
        lo=float(stress_clip_lo),
        hi=float(stress_clip_hi),
    )
    m_irr = np.clip(
        _profile_mult_on_grid(irr_csv, npts=int(npts), step_min=float(step_min)),
        0.0,
        None,
    )
    load_bind = _profile_bind_csv(
        load_csv,
        m_eff,
        prefix="parity_load_",
        step_min=float(step_min),
        npts=int(npts),
    )
    irr_bind = _profile_bind_csv(
        irr_csv,
        m_irr,
        prefix="parity_irr_",
        step_min=float(step_min),
        npts=int(npts),
    )
    return ParityProfiles(
        m_raw=m_raw,
        m_eff=m_eff,
        m_irr=m_irr,
        load_bind_csv=load_bind,
        irr_bind_csv=irr_bind,
    )


def step_load_multiplier(m_eff: np.ndarray, step_index: int, scenario_scale: float) -> float:
    i = int(step_index)
    if i < 0 or i >= int(m_eff.shape[0]):
        return 0.0
    return float(m_eff[i]) * float(scenario_scale)


def step_irradiance_multiplier(m_irr: np.ndarray, step_index: int) -> float:
    i = int(step_index)
    if i < 0 or i >= int(m_irr.shape[0]):
        return 0.0
    return float(m_irr[i])


def apply_parity_tolerance_kw(
    tolerance_kw: float | None,
    *,
    base_kva_kw: float = _DEFAULT_BASE_KVA_KW,
) -> float | None:
    """Set OpenDSS ``tolerance`` (pu on ``BasekVA``) from a kW budget; ``None`` leaves default."""
    if tolerance_kw is None:
        return None
    if tolerance_kw <= 0.0:
        raise ValueError(f"tolerance_kw must be > 0, got {tolerance_kw}")
    bkva = float(base_kva_kw)
    if bkva <= 0.0:
        raise ValueError(f"base_kva_kw must be > 0, got {bkva}")
    tol_pu = float(tolerance_kw) / bkva
    dss.Text.Command(f"set tolerance={tol_pu:g}")
    return tol_pu


def rebind_loadshape_day5min(load_csv: Path, *, npts: int, step_min: float) -> None:
    fp = load_csv.expanduser().resolve()
    interval_h = float(step_min) / 60.0
    cmd = (
        f"New Loadshape.Day5min npts={int(npts)} interval={interval_h} "
        f'mult=(file="{fp.as_posix()}", col=2, header=no)'
    )
    dss.Text.Command(cmd)
    dss.Text.Command("BatchEdit Load..* Daily=Day5min")


def compile_and_bind_parity_daily_opendss(
    profiles: ParityProfiles,
    *,
    npts: int,
    step_min: float,
    tolerance_kw: float | None = None,
    base_kva_kw: float = _DEFAULT_BASE_KVA_KW,
) -> float | None:
    """Compile PV/unbalanced feeder and bind parity load + irradiance shapes for native daily march."""
    import run_daily_aggregate_dataset_8500 as rd8500

    rd8500._compile_8500_solar_unbalanced_pv_daily_setup()
    rebind_loadshape_day5min(profiles.load_bind_csv, npts=int(npts), step_min=float(step_min))
    rebind_irradiance_loadshape_irradday001(
        profiles.irr_bind_csv, npts=int(npts), step_min=float(step_min)
    )
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    return apply_parity_tolerance_kw(tolerance_kw, base_kva_kw=float(base_kva_kw))


def apply_daily_march_solver_knobs(
    *,
    step_min: float,
    tolerance_kw: float | None = None,
    base_kva_kw: float = _DEFAULT_BASE_KVA_KW,
) -> float | None:
    """Solver settings for native ``mode=daily`` QSTS march (parity with snapshot caps)."""
    tol_pu = apply_parity_tolerance_kw(tolerance_kw, base_kva_kw=float(base_kva_kw))
    dss.Text.Command("set mode=daily")
    dss.Text.Command(f"set stepsize={float(step_min)}m")
    dss.Text.Command("set number=1")
    dss.Text.Command(f"set maxiterations={_PARITY_MAXITERATIONS}")
    dss.Text.Command(f"set maxcontroliter={_max_control_iter()}")
    dss.Text.Command("set hour=0")
    dss.Text.Command("set sec=0")
    return tol_pu


def apply_scenario_scale_to_load_nameplates(scenario_scale: float) -> None:
    """Uniform nameplate scale (daily path); snapshot applies the same factor per step in ``m_t``."""
    if abs(float(scenario_scale) - 1.0) <= 1e-9:
        return
    if dss.Loads.First():
        while True:
            nm = dss.Loads.Name()
            dss.Loads.Name(nm)
            dss.Loads.kW(float(dss.Loads.kW()) * float(scenario_scale))
            dss.Loads.kvar(float(dss.Loads.kvar()) * float(scenario_scale))
            if not dss.Loads.Next():
                break


def set_pv_pmpp_nameplate_kw(pv_base_pmpp_kw: dict[str, float]) -> None:
    for nm, p0 in pv_base_pmpp_kw.items():
        if float(p0) <= 0.0:
            continue
        try:
            dss.PVsystems.Name(str(nm).strip())
            dss.PVsystems.Pmpp(float(p0))
        except Exception:
            pass


def collect_unscaled_load_bases() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load nameplate kW/kvar after ``_detach_daily_loadshape_from_loads()`` (no ``Daily=`` double scale)."""
    import run_daily_aggregate_dataset_8500 as rd8500

    loads, _ = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    return base_names, base_kw, base_kvar
