"""Shared OpenDSS daily-march helpers for nonunique experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import opendssdirect as dss

def _resolve_repo_root() -> Path:
    """Repo root for portable runs.

    Prefer the ``GNN2_REPO_ROOT`` env var (e.g. ``/content/GNN2`` on Colab), else
    fall back to this module's directory so local Windows runs are unaffected.
    """
    env = os.environ.get("GNN2_REPO_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parent


REPO_ROOT = _resolve_repo_root()
GRID_DIR = REPO_ROOT / "8500 nodes with solar unbalanced"

DAY_HOURS = 24
NATIVE_STEP_MIN = 5
NATIVE_NPTS = int(DAY_HOURS * 60 // NATIVE_STEP_MIN)

LOW_TAP, HIGH_TAP = -16, 16

MONITOR_CANDIDATES = ["190-8593.1", "190-8581.1", "190-7361.1", "l2973163.2"]

DER_BUS = "l2801895"
DER_PROFILE_CSV = REPO_ROOT / "a representativ days" / "battery_arbitrage_der_injection.csv"

DA_GPS_RUN_DIR = REPO_ROOT / (
    "gnn2_architecture_search/attention checkpoints/"
    "da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE"
)
DA_GPS_CACHE_PT = REPO_ROOT / (
    "datasets_gnn2_from pc/"
    "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt"
)
DA_GPS_CHECKPOINT = DA_GPS_RUN_DIR / "training_last.pt"
DA_GPS_LOAD_PROFILE = GRID_DIR / "5minDayShape.csv"
DA_GPS_PV_PROFILE = GRID_DIR / "irr_day_001.csv"
# Reference profiles used by ``run_da_gps_daily_opendss_compare.py`` driver / ``da_gps_daily_compare`` notebook cell.
DA_GPS_REF_LOAD_PROFILE = REPO_ROOT / "a representativ days" / "load_day_004.csv"
DA_GPS_REF_PV_PROFILE = REPO_ROOT / "a representativ days" / "irr_day_004.csv"

_da_gps_device_logged = False


def resolve_da_gps_device(device: str | None = None) -> str:
    """Return ``cuda`` if available and not overridden, else ``cpu``."""
    if device is not None and str(device).strip():
        choice = str(device).strip().lower()
        if choice in ("cuda", "gpu"):
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if choice not in ("auto", "default", ""):
            return str(device).strip()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def log_da_gps_device(device: str) -> None:
    """Print DA-GPS inference device once per process."""
    global _da_gps_device_logged
    if _da_gps_device_logged:
        return
    _da_gps_device_logged = True
    dev = str(device).strip().lower()
    if dev == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                print(f"[DA-GPS] device=cuda (GPU: {torch.cuda.get_device_name(0)})", flush=True)
                return
        except Exception:
            pass
    try:
        import torch

        if not torch.cuda.is_available():
            print("[DA-GPS] device=cpu (CUDA not available)", flush=True)
            return
    except Exception:
        pass
    print(f"[DA-GPS] device={dev}", flush=True)


@dataclass
class DailySimConfig:
    """Notebook-aligned simulation settings."""

    repo_root: Path = field(default_factory=lambda: REPO_ROOT)
    grid_dir: Path = field(default_factory=lambda: GRID_DIR)
    step_min: int = 5
    day_hours: int = DAY_HOURS
    include_der: bool = False
    der_nominal_kw: float = 0.0
    der_nominal_kvar: float = 0.0
    der_bus: str = DER_BUS
    der_profile_csv: Path = field(default_factory=lambda: DER_PROFILE_CSV)
    monitor_candidates: list[str] = field(default_factory=lambda: list(MONITOR_CANDIDATES))
    include_da_gps: bool = True
    da_gps_run_dir: Path = field(default_factory=lambda: DA_GPS_RUN_DIR)
    da_gps_cache_pt: Path = field(default_factory=lambda: DA_GPS_CACHE_PT)
    da_gps_checkpoint: Path = field(default_factory=lambda: DA_GPS_CHECKPOINT)
    da_gps_load_profile: Path = field(default_factory=lambda: DA_GPS_LOAD_PROFILE)
    da_gps_pv_profile: Path = field(default_factory=lambda: DA_GPS_PV_PROFILE)

    @property
    def npts(self) -> int:
        if 60 % self.step_min != 0:
            raise ValueError(f"step_min={self.step_min} must divide 60 evenly")
        if (self.day_hours * 60) % self.step_min != 0:
            raise ValueError("day_hours*60 must be divisible by step_min")
        return int(self.day_hours * 60 // self.step_min)

    @property
    def hours(self) -> np.ndarray:
        return np.arange(self.npts) * (self.step_min / 60.0)

    @property
    def steps_per_hour(self) -> int:
        return 60 // self.step_min

    @property
    def step_sec(self) -> int:
        return self.step_min * 60

    @property
    def loadshape_interval_h(self) -> float:
        return self.step_min / 60.0


def tap_pu_to_tap_number(tap_pu, mintap=0.9, maxtap=1.1, n_steps=32):
    """Map winding tap (pu) to OpenDSS TapNumber scale (e.g. -16..+16)."""
    min_tap_num = -n_steps // 2
    max_tap_num = n_steps // 2
    tap_pu = np.asarray(tap_pu, dtype=float)
    tap_num = min_tap_num + (tap_pu - mintap) * (max_tap_num - min_tap_num) / (maxtap - mintap)
    return np.clip(tap_num, min_tap_num, max_tap_num)


def _nearest_source_indices(src_h: np.ndarray, tgt_h: np.ndarray) -> np.ndarray:
    return np.abs(src_h[:, None] - tgt_h[None, :]).argmin(axis=0)


def resample_daily_profile(
    values,
    *,
    npts: int,
    step_min: int,
    day_hours: int = DAY_HOURS,
    native_npts: int | None = None,
    method: str = "linear",
):
    values = np.asarray(values, dtype=float).ravel()
    if native_npts is None:
        native_npts = len(values)
    if len(values) == npts:
        return values.copy()
    src_h = np.arange(len(values), dtype=float) * (day_hours / len(values))
    tgt_h = np.arange(npts, dtype=float) * (step_min / 60.0)
    if method == "nearest":
        return values[_nearest_source_indices(src_h, tgt_h)]
    if method != "linear":
        raise ValueError(f"unsupported resample method: {method!r}")
    return np.interp(tgt_h, src_h, values)


def resample_daily_profile_2d(
    arr,
    *,
    npts: int,
    step_min: int,
    native_npts: int | None = None,
    method: str = "linear",
):
    arr = np.asarray(arr, dtype=float)
    if native_npts is None:
        native_npts = int(arr.shape[0])
    if int(arr.shape[0]) == int(npts):
        return arr.copy()
    src_h = np.arange(int(native_npts), dtype=float) * (NATIVE_STEP_MIN / 60.0)
    tgt_h = np.arange(int(npts), dtype=float) * (step_min / 60.0)
    if method == "nearest":
        return arr[_nearest_source_indices(src_h, tgt_h), :].copy()
    if method != "linear":
        raise ValueError(f"unsupported resample method: {method!r}")
    out = np.empty((int(npts), int(arr.shape[1])), dtype=float)
    for j in range(int(arr.shape[1])):
        out[:, j] = np.interp(tgt_h, src_h, arr[:, j])
    return out


def align_irrad_loadshape_to_step(cfg: DailySimConfig):
    if cfg.step_min == NATIVE_STEP_MIN:
        return
    irr_csv = cfg.grid_dir / "irr_day_001.csv"
    if irr_csv.is_file():
        data = np.loadtxt(irr_csv, delimiter=",")
        mult = data[:, 1] if data.ndim > 1 else data
        mult = resample_daily_profile(
            mult, npts=cfg.npts, step_min=cfg.step_min, day_hours=cfg.day_hours
        )
    else:
        mult = resample_daily_profile(
            read_loadshape_mult("IrradDay001", cfg=cfg),
            npts=cfg.npts,
            step_min=cfg.step_min,
            day_hours=cfg.day_hours,
        )
    mult_str = ",".join(f"{x:.8g}" for x in mult)
    dss.Text.Command(
        f"Edit Loadshape.IrradDay001 npts={cfg.npts} interval={cfg.loadshape_interval_h} mult=({mult_str})"
    )


def compile_and_setup(cfg: DailySimConfig, *, snapshot: bool = False):
    os.chdir(cfg.grid_dir)
    dss.Text.Command("clear")
    dss.Text.Command('compile "Master-PV2MW-inv.dss"')
    shape_csv = cfg.grid_dir / "5minDayShape.csv"
    if shape_csv.is_file():
        if cfg.step_min == NATIVE_STEP_MIN:
            dss.Text.Command(
                f"New Loadshape.Day5min npts={cfg.npts} interval={cfg.loadshape_interval_h} "
                "mult=(file=5minDayShape.csv, col=2, header=no)"
            )
        else:
            raw = np.loadtxt(shape_csv, delimiter=",")
            mult_col = raw[:, 1] if raw.ndim > 1 else raw
            mult = resample_daily_profile(
                mult_col, npts=cfg.npts, step_min=cfg.step_min, day_hours=cfg.day_hours
            )
            mult_str = ",".join(f"{x:.8g}" for x in mult)
            dss.Text.Command(
                f"New Loadshape.Day5min npts={cfg.npts} interval={cfg.loadshape_interval_h} mult=({mult_str})"
            )
    else:
        t = np.linspace(0, 24, cfg.npts, endpoint=False)
        mult = 0.7 + 0.25 * np.sin(2 * np.pi * (t - 6) / 24)
        mult -= 0.15 * np.exp(-0.5 * ((t - 13) / 2.5) ** 2)
        mult = np.clip(mult, 0.4, 1.0)
        mult_str = ",".join(f"{x:.8g}" for x in mult)
        dss.Text.Command(
            f"New Loadshape.Day5min npts={cfg.npts} interval={cfg.loadshape_interval_h} mult=({mult_str})"
        )
    align_irrad_loadshape_to_step(cfg)
    dss.Text.Command("BatchEdit Load..* Daily=Day5min")
    dss.Text.Command("set controlmode=static")
    dss.Text.Command("set maxcontroliter=100")
    dss.Text.Command("set maxiterations=20")
    if snapshot:
        dss.Text.Command("set mode=snapshot")
    else:
        dss.Text.Command("set mode=daily")
        dss.Text.Command(f"set stepsize={cfg.step_min}m")
        dss.Text.Command("set number=1")
    if cfg.include_der:
        install_der_generator(cfg, cfg.der_bus)


def set_controllers(low_init: bool):
    tap = LOW_TAP if low_init else HIGH_TAP
    for nm in dss.RegControls.AllNames():
        dss.RegControls.Name(nm)
        dss.RegControls.TapNumber(tap)
    for nm in dss.Capacitors.AllNames():
        dss.Capacitors.Name(nm)
        n = dss.Capacitors.NumSteps()
        dss.Capacitors.States([0] * n if low_init else [1] * n)


def inject_controller_warmstart(
    step: int,
    reg_names: list[str],
    cap_names: list[str],
    reg_tap_pu_by_name: dict[str, np.ndarray],
    cap_sigmoid_by_name: dict[str, np.ndarray],
    *,
    cap_threshold: float = 0.5,
):
    """Inject DA-GPS predicted regulator taps and capacitor bank states before ``Solve()``.

    Note: OpenDSS has no public API to inject bus voltages/angles as a PF warm-start
    (``Bus.puVmagAngle`` etc. are read-only; ``Solution.InitSnap`` is internal only).
    Daily QSTS already reuses the prior step's converged voltage vector automatically.
    """
    for nm in reg_names:
        y = reg_tap_pu_by_name.get(nm)
        if y is None:
            y = reg_tap_pu_by_name.get(str(nm).lower())
        if y is None or step >= len(y):
            continue
        tap_num = int(round(float(tap_pu_to_tap_number(y[step]))))
        dss.RegControls.Name(nm)
        dss.RegControls.TapNumber(tap_num)
    for nm in cap_names:
        y = cap_sigmoid_by_name.get(nm)
        if y is None:
            y = cap_sigmoid_by_name.get(str(nm).lower())
        if y is None or step >= len(y):
            continue
        bank_on = float(y[step]) >= float(cap_threshold)
        dss.Capacitors.Name(nm)
        n = dss.Capacitors.NumSteps()
        dss.Capacitors.States([1] * n if bank_on else [0] * n)


def resolve_monitor_nodes(candidates: list[str]):
    all_names = dss.Circuit.AllNodeNames()
    lut = {n.lower(): n for n in all_names}
    found = [lut[c.lower()] for c in candidates if c.lower() in lut]
    if len(found) < 4:
        for n in all_names:
            if n.lower() not in {x.lower() for x in found}:
                found.append(n)
            if len(found) >= 4:
                break
    return found[:4]


def node_voltages_pu(nodes):
    names = dss.Circuit.AllNodeNames()
    mags = dss.Circuit.AllBusMagPu()
    lut = {n.lower(): float(v) for n, v in zip(names, mags)}
    return np.array([lut.get(n.lower(), np.nan) for n in nodes])


def detach_daily_loadshape():
    if not dss.Loads.First():
        return
    while True:
        nm = dss.Loads.Name()
        dss.Loads.Name(nm)
        dss.Loads.Daily("")
        if not dss.Loads.Next():
            break


def neutralize_irrad_loadshape(cfg: DailySimConfig):
    ones = ",".join("1" for _ in range(int(cfg.npts)))
    dss.Text.Command(
        f"Edit Loadshape.IrradDay001 npts={cfg.npts} interval={cfg.loadshape_interval_h} mult=({ones})"
    )


def collect_load_bases():
    names, kw, kvar = [], [], []
    if dss.Loads.First():
        while True:
            nm = dss.Loads.Name()
            dss.Loads.Name(nm)
            names.append(nm)
            kw.append(dss.Loads.kW())
            kvar.append(dss.Loads.kvar())
            if not dss.Loads.Next():
                break
    return names, np.array(kw), np.array(kvar)


def collect_pv_bases():
    names, pmpp = [], []
    for nm in dss.PVsystems.AllNames():
        dss.PVsystems.Name(nm)
        names.append(nm)
        pmpp.append(dss.PVsystems.Pmpp())
    return names, np.array(pmpp)


def read_loadshape_mult(name: str, *, cfg: DailySimConfig):
    try:
        dss.LoadShape.Name(name)
        mult = np.asarray(dss.LoadShape.PMult(), dtype=float)
    except Exception:
        return np.ones(cfg.npts, dtype=float)
    if len(mult) != cfg.npts:
        mult = resample_daily_profile(
            mult, npts=cfg.npts, step_min=cfg.step_min, day_hours=cfg.day_hours
        )
    return mult[: cfg.npts]


def load_der_multiplier_profile(csv_path: Path, *, cfg: DailySimConfig):
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        mult = np.atleast_1d(data)
        times = np.arange(len(mult), dtype=float) * (NATIVE_STEP_MIN / 60.0)
    else:
        times = data[:, 0].astype(float)
        mult = data[:, 1].astype(float)
    if len(mult) == cfg.npts:
        return mult
    target = np.arange(cfg.npts, dtype=float) * (cfg.step_min / 60.0)
    return np.interp(target, times, mult)


def install_der_generator(cfg: DailySimConfig, bus: str):
    bus = str(bus).strip().lower()
    kv_ll = 12.47
    try:
        dss.Circuit.SetActiveBus(bus)
        kv_ln = float(dss.Bus.kVBase())
        if np.isfinite(kv_ln) and kv_ln > 0:
            kv_ll = kv_ln * np.sqrt(3.0)
    except Exception:
        pass
    dss.Text.Command(
        f"New Generator.DER1 phases=3 bus1={bus} conn=wye model=1 "
        f"kV={kv_ll:.6f} kW=0 kvar=0 vminpu=0.01 vmaxpu=10"
    )


def set_der_injection(cfg: DailySimConfig, t: int, der_mult: np.ndarray):
    dss.Generators.Name("DER1")
    dss.Generators.kW(float(cfg.der_nominal_kw * der_mult[t]))
    dss.Generators.kvar(float(cfg.der_nominal_kvar * der_mult[t]))


def apply_explicit_loads_pv(load_names, base_kw, base_kvar, pv_names, pv_base, m_t, ir_t):
    for nm, kw, kvar in zip(load_names, base_kw, base_kvar):
        dss.Loads.Name(nm)
        dss.Loads.kW(float(kw * m_t))
        dss.Loads.kvar(float(kvar * m_t))
    for nm, pmpp in zip(pv_names, pv_base):
        dss.PVsystems.Name(nm)
        dss.PVsystems.Pmpp(float(pmpp * ir_t))


def read_solve_iterations() -> tuple[int, int, int]:
    ctrl = pf_total = pf_most = 0
    try:
        ctrl = int(dss.Solution.ControlIterations())
    except Exception:
        pass
    try:
        pf_total = int(dss.Solution.TotalIterations())
    except Exception:
        pass
    try:
        pf_most = int(dss.Solution.MostIterationsDone())
    except Exception:
        pf_most = pf_total
    return ctrl, pf_total, pf_most


def record_step(
    t: int,
    reg_names: list[str],
    cap_names: list[str],
    monitor_nodes: list[str],
    taps: np.ndarray,
    cap_on: np.ndarray,
    volts: np.ndarray,
    converged: np.ndarray,
    control_iters: np.ndarray | None = None,
    pf_iters_total: np.ndarray | None = None,
    pf_iters_most: np.ndarray | None = None,
):
    converged[t] = bool(dss.Solution.Converged())
    for i, nm in enumerate(reg_names):
        dss.RegControls.Name(nm)
        taps[t, i] = dss.RegControls.TapNumber()
    for j, nm in enumerate(cap_names):
        dss.Capacitors.Name(nm)
        cap_on[t, j] = sum(dss.Capacitors.States())
    volts[t] = node_voltages_pu(monitor_nodes)
    if control_iters is not None:
        ctrl, pf_total, pf_most = read_solve_iterations()
        control_iters[t] = ctrl
        pf_iters_total[t] = pf_total
        if pf_iters_most is not None:
            pf_iters_most[t] = pf_most


def empty_run_arrays(cfg: DailySimConfig, reg_names: list[str], cap_names: list[str], monitor_nodes: list[str]):
    return {
        "taps": np.zeros((cfg.npts, len(reg_names))),
        "cap_on": np.zeros((cfg.npts, len(cap_names))),
        "volts": np.zeros((cfg.npts, len(monitor_nodes))),
        "converged": np.zeros(cfg.npts, dtype=bool),
        "control_iters": np.zeros(cfg.npts, dtype=int),
        "pf_iters_total": np.zeros(cfg.npts, dtype=int),
        "pf_iters_most": np.zeros(cfg.npts, dtype=int),
    }


def make_run_result(
    label: str,
    reg_names: list[str],
    cap_names: list[str],
    monitor_nodes: list[str],
    arrays: dict[str, Any],
    **extra,
) -> dict[str, Any]:
    return {
        "label": label,
        "reg_names": reg_names,
        "cap_names": cap_names,
        "monitor_nodes": monitor_nodes,
        **arrays,
        **extra,
    }
