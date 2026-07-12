"""Multi-scenario mix-and-match warm-start band evaluation (metrics only).

Samples battery capacity, DER bus location(s), and load/PV profile pairs, then
calls ``run_da_gps_warmstart_band_daily`` with plotting disabled and aggregates
``da_gps_aggregated`` across scenarios (mean / std / median).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Metric keys collected from each scenario's ``da_gps_aggregated`` group stats.
AGG_METRIC_KEYS = (
    "mean_inside_band_frac",
    "mean_cloud_proximity",
    "mean_set_distance",
    "mean_outside_distance",
)
GROUPS = ("voltage", "regulator", "capacitor", "meta_aux")

DEFAULT_CANDIDATE_CSV_NAMES = (
    "bess_candidate_buses_scattered_3ph_mv.csv",
)
DRIVE_CANDIDATE_CANDIDATES = (
    Path(
        r"K:\My Drive\datasets_gnn2\original_8500_unbalanced_chunked_with_bess_aug_2000_40"
        r"\bess_candidate_buses_scattered_3ph_mv.csv"
    ),
    Path(
        "/content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_with_bess_aug_2000_40"
        "/bess_candidate_buses_scattered_3ph_mv.csv"
    ),
)


@dataclass
class MultiScenarioEvalConfig:
    """Knobs for the multi-scenario warm-start band sweep."""

    n_scenarios: int = 8
    seed: int = 42
    n_warm_starts: int = 3
    warm_start_mode: str = "wide"
    warm_start_randomize_static_caps: bool = False
    step_min: int = 5
    daily_stress: float = 0.0
    scenario_scale: float = 1.0
    ref_sample_index: int = 0
    gnn_batch_steps: int | None = None

    # Location sampling (matches withder BESS_NUM_NODES_MIN/MAX).
    n_buses_min: int = 1
    n_buses_max: int = 3

    # Capacity: "kw_uniform" samples peak P in [der_kw_min, der_kw_max];
    # "dataset_mva" samples total MVA ~ N(mean, sigma) then kw = max(0, mva)*1000.
    capacity_mode: str = "kw_uniform"
    der_kw_min: float = 250.0
    der_kw_max: float = 2000.0
    bess_total_mva_mean: float = 4.0
    bess_total_mva_sigma: float = 0.1
    # Q as fraction of |P|; upper bound matches dataset ``bess_q_frac_max``.
    q_frac_min: float = 0.05
    q_frac_max: float = 0.44

    include_der: bool = True
    candidate_csv: Path | str | None = None
    der_profile_csv: Path | str | None = None
    day1_dir: Path | str | None = None
    out_dir: Path | str | None = None
    save_per_scenario_run_dir: bool = True
    device: str | None = None

    # Kept True so voltage aggregated metrics use cache∩circuit nodes (same as single-scenario).
    plot_all_cache_nodes: bool = True


@dataclass
class ScenarioDraw:
    scenario_id: int
    seed: int
    der_buses: list[str]
    der_nominal_kw: float
    der_nominal_kvar: float
    q_frac: float
    capacity_mode: str
    bess_total_mva: float | None
    profile_pair_id: str
    load_profile: str
    pv_profile: str
    der_profile: str


def _as_path(p: Path | str | None) -> Path | None:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def resolve_candidate_csv(repo: Path, explicit: Path | str | None = None) -> Path:
    """Locate the scattered 3-phase MV BESS candidate CSV (repo copy or Drive)."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"BESS candidate CSV not found: {p}")

    env = os.environ.get("BESS_CANDIDATE_CSV", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    for name in DEFAULT_CANDIDATE_CSV_NAMES:
        p = (repo / name).resolve()
        if p.is_file():
            return p

    for p in DRIVE_CANDIDATE_CANDIDATES:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        "BESS candidate CSV not found. Expected repo/"
        f"{DEFAULT_CANDIDATE_CSV_NAMES[0]} (shipped with the repo) or set "
        "BESS_CANDIDATE_CSV / MultiScenarioEvalConfig.candidate_csv."
    )


def load_bess_candidate_buses(
    repo: Path,
    csv_path: Path | str | None = None,
) -> tuple[list[str], Path]:
    """Return unique 3-phase bus names from the candidate CSV + resolved path."""
    import pandas as pd

    path = resolve_candidate_csv(repo, csv_path)
    df = pd.read_csv(path)
    if "bus" not in df.columns:
        raise ValueError(f"Candidate CSV missing 'bus' column: {path}")
    buses = [str(b).strip().lower() for b in df["bus"].tolist() if str(b).strip()]
    buses = list(dict.fromkeys(buses))
    if not buses:
        raise ValueError(f"No buses in candidate CSV: {path}")
    return buses, path


def list_load_pv_profile_pairs(day1: Path) -> list[dict[str, str]]:
    """Return the 4 representative load/PV pairs ``load_day_00i`` / ``irr_day_00i``."""
    pairs: list[dict[str, str]] = []
    for i in (1, 2, 3, 4):
        load_p = day1 / f"load_day_{i:03d}.csv"
        irr_p = day1 / f"irr_day_{i:03d}.csv"
        if not load_p.is_file() or not irr_p.is_file():
            raise FileNotFoundError(
                f"Missing profile pair {i}: need {load_p.name} and {irr_p.name} under {day1}"
            )
        pairs.append(
            {
                "pair_id": f"day_{i:03d}",
                "load_profile": str(load_p.resolve()),
                "pv_profile": str(irr_p.resolve()),
            }
        )
    return pairs


def list_der_injection_profiles(
    day1: Path,
    *,
    include_zero: bool = False,
) -> list[Path]:
    """Optional DER schedule CSVs under the representative-days folder.

    By default excludes ``*_zero.csv`` (ablation / off schedule) so sampled capacity
    still drives a non-trivial injection shape. Pass ``include_zero=True`` to allow it.
    """
    names = (
        "battery_arbitrage_der_injection.csv",
        "battery_arbitrage_der_injection_zero.csv",
    )
    found: list[Path] = []
    for n in names:
        p = day1 / n
        if not p.is_file():
            continue
        if (not include_zero) and "zero" in n.lower():
            continue
        found.append(p)
    if not found:
        raise FileNotFoundError(f"No DER injection profile CSVs under {day1}")
    return found


def sample_scenario_draw(
    *,
    scenario_id: int,
    rng: np.random.Generator,
    candidate_buses: list[str],
    profile_pairs: list[dict[str, str]],
    der_profiles: list[Path],
    cfg: MultiScenarioEvalConfig,
) -> ScenarioDraw:
    """Draw one mix-and-match scenario (capacity × buses × load/PV × DER profile)."""
    n_bus_hi = min(int(cfg.n_buses_max), len(candidate_buses))
    n_bus_lo = min(int(cfg.n_buses_min), n_bus_hi)
    if n_bus_lo < 1:
        raise ValueError("Need at least one candidate bus to sample DER locations.")
    n_buses = int(rng.integers(n_bus_lo, n_bus_hi + 1))
    buses = [str(x) for x in rng.choice(candidate_buses, size=n_buses, replace=False)]

    mode = str(cfg.capacity_mode).strip().lower()
    bess_mva: float | None = None
    if mode in ("dataset_mva", "mva", "mva_normal"):
        bess_mva = float(
            max(
                0.0,
                float(cfg.bess_total_mva_mean)
                * (1.0 + float(rng.normal(0.0, float(cfg.bess_total_mva_sigma)))),
            )
        )
        der_kw = float(bess_mva * 1000.0)
        mode = "dataset_mva"
    elif mode in ("kw_uniform", "kw", "uniform_kw"):
        lo, hi = float(cfg.der_kw_min), float(cfg.der_kw_max)
        if hi < lo:
            lo, hi = hi, lo
        der_kw = float(rng.uniform(lo, hi))
        mode = "kw_uniform"
    else:
        raise ValueError(f"Unknown capacity_mode={cfg.capacity_mode!r}")

    q_lo, q_hi = float(cfg.q_frac_min), float(cfg.q_frac_max)
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    q_frac = float(rng.uniform(q_lo, q_hi))
    der_kvar = float(abs(der_kw) * q_frac)

    pair = profile_pairs[int(rng.integers(0, len(profile_pairs)))]
    der_prof = der_profiles[int(rng.integers(0, len(der_profiles)))]
    scen_seed = int(rng.integers(0, 2**31 - 1))

    return ScenarioDraw(
        scenario_id=int(scenario_id),
        seed=scen_seed,
        der_buses=buses,
        der_nominal_kw=der_kw,
        der_nominal_kvar=der_kvar,
        q_frac=q_frac,
        capacity_mode=mode,
        bess_total_mva=bess_mva,
        profile_pair_id=str(pair["pair_id"]),
        load_profile=str(pair["load_profile"]),
        pv_profile=str(pair["pv_profile"]),
        der_profile=str(Path(der_prof).resolve()),
    )


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    m = np.isfinite(arr)
    if not bool(m.any()):
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    x = arr[m]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def summarize_aggregated_across_scenarios(
    scenario_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean±std (and median/min/max) of each aggregated metric across scenarios."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group in GROUPS:
        metric_map: dict[str, dict[str, float]] = {}
        for key in AGG_METRIC_KEYS:
            vals: list[float] = []
            for row in scenario_rows:
                agg = (row.get("aggregated") or {}).get(group) or {}
                if key in agg:
                    vals.append(float(agg[key]))
            if vals:
                metric_map[key] = _finite_stats(vals)
        if metric_map:
            summary[group] = metric_map
    return summary


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def run_multi_scenario_warmstart_band_eval(
    *,
    boot: Any,
    cfg: MultiScenarioEvalConfig | None = None,
    checkpoint: Path | str | None = None,
    run_dir: Path | str | None = None,
    cache_pt: Path | str | None = None,
) -> dict[str, Any]:
    """Run N mix-and-match warm-start band scenarios and aggregate cloud metrics.

    Parameters
    ----------
    boot:
        ``NotebookBootstrap`` (or duck-typed) from ``bootstrap_warmstart_notebook``.
    cfg:
        Sweep knobs (``n_scenarios``, capacity ranges, bus counts, …).
    checkpoint / run_dir / cache_pt:
        Optional overrides for the model under evaluation.
    """
    from nonunique_da_gps_warmstart_band_daily import run_da_gps_warmstart_band_daily
    from nonunique_opendss_daily import DailySimConfig

    cfg = cfg or MultiScenarioEvalConfig()
    repo = Path(boot.repo).expanduser().resolve()
    day1 = _as_path(cfg.day1_dir) or Path(boot.day1).expanduser().resolve()
    candidate_buses, candidate_csv = load_bess_candidate_buses(repo, cfg.candidate_csv)
    profile_pairs = list_load_pv_profile_pairs(day1)
    der_profiles = list_der_injection_profiles(day1)
    if cfg.der_profile_csv is not None:
        der_profiles = [_as_path(cfg.der_profile_csv)]  # type: ignore[list-item]

    ckpt = _as_path(checkpoint) or Path(boot.checkpoint).expanduser().resolve()
    rdir = _as_path(run_dir) or Path(boot.run_dir).expanduser().resolve()
    cache = _as_path(cache_pt) or Path(boot.cache_pt).expanduser().resolve()
    device = cfg.device or getattr(boot, "device", None)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _as_path(cfg.out_dir) or (repo / "warmstart_band_runs" / f"multi_scenario_{tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    n = int(cfg.n_scenarios)
    if n < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n}")

    rng = np.random.default_rng(int(cfg.seed))
    scenario_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    print("=" * 72)
    print("Multi-scenario warm-start band eval (metrics only, no plots)")
    print(f"  n_scenarios={n}  seed={cfg.seed}  n_warm_starts={cfg.n_warm_starts}")
    print(f"  capacity_mode={cfg.capacity_mode!r}  buses=[{cfg.n_buses_min},{cfg.n_buses_max}]")
    print(f"  candidates={len(candidate_buses)} from {candidate_csv.name}")
    print(f"  profile_pairs={len(profile_pairs)}  der_profiles={len(der_profiles)}")
    print(f"  checkpoint={ckpt}")
    print(f"  out_dir={out_root}")
    print("=" * 72)

    for i in range(1, n + 1):
        draw = sample_scenario_draw(
            scenario_id=i,
            rng=rng,
            candidate_buses=candidate_buses,
            profile_pairs=profile_pairs,
            der_profiles=der_profiles,
            cfg=cfg,
        )
        scen_out = out_root / f"scenario_{i:03d}" if cfg.save_per_scenario_run_dir else out_root
        print(
            f"\n--- scenario {i}/{n}: buses={draw.der_buses}  "
            f"P={draw.der_nominal_kw:.1f} kW  Q={draw.der_nominal_kvar:.1f} kvar  "
            f"pair={draw.profile_pair_id} ---",
            flush=True,
        )
        t_scen0 = time.perf_counter()
        daily_cfg = DailySimConfig(
            step_min=int(cfg.step_min),
            include_der=bool(cfg.include_der),
            der_nominal_kw=float(draw.der_nominal_kw if cfg.include_der else 0.0),
            der_nominal_kvar=float(draw.der_nominal_kvar if cfg.include_der else 0.0),
            der_bus=",".join(draw.der_buses),
            der_profile_csv=Path(draw.der_profile),
            da_gps_run_dir=rdir,
            da_gps_cache_pt=cache,
            da_gps_checkpoint=ckpt,
        )
        result = run_da_gps_warmstart_band_daily(
            daily_cfg,
            n_warm_starts=int(cfg.n_warm_starts),
            warm_start_mode=str(cfg.warm_start_mode),
            warm_start_randomize_static_caps=bool(cfg.warm_start_randomize_static_caps),
            load_profile_path=draw.load_profile,
            pv_profile_path=draw.pv_profile,
            ref_sample_index=int(cfg.ref_sample_index),
            scenario_scale=float(cfg.scenario_scale),
            daily_stress=float(cfg.daily_stress),
            plot_all_cache_nodes=bool(cfg.plot_all_cache_nodes),
            plot_all_max_nodes=1,
            out_dir=scen_out,
            plot_reg_cap=False,
            plot_meta_aux=False,
            plot_warmstart_lines=False,
            write_voltage_pngs=False,
            seed=int(draw.seed),
            show=False,
            device=device,
            gnn_batch_steps=cfg.gnn_batch_steps,
        )
        wall_scen = time.perf_counter() - t_scen0
        aggregated = result.get("da_gps_aggregated") or {}
        row = {
            "scenario_id": draw.scenario_id,
            "config": {
                "seed": draw.seed,
                "der_buses": draw.der_buses,
                "n_buses": len(draw.der_buses),
                "der_nominal_kw": draw.der_nominal_kw,
                "der_nominal_kvar": draw.der_nominal_kvar,
                "q_frac": draw.q_frac,
                "capacity_mode": draw.capacity_mode,
                "bess_total_mva": draw.bess_total_mva,
                "profile_pair_id": draw.profile_pair_id,
                "load_profile": draw.load_profile,
                "pv_profile": draw.pv_profile,
                "der_profile": draw.der_profile,
                "n_warm_starts": int(cfg.n_warm_starts),
                "warm_start_mode": str(cfg.warm_start_mode),
                "step_min": int(cfg.step_min),
            },
            "aggregated": aggregated,
            "wall_s": float(wall_scen),
            "out_dir": str(result.get("out_dir") or scen_out),
            "n_collect_nodes": int(len(result.get("collect_nodes") or [])),
        }
        scenario_rows.append(row)
        print(f"  scenario {i} done in {wall_scen:.1f}s", flush=True)
        for group, stats in aggregated.items():
            frac = float(stats.get("mean_inside_band_frac", float("nan")))
            frac_s = f"{100.0 * frac:.1f}%" if frac == frac else "n/a"
            print(f"    [{group}] inside={frac_s}", flush=True)

    wall_total = time.perf_counter() - t0
    summary = summarize_aggregated_across_scenarios(scenario_rows)
    payload = {
        "mode": "multi_scenario_warmstart_band",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wall_s_total": float(wall_total),
        "wall_s_mean_per_scenario": float(wall_total / max(1, n)),
        "checkpoint": str(ckpt),
        "run_dir": str(rdir),
        "cache_pt": str(cache),
        "candidate_csv": str(candidate_csv),
        "n_candidate_buses": len(candidate_buses),
        "eval_config": asdict(cfg),
        "n_scenarios": n,
        "n_scenarios_ok": len(scenario_rows),
        "scenarios": scenario_rows,
        "summary_across_scenarios": summary,
    }
    out_json = out_root / "multi_scenario_band_metrics.json"
    _write_json(out_json, payload)

    print("\n" + "=" * 72)
    print(f"Multi-scenario summary ({len(scenario_rows)} scenarios, {wall_total:.1f}s total)")
    print(f"  wrote {out_json}")
    for group, metrics in summary.items():
        frac = metrics.get("mean_inside_band_frac") or {}
        if not frac:
            continue
        mean = frac.get("mean", float("nan"))
        std = frac.get("std", float("nan"))
        med = frac.get("median", float("nan"))
        mean_s = f"{100.0 * mean:.1f}%" if mean == mean else "n/a"
        std_s = f"{100.0 * std:.1f}%" if std == std else "n/a"
        med_s = f"{100.0 * med:.1f}%" if med == med else "n/a"
        print(f"  [{group}] inside mean±std={mean_s}±{std_s}  median={med_s}")
    print("=" * 72)

    return payload


__all__ = [
    "MultiScenarioEvalConfig",
    "ScenarioDraw",
    "list_der_injection_profiles",
    "list_load_pv_profile_pairs",
    "load_bess_candidate_buses",
    "run_multi_scenario_warmstart_band_eval",
    "sample_scenario_draw",
    "summarize_aggregated_across_scenarios",
]
