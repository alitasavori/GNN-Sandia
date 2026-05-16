"""
IEEE 8500 daily aggregate-load dataset generation (M1-focused pipeline).

Pipeline implemented:
  1) Sweep an interval of total aggregated load scaling.
  2) Distribute aggregated load across all loads by nominal shares.
  3) Apply per-device perturbation (independent multiplicative noise).
  4) Select time stamps from daily load profile.
  5) Solve OpenDSS and save node-wise feature/target samples.

Outputs (new folder):
  datasets_gnn2/loadtype_8500_dailyagg/
    - gnn_node_index_master.csv
    - gnn_edges_phase_static.csv
    - gnn_sample_meta.csv (per sample: P/Q load totals, post-solve realized loads, losses, upstream grid)
    - gnn_node_features_and_targets.csv (per node: electrical_distance_ohm from substation, M1 P/Q, voltages)
"""
from __future__ import annotations

import csv
import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj
import run_loadtype_dataset as lt_dist
import run_loadtype_dataset_8500 as lt8500
from compare_opendss_snapshot_helpers import (
    force_snapshot_mode_for_compare_timing as _force_snapshot_mode_for_compare_timing,
    reassert_snapshot_before_each_solve as _reassert_snapshot_before_each_solve,
)

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
lt8500 = importlib.reload(lt8500)

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    # Notebook exec(open(...).read()) path
    REPO_ROOT = Path.cwd()
OUT_DIR = REPO_ROOT / "datasets_gnn2" / "loadtype_8500_dailyagg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CSV = OUT_DIR / "gnn_edges_phase_static.csv"
NODE_CSV = OUT_DIR / "gnn_node_features_and_targets.csv"
SAMPLE_CSV = OUT_DIR / "gnn_sample_meta.csv"
NODE_INDEX_CSV = OUT_DIR / "gnn_node_index_master.csv"
RUN_DSS_DAILY = REPO_ROOT / "8500-node" / "Run_8500Node_Daily_5min.dss"

# Unbalanced IEEE 8500 with PV (same tree as ``run_original_style_dataset_8500_unbalanced``).
SOLAR_UNBAL_8500_DIR = REPO_ROOT / "8500 nodes with solar unbalanced"
MASTER_PV2_INV_DSS = SOLAR_UNBAL_8500_DIR / "Master-PV2MW-inv.dss"


def _is_source_like_bus(bus_name: str) -> bool:
    b = str(bus_name).strip().lower()
    return b.startswith("sourcebus") or b.startswith("_hvmv_sub")


def _sum_loads_post_solve_kw_kvar() -> tuple[float, float]:
    """Sum realized P/Q over all Load elements (post power-flow)."""
    p_sum, q_sum = 0.0, 0.0
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        pwr = dss.CktElement.TotalPowers()
        if pwr is not None and len(pwr) >= 2:
            p_sum += float(pwr[0])
            q_sum += float(pwr[1])
        if not dss.Loads.Next():
            break
    return p_sum, q_sum


def _circuit_losses_kw_kvar() -> tuple[float, float]:
    """Total circuit losses (post solve); kW / kvar (handles W/VAR if magnitudes are large)."""
    loss = dss.Circuit.Losses()
    p_l, q_l = float(loss[0]), float(loss[1])
    if abs(p_l) > 1000.0 or abs(q_l) > 1000.0:
        p_l /= 1000.0
        q_l /= 1000.0
    return p_l, q_l


def _grid_upstream_post_kw_kvar() -> tuple[float, float]:
    """Power from upstream / slack (post solve); positive = injection into the feeder (same as injection dataset)."""
    pwr = dss.Circuit.TotalPower()
    return -float(pwr[0]), -float(pwr[1])


def _detach_daily_loadshape_from_loads() -> None:
    """Clear Daily on every Load so kW/kvar are not multiplied again at solve.

    Run_8500Node_Daily_5min.dss runs ``BatchEdit Load..* Daily=Day5min``. This script already
    applies profile[m_t] when setting kW/kvar; leaving Daily attached would scale by m_t twice,
    so post-solve TotalPowers sums would read ~m_t × the intended totals.

    Note: ``BatchEdit Load..* Daily=`` does not clear the property in OpenDSS (verified: Daily
    stays ``day5min``); clearing must use the Loads API per element.
    """
    dss.Loads.First()
    while True:
        nm = dss.Loads.Name()
        dss.Loads.Name(nm)
        dss.Loads.Daily("")
        if not dss.Loads.Next():
            break


def _compile_8500_daily_setup() -> None:
    if not RUN_DSS_DAILY.is_file():
        raise FileNotFoundError(f"Missing daily setup entrypoint: {RUN_DSS_DAILY}")
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{os.path.abspath(str(RUN_DSS_DAILY))}"')
    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=5m")
    dss.Text.Command("set number=1")
    dss.Text.Command("set maxiterations=30")
    dss.Text.Command("set maxcontroliter=20000")


def _compile_8500_solar_unbalanced_pv_daily_setup() -> None:
    """Compile the PV/unbalanced feeder (``Master-PV2MW-inv.dss``).

    Matches ``run_original_style_dataset_8500_unbalanced._compile_8500_unbalanced_daily_setup``:
    ``cd`` into the model folder, redirect the PV master, attach ``Day5min`` from that folder's
    ``5minDayShape.csv``, then apply the same solver knobs as :func:`_compile_8500_daily_setup`.

    Use this for workflows that must read ``PVSystem`` post-solve P/Q (e.g. DA-GPS daily compare
    meta columns ``pv_*_p_post_kw``). The legacy :func:`_compile_8500_daily_setup` builds
    ``8500-node/Master.dss``, which has **no** PV objects.
    """
    if not MASTER_PV2_INV_DSS.is_file():
        raise FileNotFoundError(f"Missing PV master DSS: {MASTER_PV2_INV_DSS}")
    dayshape_csv = SOLAR_UNBAL_8500_DIR / "5minDayShape.csv"
    if not dayshape_csv.is_file():
        raise FileNotFoundError(f"Missing 5minDayShape.csv next to PV master: {dayshape_csv}")
    dss.Basic.ClearAll()
    dss.Text.Command(f'cd "{os.path.abspath(str(SOLAR_UNBAL_8500_DIR))}"')
    dss.Text.Command(f'redirect "{os.path.abspath(str(MASTER_PV2_INV_DSS))}"')
    dss.Text.Command(
        f'New Loadshape.Day5min npts=288 interval=0.0833333333333333 mult=(file="{os.path.abspath(str(dayshape_csv))}", col=2, header=no)'
    )
    dss.Text.Command("BatchEdit Load..* Daily=Day5min")
    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=5m")
    dss.Text.Command("set number=1")
    dss.Text.Command("set maxiterations=30")
    dss.Text.Command("set maxcontroliter=20000")


def _resolve_daily_profile_csv(profile_csv: str | Path | None) -> Path:
    """
    Resolve path to a two-column (time, multiplier) daily shape CSV.

    - ``None`` → ``8500-node/5minDayShape.csv`` (original default).
    - Absolute path that exists → used as-is.
    - Otherwise: ``8500-node/<basename>`` (e.g. ``5minDayShape2.csv``).
    """
    default_name = "5minDayShape.csv"
    if profile_csv is None:
        csv_path = REPO_ROOT / "8500-node" / default_name
    else:
        p = Path(profile_csv)
        if p.is_file():
            csv_path = p.resolve()
        else:
            csv_path = (REPO_ROOT / "8500-node" / p.name).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing daily profile CSV: {csv_path}\n"
            f"  (requested profile_csv={profile_csv!r}; place file under {REPO_ROOT / '8500-node'} or pass an absolute path.)"
        )
    return csv_path


def _daily_profile_5min(npts: int = 288, profile_csv: str | Path | None = None) -> np.ndarray:
    csv_path = _resolve_daily_profile_csv(profile_csv)
    return inj.read_profile_csv_two_col_noheader(str(csv_path), npts=npts, debug=False).astype(np.float32)


def _collect_loads_and_maps() -> tuple[list[dict], dict[str, list[tuple[str, int, float]]]]:
    loads: list[dict] = []
    load_to_busph: dict[str, list[tuple[str, int, float]]] = {}
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        dss.Loads.Name(name)
        loads.append(
            {
                "name": name,
                "kw": float(dss.Loads.kW()),
                "kvar": float(dss.Loads.kvar()),
            }
        )
        load_to_busph[name] = lt8500._busph_fracs_load(name)
        if not dss.Loads.Next():
            break
    return loads, load_to_busph


def _discover_reg_controls() -> list[str]:
    reg_names: list[str] = []
    try:
        if dss.RegControls.First():
            while True:
                reg_names.append(str(dss.RegControls.Name()))
                if not dss.RegControls.Next():
                    break
    except Exception:
        reg_names = []
    return sorted(reg_names)


def _discover_capacitors() -> list[str]:
    cap_names: list[str] = []
    try:
        if dss.Capacitors.First():
            while True:
                cap_names.append(str(dss.Capacitors.Name()))
                if not dss.Capacitors.Next():
                    break
    except Exception:
        cap_names = []
    return sorted(cap_names)


def _read_capacitor_sample_fields(cap_names: list[str]) -> dict[str, float | int]:
    """Per capacitor: n_steps_on, nameplate kvar, post-solve Q (in column order).

    q_nominal_kvar is the OpenDSS ``kvar`` rating (DSS nameplate). q_post_kvar uses
    ``-TotalPowers()[1]`` so **positive** means VArs **injected** by the bank (same cap
    convention as ``compare_nominal_vs_realized.py``).
    """
    out: dict[str, float | int] = {}
    for nm in cap_names:
        steps: list[int] = []
        try:
            dss.Capacitors.Name(nm)
            st = dss.Capacitors.States()
            if st is None:
                steps = []
            elif isinstance(st, (list, tuple, np.ndarray)):
                steps = [int(x) for x in st]
            else:
                steps = [int(st)]
        except Exception:
            try:
                dss.Capacitors.Name(nm)
                st1 = int(dss.Capacitors.State())
                steps = [st1]
            except Exception:
                steps = []

        n_on = int(sum(1 for x in steps if int(x) > 0))
        q_nom = np.nan
        try:
            dss.Capacitors.Name(nm)
            q_nom = float(dss.Capacitors.kvar())
        except Exception:
            pass

        q_post = np.nan
        try:
            dss.Circuit.SetActiveElement(f"Capacitor.{nm}")
            pwr = dss.CktElement.TotalPowers()
            if pwr is not None and len(pwr) >= 2:
                q_post = float(-float(pwr[1]))
        except Exception:
            pass

        out[f"cap_{nm}_n_steps_on"] = n_on
        out[f"cap_{nm}_q_nominal_kvar"] = float(q_nom) if np.isfinite(q_nom) else np.nan
        out[f"cap_{nm}_q_post_kvar"] = float(q_post) if np.isfinite(q_post) else np.nan
    return out


# Notebooks / older snippets may call this name; same payload (includes Q columns).
_read_capacitor_state = _read_capacitor_sample_fields


def _read_reg_control_state(reg_names: list[str]) -> dict[str, float | int]:
    """Per RegControl: winding tap (pu) only (no discrete tap_pos)."""
    out: dict[str, float | int] = {}
    for nm in reg_names:
        tap_val = np.nan
        try:
            dss.RegControls.Name(nm)
            xfmr = str(dss.RegControls.Transformer())
            wdg = int(dss.RegControls.Winding())
            if xfmr:
                dss.Transformers.Name(xfmr)
                dss.Transformers.Wdg(wdg)
                tap_val = float(dss.Transformers.Tap())
        except Exception:
            tap_val = np.nan
        out[f"reg_{nm}_tap_pu"] = float(tap_val) if np.isfinite(tap_val) else np.nan
    return out


def generate_dataset_8500_daily_aggregate(
    *,
    n_scenarios: int = 500,
    k_snapshots_per_scenario: int = 20,
    total_load_scale_range: tuple[float, float] = (0.7, 1.3),
    total_load_scale_disjoint_ranges: Optional[list[tuple[float, float]]] = None,
    sigma_device: float = 0.03,
    master_seed: int = 20260324,
    vmin_safe_pu: float = 0.85,
    vmax_safe_pu: float = 1.15,
    include_source_in_safe_band: bool = True,
    return_node_df: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate samples by:
      - sampling a scenario-level total load scale from range
        (or from ``total_load_scale_disjoint_ranges``: pick one interval uniformly,
        then sample uniformly inside that interval — excludes the "middle" between intervals)
      - selecting time points from daily load profile
      - applying total scale * profile[t] and per-device perturbations
      - solving and recording node features/targets
    """
    if k_snapshots_per_scenario < 1:
        raise ValueError("k_snapshots_per_scenario must be >= 1")

    disjoint: Optional[list[tuple[float, float]]] = None
    if total_load_scale_disjoint_ranges:
        disjoint = [(float(a), float(b)) for a, b in total_load_scale_disjoint_ranges]
        for lo_i, hi_i in disjoint:
            if not (0 < lo_i <= hi_i):
                raise ValueError(f"Invalid disjoint interval ({lo_i}, {hi_i}); need 0 < lo <= hi.")
        lo = min(x for x, _ in disjoint)
        hi = max(y for _, y in disjoint)
        print(f"[diag] total_load_scale: DISJOINT uniform ranges {disjoint} (envelope [{lo}, {hi}])", flush=True)
    else:
        lo, hi = float(total_load_scale_range[0]), float(total_load_scale_range[1])
        if not (0 < lo <= hi):
            raise ValueError(f"Invalid total_load_scale_range={total_load_scale_range}")
        print(f"[diag] total_load_scale: single uniform range [{lo}, {hi}]", flush=True)

    # Initial compile and static artifacts (aligned with notebook Step 9 daily setup)
    _compile_8500_daily_setup()
    _detach_daily_loadshape_from_loads()
    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}
    # Safe-band evaluation aligned with Step-9 style node population:
    # all phase nodes from the compiled circuit, optionally excluding source buses.
    safe_band_eval_indices = []
    for i, n in enumerate(node_names_master):
        b = n.split(".")[0]
        if (not include_source_in_safe_band) and _is_source_like_bus(b):
            continue
        safe_band_eval_indices.append(i)
    if not safe_band_eval_indices:
        raise RuntimeError("No nodes available for safe-band evaluation.")
    n_src_excl = sum(1 for n in node_names_master if _is_source_like_bus(n.split(".")[0]))
    if include_source_in_safe_band:
        print(
            f"[diag] safe-band eval: {len(safe_band_eval_indices)}/{len(node_names_master)} nodes "
            "(all phase nodes, no kVBase filter — aligned with notebook Step 9)"
        )
    else:
        print(
            f"[diag] safe-band eval: {len(safe_band_eval_indices)}/{len(node_names_master)} nodes "
            f"(source/substation buses excluded from metric; count skipped={n_src_excl})"
        )
    pd.DataFrame({"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}).to_csv(
        NODE_INDEX_CSV, index=False
    )
    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_master,
        edge_csv_path=str(EDGE_CSV),
        excluded_buses=(),
    )
    node_to_dist = lt_dist._compute_electrical_distance_from_source(node_names_master, str(EDGE_CSV))
    dist_vals = list(node_to_dist.values())
    print(
        f"[diag] electrical_distance_ohm: min={min(dist_vals):.6g} max={max(dist_vals):.6g} "
        f"(|Z| sum along min-|Z| path from substation)"
    )

    # Baseline loads and bus-phase maps
    loads, load_to_busph = _collect_loads_and_maps()
    if not loads:
        raise RuntimeError("No loads found in 8500 feeder.")
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    p_total_base = float(base_kw.sum())
    q_total_base = float(base_kvar.sum())
    if p_total_base <= 0 or q_total_base <= 0:
        raise RuntimeError("Unexpected non-positive baseline load totals.")
    reg_names = _discover_reg_controls()
    cap_names = _discover_capacitors()
    print(f"[diag] regcontrols={len(reg_names)} capacitors={len(cap_names)}")

    mL = _daily_profile_5min(npts=inj.NPTS)
    rng_master = np.random.default_rng(master_seed)

    if not (0.0 < float(vmin_safe_pu) < float(vmax_safe_pu)):
        raise ValueError(f"Invalid safe voltage band: [{vmin_safe_pu}, {vmax_safe_pu}]")

    rows_sample: list[dict] = []
    sample_id = 0
    skipped_nonconv = 0
    skipped_badv = 0
    total_v_outside_band = 0
    n_node_rows_written = 0

    node_fieldnames = [
        "sample_id",
        "node",
        "node_idx",
        "bus",
        "phase",
        "electrical_distance_ohm",
        "p_load_kw",
        "q_load_kvar",
        "vmag_pu",
        "vang_deg",
    ]
    with open(NODE_CSV, "w", newline="", encoding="utf-8") as f_node:
        node_writer = csv.DictWriter(f_node, fieldnames=node_fieldnames)
        node_writer.writeheader()

        for s in range(n_scenarios):
            # fresh circuit each scenario
            _compile_8500_daily_setup()
            _detach_daily_loadshape_from_loads()
            # keep all loads as model 1 (M1), as requested
            dss.Loads.First()
            while True:
                nm = dss.Loads.Name()
                dss.Loads.Name(nm)
                dss.Loads.Model(1)
                if not dss.Loads.Next():
                    break

            if disjoint:
                ri = int(rng_master.integers(0, len(disjoint)))
                a, b = disjoint[ri]
                scenario_scale = float(rng_master.uniform(a, b))
            else:
                scenario_scale = float(rng_master.uniform(lo, hi))
            rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            # timestamps selected from load profile
            times = inj.select_times_anchors_equalpop(
                profile=mL,
                K=k_snapshots_per_scenario,
                B=10,
                include_anchors=True,
                rng=rng_times,
            )
            times = [int(t) for t in times]
            rng_dev = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            nonconv_this_scenario = 0
            badv_this_scenario = 0
            outside_band_this_scenario = 0
            below_band_this_scenario = 0
            above_band_this_scenario = 0
            finite_v_count_this_scenario = 0
            offender_counts: dict[str, int] = {}

            for t in times:
                # reset to scenario baseline before each timestamp application
                # (same topology, set all load values explicitly each step)
                m_t = float(mL[t])
                total_scale_t = scenario_scale * m_t

                noise_p = np.maximum(0.0, 1.0 + rng_dev.normal(0.0, sigma_device, size=len(loads)))
                noise_q = np.maximum(0.0, 1.0 + rng_dev.normal(0.0, sigma_device, size=len(loads)))
                kw_set = base_kw * total_scale_t * noise_p
                kvar_set = base_kvar * total_scale_t * noise_q

                # Apply per-device setpoints and aggregate per-node M1 features
                busphP_load: dict[tuple[str, int], float] = {}
                busphQ_load: dict[tuple[str, int], float] = {}
                for i, name in enumerate(base_names):
                    dss.Loads.Name(name)
                    dss.Loads.kW(float(kw_set[i]))
                    dss.Loads.kvar(float(kvar_set[i]))
                    for (bus, ph, w) in load_to_busph[name]:
                        busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + float(kw_set[i]) * float(w)
                        busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + float(kvar_set[i]) * float(w)

                try:
                    hr = int(t // 12)
                    sec = int((t % 12) * 300)
                    dss.Text.Command(f"set hour={hr} sec={sec}")
                    dss.Solution.Solve()
                except Exception:
                    pass
                if not dss.Solution.Converged():
                    skipped_nonconv += 1
                    nonconv_this_scenario += 1
                    continue

                # Use exact same pu extraction path as the Step-9 notebook workflow.
                vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
                vmag_arr = np.asarray(vmag_m, dtype=float)
                if not np.isfinite(vmag_arr).all():
                    skipped_badv += 1
                    badv_this_scenario += 1
                    continue

                vmag_eval = vmag_arr[safe_band_eval_indices]
                mask_below = vmag_eval < float(vmin_safe_pu)
                mask_above = vmag_eval > float(vmax_safe_pu)
                mask_out = mask_below | mask_above
                n_below = int(np.sum(mask_below))
                n_above = int(np.sum(mask_above))
                n_outside = int(np.sum(mask_out))
                n_finite = int(np.sum(np.isfinite(vmag_eval)))
                outside_band_this_scenario += n_outside
                below_band_this_scenario += n_below
                above_band_this_scenario += n_above
                finite_v_count_this_scenario += n_finite
                total_v_outside_band += n_outside

                p_load_post_kw, q_load_post_kvar = _sum_loads_post_solve_kw_kvar()
                p_loss_post_kw, q_loss_post_kvar = _circuit_losses_kw_kvar()
                p_grid_post_kw, q_grid_post_kvar = _grid_upstream_post_kw_kvar()

                # Count offending nodes for quick diagnostics.
                if n_outside > 0:
                    eval_idx = np.asarray(safe_band_eval_indices, dtype=int)
                    for local_idx in np.where(mask_out)[0].tolist():
                        nm = str(node_names_master[int(eval_idx[local_idx])]).lower()
                        offender_counts[nm] = offender_counts.get(nm, 0) + 1

                # M1-only features
                vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

                rows_sample.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": s,
                        "t_index": t,
                        "t_minutes": int(t * inj.STEP_MIN),
                        "scenario_total_load_scale": scenario_scale,
                        "m_loadshape": m_t,
                        "effective_total_scale": total_scale_t,
                        "P_load_sum_post_kw": float(p_load_post_kw),
                        "Q_load_sum_post_kvar": float(q_load_post_kvar),
                        "P_loss_total_post_kw": float(p_loss_post_kw),
                        "Q_loss_total_post_kvar": float(q_loss_post_kvar),
                        "P_grid_upstream_post_kw": float(p_grid_post_kw),
                        "Q_grid_upstream_post_kvar": float(q_grid_post_kvar),
                        "sigma_device": float(sigma_device),
                        "safe_vmin_pu": float(vmin_safe_pu),
                        "safe_vmax_pu": float(vmax_safe_pu),
                        "n_v_outside_safe_band": int(n_outside),
                        "n_v_below_safe_band": int(n_below),
                        "n_v_above_safe_band": int(n_above),
                        **_read_reg_control_state(reg_names),
                        **_read_capacitor_sample_fields(cap_names),
                    }
                )

                rows_node_this_sample: list[dict] = []
                for n in node_names_master:
                    bus, phs = n.split(".")
                    ph = int(phs)
                    vm, va = vdict_m.get(n, (np.nan, np.nan))
                    rows_node_this_sample.append(
                        {
                            "sample_id": sample_id,
                            "node": n,
                            "node_idx": int(node_to_idx_master[n]),
                            "bus": bus,
                            "phase": int(ph),
                            "electrical_distance_ohm": float(node_to_dist.get(n, 0.0)),
                            # M1-only dataset: expose load P/Q with simplified names.
                            "p_load_kw": float(busphP_load.get((bus, ph), 0.0)),
                            "q_load_kvar": float(busphQ_load.get((bus, ph), 0.0)),
                            "vmag_pu": float(vm),
                            "vang_deg": float(va),
                        }
                    )
                node_writer.writerows(rows_node_this_sample)
                n_node_rows_written += len(rows_node_this_sample)

                sample_id += 1

            pct_out = 100.0 * outside_band_this_scenario / max(finite_v_count_this_scenario, 1)
            top_off = sorted(offender_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_off_str = ", ".join([f"{k}:{v}" for k, v in top_off]) if top_off else "none"
            print(
                f"[scenario {s+1}/{n_scenarios}] scale={scenario_scale:.3f} "
                f"kept_samples={sample_id} "
                f"nonconv_this_s={nonconv_this_scenario} badV_this_s={badv_this_scenario} "
                f"v_outside_band_this_s={outside_band_this_scenario} "
                f"(below={below_band_this_scenario}, above={above_band_this_scenario}, pct={pct_out:.2f}%) "
                f"N_nodes={len(node_names_master)} "
                f"top_offenders=[{top_off_str}] "
                f"skip_nonconv_total={skipped_nonconv} skip_badV_total={skipped_badv}"
            )

    df_sample = pd.DataFrame(rows_sample)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node = pd.read_csv(NODE_CSV) if return_node_df else pd.DataFrame()

    print("\n[8500 DAILY-AGG DATASET] saved.")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  sample_meta: {SAMPLE_CSV}")
    print(f"  node_features_targets: {NODE_CSV}")
    print(f"  kept samples: {df_sample['sample_id'].nunique() if len(df_sample) else 0}")
    print(f"  skipped_nonconv={skipped_nonconv} skipped_badV={skipped_badv}")
    print(
        f"  safe_band=[{float(vmin_safe_pu):.3f}, {float(vmax_safe_pu):.3f}] "
        f"total_v_outside_safe_band={int(total_v_outside_band)}"
    )
    print(f"  node_rows_written={int(n_node_rows_written)}")
    if not return_node_df:
        print("  return_node_df=False -> skipped loading node CSV into memory")
    return df_sample, df_node


if __name__ == "__main__":
    generate_dataset_8500_daily_aggregate(
        n_scenarios=500,
        k_snapshots_per_scenario=20,
        total_load_scale_range=(0.7, 1.3),
        sigma_device=0.03,
        master_seed=20260324,
        vmin_safe_pu=0.85,
        vmax_safe_pu=1.15,
        include_source_in_safe_band=True,
    )

