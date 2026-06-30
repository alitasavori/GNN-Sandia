"""DA-GPS overlay inference for nonunique experiments."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from nonunique_opendss_daily import (
    NATIVE_NPTS,
    NATIVE_STEP_MIN,
    DailySimConfig,
    da_gps_inference_grid,
    display_hours_array,
    log_da_gps_device,
    resample_daily_profile_2d,
    resolve_da_gps_device,
)


def _restore_matplotlib_inline(inline_backend: str):
    import matplotlib.pyplot as plt

    try:
        if plt.get_backend() != inline_backend:
            ip = get_ipython() if "get_ipython" in globals() else None  # type: ignore[name-defined]
            if ip is not None:
                ip.run_line_magic("matplotlib", "inline")
            else:
                plt.switch_backend(inline_backend)
    except Exception as exc:
        print(f"[WARN] could not restore matplotlib backend: {exc}")


def load_da_gps_overlay(
    cfg: DailySimConfig,
    monitor_nodes: list[str],
    *,
    device: str | None = None,
    inline_backend: str | None = None,
) -> dict[str, Any] | None:
    """Run GNN-only DA-GPS inference and resample onto the notebook grid."""
    if not cfg.include_da_gps:
        return None

    cwd_before = os.getcwd()
    try:
        repo_root = str(cfg.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        sys.modules.pop("run_da_gps_daily_opendss_compare", None)
        from run_da_gps_daily_opendss_compare import (
            align_da_gps_trajectory_to_opendss_names,
            run_da_gps_daily_voltages,
        )

        required = {
            "run_dir": (cfg.da_gps_run_dir, Path.is_dir),
            "cache_pt": (cfg.da_gps_cache_pt, Path.is_file),
            "checkpoint": (cfg.da_gps_checkpoint, Path.is_file),
            "load_profile": (cfg.da_gps_load_profile, Path.is_file),
            "pv_profile": (cfg.da_gps_pv_profile, Path.is_file),
        }
        if cfg.include_der:
            required["der_profile"] = (cfg.der_profile_csv, Path.is_file)
        missing = [f"{k}: {p}" for k, (p, ok) in required.items() if not ok(p)]
        if missing:
            raise FileNotFoundError(
                "include_da_gps=True but required paths are missing:\n  " + "\n  ".join(missing)
            )

        print("=" * 72)
        print("DA-GPS overlay: profiles aligned with OpenDSS")
        print(f"  load/PV: {cfg.da_gps_load_profile} / {cfg.da_gps_pv_profile}")
        inf_npts, inf_step_min = da_gps_inference_grid(cfg)
        print(
            f"  steps: {inf_npts} @ {inf_step_min} min "
            f"(display grid; native training grid is {NATIVE_NPTS} @ {NATIVE_STEP_MIN} min)"
        )
        print("=" * 72)

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        os.chdir(cfg.repo_root)
        dev = resolve_da_gps_device(device)
        log_da_gps_device(dev)
        t0 = time.perf_counter()
        bundle = run_da_gps_daily_voltages(
            run_dir=cfg.da_gps_run_dir,
            cache_pt=cfg.da_gps_cache_pt,
            checkpoint=cfg.da_gps_checkpoint,
            node_names=monitor_nodes,
            load_profile_path=str(cfg.da_gps_load_profile),
            pv_irradiance_profile_path=str(cfg.da_gps_pv_profile),
            der_max_kw=float(cfg.der_nominal_kw if cfg.include_der else 0.0),
            der_buses=str(cfg.der_bus if cfg.include_der else ""),
            der_profile_path=str(cfg.der_profile_csv if cfg.include_der else "") or None,
            npts=int(inf_npts),
            step_min=int(inf_step_min),
            skip_opendss=True,
            device=dev,
        )
        voltages_native = bundle["voltages"]
        reg_native = np.asarray(bundle["reg_tap_pu"], dtype=float)
        cap_native = np.asarray(bundle["cap_sigmoid"], dtype=float)
        reg_cols = list(bundle["reg_cols"])
        cap_cols = list(bundle["cap_cols"])

        da_hours = display_hours_array(cfg)
        if int(inf_npts) == int(cfg.npts) and int(inf_step_min) == int(cfg.step_min):
            voltages = voltages_native
            reg_rs = reg_native
            cap_rs = cap_native
        else:
            da_src_h = np.arange(int(inf_npts), dtype=float) * (float(inf_step_min) / 60.0)
            voltages = {
                k: np.interp(da_hours, da_src_h, np.asarray(v, dtype=float))
                for k, v in voltages_native.items()
            }
            reg_rs = resample_daily_profile_2d(
                reg_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=int(inf_npts),
                native_step_min=int(inf_step_min),
                method="nearest",
            )
            cap_rs = resample_daily_profile_2d(
                cap_native,
                npts=cfg.npts,
                step_min=cfg.step_min,
                native_npts=int(inf_npts),
                native_step_min=int(inf_step_min),
                method="nearest",
            )

        reg_names_probe = monitor_nodes  # placeholder; caller passes reg/cap names for alignment
        _ = reg_names_probe
        print(f"DA-GPS overlay inference finished in {time.perf_counter() - t0:.1f}s")
        return {
            "voltages": voltages,
            "hours": da_hours,
            "reg_native": reg_rs,
            "cap_native": cap_rs,
            "reg_cols": reg_cols,
            "cap_cols": cap_cols,
            "align_fn": align_da_gps_trajectory_to_opendss_names,
        }
    except Exception as exc:
        print(f"[WARN] DA-GPS overlay skipped ({type(exc).__name__}: {exc})")
        return None
    finally:
        try:
            os.chdir(cwd_before)
        except Exception:
            os.chdir(cfg.grid_dir)
        if inline_backend is not None:
            _restore_matplotlib_inline(inline_backend)


def align_da_gps_devices(
    overlay: dict[str, Any] | None,
    reg_names: list[str],
    cap_names: list[str],
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    np.ndarray | None,
]:
    if overlay is None:
        return None, None, None, None
    align = overlay["align_fn"]
    reg_by_name = align(reg_names, overlay["reg_cols"], overlay["reg_native"])
    cap_by_name = align(cap_names, overlay["cap_cols"], overlay["cap_native"])
    return overlay["voltages"], reg_by_name, cap_by_name, overlay["hours"]
