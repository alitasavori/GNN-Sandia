"""Thin entry points for nonunique.ipynb."""

from __future__ import annotations

from pathlib import Path

from nonunique_da_gps_daily_compare import run_da_gps_daily_compare_and_plot
from nonunique_four_scenario_demo import run_four_scenario_demo
from nonunique_opendss_daily import (
    DA_GPS_REF_LOAD_PROFILE,
    DA_GPS_REF_PV_PROFILE,
    DailySimConfig,
)
from nonunique_warmstart_band_daily import run_warmstart_band_daily
from nonunique_warmstart_compare import run_warmstart_compare


def run_and_plot(
    *,
    mode: str = "warmstart",
    step_min: int = 5,
    include_der: bool = False,
    include_da_gps: bool = True,
    show: bool = True,
    **kwargs,
):
    """Run experiment and render plots.

    Parameters
    ----------
    mode:
        ``warmstart`` (default) — two-scenario DA-GPS warm-start compare.
        ``four_scenario`` — legacy init_low/init_high/init_default/snapshot_cold demo.
        ``da_gps_daily_compare`` — DA-GPS GNN vs OpenDSS native daily QSTS truth.
        ``warmstart_band_daily`` — per-step band from N independent random controller warm-starts.
    """
    npts_override = kwargs.pop("npts", None)
    device = kwargs.pop("device", None)
    cfg = DailySimConfig(step_min=step_min, include_der=include_der, include_da_gps=include_da_gps)
    if npts_override is not None:
        cfg.day_hours = int(npts_override * step_min / 60)
    if mode == "four_scenario":
        return run_four_scenario_demo(cfg, show=show, device=device)
    if mode == "warmstart":
        return run_warmstart_compare(cfg, show=show, device=device)
    if mode == "warmstart_band_daily":
        return run_warmstart_band_daily(
            cfg,
            n_warm_starts=int(kwargs.pop("n_warm_starts", 10)),
            monitor_nodes=kwargs.pop("monitor_nodes", None),
            include_da_gps=kwargs.pop("include_da_gps", include_da_gps),
            plot_reg_cap=bool(kwargs.pop("plot_reg_cap", True)),
            plot_warmstart_lines=bool(kwargs.pop("plot_warmstart_lines", True)),
            seed=kwargs.pop("seed", 42),
            show=show,
            device=device,
        )
    if mode == "da_gps_daily_compare":
        if not include_da_gps:
            raise ValueError("da_gps_daily_compare requires include_da_gps=True")
        cfg.da_gps_load_profile = Path(
            kwargs.pop("load_profile_path", DA_GPS_REF_LOAD_PROFILE)
        )
        cfg.da_gps_pv_profile = Path(kwargs.pop("pv_profile_path", DA_GPS_REF_PV_PROFILE))
        return run_da_gps_daily_compare_and_plot(
            cfg,
            show=show,
            plot_all_cache_nodes=bool(kwargs.pop("plot_all_cache_nodes", False)),
            plot_all_max_nodes=int(kwargs.pop("plot_all_max_nodes", 0)),
            out_dir=kwargs.pop("out_dir", None),
            load_profile_path=cfg.da_gps_load_profile,
            pv_profile_path=cfg.da_gps_pv_profile,
            ref_sample_index=int(kwargs.pop("ref_sample_index", 0)),
            scenario_scale=float(kwargs.pop("scenario_scale", 1.0)),
            daily_stress=float(kwargs.pop("daily_stress", 0.0)),
            device=device,
            **kwargs,
        )
    raise ValueError(
        f"Unknown mode={mode!r}; use 'warmstart', 'four_scenario', "
        f"'da_gps_daily_compare', or 'warmstart_band_daily'"
    )


__all__ = [
    "DailySimConfig",
    "run_and_plot",
    "run_da_gps_daily_compare_and_plot",
    "run_four_scenario_demo",
    "run_warmstart_compare",
    "run_warmstart_band_daily",
]
