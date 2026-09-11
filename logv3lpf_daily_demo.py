"""Compare Log(v) 3LPF vs OpenDSS using local GNN2 DSS masters."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Callable

try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except Exception:
    pass


def ensure_logv3lpf(repo: Path, *, reload: bool = False) -> Path:
    """Make ``logv3lpf`` importable via the local Log-v-3LPF clone.

    Parameters
    ----------
    reload :
        If True, drop every ``logv3lpf*`` entry from ``sys.modules`` and re-import
        so disk edits (e.g. transformer Yprim Vbase scaling) take effect without a
        kernel restart. Notebook cells that only ``reload`` a wrapper module still
        keep a stale in-memory ``logv3lpf.utils``.
    """
    pkg_root = (Path(repo) / "Log-v-3LPF").resolve()
    pkg_dir = pkg_root / "logv3lpf"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {pkg_dir}. Clone "
            "https://github.com/krishnasandeepaxe190/Log-v-3LPF into the repo as Log-v-3LPF/"
        )
    root = str(pkg_root)
    # Prefer the local clone over any earlier sys.path entry / stale import.
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    if reload:
        for name in list(sys.modules):
            if name == "logv3lpf" or name.startswith("logv3lpf."):
                del sys.modules[name]
    import logv3lpf  # noqa: F401

    return Path(logv3lpf.__path__[0]).resolve()


def compile_cmd(dss_path: Path) -> str:
    return f'Compile "{Path(dss_path).resolve()}"'


def _local(repo: Path, *parts: str) -> Path:
    return (Path(repo).resolve().joinpath(*parts)).resolve()


# Local masters (same files used elsewhere in GNN2) - not the upstream package copies.
FEEDERS: dict[str, dict[str, Any]] = {
    "ieee34": {
        "aliases": ("34", "ieee34", "ieee-34"),
        "dss": lambda repo: _local(repo, "new dss from dr mirzaei", "IEEE34_PV.dss"),
        "sourcebus": "sourcebus",
        "refvm": 1.05,
    },
    "906": {
        # Default 906 = PV Volt-Var copy (PV906 @ 906.1, PV458 @ 458.3).
        # Stock no-PV feeder remains available as feeder="906_stock".
        "aliases": ("906", "lvtestcase", "lv", "europeanlv", "906_pv", "906_voltvar"),
        "dss": lambda repo: _local(
            repo,
            "906 bus system",
            "LVTestCase_PV_voltvar",
            "Master_snapshot_PV_voltvar.dss",
        ),
        "sourcebus": "sourcebus",
        "refvm": 1.05,
    },
    "906_stock": {
        "aliases": ("906_stock", "lvtestcase_stock", "906_nopv"),
        "dss": lambda repo: _local(
            repo,
            "906 bus system",
            "OpenDSS-master",
            "OpenDSS-master",
            "Distrib",
            "IEEETestCases",
            "LVTestCase",
            # Snapshot-only master (no yearly 1440-step DemandInterval solve).
            "Master_snapshot.dss",
        ),
        "sourcebus": "sourcebus",
        "refvm": 1.05,
    },
    "8500": {
        "aliases": ("8500", "ieee8500", "ieee-8500"),
        "dss": lambda repo: _local(
            repo, "8500 nodes with solar unbalanced", "Master-PV2MW-inv.dss"
        ),
        "sourcebus": "sourcebus",
        "refvm": 1.05,
    },
}


def resolve_feeder(feeder: str) -> str:
    key = str(feeder).strip().lower()
    for canon, meta in FEEDERS.items():
        if key == canon or key in meta["aliases"]:
            return canon
    raise ValueError(
        f"Unknown feeder {feeder!r}. Use one of: {sorted(FEEDERS)} "
        f"(aliases: 34, 906/lvtestcase/906_pv, 906_stock, 8500)."
    )


def reset_invcontrol_pv_to_pf1(pv_names: list[str] | None = None) -> int:
    """Shared fixedctrlinit start: clear residual InvControl Q (PF=1, kvar=0).

    Matches ``run_original_style_dataset_906_lvtestcase._reset_invcontrol_pv_to_compile_defaults``.
    Call before OpenDSS Static so Volt-Var settles from the same reference each sample.
    Returns number of PVSystems edited.
    """
    import opendssdirect as dss

    try:
        names = (
            [str(n) for n in pv_names]
            if pv_names is not None
            else [str(n) for n in (dss.PVsystems.AllNames() or []) if n and n != "NONE"]
        )
    except Exception:
        return 0
    n = 0
    for nm in names:
        if not nm or str(nm).upper() == "NONE":
            continue
        try:
            dss.Text.Command(f"Edit PVSystem.{nm} PF=1.0 kvar=0")
            n += 1
        except Exception:
            continue
    return n


def voltage_metrics(case) -> dict[str, float]:
    import numpy as np

    errs = []
    for bus, vm_l in case.results["logv3lpf"]["vm"].items():
        vm_o = case.results["openDSS"]["vm"].get(bus)
        if vm_o is None:
            continue
        a = np.asarray(vm_l, float)
        b = np.asarray(vm_o, float)
        n = min(len(a), len(b))
        if n:
            errs.append(np.abs(a[:n] - b[:n]))
    if not errs:
        return {"mae": float("nan"), "rmse": float("nan"), "max": float("nan"), "n": 0}
    e = np.concatenate(errs)
    return {
        "mae": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "max": float(np.max(e)),
        "n": int(e.size),
    }


def _angle_diff_deg(a_deg, b_deg):
    """Smallest signed angle difference in degrees, in (-180, 180]."""
    import numpy as np

    d = (np.asarray(a_deg, float) - np.asarray(b_deg, float) + 180.0) % 360.0 - 180.0
    return d


def paper_accuracy_metrics(case, *, exclude_source: bool = False) -> dict[str, float]:
    """Paper Table IV-style accuracy: |V| RMSE/MAPE and angle RMSE over bus-phases.

    ``va_rmse_deg`` is the primary angle metric (Log(v) angles are source-anchored
    to OpenDSS in ``process_logv3lpf_solution``). ``va_rmse_deg_global_aligned``
    remains a diagnostic only.
    """
    import numpy as np

    vm_l = case.results.get("logv3lpf", {}).get("vm", {})
    vm_o = case.results.get("openDSS", {}).get("vm", {})
    va_l = case.results.get("logv3lpf", {}).get("va", {})
    va_o = case.results.get("openDSS", {}).get("va", {})
    src = str(getattr(case, "sourcebus", "sourcebus")).lower()
    src_offset = case.results.get("logv3lpf", {}).get("va_source_offset_deg")

    dv, mape_terms, dang = [], [], []
    for bus, a in vm_l.items():
        if exclude_source and str(bus).split(".")[0].lower() == src:
            continue
        b = vm_o.get(bus)
        if b is None:
            key = next((k for k in vm_o if str(k).lower() == str(bus).lower()), None)
            if key is None:
                continue
            b = vm_o[key]
            bus_o = key
        else:
            bus_o = bus
        aa = np.asarray(a, float)
        bb = np.asarray(b, float)
        n = min(len(aa), len(bb))
        if n <= 0:
            continue
        dv.append(np.abs(aa[:n] - bb[:n]))
        denom = np.maximum(np.abs(bb[:n]), 1e-12)
        mape_terms.append(np.abs(aa[:n] - bb[:n]) / denom)

        al = va_l.get(bus, va_l.get(bus_o))
        ao = va_o.get(bus_o, va_o.get(bus))
        if al is None or ao is None:
            continue
        al = np.asarray(al, float)
        ao = np.asarray(ao, float)
        m = min(n, len(al), len(ao))
        if m > 0:
            dang.append(_angle_diff_deg(al[:m], ao[:m]))

    if not dv:
        return {
            "n_phase_nodes": 0,
            "vm_rmse_pu": float("nan"),
            "vm_mape_pct": float("nan"),
            "vm_mae_pu": float("nan"),
            "va_rmse_deg": float("nan"),
            "va_mae_deg": float("nan"),
            "va_rmse_deg_global_aligned": float("nan"),
            "va_source_offset_deg": float("nan"),
        }
    e = np.concatenate(dv)
    mp = np.concatenate(mape_terms)
    out = {
        "n_phase_nodes": int(e.size),
        "vm_rmse_pu": float(np.sqrt(np.mean(e**2))),
        "vm_mape_pct": float(100.0 * np.mean(mp)),
        "vm_mae_pu": float(np.mean(e)),
        "va_rmse_deg": float("nan"),
        "va_mae_deg": float("nan"),
        "va_rmse_deg_global_aligned": float("nan"),
        "va_source_offset_deg": float(src_offset) if src_offset is not None else float("nan"),
        "vsource_angle_deg": float(
            case.results.get("logv3lpf", {}).get(
                "vsource_angle_deg", getattr(case, "vsource_angle_deg", float("nan"))
            )
        ),
    }
    if dang:
        de = np.concatenate(dang)
        out["va_mae_deg"] = float(np.mean(np.abs(de)))
        out["va_rmse_deg"] = float(np.sqrt(np.mean(de**2)))
        bias = float(np.mean(de))
        out["va_rmse_deg_global_aligned"] = float(np.sqrt(np.mean((de - bias) ** 2)))
    return out


def paper_flop_estimates(case) -> dict[str, float]:
    """Paper Table II-style inverse-update FLOP estimates (not wall-clock).

    Matches ``linpf.check_logv3lpf_performance``:
      system size for cubic cost is ``2 * Nn`` (log|V| and angle),
      while reported ``n_bus_phases`` is ``Nn``.
    Regulator phases use ``regulator_names`` (includes Mirzaei regs without RegControl).
    Also reports ``n_excl_source`` (paper 8500 n=8531 ≈ Nn-3).
    """
    import numpy as np

    loads = case.loads
    k_load = 0
    k_cap = 0
    k_reg = 0
    if loads is not None and len(loads):
        non_p = loads[loads.model != 1]
        if len(non_p) and len(non_p.phases):
            k_load = int(len(np.hstack(non_p.phases.to_numpy())))
    caps = case.capacitors
    if caps is not None and len(caps) and len(caps.phases):
        k_cap = int(len(np.hstack(caps.phases.to_numpy())))

    reg_names = list(getattr(case, "regulator_names", None) or [])
    if not reg_names:
        regs = getattr(case, "regcontrols", None)
        if regs is not None and len(regs):
            if "transformer" in regs.columns:
                reg_names = [str(x) for x in regs.transformer.values]
            else:
                # Upstream bug: looking up transformer by RegControl *name* finds nothing.
                reg_names = [str(x) for x in regs.name.values]
    for name in reg_names:
        hit = case.transformers[case.transformers.name == name]
        for phases in hit.phases:
            el = np.asarray(phases)
            # Package / Table II: all winding phase slots on regulator transformers.
            k_reg += int(np.sum(el != 0))
    k_reg_primary = 0
    for name in reg_names:
        hit = case.transformers[case.transformers.name == name]
        for phases in hit.phases:
            el = np.asarray(phases[0] if len(phases) else phases)
            k_reg_primary += int(np.sum(np.asarray(el) != 0))
            break

    k_phases = k_load + k_cap + k_reg
    k = k_phases * 2
    # Actual Woodbury U width (non-const-P loads + caps only; regs live in base A).
    k_woodbury = (k_load + k_cap) * 2

    n_bus_phases = int(case.Nn)
    src = getattr(case, "sourcebus", None)
    n_src = 0
    if src is not None and hasattr(case, "bus_phases"):
        for b, ph in case.bus_phases.items():
            if str(b).lower() == str(src).lower():
                n_src = len(ph)
                break
    n_excl_source = n_bus_phases - n_src
    # Paper Table II cubic uses n = bus-phases (then 2n for log|V|+angle in package).
    n_sys = n_bus_phases * 2
    n_sys_excl = n_excl_source * 2
    o_opendss = int((2.0 / 3.0) * (n_sys**3) + (2 * n_sys) ** 2)
    o_opendss_excl = int((2.0 / 3.0) * (n_sys_excl**3) + (2 * n_sys_excl) ** 2)
    if k == 0:
        o_logv_update = 0
        o_logv_apply = int(n_sys**2)
    else:
        o_logv_update = int(3 * (k**3) + (2 * n_sys * (k**2)) + n_sys**2 + k)
        o_logv_apply = o_logv_update
    ratio = float("nan")
    if o_logv_update > 0:
        ratio = float(o_opendss) / float(o_logv_update)
    return {
        "n_bus_phases": float(n_bus_phases),
        "n_excl_source": float(n_excl_source),
        "n_source_phases": float(n_src),
        "n_sys": float(n_sys),
        "k": float(k),
        "k_load_phases": float(k_load),
        "k_cap_phases": float(k_cap),
        "k_reg_phases": float(k_reg),
        "k_reg_primary_phases": float(k_reg_primary),
        "k_woodbury_U": float(k_woodbury),
        "n_regulators": float(len(reg_names)),
        "opendss_lu_flops": float(o_opendss),
        "opendss_lu_flops_excl_source": float(o_opendss_excl),
        "logv_woodbury_update_flops": float(o_logv_update),
        "logv_apply_flops_lower_bound": float(o_logv_apply),
        "flop_ratio_opendss_over_logv_update": ratio,
    }


PAPER_TABLE_IV = {
    "ieee34": {"vm_rmse_pu": 0.03, "vm_mape_pct": 2.46, "va_rmse_deg": 2.10},
    "906": {"vm_rmse_pu": 0.00, "vm_mape_pct": 0.09, "va_rmse_deg": 0.15},
    "8500": {"vm_rmse_pu": 0.02, "vm_mape_pct": 2.45, "va_rmse_deg": 3.44},
}
PAPER_TABLE_II = {
    "ieee34": {"n": 95, "k": 114, "opendss_flops": 4.72e6, "logv_flops": 9.42e6, "ratio": 0.50},
    "906": {"n": 2721, "k": 0, "opendss_flops": 1.08e11, "logv_flops": 0.0, "ratio": float("nan")},
    "8500": {"n": 8531, "k": 72, "opendss_flops": 3.31e12, "logv_flops": 4.69e8, "ratio": 7060.86},
}


def plot_voltage_compare(case, feeder: str, out_png: Path | None = None, *, save: bool = False):
    """Always plot OpenDSS vs Log(v) 3LPF |V| inline; optional save."""
    import matplotlib.pyplot as plt
    import numpy as np
    from logv3lpf.plotting import plot_results

    n_bus = len(case.bus_phases)
    metrics = voltage_metrics(case)

    if n_bus <= 120:
        plot_results(case, "openDSS", "logv3lpf", "vm")
        fig = plt.gcf()
        fig.suptitle(f"OpenDSS vs Log(v) 3LPF |V| - {feeder}", y=1.02)
        if save and out_png is not None:
            fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.show()
        return metrics

    od = case.results["openDSS"]["vm"]
    lv = case.results["logv3lpf"]["vm"]
    xo, xl = [], []
    for bus in lv:
        if bus not in od:
            continue
        a = np.asarray(lv[bus], float)
        b = np.asarray(od[bus], float)
        n = min(len(a), len(b))
        for i in range(n):
            xl.append(a[i])
            xo.append(b[i])
    xo = np.asarray(xo)
    xl = np.asarray(xl)
    err = np.abs(xl - xo)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(xo, xl, s=8, alpha=0.35, label="phase nodes")
    lims = [float(min(xo.min(), xl.min())), float(max(xo.max(), xl.max()))]
    axes[0].plot(lims, lims, "k--", lw=1, label="y = x")
    axes[0].set_xlabel("OpenDSS |V| (pu)")
    axes[0].set_ylabel("Log(v) 3LPF |V| (pu)")
    axes[0].set_title(f"{feeder}: OpenDSS vs Log(v)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    order = np.argsort(err)[::-1]
    axes[1].plot(err[order], lw=0.8)
    axes[1].set_xlabel("phase-node (sorted by |OpenDSS − Log(v)|)")
    axes[1].set_ylabel("|dV| (pu)")
    axes[1].set_title(
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  max={metrics['max']:.4f}"
    )
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"OpenDSS vs Log(v) 3LPF - {feeder} ({n_bus} buses)", fontsize=12
    )
    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()
    return metrics


def plot_signed_residual_vs_opendss(
    case,
    feeder: str,
    out_png: Path | None = None,
    *,
    save: bool = False,
    ve: dict[str, Any] | None = None,
) -> None:
    """Plot signed |V| residual vs OpenDSS voltage (inline).

    For each bus-phase node i:
        e_i = |V_i|_Log(v) - |V_i|_OpenDSS
    plotted against |V_i|_OpenDSS.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ve is None:
        ve = collect_voltage_errors(case)
    vod = np.asarray(ve.get("opendss_vm", []), float)
    signed = np.asarray(ve.get("signed", []), float)
    if vod.size == 0 or signed.size == 0 or vod.size != signed.size:
        return

    bias = float(ve.get("signed_mean", np.mean(signed)))
    mae = float(np.mean(np.abs(signed)))
    rmse = float(np.sqrt(np.mean(signed**2)))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(vod, signed, s=10, alpha=0.35, c="C0", edgecolors="none", label="bus-phase nodes")
    ax.axhline(0.0, color="k", lw=1.0, ls="--", label="e = 0")
    ax.axhline(bias, color="C3", lw=1.2, ls=":", label=f"mean bias = {bias:+.4f} pu")
    ax.set_xlabel(r"OpenDSS $|V|$ (pu)")
    ax.set_ylabel(r"$e = |V|_{\mathrm{Log(v)}} - |V|_{\mathrm{OpenDSS}}$ (pu)")
    ax.set_title(
        f"{feeder}: signed |V| residual vs OpenDSS\n"
        f"MAE={mae:.4f} pu   RMSE={rmse:.4f} pu   n={vod.size}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()


def _opendss_reg_taps() -> dict[str, float]:
    import opendssdirect as dss

    out = {}
    try:
        names = [n for n in dss.RegControls.AllNames() if n and n != "NONE"]
    except Exception:
        return out
    for name in names:
        try:
            dss.RegControls.Name(name)
            out[str(name)] = 1.0 + float(dss.RegControls.TapNumber()) * 0.00625
        except Exception:
            pass
    return out


def _opendss_control_inventory() -> dict[str, int]:
    import opendssdirect as dss

    def _n(call):
        try:
            names = [x for x in call() if x and x != "NONE"]
            return len(names)
        except Exception:
            return 0

    n_inv = 0
    try:
        if hasattr(dss, "InvControls"):
            n_inv = _n(dss.InvControls.AllNames)
    except Exception:
        n_inv = 0

    return {
        "RegControls": _n(dss.RegControls.AllNames),
        "CapControls": _n(dss.CapControls.AllNames),
        "InvControls": n_inv,
        "Capacitors": _n(dss.Capacitors.AllNames),
        "PVSystems": _n(dss.PVsystems.AllNames),
    }


def collect_voltage_errors(case) -> dict[str, Any]:
    """Per-node |V| OpenDSS vs Log(v) for audits."""
    import numpy as np

    rows = []
    vm_l = case.results.get("logv3lpf", {}).get("vm", {})
    vm_o = case.results.get("openDSS", {}).get("vm", {})
    for bus, a in vm_l.items():
        b = vm_o.get(bus)
        if b is None:
            bl = str(bus).lower()
            bkey = next((k for k in vm_o if str(k).lower() == bl), None)
            if bkey is None:
                continue
            b = vm_o[bkey]
        aa = np.asarray(a, float)
        bb = np.asarray(b, float)
        n = min(len(aa), len(bb))
        phases = list(case.bus_phases.get(bus, list(range(1, n + 1))))
        for i in range(n):
            ph = int(phases[i]) if i < len(phases) else i + 1
            rows.append(
                {
                    "bus": str(bus),
                    "phase": ph,
                    "node": f"{str(bus).lower()}.{ph}",
                    "opendss_vm": float(bb[i]),
                    "logv_vm": float(aa[i]),
                    "abs_err": float(abs(aa[i] - bb[i])),
                    "signed_err": float(aa[i] - bb[i]),
                }
            )
    if not rows:
        return {
            "n": 0,
            "err": np.array([]),
            "signed": np.array([]),
            "opendss_vm": np.array([]),
            "logv_vm": np.array([]),
            "worst": [],
            "vmin_od": float("nan"),
            "vmax_od": float("nan"),
            "vmin_lv": float("nan"),
            "vmax_lv": float("nan"),
            "frac_outside_od": float("nan"),
            "frac_outside_lv": float("nan"),
            "signed_mean": float("nan"),
            "signed_std": float("nan"),
            "mae_bias_corrected": float("nan"),
            "near_constant_offset": False,
        }
    err = np.array([r["abs_err"] for r in rows], float)
    signed = np.array([r["signed_err"] for r in rows], float)
    vod = np.array([r["opendss_vm"] for r in rows], float)
    vlv = np.array([r["logv_vm"] for r in rows], float)
    order = np.argsort(err)[::-1]
    worst = [rows[int(i)] for i in order[:12]]
    signed_mean = float(np.mean(signed))
    signed_std = float(np.std(signed))
    mae = float(np.mean(err))
    # After removing global bias, remaining MAE shows true node-to-node mismatch.
    mae_bc = float(np.mean(np.abs(signed - signed_mean)))
    near_const = bool(
        mae > 1e-4 and signed_std < 0.25 * abs(signed_mean) and signed_std < 0.005
    )
    return {
        "n": len(rows),
        "err": err,
        "signed": signed,
        "opendss_vm": vod,
        "logv_vm": vlv,
        "worst": worst,
        "vmin_od": float(np.min(vod)),
        "vmax_od": float(np.max(vod)),
        "vmin_lv": float(np.min(vlv)),
        "vmax_lv": float(np.max(vlv)),
        "frac_outside_od": float(np.mean((vod < 0.95) | (vod > 1.05))),
        "frac_outside_lv": float(np.mean((vlv < 0.95) | (vlv > 1.05))),
        "signed_mean": signed_mean,
        "signed_std": signed_std,
        "mae_bias_corrected": mae_bc,
        "near_constant_offset": near_const,
        "corr_od_lv": float(np.corrcoef(vod, vlv)[0, 1]) if len(vod) > 1 else float("nan"),
    }


def transformer_drop_audit(case) -> list[dict[str, Any]]:
    """Compare OpenDSS vs Log(v) |V| drop across each 2-winding transformer."""
    import numpy as np

    out = []
    vm_o = case.results.get("openDSS", {}).get("vm", {})
    vm_l = case.results.get("logv3lpf", {}).get("vm", {})
    if case.transformers is None:
        return out

    def _mean_vm(vm, bus):
        key = bus if bus in vm else next((k for k in vm if str(k).lower() == str(bus).lower()), None)
        if key is None:
            return float("nan")
        return float(np.mean(np.asarray(vm[key], float)))

    for i in range(len(case.transformers)):
        buses = list(case.transformers.buses[i])
        if len(buses) < 2:
            continue
        b0, b1 = buses[0], buses[1]
        od0, od1 = _mean_vm(vm_o, b0), _mean_vm(vm_o, b1)
        lv0, lv1 = _mean_vm(vm_l, b0), _mean_vm(vm_l, b1)
        conns = list(case.transformers.Conn[i]) if "Conn" in case.transformers.columns else []
        out.append(
            {
                "name": str(case.transformers.name[i]),
                "buses": f"{b0}->{b1}",
                "conn": conns,
                "od_drop": od0 - od1,
                "lv_drop": lv0 - lv1,
                "od_from": od0,
                "od_to": od1,
                "lv_from": lv0,
                "lv_to": lv1,
            }
        )
    return out



def print_audit_report(report: dict[str, Any]) -> None:
    """Human-readable trust checklist (what ran, what differed)."""
    sep = "=" * 64
    print(sep)
    print("AUDIT - what this cell actually ran")
    print(sep)
    print(f"  feeder          : {report.get('feeder')}")
    print(f"  DSS             : {report.get('dss')}")
    print(f"  control_mode    : {report.get('control_mode')}")
    print()
    print("  WHAT Log(v) IS")
    print("    linearized power-flow (rank-k / Woodbury linear algebra)")
    print("    NOT a linear program / optimizer")
    print("    paper: v ~ Delta*(1+u+j*theta_tilde); Ytilde=Delta*Y*.*Delta^H; ytilde=Ytilde*1")
    print("    controller limits are NOT encoded as LP constraints")
    notes = report.get("model_notes") or []
    if notes:
        print("  MODEL DISCLOSURES")
        for line in notes:
            print(f"    - {line}")
    print()
    print("  WHAT OpenDSS DID")
    print(f"    ControlMode     : {report.get('opendss_control_mode')}")
    print(f"    Solution converged: {report.get('opendss_converged')}")
    inv = report.get("opendss_inventory") or {}
    print(
        f"    inventory       : RegControls={inv.get('RegControls', 0)}  "
        f"CapControls={inv.get('CapControls', 0)}  "
        f"InvControls={inv.get('InvControls', 0)}  "
        f"PV={inv.get('PVSystems', 0)}"
    )
    print()
    print("  WHAT Log(v) DID")
    cm = report.get("control_mode")
    if cm == "synced":
        print("    mode: synced - OD taps baked into Log(v) A; one PF (oracle states)")
        print(f"    taps applied from OpenDSS: {report.get('reg_iters', 0)}")
    elif cm == "static":
        print("    mode: static - Log(v) RegControl heuristic (states may differ from OD)")
        print(f"    RegControl iterations: {report.get('reg_iters', 0)}")
    else:
        print("    mode: off - frozen taps; pure PF/model compare")
        print(f"    RegControl iterations: {report.get('reg_iters', 0)}")
    print(f"    RegControls in model : {report.get('n_reg', 0)}")
    print(f"    CapControl / InvControl in Log(v): no (not implemented)")
    print()
    print("  |V| COMPARE (OpenDSS vs Log(v))")
    print("    Residual mixes linearization + transformer/shunt/source/load model diffs")
    print("    (not 'linearization only' unless control_mode=off and models match).")
    m = report.get("metrics") or {}
    print(
        f"    MAE={m.get('mae')}  RMSE={m.get('rmse')}  "
        f"max={m.get('max')}  n_nodes={m.get('n')}"
    )
    ve = report.get("voltage_errors") or {}
    print(
        f"    OpenDSS |V| range : [{ve.get('vmin_od')}, {ve.get('vmax_od')}] pu"
    )
    print(
        f"    Log(v)  |V| range : [{ve.get('vmin_lv')}, {ve.get('vmax_lv')}] pu"
    )
    print(
        f"    frac outside [0.95,1.05]: OpenDSS={ve.get('frac_outside_od')}  "
        f"Log(v)={ve.get('frac_outside_lv')}"
    )
    print(
        f"    signed bias (Log(v)-OpenDSS): mean={ve.get('signed_mean')}  "
        f"std={ve.get('signed_std')}"
    )
    print(
        f"    MAE after removing bias     : {ve.get('mae_bias_corrected')}  "
        f"(node-to-node shape error)"
    )
    print(f"    corr(OpenDSS, Log(v))       : {ve.get('corr_od_lv')}")
    if ve.get("near_constant_offset"):
        print()
        print("  *** WARNING: |dV| is nearly CONSTANT across nodes ***")
        print("      That is NOT normal linearization error (which varies by bus).")
        print("      Likely a systematic bias (often the HV/LV substation transformer).")
        print("      Check TRANSFORMER DROPS below.")
    print()
    xf = report.get("transformer_drops") or []
    if xf:
        print("  TRANSFORMER |V| DROPS (mean across phases)")
        print(
            f"    {'name':<12} {'buses':<22} {'OD_drop':>10} {'LV_drop':>10} {'conn'}"
        )
        for row in xf:
            print(
                f"    {row['name']:<12} {row['buses']:<22} "
                f"{row['od_drop']:10.5f} {row['lv_drop']:10.5f} {row.get('conn')}"
            )
    print()
    taps = report.get("taps") or {}
    if taps.get("rows"):
        print("  REGULATOR TAPS (start -> end)")
        print(
            f"    {'name':<28} {'OD_start':>9} {'OD_end':>9} "
            f"{'LV_start':>9} {'LV_end':>9} {'|d|':>8}"
        )
        for row in taps["rows"]:
            print(
                f"    {row['name']:<28} {row['od_start']:9.5f} {row['od_end']:9.5f} "
                f"{row['lv_start']:9.5f} {row['lv_end']:9.5f} {row['abs_diff']:8.5f}"
            )
        print(f"    tap MAE (final OD vs LV) = {taps.get('mae')}")
    else:
        print("  REGULATOR TAPS")
        print("    none (no RegControl objects in this DSS - tap autonomy compare N/A)")
    print()
    lm = report.get("load_models") or {}
    print("  LOAD / ZIP MODEL ERRORS (if applicable)")
    if not lm.get("counts_dss") and not lm.get("counts_labeled"):
        print("    no loads parsed")
    else:
        print(f"    DSS counts : {lm.get('counts_labeled') or lm.get('counts_dss')}")
        print(f"    remapped -> constP: {lm.get('n_remapped')}")
        if lm.get("note"):
            print(f"    note: {lm['note']}")
        mg = lm.get("mae_by_group") or {}
        if lm.get("n_remapped"):
            print(
                f"    |V| MAE native P/Z/I buses : {mg.get('native_PZI_load_buses')}"
            )
            print(
                f"    |V| MAE remapped->P buses  : {mg.get('remapped_load_buses')}"
            )
        mm = lm.get("mae_by_dss_model") or {}
        if mm and (lm.get("has_mixed_pzi") or lm.get("has_zip_or_unsupported")):
            print("    |V| MAE by DSS model buses:")
            for k, v in mm.items():
                print(f"      {k:<20} {v}")
        if not lm.get("applicable"):
            if int(lm.get("n_remapped") or 0) == 0:
                print(
                    "    (DSS models already in Log(v)-supported {1,2,5} - "
                    "no unsupported-model remap)"
                )
            else:
                print(
                    "    (ZIP-combined / unsupported models remapped → const-P; "
                    "see remapped counts above - do not call this 'no ZIP error')"
                )
    print()

    ctrl = report.get("controllers") or {}
    print("  CONTROLLER / DER STATE ERRORS (only what exists on this feeder)")
    reg = ctrl.get("regcontrols") or {}
    if reg.get("applicable"):
        print(
            f"    RegControls: n={reg.get('n')}  tap MAE={reg.get('mae_tap')}  "
            f"({reg.get('note', '')})"
        )
        print(
            f"      {'name':<24} {'OD_tap':>9} {'LV_tap':>9} {'|err|':>9}"
        )
        for r in (reg.get("rows") or [])[:12]:
            print(
                f"      {r['name']:<24} {r['opendss_tap']:9.5f} "
                f"{r['logv_tap']:9.5f} {r['abs_err']:9.5f}"
            )
    else:
        print("    RegControls: N/A (none on this DSS)")

    cap = ctrl.get("capcontrols") or {}
    if cap.get("applicable"):
        print(
            f"    Capacitors: n={cap.get('n')}  CapControls={cap.get('n_capcontrols')}  "
            f"on/off disagree={cap.get('n_disagree_onoff')}  ({cap.get('note', '')})"
        )
        print(
            f"      {'name':<20} {'OD_on':>6} {'LV_on':>6} {'OD_kvar':>10} {'LV_kvar':>10}"
        )
        for r in (cap.get("rows") or [])[:12]:
            print(
                f"      {r['name']:<20} {r['opendss_on']:6.0f} {r['logv_on']:6.0f} "
                f"{r['opendss_kvar']:10.2f} {r['logv_kvar']:10.2f}"
            )
    else:
        print("    Capacitors/CapControls: N/A")

    inv = ctrl.get("inv_pv") or {}
    if inv.get("applicable"):
        print(
            f"    PV/Inv: n={inv.get('n')}  MAE_P={inv.get('mae_kw')} kW  "
            f"MAE_Q={inv.get('mae_kvar')} kvar  ({inv.get('note', '')})"
        )
        print(
            f"      {'name':<16} {'OD_kW':>10} {'LV_kW':>10} {'OD_kvar':>10} {'LV_kvar':>10}"
        )
        for r in inv.get("rows") or []:
            print(
                f"      {r['name']:<16} {r['opendss_kw']:10.3f} {r['logv_kw']:10.3f} "
                f"{r['opendss_kvar']:10.3f} {r['logv_kvar']:10.3f}"
            )
    else:
        print("    PV/InvControl: N/A")
    print()
    worst = ve.get("worst") or []
    if worst:
        print("  WORST |V| NODES (OpenDSS vs Log(v))")
        print(f"    {'node':<22} {'OpenDSS':>9} {'Log(v)':>9} {'|err|':>9}")
        for r in worst[:10]:
            print(
                f"    {r['node']:<22} {r['opendss_vm']:9.5f} "
                f"{r['logv_vm']:9.5f} {r['abs_err']:9.5f}"
            )
    print(sep)


def plot_audit_diagnostics(
    case,
    report: dict[str, Any],
    *,
    out_png: Path | None = None,
    save: bool = False,
) -> None:
    """Verification plots: |dV|, signed bias, bias-corrected error, |V| ranges."""
    import matplotlib.pyplot as plt
    import numpy as np

    ve = report.get("voltage_errors") or {}
    err = np.asarray(ve.get("err", []), float)
    signed = np.asarray(ve.get("signed", []), float)
    if err.size == 0:
        return
    feeder = report.get("feeder", "")
    bias = float(ve.get("signed_mean", np.mean(signed))) if signed.size else 0.0
    err_bc = np.abs(signed - bias) if signed.size else err

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(err, bins=min(40, max(10, err.size // 20)), color="C0", edgecolor="white")
    axes[0, 0].axvline(float(np.mean(err)), color="C3", ls="--", label=f"MAE={np.mean(err):.4f}")
    axes[0, 0].set_xlabel("|OpenDSS - Log(v)| (pu)")
    axes[0, 0].set_ylabel("count")
    title0 = f"{feeder}: |dV| histogram"
    if ve.get("near_constant_offset"):
        title0 += "  [NEAR-CONSTANT OFFSET]"
    axes[0, 0].set_title(title0)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    if signed.size:
        axes[0, 1].hist(
            signed, bins=min(40, max(10, signed.size // 20)), color="C2", edgecolor="white"
        )
        axes[0, 1].axvline(bias, color="C3", ls="--", label=f"bias={bias:.4f}")
        axes[0, 1].set_xlabel("Log(v) - OpenDSS (pu)")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].set_title(
            f"{feeder}: signed error  (std={float(np.std(signed)):.5f})"
        )
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].axis("off")

    axes[1, 0].hist(
        err_bc, bins=min(40, max(10, err_bc.size // 20)), color="C1", edgecolor="white"
    )
    axes[1, 0].axvline(
        float(np.mean(err_bc)), color="C3", ls="--", label=f"MAE_bc={np.mean(err_bc):.5f}"
    )
    axes[1, 0].set_xlabel("|dV - bias| (pu)")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title(f"{feeder}: bias-corrected |dV| (true node variation)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    vod = np.asarray(ve.get("opendss_vm", []), float)
    vlv = np.asarray(ve.get("logv_vm", []), float)
    axes[1, 1].scatter(vod, vlv, s=8, alpha=0.25, label="nodes")
    lims = [float(min(vod.min(), vlv.min())), float(max(vod.max(), vlv.max()))]
    axes[1, 1].plot(lims, lims, "k--", lw=1, label="y=x")
    if abs(bias) > 1e-6:
        axes[1, 1].plot(
            lims, [x + bias for x in lims], "C3:", lw=1.2, label=f"y=x+bias ({bias:.4f})"
        )
    axes[1, 1].set_xlabel("OpenDSS |V| (pu)")
    axes[1, 1].set_ylabel("Log(v) |V| (pu)")
    axes[1, 1].set_title(f"{feeder}: OpenDSS vs Log(v)  (corr={ve.get('corr_od_lv', float('nan')):.4f})")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Verification - {feeder} [{report.get('control_mode')}]  "
        "(linear PF compare, not an optimizer)",
        fontsize=11,
    )
    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()


def plot_reg_compare(
    case,
    feeder: str,
    out_png: Path | None = None,
    *,
    save: bool = False,
    title_suffix: str = "",
):
    """OpenDSS vs Log(v) regulator taps (skip if no pair)."""
    import matplotlib.pyplot as plt
    import numpy as np

    reg = getattr(case, "regcontrols", None)
    if reg is None or len(reg) == 0:
        return

    od_tap = case.results.get("openDSS", {}).get("regtap")
    lv_tap = case.results.get("logv3lpf", {}).get("regtap")
    if not isinstance(od_tap, dict) or not od_tap:
        return

    names = [str(n) for n in reg.name]
    if isinstance(lv_tap, dict) and lv_tap:
        tap_lv = np.array([float(lv_tap.get(n, np.nan)) for n in names], dtype=float)
    elif "tap" in reg.columns:
        tap_lv = np.asarray(reg.tap, float)
    else:
        tap_lv = np.full(len(names), np.nan)

    tap_od = np.array([float(od_tap.get(n, np.nan)) for n in names], dtype=float)
    if not np.isfinite(tap_od).any() or not np.isfinite(tap_lv).any():
        return

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(names) + 2), 4))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, tap_od, width=w, label="OpenDSS (Static)")
    ax.bar(x + w / 2, tap_lv, width=w, label="Log(v) RegControl")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("tap (pu)")
    ttl = f"{feeder}: autonomous regulator taps - OpenDSS vs Log(v)"
    if title_suffix:
        ttl = f"{ttl} ({title_suffix})"
    ax.set_title(ttl)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()


def plot_reg_cap(case, feeder: str, out_png: Path | None = None, *, save: bool = False):
    return plot_reg_compare(case, feeder, out_png=out_png, save=save)


def _quiet():
    """Silence package chatter during parse/solve."""
    import contextlib
    import io

    return contextlib.redirect_stdout(io.StringIO())


def _reset_reg_taps_opendss() -> None:
    import opendssdirect as dss

    try:
        names = [n for n in dss.RegControls.AllNames() if n and n != "NONE"]
    except Exception:
        return
    for name in names:
        try:
            dss.RegControls.Name(name)
            dss.RegControls.TapNumber(0)
        except Exception:
            try:
                xf = dss.RegControls.Transformer()
                dss.Transformers.Name(xf)
                dss.Transformers.Wdg(2)
                dss.Transformers.Tap(1.0)
            except Exception:
                pass


def _push_logv_regtaps_to_opendss(case) -> int:
    """Write Log(v) regulator transformer taps into live OpenDSS (YPrim source).

    After OD Static, DSS still holds settled taps. Log(v) autonomous RegControl
    rebuilds Y from OpenDSS YPrim — if DSS taps are left at the Static solution,
    the first rebuild collapses voltages and every tap runs to max (1.10).
    Call this with case taps (usually 1.0) before ``run_logv_autonomous``.
    """
    import opendssdirect as dss

    if getattr(case, "regcontrols", None) is None or len(case.regcontrols) == 0:
        return 0
    n = 0
    for i in range(len(case.regcontrols)):
        xfmr = (
            case.regcontrols.transformer[i]
            if "transformer" in case.regcontrols.columns
            else case.regcontrols.name[i]
        )
        hit = case.transformers.index[
            case.transformers.name.astype(str).str.lower() == str(xfmr).lower()
        ]
        if len(hit) == 0:
            continue
        j = int(hit[0])
        taps = list(case.transformers.taps[j])
        try:
            dss.Transformers.Name(str(case.transformers.name[j]))
        except Exception:
            continue
        for wdg, tap in enumerate(taps, start=1):
            try:
                dss.Transformers.Wdg(int(wdg))
                dss.Transformers.Tap(float(tap))
                n += 1
            except Exception:
                continue
        if "tap" in case.regcontrols.columns:
            try:
                dss.RegControls.Name(str(case.regcontrols.name[i]))
                # Keep RegControl tap number consistent with winding tap.
                if "winding" in case.regcontrols.columns:
                    wdg_i = max(0, min(len(taps) - 1, int(case.regcontrols.winding[i]) - 1))
                else:
                    wdg_i = 1 if len(taps) > 1 else 0
                tn = int(round((float(taps[wdg_i]) - 1.0) / 0.00625))
                dss.RegControls.TapNumber(tn)
            except Exception:
                pass
    return n


def _reset_reg_taps_logv(case) -> None:
    import logv3lpf.linpf as linpf

    if getattr(case, "regcontrols", None) is None or len(case.regcontrols) == 0:
        return
    changed = False
    for i in range(len(case.regcontrols)):
        xfmr = (
            case.regcontrols.transformer[i]
            if "transformer" in case.regcontrols.columns
            else case.regcontrols.name[i]
        )
        hit = case.transformers.index[
            case.transformers.name.astype(str).str.lower() == str(xfmr).lower()
        ]
        if len(hit) == 0:
            continue
        j = int(hit[0])
        taps = list(case.transformers.taps[j])
        if len(taps) >= 2:
            taps[1] = 1.0
        elif taps:
            taps[0] = 1.0
        case.transformers.at[j, "taps"] = taps
        if "tap" in case.regcontrols.columns:
            case.regcontrols.at[i, "tap"] = 1.0
        changed = True
    if changed:
        linpf.get_transformer_matrices(case)
        linpf.calculate_base_matrices(case)


def _store_logv_regtaps(case) -> None:
    """Mirror OpenDSS regtap dict from Log(v) transformer taps after autonomous settle.

    Prefer live transformer.taps (source of truth for the PF). The regcontrols.tap
    column is a cache that can be stale/NaN; never leave LV as nan when the xfmr exists.
    """
    import math

    taps = {}
    if not hasattr(case, "results") or case.results is None:
        case.results = {}
    if getattr(case, "regcontrols", None) is None:
        case.results.setdefault("logv3lpf", {})["regtap"] = taps
        return
    for i in range(len(case.regcontrols)):
        name = str(case.regcontrols.name[i])
        xfmr = (
            case.regcontrols.transformer[i]
            if "transformer" in case.regcontrols.columns
            else name
        )
        hit = case.transformers.index[
            case.transformers.name.astype(str).str.lower() == str(xfmr).lower()
        ]
        if len(hit) == 0:
            hit = case.transformers.index[
                case.transformers.name.astype(str).str.lower() == name.lower()
            ]
        tap_val = float("nan")
        if len(hit) > 0:
            tt = list(case.transformers.taps[int(hit[0])])
            if "winding" in case.regcontrols.columns:
                wdg_i = max(0, min(len(tt) - 1, int(case.regcontrols.winding[i]) - 1))
            else:
                wdg_i = 1 if len(tt) > 1 else 0
            tap_val = float(tt[wdg_i])
        elif "tap" in case.regcontrols.columns:
            try:
                tap_val = float(case.regcontrols.tap[i])
            except Exception:
                tap_val = float("nan")
        if math.isfinite(tap_val):
            taps[name] = tap_val
            if "tap" in case.regcontrols.columns:
                try:
                    case.regcontrols.at[i, "tap"] = tap_val
                except Exception:
                    pass
    case.results.setdefault("logv3lpf", {})["regtap"] = taps


def _apply_opendss_regtaps_to_logv(case) -> int:
    """Copy OpenDSS-resolved RegControl taps into ``case.transformers.taps``.

    Callers that need those taps in the linear model must rebuild ``A`` afterward
    via ``get_transformer_matrices`` + ``calculate_base_matrices`` (analytical Tau;
    regulators use Ỹ in A and ỹ on the RHS — see ``linpf.get_transformer_matrices``).
    """
    import opendssdirect as dss

    if getattr(case, "regcontrols", None) is None or len(case.regcontrols) == 0:
        return 0
    n_applied = 0
    od = (case.results.get("openDSS") or {}).get("regtap") or {}
    for i in range(len(case.regcontrols)):
        name = str(case.regcontrols.name[i])
        tap = od.get(name)
        if tap is None:
            try:
                dss.RegControls.Name(name)
                tap = 1.0 + float(dss.RegControls.TapNumber()) * 0.00625
            except Exception:
                continue
        xfmr = (
            case.regcontrols.transformer[i]
            if "transformer" in case.regcontrols.columns
            else name
        )
        hit = case.transformers.index[
            case.transformers.name.astype(str).str.lower() == str(xfmr).lower()
        ]
        if len(hit) == 0:
            hit = case.transformers.index[
                case.transformers.name.astype(str).str.lower() == name.lower()
            ]
        if len(hit) == 0:
            continue
        j = int(hit[0])
        taps = list(case.transformers.taps[j])
        if "winding" in case.regcontrols.columns:
            wdg_i = max(0, min(len(taps) - 1, int(case.regcontrols.winding[i]) - 1))
        else:
            wdg_i = 1 if len(taps) > 1 else 0
        taps[wdg_i] = float(tap)
        case.transformers.at[j, "taps"] = taps
        if "tap" in case.regcontrols.columns:
            case.regcontrols.at[i, "tap"] = float(tap)
        n_applied += 1
    return n_applied


def _rebuild_logv_A_from_case_taps(case) -> None:
    """Rebuild Log(v) base ``A`` from ``case.transformers.taps`` (analytical Tau)."""
    import logv3lpf.linpf as linpf

    with _quiet():
        linpf.get_transformer_matrices(case)
        linpf.calculate_base_matrices(case)


def run_logv_synced_from_opendss(case) -> int:
    """One Log(v) PF with OpenDSS-resolved taps baked into ``A``."""
    import logv3lpf.linpf as linpf

    n = _apply_opendss_regtaps_to_logv(case)
    _rebuild_logv_A_from_case_taps(case)
    with _quiet():
        linpf.rank_k_correction_solve(case, True)
    _store_logv_regtaps(case)
    return n


def _apply_opendss_cap_pv_to_logv(case) -> dict[str, int]:
    """Copy OpenDSS-settled capacitor states and PV P/Q into Log(v).

    CapControl / InvControl are not looped inside Log(v); for ``static``/``synced``
    we freeze OD Static outcomes into the linear model so only RegControl (or PF)
    disagreement remains.
    """
    import logv3lpf.linpf as linpf
    import numpy as np
    import opendssdirect as dss

    out = {"caps": 0, "pvs": 0}
    caps_changed = False

    # ---- Capacitors ----
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
                caps_changed = True
            out["caps"] += 1

    # ---- PVSystems -> pv_* loads (includes InvControl Q after OD Static) ----
    if getattr(case, "loads", None) is not None and len(case.loads):
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
            hit = case.loads.index[case.loads.name.astype(str).str.lower() == row.lower()]
            if len(hit) == 0:
                continue
            j = int(hit[0])
            case.loads.at[j, "kW"] = p_kw
            case.loads.at[j, "kvar"] = q_kvar
            out["pvs"] += 1

    if caps_changed:
        with _quiet():
            linpf.calculate_base_matrices(case)
    return out


def run_logv_autonomous(case, *, max_iter: int = 40) -> int:
    """Iterate Log(v) RegControl until taps settle. Returns iteration count."""
    import logv3lpf.linpf as linpf
    from logv3lpf.Controllers import apply_controls

    if getattr(case, "regcontrols", None) is None or len(case.regcontrols) == 0:
        with _quiet():
            linpf.rank_k_correction_solve(case, True)
        _store_logv_regtaps(case)
        return 0

    def _rebuild_y_from_logv_taps() -> None:
        # Regulators use analytical Tau(case.taps); no OpenDSS Solve needed.
        with _quiet():
            linpf.get_transformer_matrices(case)
            linpf.calculate_base_matrices(case)

    _rebuild_y_from_logv_taps()

    n_it = 0
    for n_it in range(1, max_iter + 1):
        with _quiet():
            linpf.rank_k_correction_solve(case, True)
        changes = apply_controls(case)
        if not changes or all(int(c) == 0 for c in changes):
            break
        _rebuild_y_from_logv_taps()
    _store_logv_regtaps(case)
    return n_it


_MODEL_LABEL = {
    1: "constP",
    2: "constZ",
    3: "constPQZ",
    4: "exp",
    5: "constI",
    6: "constP_Z",
    7: "constP_I",
    8: "ZIP",
}


def load_model_breakdown(case) -> dict[str, Any]:
    """ZIP / P/Z/I load-model errors: counts, remaps, |V| MAE by DSS model group."""
    import numpy as np

    loads = case.loads
    empty = {
        "applicable": False,
        "counts_dss": {},
        "counts_labeled": {},
        "n_remapped": 0,
        "mae_by_group": {},
        "mae_by_dss_model": {},
        "n_remapped_buses": 0,
        "n_native_buses": 0,
        "has_zip_or_unsupported": False,
        "has_mixed_pzi": False,
    }
    if loads is None or len(loads) == 0:
        return empty

    names = loads.name.astype(str)
    keep = ~names.str.lower().str.startswith("pv_")
    idx_keep = np.where(keep.to_numpy())[0]
    if len(idx_keep) == 0:
        return empty

    dss_m = (
        np.asarray(loads.dss_model, int)
        if "dss_model" in loads.columns
        else np.asarray(loads.model, int)
    )
    log_m = np.asarray(loads.model, int)
    dss_k = dss_m[idx_keep]
    log_k = log_m[idx_keep]

    counts: dict[str, int] = {}
    counts_labeled: dict[str, int] = {}
    for m in dss_k:
        key = f"model_{int(m)}"
        counts[key] = counts.get(key, 0) + 1
        lab = f"M{int(m)}({_MODEL_LABEL.get(int(m), '?')})"
        counts_labeled[lab] = counts_labeled.get(lab, 0) + 1
    n_remapped = int(np.sum(dss_k != log_k))
    has_zip = bool(np.any(dss_k == 8))
    has_unsupported = bool(np.any(~np.isin(dss_k, [1, 2, 5])))
    has_mixed = len(set(int(x) for x in dss_k)) > 1

    # buses by original DSS model
    buses_by_model: dict[int, set[str]] = {}
    remapped_buses: set[str] = set()
    native_buses: set[str] = set()
    for i in idx_keep:
        bus = str(loads.bus[i])
        dm = int(dss_m[i])
        lm = int(log_m[i])
        buses_by_model.setdefault(dm, set()).add(bus)
        if dm != lm:
            remapped_buses.add(bus)
        else:
            native_buses.add(bus)
    native_buses -= remapped_buses

    def _mae_on(buses: set[str]) -> float:
        errs = []
        vm_l = case.results.get("logv3lpf", {}).get("vm", {})
        vm_o = case.results.get("openDSS", {}).get("vm", {})
        for bus in buses:
            key_l = bus if bus in vm_l else next(
                (k for k in vm_l if str(k).lower() == bus.lower()), None
            )
            key_o = bus if bus in vm_o else next(
                (k for k in vm_o if str(k).lower() == bus.lower()), None
            )
            if key_l is None or key_o is None:
                continue
            a = np.asarray(vm_l[key_l], float)
            b = np.asarray(vm_o[key_o], float)
            n = min(len(a), len(b))
            if n:
                errs.append(np.abs(a[:n] - b[:n]))
        if not errs:
            return float("nan")
        return float(np.mean(np.concatenate(errs)))

    mae_by_dss: dict[str, float] = {}
    for dm, buses in sorted(buses_by_model.items()):
        # isolate: buses that only have this model (approx: all buses hosting it)
        lab = f"M{dm}({_MODEL_LABEL.get(dm, '?')})"
        mae_by_dss[lab] = _mae_on(buses)

    return {
        "applicable": bool(has_mixed or has_unsupported or n_remapped),
        "counts_dss": counts,
        "counts_labeled": counts_labeled,
        "n_remapped": n_remapped,
        "mae_by_group": {
            "remapped_load_buses": _mae_on(remapped_buses),
            "native_PZI_load_buses": _mae_on(native_buses),
        },
        "mae_by_dss_model": mae_by_dss,
        "n_remapped_buses": len(remapped_buses),
        "n_native_buses": len(native_buses),
        "has_zip_or_unsupported": bool(has_zip or has_unsupported),
        "has_mixed_pzi": has_mixed,
        "note": (
            "Log(v) supports DSS models 1/2/5 (P/Z/I) only; "
            "ZIP(model8) and others are remapped to const-P before solve."
        ),
    }


def collect_controller_state_errors(case) -> dict[str, Any]:
    """OpenDSS vs Log(v) controller states that exist on this feeder."""
    import numpy as np
    import opendssdirect as dss

    out: dict[str, Any] = {
        "regcontrols": {"applicable": False, "rows": [], "mae_tap": float("nan"), "n": 0},
        "capcontrols": {"applicable": False, "rows": [], "n_disagree_onoff": 0, "n": 0},
        "inv_pv": {"applicable": False, "rows": [], "mae_kw": float("nan"), "mae_kvar": float("nan"), "n": 0},
    }

    # ---- RegControls ----
    od_tap = case.results.get("openDSS", {}).get("regtap") or {}
    lv_tap = case.results.get("logv3lpf", {}).get("regtap") or {}
    reg_names = sorted(set(od_tap) | set(lv_tap))
    if getattr(case, "regcontrols", None) is not None and len(case.regcontrols):
        reg_names = sorted(set(reg_names) | set(str(n) for n in case.regcontrols.name))
    rows_reg = []
    diffs = []
    for name in reg_names:
        od = float(od_tap.get(name, np.nan))
        lv = float(lv_tap.get(name, np.nan))
        d = abs(od - lv) if np.isfinite(od) and np.isfinite(lv) else float("nan")
        if np.isfinite(d):
            diffs.append(d)
        rows_reg.append({"name": name, "opendss_tap": od, "logv_tap": lv, "abs_err": d})
    if rows_reg:
        out["regcontrols"] = {
            "applicable": True,
            "rows": rows_reg,
            "mae_tap": float(np.mean(diffs)) if diffs else float("nan"),
            "n": len(rows_reg),
            "note": "OpenDSS RegControl (Static) vs Log(v) RegControl loop taps",
        }

    # ---- Capacitors / CapControls ----
    cap_rows = []
    try:
        cap_names = [n for n in dss.Capacitors.AllNames() if n and n != "NONE"]
    except Exception:
        cap_names = []
    try:
        cc_names = [n for n in dss.CapControls.AllNames() if n and n != "NONE"]
    except Exception:
        cc_names = []

    for cn in cap_names:
        try:
            dss.Capacitors.Name(cn)
            st = np.asarray(dss.Capacitors.States(), dtype=float)
            od_on = float(1.0 if np.sum(st > 0) > 0.5 else 0.0)
            kvar_od = float(dss.Capacitors.kvar()) if hasattr(dss.Capacitors, "kvar") else float("nan")
        except Exception:
            od_on = float("nan")
            kvar_od = float("nan")
        lv_kvar = float("nan")
        lv_on = float("nan")
        if getattr(case, "capacitors", None) is not None and len(case.capacitors):
            hit = case.capacitors.index[
                case.capacitors.name.astype(str).str.lower() == str(cn).lower()
            ]
            if len(hit):
                lv_kvar = float(case.capacitors.kvar[int(hit[0])])
                # Log(v) has no CapControl: bank modeled as fixed shunt kvar
                lv_on = float(1.0 if abs(lv_kvar) > 1e-6 else 0.0)
        cap_rows.append(
            {
                "name": cn,
                "opendss_on": od_on,
                "logv_on": lv_on,
                "opendss_kvar": kvar_od,
                "logv_kvar": lv_kvar,
                "on_mismatch": (
                    int(od_on != lv_on)
                    if np.isfinite(od_on) and np.isfinite(lv_on)
                    else None
                ),
            }
        )
    if cap_rows:
        n_mis = sum(1 for r in cap_rows if r["on_mismatch"] == 1)
        out["capcontrols"] = {
            "applicable": True,
            "n_capcontrols": len(cc_names),
            "rows": cap_rows,
            "n": len(cap_rows),
            "n_disagree_onoff": n_mis,
            "note": (
                "OpenDSS CapControl Static states copied into Log(v) "
                "(Log(v) has no CapControl loop)."
            ),
        }

    # ---- PV / InvControl (OpenDSS injection vs Log(v) pv_* loads) ----
    pv_rows = []
    try:
        pv_names = [n for n in dss.PVsystems.AllNames() if n and n != "NONE"]
    except Exception:
        pv_names = []
    for pv in pv_names:
        try:
            dss.PVsystems.Name(pv)
            powers = np.asarray(dss.CktElement.Powers(), dtype=float)
            # into-element: generation negative
            od_kw = float(powers[0::2].sum())
            od_kvar = float(powers[1::2].sum())
        except Exception:
            od_kw = od_kvar = float("nan")
        lv_kw = lv_kvar = float("nan")
        row = f"pv_{pv}"
        if getattr(case, "loads", None) is not None:
            hit = case.loads.index[case.loads.name.astype(str).str.lower() == row.lower()]
            if len(hit):
                j = int(hit[0])
                lv_kw = float(case.loads.kW[j])
                lv_kvar = float(case.loads.kvar[j])
        pv_rows.append(
            {
                "name": pv,
                "opendss_kw": od_kw,
                "logv_kw": lv_kw,
                "opendss_kvar": od_kvar,
                "logv_kvar": lv_kvar,
                "abs_err_kw": abs(od_kw - lv_kw) if np.isfinite(od_kw) and np.isfinite(lv_kw) else float("nan"),
                "abs_err_kvar": abs(od_kvar - lv_kvar)
                if np.isfinite(od_kvar) and np.isfinite(lv_kvar)
                else float("nan"),
            }
        )
    if pv_rows:
        ekw = [r["abs_err_kw"] for r in pv_rows if np.isfinite(r["abs_err_kw"])]
        eq = [r["abs_err_kvar"] for r in pv_rows if np.isfinite(r["abs_err_kvar"])]
        out["inv_pv"] = {
            "applicable": True,
            "rows": pv_rows,
            "n": len(pv_rows),
            "mae_kw": float(np.mean(ekw)) if ekw else float("nan"),
            "mae_kvar": float(np.mean(eq)) if eq else float("nan"),
            "note": (
                "OpenDSS PV(+InvControl) P/Q copied into Log(v) pv_* loads after OD Static "
                "(Log(v) has no InvControl loop)."
            ),
        }

    return out


def plot_controller_state_compare(
    report: dict[str, Any],
    feeder: str,
    *,
    out_png: Path | None = None,
    save: bool = False,
) -> None:
    """Plot applicable controller-state OpenDSS vs Log(v) errors."""
    import matplotlib.pyplot as plt
    import numpy as np

    ctrl = report.get("controllers") or {}
    panels = []
    reg = ctrl.get("regcontrols") or {}
    cap = ctrl.get("capcontrols") or {}
    inv = ctrl.get("inv_pv") or {}
    if reg.get("applicable") and reg.get("rows"):
        panels.append("reg")
    if cap.get("applicable") and cap.get("rows"):
        panels.append("cap")
    if inv.get("applicable") and inv.get("rows"):
        panels.append("inv")
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    ax_i = 0

    if "reg" in panels:
        ax = axes[ax_i]
        rows = reg["rows"]
        names = [r["name"] for r in rows]
        od = [r["opendss_tap"] for r in rows]
        lv = [r["logv_tap"] for r in rows]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, od, width=w, label="OpenDSS")
        ax.bar(x + w / 2, lv, width=w, label="Log(v)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("tap (pu)")
        ax.set_title(f"{feeder}: RegControl taps  MAE={reg.get('mae_tap', float('nan')):.5f}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax_i += 1

    if "cap" in panels:
        ax = axes[ax_i]
        rows = cap["rows"]
        names = [r["name"] for r in rows]
        od = [r["opendss_on"] for r in rows]
        lv = [r["logv_on"] for r in rows]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, od, width=w, label="OpenDSS on")
        ax.bar(x + w / 2, lv, width=w, label="Log(v) on*")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("bank on (0/1)")
        ax.set_title(
            f"{feeder}: caps  disagree={cap.get('n_disagree_onoff', 0)}/{cap.get('n', 0)}\n"
            f"*Log(v)=fixed kvar, no CapControl"
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax_i += 1

    if "inv" in panels:
        ax = axes[ax_i]
        rows = inv["rows"]
        names = [r["name"] for r in rows]
        x = np.arange(len(names))
        w = 0.2
        ax.bar(x - 1.5 * w, [r["opendss_kw"] for r in rows], width=w, label="OD P")
        ax.bar(x - 0.5 * w, [r["logv_kw"] for r in rows], width=w, label="LV P")
        ax.bar(x + 0.5 * w, [r["opendss_kvar"] for r in rows], width=w, label="OD Q")
        ax.bar(x + 1.5 * w, [r["logv_kvar"] for r in rows], width=w, label="LV Q")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("kW / kvar (into element)")
        ax.set_title(
            f"{feeder}: PV P/Q  MAE_P={inv.get('mae_kw', float('nan')):.3f}  "
            f"MAE_Q={inv.get('mae_kvar', float('nan')):.3f}"
        )
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{feeder}: autonomous controller / DER state errors", fontsize=11)
    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()


def plot_load_model_compare(case, feeder: str, breakdown: dict, out_png: Path | None = None, *, save: bool = False):
    """Bar: DSS load-model counts + |V| MAE by model / remap group."""
    import matplotlib.pyplot as plt
    import numpy as np

    counts = breakdown.get("counts_labeled") or breakdown.get("counts_dss") or {}
    mae_g = breakdown.get("mae_by_group") or {}
    mae_m = breakdown.get("mae_by_dss_model") or {}
    if not counts:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    axes[0].bar(np.arange(len(labels)), vals, color="C0")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("# loads")
    axes[0].set_title(f"{feeder}: DSS load models (ZIP=M8; Log(v) keeps 1/2/5)")
    axes[0].grid(True, axis="y", alpha=0.3)

    # prefer per-model MAE if >1 model; else remap split
    if len(mae_m) > 1:
        mlab = list(mae_m.keys())
        mval = [mae_m[k] for k in mlab]
        axes[1].bar(np.arange(len(mlab)), mval, color="C3")
        axes[1].set_xticks(np.arange(len(mlab)))
        axes[1].set_xticklabels(mlab, rotation=25, ha="right", fontsize=8)
        axes[1].set_title(f"{feeder}: |V| MAE by DSS load-model buses")
    else:
        g_labels = ["native P/Z/I", "remapped->P"]
        g_vals = [
            mae_g.get("native_PZI_load_buses", float("nan")),
            mae_g.get("remapped_load_buses", float("nan")),
        ]
        axes[1].bar(np.arange(2), g_vals, color=["C2", "C3"])
        axes[1].set_xticks(np.arange(2))
        axes[1].set_xticklabels(g_labels, fontsize=9)
        axes[1].set_title(
            f"{feeder}: |V| MAE by remap  (n_remap={breakdown.get('n_remapped', 0)})"
        )
    axes[1].set_ylabel("|V| MAE (pu)")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save and out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()


def _time_calls(
    fn,
    *,
    repeats: int = 5,
    warmup: int = 0,
) -> dict[str, float]:
    """Wall-clock timing helper with optional warm-up.

    Returns min/median/mean/max and IQR (p25/p75) over timed repeats only.
    """
    import time

    import numpy as np

    warm = max(0, int(warmup))
    for _ in range(warm):
        fn()
    reps = max(1, int(repeats))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, float)
    return {
        "warmup": float(warm),
        "repeats": float(reps),
        "sec_min": float(np.min(arr)),
        "sec_p25": float(np.percentile(arr, 25)),
        "sec_median": float(np.median(arr)),
        "sec_p75": float(np.percentile(arr, 75)),
        "sec_mean": float(np.mean(arr)),
        "sec_max": float(np.max(arr)),
    }


def print_paper_replication_report(
    feeder: str,
    *,
    paper_acc: dict[str, float],
    ours_acc: dict[str, float],
    paper_flops: dict[str, float],
    ours_flops: dict[str, float],
    wall: dict[str, Any] | None = None,
) -> None:
    """Print Table II / IV style replication vs published Log(v) 3LPF numbers."""
    key = resolve_feeder(feeder)
    sep = "-" * 64
    print(sep)
    print(f"PAPER REPLICATION - {key}  (compare to Carreno et al. Tables II & IV)")
    print(sep)
    print("  Accuracy (Table IV)")
    print(
        f"    {'metric':<16} {'paper':>12} {'ours':>12} {'delta':>12}"
    )
    for label, pk, ok in (
        ("|V| RMSE (pu)", "vm_rmse_pu", "vm_rmse_pu"),
        ("|V| MAPE (%)", "vm_mape_pct", "vm_mape_pct"),
        ("angle RMSE (deg)", "va_rmse_deg", "va_rmse_deg"),
        ("angle RMSE diag", None, "va_rmse_deg_global_aligned"),
        ("|V| MAE (pu)", None, "vm_mae_pu"),
    ):
        ov = ours_acc.get(ok, float("nan"))
        if pk is None:
            print(f"    {label:<16} {'(not pub.)':>12} {ov:12.5g} {'':>12}")
            continue
        pv = paper_acc.get(pk, float("nan"))
        dv = ov - pv if (pv == pv and ov == ov) else float("nan")
        print(f"    {label:<16} {pv:12.5g} {ov:12.5g} {dv:12.5g}")
    off = ours_acc.get("va_source_offset_deg", float("nan"))
    vs = ours_acc.get("vsource_angle_deg", float("nan"))
    if off == off or vs == vs:
        print(
            f"    note: angle frame = linearization (phase A=0 + Dy Lag -30 LV) "
            f"+ Vsource.Angle={vs if vs==vs else 0:+.1f} deg (reporting). "
            f"residual source offset={off if off==off else 0:+.4f} deg "
            f"(extra rotate only if |offset|>0.5). "
            f"'angle RMSE diag' = global-mean alignment diagnostic only."
        )
    print()
    print("  Inverse-update FLOPs (Table II) - theoretical, not wall-clock")
    print(
        f"    {'':<22} {'paper':>14} {'ours':>14}"
    )
    print(
        f"    {'n (bus-phases)':<22} {paper_flops.get('n', float('nan')):14.0f} "
        f"{ours_flops.get('n_bus_phases', float('nan')):14.0f}"
    )
    n_ex = ours_flops.get("n_excl_source", float("nan"))
    if n_ex == n_ex:
        print(f"    {'n excl. source':<22} {'(8500 paper~)':>14} {n_ex:14.0f}")
    print(
        f"    {'k (corr. rank)':<22} {paper_flops.get('k', float('nan')):14.0f} "
        f"{ours_flops.get('k', float('nan')):14.0f}"
    )
    print(
        f"    {'  load/cap/reg ph':<22} {'':>14} "
        f"{int(ours_flops.get('k_load_phases', 0))}/"
        f"{int(ours_flops.get('k_cap_phases', 0))}/"
        f"{int(ours_flops.get('k_reg_phases', 0))} "
        f"(x2->k; Woodbury U={int(ours_flops.get('k_woodbury_U', 0))})"
    )
    print(
        f"    {'OpenDSS/LU FLOPs':<22} {paper_flops.get('opendss_flops', float('nan')):14.3e} "
        f"{ours_flops.get('opendss_lu_flops', float('nan')):14.3e}"
    )
    print(
        f"    {'Log(v) update FLOPs':<22} {paper_flops.get('logv_flops', float('nan')):14.3e} "
        f"{ours_flops.get('logv_woodbury_update_flops', float('nan')):14.3e}"
    )
    pr = paper_flops.get("ratio", float("nan"))
    orat = ours_flops.get("flop_ratio_opendss_over_logv_update", float("nan"))
    print(f"    {'FLOP ratio OD/Log(v)':<22} {pr:14.3g} {orat:14.3g}")
    print(
        "    note: paper ratio is inverse-update arithmetic only; k=0 => update FLOPs=0 "
        "(A fixed), not zero PF cost."
    )
    if wall:
        od = wall.get("opendss_solve") or {}
        lv = wall.get("logv_solve") or {}
        if "sec_median" in od or "sec_median" in lv:
            print()
            print("  Wall-clock STOCK Log(v) path (get_loads every solve) — NOT the fast speedup")
            print(
                f"    OpenDSS solve median : {od.get('sec_median', float('nan')):.6f} s "
                f"(p25={od.get('sec_p25', float('nan')):.6f}, "
                f"p75={od.get('sec_p75', float('nan')):.6f}, "
                f"warmup={int(od.get('warmup', 0))}, n={int(od.get('repeats', 0))})"
            )
            print(
                f"    stock Log(v) median  : {lv.get('sec_median', float('nan')):.6f} s "
                f"(p25={lv.get('sec_p25', float('nan')):.6f}, "
                f"p75={lv.get('sec_p75', float('nan')):.6f}, "
                f"warmup={int(lv.get('warmup', 0))}, n={int(lv.get('repeats', 0))})"
            )
            od_m = float(od.get("sec_median", float("nan")))
            lv_m = float(lv.get("sec_median", float("nan")))
            if od_m == od_m and lv_m == lv_m and lv_m > 0:
                print(
                    f"    stock ratio OD/Log(v): {od_m / lv_m:.3f}x  "
                    f"(values <1 mean OpenDSS is faster than stock Log(v) — expected)"
                )
            print(
                "    *** For sequential FastLogv speedups, run benchmark_fast_vs_opendss "
                "/ set RUN_SPEED=True and scroll to [SEQUENTIAL ONLY] / SPEED SUMMARY. ***"
            )
            print(
                "    timing excludes DSS compile + Log(v) matrix build; "
                "each timed OpenDSS call nudges LoadMult."
            )
    print(sep)


def compare_logv3lpf_vs_opendss(
    repo: Path | None = None,
    *,
    feeder: str = "ieee34",
    plot: bool = True,
    save_plots: bool = False,
    out_dir: Path | None = None,
    control_mode: str = "static",
    max_reg_iter: int = 40,
    diagnostics: bool = True,
    timing_repeats: int = 5,
    timing_warmup: int | None = None,
):
    """OpenDSS vs Log(v) with explicit control-state modes.

    ``control_mode``:
      - ``off``: both frozen (ControlMode=OFF) - pure linearization / model compare.
      - ``synced``: OpenDSS Static to settle devices; copy OD taps into Log(v);
        one Log(v) PF. Isolates PF/model error from controller disagreement.
      - ``static``: OpenDSS Static vs Log(v) own RegControl heuristic loop
        (device states may differ - not a pure linearization test).

    ``diagnostics`` (default True): print audit checklist + verification plots.
    ``timing_repeats`` / ``timing_warmup``: wall-clock protocol (paper reported FLOPs
    only). Default warmup = max(5, repeats//5).
    """
    import opendssdirect as dss
    import numpy as np
    import logv3lpf
    import logv3lpf.linpf as linpf
    from logv3lpf.DSSParser import DSScase

    repo = Path(repo or Path.cwd()).resolve()
    ensure_logv3lpf(repo)
    key = resolve_feeder(feeder)
    meta = FEEDERS[key]
    dss_path: Path = Path(meta["dss"](repo)).resolve()
    if not dss_path.is_file():
        raise FileNotFoundError(dss_path)

    mode = str(control_mode).strip().lower()
    if mode not in ("static", "off", "synced"):
        raise ValueError("control_mode must be 'static', 'off', or 'synced'")

    # OpenDSS Compile often chdirs into the DSS folder - pin and restore.
    import os

    _cwd0 = Path.cwd()
    try:
        os.chdir(repo)
    except Exception:
        pass

    with _quiet():
        case = logv3lpf.case(
            compile_cmd(dss_path),
            meta["sourcebus"],
            float(meta["refvm"]),
            0,
        )

    n_reg = int(len(case.regcontrols)) if getattr(case, "regcontrols", None) is not None else 0
    n_it = 0
    inventory = _opendss_control_inventory()
    od_taps_start: dict[str, float] = {}
    lv_taps_start: dict[str, float] = {}
    wall: dict[str, Any] = {}
    _od_solve_i = {"i": 0}
    _do_stock_timing = int(timing_repeats) > 0
    _warm = (
        int(timing_warmup)
        if timing_warmup is not None
        else (max(5, int(timing_repeats) // 5) if _do_stock_timing else 0)
    )

    def _od_solve_once():
        # Force a real re-solve: OpenDSS returns instantly if nothing changed.
        with _quiet():
            _od_solve_i["i"] += 1
            pert = 1.0 + 1e-9 * float(_od_solve_i["i"])
            try:
                dss.Text.Command(f"Set LoadMult={pert}")
            except Exception:
                pass
            dss.Solution.Solve()

    def _lv_solve_once():
        with _quiet():
            linpf.rank_k_correction_solve(case, True)

    def _finalize_opendss_nominal():
        with _quiet():
            try:
                dss.Text.Command("Set LoadMult=1")
            except Exception:
                pass
            dss.Solution.Solve()
            DSScase.process_openDSS_solution(case)

    def _time_od():
        if not _do_stock_timing:
            _od_solve_once()
            return {}
        return _time_calls(_od_solve_once, repeats=timing_repeats, warmup=_warm)

    def _time_lv():
        if not _do_stock_timing:
            _lv_solve_once()
            return {}
        return _time_calls(_lv_solve_once, repeats=timing_repeats, warmup=_warm)

    if mode == "static":
        _reset_reg_taps_opendss()
        _reset_reg_taps_logv(case)
        od_taps_start = _opendss_reg_taps()
        _store_logv_regtaps(case)
        lv_taps_start = dict(case.results.get("logv3lpf", {}).get("regtap") or {})
        try:
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Set ControlMode=Static")
            dss.Text.Command("Set MaxControlIter=100")
        except Exception:
            pass
        # fixedctrlinit parity: clear residual InvControl Q before Static settle.
        reset_invcontrol_pv_to_pf1()
        wall["opendss_solve"] = _time_od()
        _finalize_opendss_nominal()
        # Freeze OD CapControl / InvControl outcomes into Log(v); RegControl still autonomous.
        _apply_opendss_cap_pv_to_logv(case)
        # OD Static left DSS taps settled; snap DSS YPrim back to Log(v) taps (1.0)
        # before the autonomous loop rebuilds from OpenDSS YPrim.
        _push_logv_regtaps_to_opendss(case)
        n_it = run_logv_autonomous(case, max_iter=int(max_reg_iter))
        # Autonomous path already solved; time one additional fixed-tap Log(v) solve.
        # process_logv3lpf_solution() rebuilds results["logv3lpf"] and drops regtap —
        # re-store after timing so the controller audit is not all-NaN.
        wall["logv_solve"] = _time_lv()
        _store_logv_regtaps(case)
    elif mode == "synced":
        _reset_reg_taps_opendss()
        _reset_reg_taps_logv(case)
        od_taps_start = _opendss_reg_taps()
        _store_logv_regtaps(case)
        lv_taps_start = dict(case.results.get("logv3lpf", {}).get("regtap") or {})
        try:
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Set ControlMode=Static")
            dss.Text.Command("Set MaxControlIter=100")
        except Exception:
            pass
        # fixedctrlinit parity: clear residual InvControl Q before Static settle.
        reset_invcontrol_pv_to_pf1()
        wall["opendss_solve"] = _time_od()
        _finalize_opendss_nominal()
        _apply_opendss_cap_pv_to_logv(case)
        n_it = _apply_opendss_regtaps_to_logv(case)
        _rebuild_logv_A_from_case_taps(case)
        wall["logv_solve"] = _time_lv()
        _store_logv_regtaps(case)
    else:
        try:
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Set ControlMode=OFF")
            dss.Text.Command("Set Number=1")
            dss.Text.Command("Set Hour=0")
            dss.Text.Command("Set Sec=0")
        except Exception:
            pass
        od_taps_start = _opendss_reg_taps()
        _store_logv_regtaps(case)
        lv_taps_start = dict(case.results.get("logv3lpf", {}).get("regtap") or {})
        wall["opendss_solve"] = _time_od()
        _finalize_opendss_nominal()
        wall["logv_solve"] = _time_lv()
        _store_logv_regtaps(case)

    metrics = voltage_metrics(case)
    paper_acc = paper_accuracy_metrics(case)
    flops = paper_flop_estimates(case)
    zip_info = load_model_breakdown(case)
    ve = collect_voltage_errors(case)
    controllers = collect_controller_state_errors(case)

    od_t = case.results.get("openDSS", {}).get("regtap") or {}
    lv_t = case.results.get("logv3lpf", {}).get("regtap") or {}
    tap_rows = []
    tap_mae = float("nan")
    names = sorted(set(od_t) | set(lv_t) | set(od_taps_start) | set(lv_taps_start))
    diffs = []
    for n in names:
        od_s = float(od_taps_start.get(n, np.nan))
        od_e = float(od_t.get(n, np.nan))
        lv_s = float(lv_taps_start.get(n, np.nan))
        lv_e = float(lv_t.get(n, np.nan))
        d = abs(od_e - lv_e) if np.isfinite(od_e) and np.isfinite(lv_e) else float("nan")
        if np.isfinite(d):
            diffs.append(d)
        tap_rows.append(
            {
                "name": n,
                "od_start": od_s,
                "od_end": od_e,
                "lv_start": lv_s,
                "lv_end": lv_e,
                "abs_diff": d,
            }
        )
    if diffs:
        tap_mae = float(np.mean(diffs))

    try:
        od_converged = bool(dss.Solution.Converged())
    except Exception:
        od_converged = None

    od_ctrl_label = {
        "static": "Static",
        "synced": "Static (then taps copied into Log(v))",
        "off": "OFF",
    }[mode]

    report = {
        "feeder": key,
        "dss": str(dss_path),
        "control_mode": mode,
        "opendss_control_mode": od_ctrl_label,
        "opendss_converged": od_converged,
        "opendss_inventory": inventory,
        "n_reg": n_reg,
        "reg_iters": n_it,
        "metrics": metrics,
        "paper_accuracy": paper_acc,
        "paper_flops": flops,
        "wall_clock": wall,
        "voltage_errors": {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in ve.items()
            if k not in ("err", "signed", "opendss_vm", "logv_vm")
        },
        "transformer_drops": transformer_drop_audit(case),
        "taps": {"rows": tap_rows, "mae": tap_mae},
        "load_models": zip_info,
        "controllers": controllers,
        "model_notes": [
            "Log(v) uses paper Ytilde:=Delta Y* Delta^H and ytilde:=Ytilde*1 (row sums).",
            "No Delta-Wye ytilde zeroing; no default A ridge (ainv_ridge=0).",
            "2-winding transformer Yprim from OpenDSS CktElement.YPrim (pu), not ad-hoc ytilde zeroing.",
            "Analytical Yprim is fallback only; magnetizing branch omitted in fallback.",
            "Unsupported DSS load models remapped -> const-P (see load_models).",
            "PVSystems -> negative const-P loads (no InvControl in Log(v)).",
            "Paper Table II FLOPs are inverse-update estimates; wall-clock is measured separately.",
        ],
    }
    # restore arrays for plotting
    report["voltage_errors"].update(
        {
            "err": ve["err"],
            "signed": ve.get("signed", np.array([])),
            "opendss_vm": ve["opendss_vm"],
            "logv_vm": ve["logv_vm"],
            "worst": ve["worst"],
        }
    )

    if diagnostics:
        print_audit_report(report)
        print_paper_replication_report(
            key,
            paper_acc=PAPER_TABLE_IV.get(key, {}),
            ours_acc=paper_acc,
            paper_flops=PAPER_TABLE_II.get(key, {}),
            ours_flops=flops,
            wall=wall,
        )

    if plot:
        out_dir_p = Path(out_dir) if out_dir is not None else None
        if save_plots and out_dir_p is not None:
            out_dir_p.mkdir(parents=True, exist_ok=True)
        if diagnostics:
            a_png = (
                (out_dir_p / f"opendss_vs_logv3lpf_{key}_audit.png")
                if (save_plots and out_dir_p)
                else None
            )
            plot_audit_diagnostics(
                case, report, out_png=a_png, save=bool(save_plots)
            )
        v_png = (
            (out_dir_p / f"opendss_vs_logv3lpf_{key}_voltage.png")
            if (save_plots and out_dir_p)
            else None
        )
        plot_voltage_compare(case, key, out_png=v_png, save=bool(save_plots))
        # Signed residual e = |V|_Log(v) - |V|_OpenDSS vs OpenDSS |V|
        e_png = (
            (out_dir_p / f"opendss_vs_logv3lpf_{key}_signed_residual.png")
            if (save_plots and out_dir_p)
            else None
        )
        plot_signed_residual_vs_opendss(
            case,
            key,
            out_png=e_png,
            save=bool(save_plots),
            ve=ve,
        )
        # Always plot controller/ZIP panels when applicable on this feeder
        ctrl_applicable = any(
            (controllers.get(k) or {}).get("applicable")
            for k in ("regcontrols", "capcontrols", "inv_pv")
        )
        if ctrl_applicable:
            c_png = (
                (out_dir_p / f"opendss_vs_logv3lpf_{key}_controllers.png")
                if (save_plots and out_dir_p)
                else None
            )
            plot_controller_state_compare(
                report, key, out_png=c_png, save=bool(save_plots)
            )
        elif n_reg and mode == "static":
            r_png = (
                (out_dir_p / f"opendss_vs_logv3lpf_{key}_reg.png")
                if (save_plots and out_dir_p)
                else None
            )
            plot_reg_compare(
                case,
                key,
                out_png=r_png,
                save=bool(save_plots),
                title_suffix=f"iters={n_it}",
            )
        if zip_info.get("counts_dss") and (
            zip_info.get("applicable")
            or zip_info.get("n_remapped", 0) > 0
            or len(zip_info.get("counts_dss") or {}) > 1
        ):
            z_png = (
                (out_dir_p / f"opendss_vs_logv3lpf_{key}_load_models.png")
                if (save_plots and out_dir_p)
                else None
            )
            plot_load_model_compare(
                case, key, zip_info, out_png=z_png, save=bool(save_plots)
            )

    metrics = dict(metrics)
    metrics["control_mode"] = mode
    metrics["n_reg"] = n_reg
    metrics["reg_iters"] = n_it
    metrics["tap_mae"] = tap_mae
    metrics["paper_accuracy"] = paper_acc
    metrics["paper_flops"] = flops
    metrics["wall_clock"] = wall
    metrics["load_models"] = zip_info
    metrics["controllers"] = controllers
    metrics["audit"] = {
        k: v
        for k, v in report.items()
        if k not in ("_ve_arrays",)
        and k != "voltage_errors"
    }
    metrics["audit"]["voltage_errors"] = {
        k: v
        for k, v in report["voltage_errors"].items()
        if k not in ("err", "opendss_vm", "logv_vm", "signed")
    }
    try:
        os.chdir(_cwd0)
    except Exception:
        pass
    return case, metrics


def replicate_paper_benchmarks(
    repo: Path | None = None,
    *,
    feeders: tuple[str, ...] = ("ieee34", "906", "8500"),
    control_mode: str = "off",
    timing_repeats: int = 50,
    timing_warmup: int | None = None,
    plot: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run paper-style Table II/IV replication for one or more feeders.

    Uses ``control_mode='off'`` by default (fixed device states) so results are a
    voltage-accuracy / FLOP / wall-clock test - not controller-trajectory validation.
    Angle RMSE is source-anchored to OpenDSS (not post-hoc global alignment).
    """
    repo = Path(repo or Path.cwd()).resolve()
    out: dict[str, dict[str, Any]] = {}
    print("=" * 72)
    print(
        f"Log(v) 3LPF paper replication  control_mode={control_mode!r}  "
        f"timing_repeats={timing_repeats}"
    )
    print(
        "Note: Table IV uses fixed-state linearization (control_mode=off|synced). "
        "Use control_mode=static separately for autonomous RegControl disagreement."
    )
    print("=" * 72)
    for f in feeders:
        print(f"\n>>> feeder={f}")
        _, metrics = compare_logv3lpf_vs_opendss(
            repo,
            feeder=f,
            plot=plot,
            save_plots=False,
            control_mode=control_mode,
            diagnostics=True,
            timing_repeats=timing_repeats,
            timing_warmup=timing_warmup,
        )
        out[resolve_feeder(f)] = metrics
    # compact summary table
    print("\n" + "=" * 72)
    print("SUMMARY vs paper Tables II & IV")
    print("=" * 72)
    hdr = (
        f"{'feeder':<8} {'n':>6} {'n-src':>6} {'k':>5} {'RMSE_V':>8} {'MAPE%':>7} "
        f"{'RMSE_a':>7} {'FLOPr':>8} {'wall_x':>8}"
    )
    print(hdr)
    for f, m in out.items():
        acc = m.get("paper_accuracy") or {}
        fl = m.get("paper_flops") or {}
        wall = m.get("wall_clock") or {}
        od = (wall.get("opendss_solve") or {}).get("sec_median", float("nan"))
        lv = (wall.get("logv_solve") or {}).get("sec_median", float("nan"))
        wx = (od / lv) if (isinstance(od, float) and isinstance(lv, float) and lv > 0) else float("nan")
        print(
            f"{f:<8} {int(fl.get('n_bus_phases', 0)):6d} "
            f"{int(fl.get('n_excl_source', 0)):6d} {int(fl.get('k', 0)):5d} "
            f"{acc.get('vm_rmse_pu', float('nan')):8.4f} "
            f"{acc.get('vm_mape_pct', float('nan')):7.3f} "
            f"{acc.get('va_rmse_deg', float('nan')):7.3f} "
            f"{fl.get('flop_ratio_opendss_over_logv_update', float('nan')):8.2g} "
            f"{wx:8.2f}"
        )
    print(
        "RMSE_a = angle RMSE after source-frame anchor to OpenDSS (primary). "
        "n-src = Nn excluding source phases (paper 8500 n≈8531). "
        "wall_x = OpenDSS_solve_median / Log(v)_solve_median."
    )
    return out


def benchmark_fast_logv(repo: Path | None = None, **kwargs):
    """Sequential latency benchmark for FastLogvSolver vs OpenDSS.

    One scenario at a time (no multi-RHS batching). See
    ``logv_fast_solver.benchmark_fast_vs_opendss``.
    """
    from logv_fast_solver import benchmark_all_feeders, benchmark_fast_vs_opendss

    kwargs.pop("batch_sizes", None)  # ignore legacy kwarg
    if kwargs.get("feeder"):
        return benchmark_fast_vs_opendss(repo, **kwargs)
    return benchmark_all_feeders(repo, **kwargs)
