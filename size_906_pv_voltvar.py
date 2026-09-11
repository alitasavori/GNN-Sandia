#!/usr/bin/env python3
"""Size Volt-Var PV906 on the European LV feeder copy and score voltage regulation.

Baseline = stock snapshot (no PV).
Candidates = Pmpp sweep with InvControl VOLTVAR (deadband 0.98–1.02, |Q|=0.44*kVA).
Also reports MPPT-only at the same size.

On this LV feeder, active power raises remote V; Volt-Var helps mainly vs uncontrolled
PV (absorb Q). We oversize kVA vs Pmpp (IEEE34-style headroom) so |Q|_max is useful,
and pick Pmpp that maximizes Volt-Var benefit vs MPPT while keeping feeder voltages
reasonable.

Note: PV_voltvar_906.dss also defines PV639 @ 639.2 (LOAD35) with the same shared
vv_curve_044 and starting size (12 kW / 15 kVA). This script still sweeps PV906 only;
PV639 is not re-swept here.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import opendssdirect as dss
from dss import DSSException

REPO = Path(__file__).resolve().parent
DSS_DIR = REPO / "906 bus system" / "LVTestCase_PV_voltvar"
BASE_MASTER = DSS_DIR / "Master_snapshot.dss"
PV_MASTER = DSS_DIR / "Master_snapshot_PV_voltvar.dss"
OUT_JSON = REPO / "outputs" / "906_pv_voltvar_sizing.json"
OUT_CSV = REPO / "outputs" / "906_pv_voltvar_sizing.csv"

# Pmpp candidates (kW). kVA = Pmpp / KVA_PMPP_RATIO so |Q|max = 0.44*kVA has headroom.
PMPP_CANDIDATES_KW = [2, 5, 8, 10, 12, 15, 20, 25]
KVA_PMPP_RATIO = 0.80  # kVA = Pmpp/0.80 → ~25% inverter oversize (Q headroom)

SCENARIOS = [
    {"name": "nominal", "loadmult": 1.0, "vsource_pu": 1.05},
    {"name": "heavy_load", "loadmult": 2.0, "vsource_pu": 1.05},
    {"name": "light_load_high_v", "loadmult": 0.5, "vsource_pu": 1.06},
    {"name": "midday_pv_stress", "loadmult": 0.7, "vsource_pu": 1.05, "hour": 12},
    {"name": "undervolt_support", "loadmult": 2.5, "vsource_pu": 1.00},
]


def _kva_for(pmpp_kw: float) -> float:
    return float(pmpp_kw) / float(KVA_PMPP_RATIO)


def _compile(master: Path) -> None:
    dss.Basic.ClearAll()
    dss.Text.Command(f'cd "{os.path.abspath(str(DSS_DIR))}"')
    dss.Text.Command(f'Redirect "{master.name}"')
    dss.Text.Command("Set MaxControlIter=200")
    dss.Text.Command("Set MaxIterations=100")


def _set_scenario(sc: dict) -> None:
    dss.Text.Command(f"Set LoadMult={float(sc['loadmult'])}")
    dss.Text.Command(f"Edit Vsource.Source pu={float(sc['vsource_pu'])}")
    hour = int(sc.get("hour", 0))
    dss.Text.Command(f"Set Hour={hour}")
    dss.Text.Command("Set Sec=0")
    if any(str(n).lower() == "pv906" for n in (dss.PVsystems.AllNames() or [])):
        dss.Text.Command("Edit PVSystem.PV906 irradiance=1.0")


def _disable_invcontrol() -> None:
    dss.Text.Command("Edit InvControl.PV906_VV enabled=no")
    # Clear residual Q from prior Volt-Var solve
    dss.Text.Command("Edit PVSystem.PV906 PF=1.0 kvar=0")


def _enable_invcontrol() -> None:
    dss.Text.Command("Edit InvControl.PV906_VV enabled=yes")


def _set_pmpp(pmpp_kw: float) -> None:
    kva = _kva_for(pmpp_kw)
    dss.Text.Command(
        f"Edit PVSystem.PV906 Pmpp={float(pmpp_kw)} kVA={float(kva)} PF=1.0"
    )


def _solve(control_static: bool) -> bool:
    dss.Text.Command(f"Set ControlMode={'STATIC' if control_static else 'OFF'}")
    try:
        dss.Text.Command("Solve")
    except DSSException as exc:
        print(f"  solve warning: {exc}")
        return bool(dss.Solution.Converged())
    return bool(dss.Solution.Converged())


def _bus_vmag_pu() -> tuple[np.ndarray, list[str]]:
    names = [str(n) for n in dss.Circuit.AllNodeNames()]
    v = np.asarray(dss.Circuit.AllBusMagPu(), dtype=float)
    keep, vals = [], []
    for nm, vv in zip(names, v):
        if str(nm).lower().startswith("sourcebus"):
            continue
        if not np.isfinite(vv) or vv <= 0:
            continue
        keep.append(str(nm))
        vals.append(float(vv))
    return np.asarray(vals, dtype=float), keep


def _node_v(node: str, names: list[str], vals: np.ndarray) -> float:
    want = node.lower()
    for nm, vv in zip(names, vals):
        if nm.lower() == want:
            return float(vv)
    return float("nan")


def _pv_pq() -> tuple[float, float]:
    names = [str(n).lower() for n in (dss.PVsystems.AllNames() or [])]
    if "pv906" not in names:
        return 0.0, 0.0
    dss.PVsystems.Name("pv906")
    return float(dss.PVsystems.kW()), float(dss.PVsystems.kvar())


def score_voltages(vals: np.ndarray) -> dict[str, float]:
    dev = np.abs(vals - 1.0)
    return {
        "vmin": float(vals.min()),
        "vmax": float(vals.max()),
        "vmean": float(vals.mean()),
        "vrange": float(vals.max() - vals.min()),
        "mae_to_1pu": float(dev.mean()),
        "rms_to_1pu": float(np.sqrt(np.mean(dev**2))),
        "n_out_0.95_1.05": int(np.sum((vals < 0.95) | (vals > 1.05))),
        "n_out_0.90_1.10": int(np.sum((vals < 0.90) | (vals > 1.10))),
        "n_nodes": int(vals.size),
    }


def run_case(*, master: Path, sc: dict, pmpp_kw: float | None, mode: str) -> dict:
    _compile(master)
    if pmpp_kw is not None:
        _set_pmpp(pmpp_kw)
    if mode == "mppt":
        _disable_invcontrol()
    elif mode == "voltvar":
        _enable_invcontrol()
    _set_scenario(sc)
    ok = _solve(control_static=(mode == "voltvar"))
    vals, names = _bus_vmag_pu()
    met = score_voltages(vals)
    p_inj, q_inj = _pv_pq() if mode != "baseline" else (0.0, 0.0)
    return {
        "scenario": sc["name"],
        "mode": mode,
        "pmpp_kw": None if pmpp_kw is None else float(pmpp_kw),
        "kva": None if pmpp_kw is None else _kva_for(float(pmpp_kw)),
        "converged": bool(ok),
        "v_906_1": _node_v("906.1", names, vals),
        "pv_p_inj_kw": p_inj,
        "pv_q_inj_kvar": q_inj,
        **met,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for sc in SCENARIOS:
        r = run_case(master=BASE_MASTER, sc=sc, pmpp_kw=None, mode="baseline")
        rows.append(r)
        print(
            f"[baseline] {sc['name']}: vmax={r['vmax']:.4f} vmin={r['vmin']:.4f} "
            f"mae={r['mae_to_1pu']:.5f} v906={r['v_906_1']:.4f}"
        )

    for pmpp in PMPP_CANDIDATES_KW:
        for sc in SCENARIOS:
            for mode in ("voltvar", "mppt"):
                r = run_case(
                    master=PV_MASTER, sc=sc, pmpp_kw=float(pmpp), mode=mode
                )
                rows.append(r)
                print(
                    f"[{mode:7s}] Pmpp={pmpp:4.0f} kVA={_kva_for(pmpp):5.1f} "
                    f"{sc['name']:20s}: vmax={r['vmax']:.4f} mae={r['mae_to_1pu']:.5f} "
                    f"v906={r['v_906_1']:.4f} Q={r['pv_q_inj_kvar']:+.3f} ok={r['converged']}"
                )

    # Pair voltvar vs mppt per (pmpp, scenario)
    mppt_idx = {
        (r["pmpp_kw"], r["scenario"]): r
        for r in rows
        if r["mode"] == "mppt" and r["converged"]
    }
    base_idx = {r["scenario"]: r for r in rows if r["mode"] == "baseline"}

    by_pmpp: dict[float, list[dict]] = {float(p): [] for p in PMPP_CANDIDATES_KW}
    for r in rows:
        if r["mode"] != "voltvar" or not r["converged"]:
            continue
        p = float(r["pmpp_kw"])
        m = mppt_idx.get((p, r["scenario"]))
        b = base_idx[r["scenario"]]
        if m is None:
            continue
        by_pmpp[p].append(
            {
                "scenario": r["scenario"],
                "d_mae_vs_mppt": float(m["mae_to_1pu"]) - float(r["mae_to_1pu"]),
                "d_vmax_vs_mppt": float(m["vmax"]) - float(r["vmax"]),
                "d_mae_vs_base": float(b["mae_to_1pu"]) - float(r["mae_to_1pu"]),
                "d_vmax_vs_base": float(b["vmax"]) - float(r["vmax"]),
                "v906": float(r["v_906_1"]),
                "q": float(r["pv_q_inj_kvar"]),
            }
        )

    ranking = []
    for p, imps in by_pmpp.items():
        if not imps:
            continue
        mean_mae_mppt = float(np.mean([x["d_mae_vs_mppt"] for x in imps]))
        mean_vmax_mppt = float(np.mean([x["d_vmax_vs_mppt"] for x in imps]))
        mean_mae_base = float(np.mean([x["d_mae_vs_base"] for x in imps]))
        # Reconstruct mean VV vmax from mppt vmax - delta
        # Prefer help vs MPPT, but penalize large absolute overvoltage.
        mean_vv_vmax = float(
            np.mean(
                [
                    mppt_idx[(p, x["scenario"])]["vmax"] - x["d_vmax_vs_mppt"]
                    for x in imps
                ]
            )
        )
        n_help_mppt = int(sum(1 for x in imps if x["d_vmax_vs_mppt"] > 1e-4))
        over_pen = max(0.0, mean_vv_vmax - 1.07)
        score = (
            mean_vmax_mppt
            + 0.5 * mean_mae_mppt
            + 0.15 * mean_mae_base
            - 2.0 * over_pen
        )
        ranking.append(
            {
                "pmpp_kw": p,
                "kva": _kva_for(p),
                "score": score,
                "mean_d_vmax_vs_mppt": mean_vmax_mppt,
                "mean_d_mae_vs_mppt": mean_mae_mppt,
                "mean_d_mae_vs_baseline": mean_mae_base,
                "mean_vv_vmax": mean_vv_vmax,
                "n_scenarios_vmax_helped_vs_mppt": n_help_mppt,
                "per_scenario": imps,
            }
        )
    ranking.sort(key=lambda t: t["score"], reverse=True)

    print("\n=== ranking (higher score = Volt-Var helps more vs MPPT) ===")
    for row in ranking:
        print(
            f"  Pmpp={row['pmpp_kw']:5.1f} kW  kVA={row['kva']:5.1f}  "
            f"score={row['score']:+.5f}  "
            f"dvmax_vs_mppt={row['mean_d_vmax_vs_mppt']:+.5f}  "
            f"dmae_vs_mppt={row['mean_d_mae_vs_mppt']:+.5f}  "
            f"dmae_vs_base={row['mean_d_mae_vs_baseline']:+.5f}  "
            f"#vmax+={row['n_scenarios_vmax_helped_vs_mppt']}"
        )

    best = ranking[0] if ranking else {"pmpp_kw": 10.0, "kva": _kva_for(10.0)}
    best_pmpp = float(best["pmpp_kw"])
    best_kva = float(best["kva"])

    # Write chosen size into PV_voltvar_906.dss
    pv_dss = DSS_DIR / "PV_voltvar_906.dss"
    text = pv_dss.read_text(encoding="utf-8")
    text2 = re.sub(
        r"(New PVSystem\.PV906[\s\S]*?~ kVA=)\S+(\s+Pmpp=)\S+",
        rf"\g<1>{best_kva:g}\g<2>{best_pmpp:g}",
        text,
        count=1,
    )
    text2 = re.sub(
        r"Default size after sweep: Pmpp=\S+ kW",
        f"Default size after sweep: Pmpp={best_pmpp:g} kW (kVA={best_kva:g})",
        text2,
    )
    # Keep comment accurate about oversize
    if "kVA = Pmpp so Q headroom" in text2:
        text2 = text2.replace(
            "kVA = Pmpp so Q headroom is fully from VARMAX=0.44*kVA",
            f"kVA = Pmpp/{KVA_PMPP_RATIO:g} (~{(1/KVA_PMPP_RATIO-1)*100:.0f}% oversize); "
            f"|Q|max = 0.44*kVA",
        )
    pv_dss.write_text(text2, encoding="utf-8")
    print(f"\nWrote Pmpp={best_pmpp:g} kW, kVA={best_kva:g} into {pv_dss}")

    # Verify locked size once more (nominal + light_load_high_v)
    verify = []
    for sc in SCENARIOS:
        for mode in ("baseline", "voltvar", "mppt"):
            master = BASE_MASTER if mode == "baseline" else PV_MASTER
            pmpp = None if mode == "baseline" else best_pmpp
            verify.append(
                run_case(master=master, sc=sc, pmpp_kw=pmpp, mode=mode)
            )
    print("\n=== verification at chosen size ===")
    for sc in SCENARIOS:
        b = next(r for r in verify if r["scenario"] == sc["name"] and r["mode"] == "baseline")
        v = next(r for r in verify if r["scenario"] == sc["name"] and r["mode"] == "voltvar")
        m = next(r for r in verify if r["scenario"] == sc["name"] and r["mode"] == "mppt")
        print(
            f"  {sc['name']:20s}  base vmax={b['vmax']:.4f}  "
            f"VV vmax={v['vmax']:.4f} (Q={v['pv_q_inj_kvar']:+.2f})  "
            f"MPPT vmax={m['vmax']:.4f}  "
            f"dvmax(VV-MPPT)={v['vmax']-m['vmax']:+.4f}"
        )

    summary = {
        "best_pmpp_kw": best_pmpp,
        "best_kva": best_kva,
        "kva_pmpp_ratio": KVA_PMPP_RATIO,
        "ranking": ranking,
        "curve": {
            "deadband_pu": [0.98, 1.02],
            "q_limit_pu_kva": 0.44,
            "xarray": [0.92, 0.98, 1.02, 1.08],
            "yarray": [0.44, 0.0, 0.0, -0.44],
            "RefReactivePower": "VARMAX",
        },
        "bus": "906.1",
        "dss_dir": str(DSS_DIR),
        "note": (
            "Stock LVTestCase voltages are already high (~1.03–1.05). "
            "Adding P raises remote V; Volt-Var absorbs Q and reduces overvoltage "
            "vs the same-kW MPPT unit. Absolute MAE vs no-PV baseline may still "
            "worsen on resistive LV unless Pmpp is small."
        ),
        "verification": verify,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
