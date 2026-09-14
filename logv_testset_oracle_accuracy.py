"""Log(v) |V|/angle accuracy on the GNN test split with pregiven (oracle) controllers.

Same protocol as FINAL.ipynb cell 63's Log(v) column, plus voltage metrics:

  1. Reset controllers to the fixedctrlinit start.
  2. Apply that sample's load/PV injections.
  3. OpenDSS ``ControlMode=Static`` ``Solve()`` (controllers settle).
  4. Copy settled taps / caps / P/Q into Log(v); rebuild ``A`` / refresh the
     factor only if the device fingerprint changed (untimed).
  5. One ``FastLogvSolver.solve(P, Q)``.
  6. Score |V| and angle vs that same live OpenDSS solution.

Log(v) does **not** run RegControl. Residual error is linearization, not missed taps.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREFERRED_DATA_ID = "1H5pVM6GkzBDbt8KgIpdy8ow8A0Xks4F3"

CASES: dict[str, dict[str, Any]] = {
    "ieee34": {
        "chunk_parent": "original_ieee34_draft_yzip_v3_m4fix_H_fixedzip_chunked",
        "chunk_csv_fallback_parent": "original_ieee34_mirzaei_chunked_fixedctrlinit",
        "cache_dir": "da_gps_chunked_ieee34_draft_yzip_fixedzip_full_gine",
        "dss_masters": [
            "new dss from dr mirzaei/IEEE34_PV.dss",
            "IEEE34/IEEE34_PV.dss",
        ],
        "meta_csv": "gnn_sample_meta.csv",
    },
    "906": {
        "chunk_parent": "original_906_lvtestcase_pv_voltvar_fixedctrlinit_yedges",
        "chunk_csv_fallback_parent": "original_906_lvtestcase_pv_voltvar_fixedctrlinit",
        "cache_dir": "da_gps_chunked_906_fixedctrlinit_draft_yedges_pqp_full_gine",
        "dss_masters": [
            "906 bus system/LVTestCase_PV_voltvar/Master_snapshot_PV_voltvar.dss",
            "906 bus system/LVTestCase_PV_voltvar/Master.dss",
            "906 bus system/OpenDSS-master/OpenDSS-master/Distrib/IEEETestCases/LVTestCase/Master.dss",
        ],
        "meta_csv": "gnn_sample_meta.csv",
    },
    "8500": {
        "chunk_parent": "original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40_fixedctrlinit_yedges",
        "chunk_csv_fallback_parent": "original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40_fixedctrlinit",
        "cache_dir": "da_gps_chunked_mvagg_fixedctrlinit_yedges_full_gine",
        "dss_masters": [
            "8500 nodes with solar unbalanced/Master-PV2MW-inv.dss",
            "8500-node/Master.dss",
        ],
        "meta_csv": "gnn_sample_meta.csv",
    },
}


def _chunk_permutation_split(
    n: int,
    *,
    split_seed: int,
    chunk_idx: int,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(split_seed) + int(chunk_idx) * 100_003)
    perm = rng.permutation(n)
    n_train = int(n * float(train_frac))
    n_val = int(n * float(val_frac))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Invalid train/val/test split for chunk index {chunk_idx} (n={n}).")
    return (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val :],
    )


def data_roots(repo: Path) -> list[Path]:
    preferred = Path(f"/content/drive/.shortcut-targets-by-id/{PREFERRED_DATA_ID}/datasets_gnn2")
    ordered: list[Path] = [
        preferred,
        Path(f"K:/.shortcut-targets-by-id/{PREFERRED_DATA_ID}/datasets_gnn2"),
        Path(f"G:/.shortcut-targets-by-id/{PREFERRED_DATA_ID}/datasets_gnn2"),
        Path("/content/drive/MyDrive/datasets_gnn2"),
        Path(r"K:\My Drive\datasets_gnn2"),
        Path(r"H:\My Drive\datasets_gnn2"),
        repo / "datasets_gnn2",
        repo / "datasets_gnn2_from pc",
    ]
    shortcut = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut.is_dir():
        ordered.extend(sorted(shortcut.glob("*/datasets_gnn2")))
    try:
        from nonunique_notebook_bootstrap import _datasets_gnn2_roots

        ordered.extend(_datasets_gnn2_roots(repo))
    except Exception:
        pass
    out: list[Path] = []
    seen: set[Path] = set()
    for r in ordered:
        try:
            if not r.is_dir():
                continue
            rr = r.resolve()
        except Exception:
            continue
        if rr not in seen:
            seen.add(rr)
            out.append(rr)
    return out


def _fuse_wake(path: Path) -> None:
    try:
        if path.is_dir():
            _ = sorted(path.iterdir())[:8]
        elif path.parent.is_dir():
            _ = sorted(path.parent.iterdir())[:8]
    except Exception:
        pass


def chunk_parent(repo: Path, spec: dict) -> Path:
    names = [str(spec["chunk_parent"])]
    fb = str(spec.get("chunk_csv_fallback_parent") or "").strip()
    if fb and fb not in names:
        names.append(fb)
    tried: list[str] = []
    for root in data_roots(repo):
        for name in names:
            p = root / name
            tried.append(str(p))
            _fuse_wake(p)
            if p.is_dir() and any(p.glob("run_*")):
                return p.resolve()
    raise FileNotFoundError("chunk parent not found:\n  " + "\n  ".join(tried[:12]))


def cache_dir(repo: Path, spec: dict) -> Path | None:
    name = str(spec.get("cache_dir") or "")
    if not name:
        return None
    for root in data_roots(repo):
        p = root / "cache" / name
        if p.is_dir():
            return p.resolve()
    return None


def _cache_for_chunk(cdir: Path | None, chunk_name: str) -> Path | None:
    if cdir is None:
        return None
    hits = sorted(cdir.glob(f"{chunk_name}__*.pt")) or sorted(cdir.glob(f"{chunk_name}*.pt"))
    return hits[-1].resolve() if hits else None


def resolve_meta_csv(repo: Path, spec: dict, chunk_name: str) -> Path:
    meta_name = str(spec.get("meta_csv", "gnn_sample_meta.csv"))
    names = [str(spec["chunk_parent"])]
    fb = str(spec.get("chunk_csv_fallback_parent") or "").strip()
    if fb and fb not in names:
        names.append(fb)
    tried: list[str] = []
    for root in data_roots(repo):
        for parent_name in names:
            cand = root / parent_name / chunk_name
            meta_p = cand / meta_name
            tried.append(str(meta_p))
            _fuse_wake(cand)
            if meta_p.is_file():
                return meta_p.resolve()
            _fuse_wake(cand)
            if meta_p.is_file():
                return meta_p.resolve()
    raise FileNotFoundError(f"Missing {meta_name} for {chunk_name} (tried {len(tried)} paths)")


def sample_ids_for_split(meta_path: Path, cache_pt: Path | None) -> list[int]:
    if cache_pt is not None and cache_pt.is_file():
        try:
            import torch

            z = torch.load(cache_pt, map_location="cpu", weights_only=False)
            return [int(s) for s in z["sample_ids"]]
        except Exception:
            pass
    meta = pd.read_csv(meta_path, usecols=["sample_id"])
    return sorted({int(s) for s in meta["sample_id"].tolist()})


def dss_master(repo: Path, spec: dict) -> Path:
    for rel in spec["dss_masters"]:
        p = repo / rel
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("DSS master not found:\n  " + "\n  ".join(spec["dss_masters"]))


def _ensure_906_daily_profiles(repo: Path, model_dir: Path) -> None:
    dest = model_dir / "Daily_1min_100profiles"
    probe = dest / "load_profile_1.txt"
    if probe.is_file():
        return
    src = (
        repo
        / "906 bus system"
        / "OpenDSS-master"
        / "OpenDSS-master"
        / "Distrib"
        / "IEEETestCases"
        / "LVTestCase"
        / "Daily_1min_100profiles"
    )
    if not (src / "load_profile_1.txt").is_file():
        raise FileNotFoundError(f"Missing stock profiles: {src}")
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        try:
            if not any(dest.iterdir()):
                dest.rmdir()
        except OSError:
            pass
    if dest.exists():
        raise FileNotFoundError(f"{dest} exists but has no load_profile_1.txt")
    dest.symlink_to(src.resolve(), target_is_directory=True)
    print(f"[diag] linked {dest} -> {src.resolve()}", flush=True)

    bc_lo = model_dir / "buscoords.txt"
    if not bc_lo.is_file():
        for cand in ("Buscoords.txt", "BusCoords.txt", "BUSCOORDS.TXT"):
            hi = model_dir / cand
            if hi.is_file():
                try:
                    bc_lo.symlink_to(hi.name)
                except OSError:
                    import shutil

                    shutil.copy2(hi, bc_lo)
                break


def compile_dss(repo: Path, master: Path, *, feeder: str | None = None) -> None:
    import opendssdirect as dss
    from compare_opendss_snapshot_helpers import setup_da_gps_snapshot_opendss

    if feeder == "906" or "LVTestCase_PV_voltvar" in str(master).replace("\\", "/"):
        _ensure_906_daily_profiles(repo, master.parent)

    dss.Basic.ClearAll()
    dss.Text.Command(f'cd "{master.parent}"')
    dss.Text.Command(f'redirect "{master}"')
    setup_da_gps_snapshot_opendss(npts=288, step_min=5.0)
    try:
        dss.Text.Command("Set Mode=Snapshot")
        dss.Text.Command("Set ControlMode=Static")
        dss.Text.Command("Set MaxControlIter=120")
        dss.Text.Command("Set MaxIterations=100")
    except Exception:
        pass


def reset_fixed_init(feeder: str) -> None:
    if feeder in ("8500", "ieee34"):
        import run_original_style_dataset_8500_unbalanced as ds8500

        ds8500._reset_controllers_to_compile_defaults()
    if feeder == "906":
        try:
            import run_original_style_dataset_906_lvtestcase as ds906

            if hasattr(ds906, "_reset_pv_invcontrol_to_pf1"):
                ds906._reset_pv_invcontrol_to_pf1()
            elif hasattr(ds906, "reset_invcontrol_pv_to_pf1"):
                ds906.reset_invcontrol_pv_to_pf1()
        except Exception:
            pass
        try:
            from logv3lpf_daily_demo import reset_invcontrol_pv_to_pf1

            reset_invcontrol_pv_to_pf1()
        except Exception:
            pass
    if feeder == "ieee34":
        try:
            import run_original_style_dataset_ieee34_mirzaei as ds34

            if hasattr(ds34, "_reset_pv_invcontrol_to_pf1"):
                ds34._reset_pv_invcontrol_to_pf1()
        except Exception:
            pass


def apply_sample_injections(meta_row: pd.Series, bases: dict) -> None:
    import opendssdirect as dss

    def _f(key: str, default: float = float("nan")) -> float:
        v = meta_row.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    p_time = _f("P_load_time_kw")
    q_time = _f("Q_load_time_kvar")
    m_load = _f("m_loadshape", 1.0)
    if not np.isfinite(p_time):
        p_scen = _f("P_load_total_kw")
        p_time = (p_scen * m_load) if np.isfinite(p_scen) else (float(bases["sum_kw"]) * m_load)
    if not np.isfinite(q_time):
        q_scen = _f("Q_load_total_kvar")
        q_time = (q_scen * m_load) if np.isfinite(q_scen) else (float(bases["sum_kvar"]) * m_load)

    sum_kw = float(bases["sum_kw"])
    sum_kvar = float(bases["sum_kvar"])
    s_p = (p_time / sum_kw) if sum_kw > 1e-12 else 1.0
    s_q = (q_time / sum_kvar) if sum_kvar > 1e-12 else s_p

    kw_set = np.asarray(bases["kw"], dtype=float) * float(s_p)
    kvar_set = np.asarray(bases["kvar"], dtype=float) * float(s_q)
    for j, name in enumerate(bases["names"]):
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw_set[j]))
        dss.Loads.kvar(float(kvar_set[j]))

    m_irr = _f("m_irradshape", 0.0)
    p_pv_time = _f("P_pv_time_kw")
    if not np.isfinite(p_pv_time):
        p_pv_scen = _f("P_pv_total_kw", 0.0)
        p_pv_time = (p_pv_scen * m_irr) if np.isfinite(p_pv_scen) else 0.0

    pv_names = list(bases["pv_names"])
    pv_base = bases["pv_base"]
    base_pmpp_sum = float(sum(float(pv_base.get(str(n).strip(), 0.0)) for n in pv_names))
    if base_pmpp_sum > 1e-12 and np.isfinite(p_pv_time):
        pmpp_scale = float(p_pv_time) / base_pmpp_sum
    else:
        pmpp_scale = float(m_irr) if np.isfinite(m_irr) else 0.0

    for pv_nm in pv_names:
        b0 = float(pv_base.get(str(pv_nm).strip(), 0.0))
        if b0 <= 0.0:
            continue
        try:
            dss.PVsystems.Name(str(pv_nm).strip())
            dss.PVsystems.Pmpp(float(b0) * float(pmpp_scale))
        except Exception:
            pass

    t_idx = meta_row.get("t_index", None)
    if t_idx is not None and np.isfinite(float(t_idx)):
        ti = int(t_idx)
        hr = int(ti // 12)
        sec = int((ti % 12) * 300)
        try:
            dss.Text.Command(f"set hour={hr} sec={sec}")
        except Exception:
            pass


def collect_load_bases() -> dict:
    from compare_opendss_snapshot_helpers import (
        collect_unscaled_load_bases,
        discover_pv_system_names,
        read_pv_base_pmpp_kw,
    )

    names, kw, kvar = collect_unscaled_load_bases()
    pv_names = discover_pv_system_names()
    pv_base = read_pv_base_pmpp_kw(pv_names)
    return {
        "names": names,
        "kw": np.asarray(kw, dtype=float),
        "kvar": np.asarray(kvar, dtype=float),
        "sum_kw": float(np.sum(kw)),
        "sum_kvar": float(np.sum(kvar)),
        "pv_names": pv_names,
        "pv_base": pv_base,
    }


def init_fastlogv(repo: Path, feeder: str, master: Path):
    import importlib

    import opendssdirect as dss
    from compare_opendss_snapshot_helpers import setup_da_gps_snapshot_opendss
    from logv3lpf_daily_demo import FEEDERS as LOGV_FEEDERS
    from logv3lpf_daily_demo import _quiet, compile_cmd, ensure_logv3lpf, resolve_feeder

    import logv_fast_solver as _lfs

    importlib.reload(_lfs)
    FastLogvSolver = _lfs.FastLogvSolver
    ensure_logv3lpf(repo)
    import logv3lpf
    from logv3lpf.DSSParser import DSScase

    key = resolve_feeder(feeder)
    meta = LOGV_FEEDERS[key]
    print("  [logv] building FastLogvSolver (once) ...", flush=True)
    t0 = time.perf_counter()
    with _quiet():
        case = logv3lpf.case(
            compile_cmd(master),
            meta["sourcebus"],
            float(meta["refvm"]),
            0,
        )
    setup_da_gps_snapshot_opendss(npts=288, step_min=5.0)
    try:
        dss.Text.Command("Set Mode=Snapshot")
        dss.Text.Command("Set ControlMode=Static")
        dss.Text.Command("Set MaxControlIter=120")
        dss.Text.Command("Set MaxIterations=100")
        dss.Solution.Solve()
        DSScase.process_openDSS_solution(case)
    except Exception:
        pass
    solver = FastLogvSolver(case)
    try:
        solver.solve(solver.P0, solver.Q0)
    except Exception:
        pass
    print(
        f"  [logv] ready in {time.perf_counter() - t0:.1f}s  "
        f"n_cp={solver.n_cp} has_H={bool(solver.has_H)} k={int(solver.k)}",
        flush=True,
    )
    return case, solver


def _mean_finite(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    m = np.isfinite(arr)
    return float(np.mean(arr[m])) if m.any() else float("nan")


def evaluate_feeder(
    repo: Path,
    feeder: str,
    *,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    max_samples: int | None = None,
) -> dict[str, Any]:
    import opendssdirect as dss
    from logv3lpf.DSSParser import DSScase
    from logv3lpf_daily_demo import paper_accuracy_metrics
    from logv3lpf_daily_opendss_compare import (
        _apply_fast_solution_to_case,
        _cap_fingerprint,
        _rebuild_logv_A_from_case_taps,
        _sync_loads_pv_caps_taps_from_opendss,
        _tap_fingerprint,
    )

    spec = CASES[feeder]
    master = dss_master(repo, spec)
    parent = chunk_parent(repo, spec)
    cdir = cache_dir(repo, spec)
    print(f"  DSS master={master}", flush=True)
    print(f"  CHUNK_PARENT={parent}", flush=True)

    compile_dss(repo, master, feeder=feeder)
    case, fast_solver = init_fastlogv(repo, feeder, master)
    bases = collect_load_bases()

    chunks = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not chunks:
        raise FileNotFoundError(f"No run_* under {parent}")

    rows: list[dict[str, Any]] = []
    n_seen = n_ok = n_fail = n_fail_nonconv = n_fail_ctrl = n_fail_exc = 0
    n_logv_fail = n_refresh = 0
    prev_tap_fp = None
    prev_cap_fp = None

    for ci, ch in enumerate(chunks):
        meta_path = resolve_meta_csv(repo, spec, ch.name)
        cache_pt = _cache_for_chunk(cdir, ch.name)
        sample_ids = sample_ids_for_split(meta_path, cache_pt)
        n = len(sample_ids)
        _tr, _va, idx_te = _chunk_permutation_split(
            n, split_seed=int(seed), chunk_idx=int(ci), train_frac=float(train_frac), val_frac=float(val_frac)
        )
        idx_te = np.asarray(idx_te, dtype=int)
        if max_samples is not None:
            remain = int(max_samples) - n_seen
            if remain <= 0:
                break
            idx_te = idx_te[:remain]

        meta = pd.read_csv(meta_path)
        meta["sample_id"] = meta["sample_id"].astype(int)
        meta_by = meta.set_index("sample_id", drop=False)

        for li in idx_te:
            sid = int(sample_ids[int(li)])
            n_seen += 1
            if sid not in meta_by.index:
                n_fail += 1
                n_fail_exc += 1
                continue
            row = meta_by.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            try:
                apply_sample_injections(row, bases)
                reset_fixed_init(feeder)
                try:
                    dss.Text.Command("Set ControlMode=Static")
                    dss.Text.Command("Set MaxControlIter=120")
                except Exception:
                    pass
                dss.Solution.Solve()
                converged = bool(dss.Solution.Converged())
                try:
                    n_ctrl = int(dss.Solution.ControlIterations())
                except Exception:
                    n_ctrl = -1
                if not (converged and n_ctrl < 120):
                    n_fail += 1
                    if converged and n_ctrl >= 120:
                        n_fail_ctrl += 1
                    else:
                        n_fail_nonconv += 1
                    continue

                _sync_loads_pv_caps_taps_from_opendss(case, sync_taps=True)
                tap_fp = _tap_fingerprint(case)
                cap_fp = _cap_fingerprint(case)
                need_refresh = (tap_fp != prev_tap_fp) or (cap_fp != prev_cap_fp)
                if tap_fp != prev_tap_fp:
                    _rebuild_logv_A_from_case_taps(case)
                    prev_tap_fp = tap_fp
                if cap_fp != prev_cap_fp:
                    prev_cap_fp = cap_fp
                if need_refresh:
                    fast_solver.refresh_zip_cap_and_factor(case)
                    n_refresh += 1
                    try:
                        fast_solver.solve(fast_solver.P0, fast_solver.Q0)
                    except Exception:
                        pass
                P = np.asarray(
                    [float(case.loads.kW[int(j)]) for j in fast_solver.cp_idx],
                    dtype=float,
                )
                Q = np.asarray(
                    [float(case.loads.kvar[int(j)]) for j in fast_solver.cp_idx],
                    dtype=float,
                )
                vm_f, va_f = fast_solver.solve(P, Q)
                DSScase.process_openDSS_solution(case)
                _apply_fast_solution_to_case(case, vm_f, va_f)
                m = paper_accuracy_metrics(case, exclude_source=False)
                rows.append(
                    {
                        "feeder": feeder,
                        "chunk": ch.name,
                        "sample_id": sid,
                        "n_ctrl_iter": int(n_ctrl),
                        "n_phase_nodes": int(m.get("n_phase_nodes", 0)),
                        "mae_vm_pu": float(m.get("vm_mae_pu", float("nan"))),
                        "rmse_vm_pu": float(m.get("vm_rmse_pu", float("nan"))),
                        "mae_va_deg": float(m.get("va_mae_deg", float("nan"))),
                        "rmse_va_deg": float(m.get("va_rmse_deg", float("nan"))),
                    }
                )
                n_ok += 1
            except Exception:
                n_fail += 1
                n_logv_fail += 1
                n_fail_exc += 1
            if (n_seen % 10) == 0:
                mae_so = _mean_finite([r["mae_vm_pu"] for r in rows])
                ang_so = _mean_finite([r["mae_va_deg"] for r in rows])
                print(
                    f"  [{feeder}] {n_seen} samples  ok={n_ok} fail={n_fail}  "
                    f"mean |V| MAE={mae_so:.6f} pu  mean ang MAE={ang_so:.4f} deg",
                    flush=True,
                )
        if max_samples is not None and n_seen >= int(max_samples):
            break

    df = pd.DataFrame(rows)
    out = {
        "feeder": feeder,
        "chunk_parent": str(parent),
        "dss_master": str(master),
        "n_chunks": int(len(chunks)),
        "n_test_attempted": int(n_seen),
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_fail_nonconv": int(n_fail_nonconv),
        "n_fail_ctrl": int(n_fail_ctrl),
        "n_fail_exc": int(n_fail_exc),
        "n_logv_fail": int(n_logv_fail),
        "n_A_or_cap_refresh": int(n_refresh),
        "mean_mae_vm_pu": _mean_finite([r["mae_vm_pu"] for r in rows]),
        "mean_rmse_vm_pu": _mean_finite([r["rmse_vm_pu"] for r in rows]),
        "mean_mae_va_deg": _mean_finite([r["mae_va_deg"] for r in rows]),
        "mean_rmse_va_deg": _mean_finite([r["rmse_va_deg"] for r in rows]),
        "median_mae_vm_pu": float(np.nanmedian(df["mae_vm_pu"])) if len(df) else float("nan"),
        "median_mae_va_deg": float(np.nanmedian(df["mae_va_deg"])) if len(df) else float("nan"),
    }
    return {"summary": out, "samples": df}


def run_all(
    repo: Path,
    *,
    feeders: list[str] | None = None,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    smoke: bool = False,
    max_test_samples: int | None = None,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    repo = Path(repo).resolve()
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "Log-v-3LPF"))

    feeders = list(feeders or ["ieee34", "906", "8500"])
    max_n = 8 if smoke else max_test_samples
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) if out_dir is not None else (repo / "outputs" / "logv3lpf_testset_oracle" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Test-set Log(v) voltage (oracle / pregiven controllers) ===")
    print(f"REPO={repo}")
    print(f"split seed={seed} train/val={train_frac}/{val_frac}  SMOKE={smoke} MAX_TEST={max_n}")
    print("Log(v) does not run RegControl; taps/caps copied from OpenDSS Static Solve.")
    print(f"OUT={out_dir}")
    print("=" * 78)

    summaries: list[dict[str, Any]] = []
    sample_tables: dict[str, pd.DataFrame] = {}
    for feeder in feeders:
        print("\n" + "#" * 78)
        print(f"FEEDER={feeder}", flush=True)
        try:
            pack = evaluate_feeder(
                repo,
                feeder,
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                max_samples=max_n,
            )
        except Exception as e:
            print(f"  SKIP — {e}", flush=True)
            summaries.append({"feeder": feeder, "error": str(e)[:400]})
            continue
        summ = pack["summary"]
        df = pack["samples"]
        summaries.append(summ)
        sample_tables[feeder] = df
        csv_p = out_dir / f"{feeder}_per_sample.csv"
        df.to_csv(csv_p, index=False)
        print(
            f"  mean |V| MAE  : {summ['mean_mae_vm_pu']:.6f} pu   "
            f"(median {summ['median_mae_vm_pu']:.6f})",
            flush=True,
        )
        print(
            f"  mean angle MAE: {summ['mean_mae_va_deg']:.6f} deg  "
            f"(median {summ['median_mae_va_deg']:.6f})",
            flush=True,
        )
        print(f"  ok={summ['n_ok']}/{summ['n_test_attempted']}  wrote {csv_p}", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 78)
    print("=== SUMMARY (mean over TEST samples, Log(v) vs live OpenDSS, oracle taps/caps) ===")
    cols = [
        "feeder",
        "n_ok",
        "n_test_attempted",
        "mean_mae_vm_pu",
        "mean_mae_va_deg",
        "mean_rmse_vm_pu",
        "mean_rmse_va_deg",
    ]
    show = [c for c in cols if c in summary_df.columns]
    if show:
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(summary_df[show].to_string(index=False))
    print(f"\nWrote {summary_csv}")
    return summary_df, sample_tables


if __name__ == "__main__":
    root = Path(os.environ.get("GNN2_REPO_ROOT", "")).expanduser()
    if not root.is_dir():
        root = Path(__file__).resolve().parent
    run_all(root, smoke="--smoke" in sys.argv)
