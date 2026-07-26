"""Method A latency: OpenDSS Solve() vs PowerFlowMultiNet batch-1 forward.

Paper-style timing (same buckets as DA-GPS Method A):
  - OpenDSS: ``Solve()`` only per 5-min snapshot step (apply / reassert excluded from
    the primary ``dss_solve_ms`` line; still reported for completeness).
  - PFMN: ``model(...)`` only, ``torch.no_grad()``, CUDA sync after forward when on GPU.
  - Oracle taps/caps come from a PFMN pack sample (not live DSS reads) — latency-only.

Usage (Colab / local)::

  python -u run_pfmn_method_a_timing.py \\
    --feeder ieee34 \\
    --run-dir /path/to/pfmn_oracle_ieee34_l12_h128_... \\
    --cache-dir /path/to/pfmn_chunked_ieee34_oracle_v2 \\
    --device auto --npts 288 --out-dir ./pfmn_method_a_timing/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import opendssdirect as dss
import torch

from build_powerflowmultinet_graph import EDGE_CONT_IDX, NODE_CONT_IDX, materialize_edge_attr
from compare_mv_daily_timing import (
    compute_mv_daily_timing_metrics,
    print_mv_daily_timing_summary,
)
from compare_opendss_snapshot_helpers import (
    apply_explicit_loads_and_pv_pmpp,
    prepare_parity_profiles,
    reassert_snapshot_and_set_clock,
    setup_da_gps_snapshot_opendss,
    step_irradiance_multiplier,
    step_load_multiplier,
)
from nonunique_opendss_daily import resolve_da_gps_device
from powerflowmultinet_model import PowerFlowMultiNet
from run_da_gps_daily_opendss_compare import (
    _align_method_a_load_bases_for_feeder,
    _compile_feeder_master,
    _default_feeder_profiles,
    _discover_pv_system_names,
    _ieee34_training_aligned_pv_bases,
    _read_pv_base_pmpp_kw,
    _resolve_profile_csv_path,
    normalize_feeder,
)
from train_powerflowmultinet import _CACHE_LEGACY_SUFFIXES, _CACHE_SUFFIX, _normalize_pack_node_feats

try:
    import run_daily_aggregate_dataset_8500 as rd8500
except Exception:  # pragma: no cover
    rd8500 = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _find_pack(cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"--cache-dir is not a directory: {cache_dir}")
    for suf in (_CACHE_SUFFIX, *_CACHE_LEGACY_SUFFIXES):
        hits = sorted(cache_dir.glob(f"run_001*{suf}"))
        if hits:
            return hits[0]
        hits = sorted(cache_dir.glob(f"*{suf}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No PFMN pack (*{_CACHE_SUFFIX} or legacy) under {cache_dir}"
    )


def _resolve_pfmn_checkpoint(run_dir: Path, explicit: str = "") -> Path:
    if str(explicit).strip():
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    best = run_dir / "pfmn_oracle_best.pt"
    last = run_dir / "training_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"No pfmn_oracle_best.pt / training_last.pt in {run_dir}")


def _build_sample_tensors(pack: dict, norms: dict, sample_i: int, device: torch.device):
    x = pack["x"][sample_i].clone()
    for j in NODE_CONT_IDX:
        x[:, j] = (x[:, j] - norms["x_mean"][j]) / norms["x_std"][j]
    ea = materialize_edge_attr(
        pack["edge_attr_static"], pack["edge_tap_reg_idx"], pack["reg_taps"][sample_i]
    )
    for j in EDGE_CONT_IDX:
        ea[:, j] = (ea[:, j] - norms["e_mean"][j]) / norms["e_std"][j]
    ds = pack["device_state"][sample_i]
    if ds.dim() == 1:
        ds = ds.unsqueeze(0)
    ei = pack["edge_index"]
    return (
        x.to(device),
        ei.to(device),
        ea.to(device),
        ds.to(device),
    )


def run_pfmn_method_a(
    *,
    feeder: str,
    run_dir: Path,
    cache_dir: Path,
    checkpoint: Path | None = None,
    device: str = "auto",
    npts: int = 288,
    step_min: float = 5.0,
    scenario_scale: float = 1.0,
    daily_stress: float = 0.0,
    load_profile_path: str = "",
    load_profile_filename: str = "5minDayShape.csv",
    pv_irradiance_profile_path: str = "",
    pv_irradiance_filename: str = "irr_day_001.csv",
    out_dir: Path | None = None,
    warmup: int = 3,
) -> dict:
    feeder_key = normalize_feeder(feeder)
    run_dir = Path(run_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    ckpt_path = _resolve_pfmn_checkpoint(run_dir, str(checkpoint or ""))
    pack_path = _find_pack(cache_dir)

    if out_dir is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / f"pfmn_method_a_{feeder_key}_timing_runs" / tag
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_s = resolve_da_gps_device(device)
    dev = torch.device(dev_s)
    print(f"[pfmn_method_a] feeder={feeder_key} device={dev} ckpt={ckpt_path.name}", flush=True)
    print(f"[pfmn_method_a] pack={pack_path.name}", flush=True)

    bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    norms_raw = bundle.get("norms") or {}
    if not norms_raw:
        raise RuntimeError(f"Checkpoint missing norms: {ckpt_path}")
    norms = {k: v.float() if torch.is_tensor(v) else torch.as_tensor(v).float() for k, v in norms_raw.items()}

    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    pack = _normalize_pack_node_feats(pack, src_name=pack_path.name)
    n_samp = int(pack["x"].shape[0])
    node_dim = int(bundle.get("node_dim", pack.get("node_dim", pack["x"].shape[-1])))
    edge_dim = int(bundle.get("edge_dim", pack.get("edge_dim", pack["edge_attr_static"].shape[-1])))
    state_dim = int(bundle.get("state_dim", pack.get("state_dim", pack["device_state"].shape[-1])))
    hidden = int(bundle.get("hidden", 128))
    layers = int(bundle.get("layers", 12))
    dropout = float(bundle.get("dropout", 0.0))

    model = PowerFlowMultiNet(
        node_dim,
        edge_dim,
        state_dim,
        hidden=hidden,
        num_layers=layers,
        dropout=0.0,  # inference
        predict_substation=True,
    )
    state = bundle.get("model_state_dict") or bundle.get("state_dict")
    if state is None:
        raise RuntimeError(f"No model_state_dict in {ckpt_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(dev)

    # Warmup a few pack samples (build on the fly — 8500 packs are large).
    sample_ids = [i % n_samp for i in range(int(npts))]
    with torch.no_grad():
        for k in range(min(int(warmup), len(sample_ids))):
            x, ei, ea, ds = _build_sample_tensors(pack, norms, sample_ids[k], dev)
            _ = model(x, ei, ea, ds, batch=None)
            _sync(dev)

    compile_meta = _compile_feeder_master(feeder_key)
    print(
        f"[pfmn_method_a] OpenDSS compile: {compile_meta['master_dss']} "
        f"(cwd {compile_meta['model_dir']})",
        flush=True,
    )
    fb_load, fb_irr = _default_feeder_profiles(
        feeder_key, out_dir=out_dir, npts=int(npts), step_min=float(step_min)
    )
    if str(pv_irradiance_profile_path or "").strip():
        irr_csv = _resolve_profile_csv_path(
            pv_irradiance_profile_path,
            default_if_dir=str(pv_irradiance_filename),
            fallback_file=fb_irr,
        )
    else:
        irr_csv = fb_irr
    if str(load_profile_path or "").strip():
        prof_resolved = _resolve_profile_csv_path(
            load_profile_path,
            default_if_dir=str(load_profile_filename),
            fallback_file=fb_load,
        )
    else:
        prof_resolved = fb_load

    parity = prepare_parity_profiles(
        prof_resolved,
        irr_csv,
        npts=int(npts),
        step_min=float(step_min),
        daily_stress=float(daily_stress),
        stress_clip_lo=0.05,
        stress_clip_hi=3.0,
    )
    m_eff, m_irr = parity.m_eff, parity.m_irr
    setup_da_gps_snapshot_opendss(npts=int(npts), step_min=float(step_min))

    if rd8500 is None:
        raise RuntimeError("run_daily_aggregate_dataset_8500 import failed")
    loads, _ = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    base_kw, base_kvar, align_meta = _align_method_a_load_bases_for_feeder(
        feeder_key, base_names=base_names, base_kw=base_kw, base_kvar=base_kvar
    )
    print(
        f"[pfmn_method_a] load alignment feeder={feeder_key}: "
        f"physical_P_kw≈{align_meta['physical_P_kw']:.4g} ratio={align_meta['ratio_p']:.4g}",
        flush=True,
    )
    pv_names = _discover_pv_system_names()
    pv_base = _read_pv_base_pmpp_kw(pv_names)
    if feeder_key == "ieee34":
        pv_base = _ieee34_training_aligned_pv_bases(pv_base)

    open_apply_s = 0.0
    open_reassert_s = 0.0
    open_solve_s = 0.0
    open_get_s = 0.0
    pfmn_fwd_s = 0.0
    feature_s = 0.0  # pack features are prebuilt (oracle) — 0 for Method A forward-only
    n_ok = 0
    n_nonconv = 0

    for i in range(int(npts)):
        m_t = step_load_multiplier(m_eff, i, scenario_scale)
        ir_t = step_irradiance_multiplier(m_irr, i)

        t0 = time.perf_counter()
        apply_explicit_loads_and_pv_pmpp(
            base_names=base_names,
            base_kw=base_kw,
            base_kvar=base_kvar,
            m_t=m_t,
            pv_names=pv_names,
            pv_base_pmpp_kw=pv_base,
            ir_t=ir_t,
        )
        open_apply_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        reassert_snapshot_and_set_clock(i, step_min=float(step_min))
        open_reassert_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        dss.Solution.Solve()
        open_solve_s += time.perf_counter() - t0

        if not dss.Solution.Converged():
            n_nonconv += 1
            continue
        n_ok += 1

        # Collect V only to keep OpenDSS wall similar; not used for MAE here.
        t0 = time.perf_counter()
        try:
            _ = dss.Circuit.AllBusVmagPu()
        except Exception:
            pass
        open_get_s += time.perf_counter() - t0

        x, ei, ea, ds = _build_sample_tensors(pack, norms, sample_ids[i], dev)
        with torch.no_grad():
            t0 = time.perf_counter()
            _ = model(x, ei, ea, ds, batch=None)
            _sync(dev)
            pfmn_fwd_s += time.perf_counter() - t0

    print_mv_daily_timing_summary(
        n_ok=n_ok,
        npts=int(npts),
        n_nonconv=n_nonconv,
        open_apply_s_total=open_apply_s,
        open_reassert_s_total=open_reassert_s,
        open_solve_only_s_total=open_solve_s,
        open_get_s_total=open_get_s,
        feature_build_s_total=feature_s,
        gnn_forward_only_s_total=pfmn_fwd_s,
        gnn_bucket_s_total=pfmn_fwd_s,
        device=str(dev),
        title=f"Daily Timing Summary (PFMN oracle vs OpenDSS) — {feeder_key}",
        feature_label="PFMN feature build (prebuilt pack / oracle; not timed)",
        log_prefix="[pfmn_method_a]",
    )

    metrics = compute_mv_daily_timing_metrics(
        n_ok=n_ok,
        open_apply_s_total=open_apply_s,
        open_solve_only_s_total=open_solve_s,
        open_get_s_total=open_get_s,
        feature_build_s_total=feature_s,
        gnn_forward_only_s_total=pfmn_fwd_s,
    )
    report = {
        "feeder": feeder_key,
        "device": str(dev),
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "pack": str(pack_path),
        "npts": int(npts),
        "n_ok": int(n_ok),
        "n_nonconv": int(n_nonconv),
        "timing_ms_per_ok_step": {
            "dss_solve_ms": metrics["dss_solve_ms"],
            "pfmn_forward_ms": metrics["gnn_forward_ms"],
            "dss_apply_ms": metrics["dss_apply_ms"],
            "dss_collect_ms": metrics["dss_collect_ms"],
            "ratio_od_solve_over_pfmn_fwd": (
                float(metrics["dss_solve_ms"]) / max(float(metrics["gnn_forward_ms"]), 1e-12)
            ),
        },
        "method": "Method A: OpenDSS Solve() vs PFMN forward-only (oracle pack states)",
        "hidden": hidden,
        "layers": layers,
    }
    out_json = out_dir / "pfmn_method_a_timing.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pfmn_method_a] wrote {out_json}", flush=True)
    print(
        f"[pfmn_method_a] PAPER LINE feeder={feeder_key} device={dev} | "
        f"OD Solve={metrics['dss_solve_ms']:.3f} ms | "
        f"PFMN fwd={metrics['gnn_forward_ms']:.3f} ms | "
        f"ratio={report['timing_ms_per_ok_step']['ratio_od_solve_over_pfmn_fwd']:.3f}",
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="PFMN Method A timing vs OpenDSS Solve()")
    p.add_argument("--feeder", required=True, choices=("8500", "ieee34", "906", "ieee8500", "34"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-dir", required=True, help="Folder of *__pfmn_oracle_v*.pt packs")
    p.add_argument("--checkpoint", default="", help="Optional explicit .pt (default: best then last)")
    p.add_argument("--device", default="auto")
    p.add_argument("--npts", type=int, default=288)
    p.add_argument("--step-min", type=float, default=5.0)
    p.add_argument("--scenario-scale", type=float, default=1.0)
    p.add_argument("--daily-stress", type=float, default=0.0)
    p.add_argument("--load-profile-path", default="")
    p.add_argument("--load-profile-filename", default="5minDayShape.csv")
    p.add_argument("--pv-irradiance-profile-path", default="")
    p.add_argument("--pv-irradiance-filename", default="irr_day_001.csv")
    p.add_argument("--out-dir", default="")
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    run_pfmn_method_a(
        feeder=args.feeder,
        run_dir=Path(args.run_dir),
        cache_dir=Path(args.cache_dir),
        checkpoint=Path(args.checkpoint) if str(args.checkpoint).strip() else None,
        device=args.device,
        npts=int(args.npts),
        step_min=float(args.step_min),
        scenario_scale=float(args.scenario_scale),
        daily_stress=float(args.daily_stress),
        load_profile_path=args.load_profile_path,
        load_profile_filename=args.load_profile_filename,
        pv_irradiance_profile_path=args.pv_irradiance_profile_path,
        pv_irradiance_filename=args.pv_irradiance_filename,
        out_dir=Path(args.out_dir) if str(args.out_dir).strip() else None,
        warmup=int(args.warmup),
    )


if __name__ == "__main__":
    main()
