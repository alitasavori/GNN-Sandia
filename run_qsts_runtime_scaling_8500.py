#!/usr/bin/env python3
"""Warm-started QSTS runtime scaling on IEEE 8500 (Method B / native daily).

Sweeps display resolutions and devices (cuda/cpu).
OpenDSS: one compile, sequential daily Solve() with state carry-forward.
DA-GPS: per-step forward from current inputs (no previous prediction).

Example:
  python -u run_qsts_runtime_scaling_8500.py --device cuda,cpu --smoke
  python -u run_qsts_runtime_scaling_8500.py --device cuda,cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    env = os.environ.get("GNN2_REPO_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parent


# Paper sweep: step_min must divide 60 (DailySimConfig). Coarse → fine.
RESOLUTIONS = (
    # (step_min, npts) — 24h day
    (60, 24),
    (30, 48),
    (20, 72),
    (15, 96),
    (10, 144),
    (5, 288),
    (2, 720),
)

# Match plots/presentation palette
COLOR_OPENDSS = "#4c78a8"
COLOR_DAGPS_CUDA = "#e45756"
COLOR_DAGPS_CPU = "#f58518"
FONT_FAMILY = "Times New Roman"


def _setup_paper_fonts() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    # Prefer true Times New Roman when present (Windows / Colab with mscorefonts).
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = FONT_FAMILY if FONT_FAMILY in available else "serif"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,  # TrueType in PDF (editable in Illustrator)
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _plot_scaling(rows: list[dict], out_pdf: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_fonts()

    by_dev: dict[str, list[dict]] = {}
    for r in rows:
        by_dev.setdefault(str(r["device"]), []).append(r)

    # Single-column-ish IEEE width; no title (LaTeX caption).
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    od_by_res: dict[int, list[float]] = {}
    for r in rows:
        od_by_res.setdefault(int(r["step_min"]), []).append(float(r["opendss_ms_per_eval"]))
    res_sorted = sorted(od_by_res)  # ascending; inverted → 60 left, 2 right
    od_ms = [float(np.mean(od_by_res[s])) for s in res_sorted]
    ax.plot(
        res_sorted,
        od_ms,
        "o-",
        color=COLOR_OPENDSS,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="OpenDSS",
        zorder=3,
    )

    style_by_dev = {
        "cuda": ("s-", COLOR_DAGPS_CUDA, "DA-GPS (GPU)"),
        "gpu": ("s-", COLOR_DAGPS_CUDA, "DA-GPS (GPU)"),
        "cpu": ("^-", COLOR_DAGPS_CPU, "DA-GPS (CPU)"),
    }
    for dev, rs in sorted(by_dev.items()):
        rs = sorted(rs, key=lambda x: int(x["step_min"]))
        xs = [int(r["step_min"]) for r in rs]
        ys = [float(r["dagps_ms_per_eval"]) for r in rs]
        fmt, color, label = style_by_dev.get(
            str(dev).lower(), ("D-", "#54a24b", f"DA-GPS ({dev})")
        )
        ax.plot(
            xs,
            ys,
            fmt,
            color=color,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=label,
            zorder=4,
        )

    ax.set_xlabel("Resolution (min)")
    ax.set_ylabel("Latency (ms/step)")
    ax.set_xticks(res_sorted)
    ax.set_xticklabels([str(s) for s in res_sorted])
    ax.invert_xaxis()  # finer resolution (smaller step) to the right
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", handlelength=1.8, borderaxespad=0.2)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)


def run_sweep(
    *,
    repo: Path,
    run_dir: Path,
    cache_pt: Path,
    checkpoint: Path,
    load_profile: Path,
    irr_profile: Path,
    devices: list[str],
    resolutions: list[tuple[int, int]],
    out_dir: Path,
    show_plots: bool = False,
) -> dict:
    os.environ.setdefault("GNN2_REPO_ROOT", str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from nonunique_da_gps_daily_compare import run_da_gps_daily_compare_and_plot
    from nonunique_opendss_daily import DailySimConfig

    import inspect

    if "skip_plots" not in inspect.signature(run_da_gps_daily_compare_and_plot).parameters:
        raise RuntimeError(
            "Stale nonunique_da_gps_daily_compare.py (missing skip_plots=). "
            "Run: cd /content/GNN-Sandia && git pull"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 72, flush=True)
    print("[qsts_runtime] Warm-started daily QSTS scaling (IEEE 8500)", flush=True)
    print(f"  run_dir={run_dir}", flush=True)
    print(f"  cache_pt={cache_pt}", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  devices={devices}", flush=True)
    print(f"  resolutions={resolutions}", flush=True)
    print("=" * 72, flush=True)

    for device in devices:
        for step_min, npts in resolutions:
            print("\n" + "-" * 72, flush=True)
            print(
                f"[qsts_runtime] device={device}  step_min={step_min}  npts={npts}",
                flush=True,
            )
            cfg = DailySimConfig(
                step_min=int(step_min),
                day_hours=24,
                include_der=False,
                include_da_gps=True,
                da_gps_run_dir=run_dir,
                da_gps_cache_pt=cache_pt,
                da_gps_checkpoint=checkpoint,
                da_gps_load_profile=load_profile,
                da_gps_pv_profile=irr_profile,
            )
            if int(cfg.npts) != int(npts):
                raise RuntimeError(
                    f"npts mismatch: cfg.npts={cfg.npts} vs requested={npts} "
                    f"(step_min={step_min})"
                )

            try:
                summ = run_da_gps_daily_compare_and_plot(
                    cfg,
                    show=False,
                    plot_all_cache_nodes=False,
                    skip_plots=True,
                    out_dir=out_dir / f"detail_{device}_step{step_min}_npts{npts}",
                    load_profile_path=load_profile,
                    pv_profile_path=irr_profile,
                    ref_sample_index=0,
                    scenario_scale=1.0,
                    daily_stress=0.0,
                    device=device,
                )
            except Exception:
                import traceback

                print(
                    f"[qsts_runtime] FAILED device={device} step_min={step_min} npts={npts}",
                    flush=True,
                )
                traceback.print_exc()
                raise
            tms = summ.get("timing_ms_per_ok_step") or {}
            od_ms = float(tms.get("dss_solve_ms", float("nan")))
            # Prefer forward-only; fall back to feature+forward then wall/npts
            gnn_ms = float(tms.get("gnn_forward_ms", float("nan")))
            if not np.isfinite(gnn_ms) or gnn_ms <= 0:
                gnn_ms = float(tms.get("gnn_feature_plus_forward_ms", float("nan")))
            if not np.isfinite(gnn_ms) or gnn_ms <= 0:
                deploy = summ.get("gnn_deployment_wall_s")
                if deploy is not None and float(deploy) > 0:
                    gnn_ms = 1000.0 * float(deploy) / max(int(npts), 1)

            od_wall = float(summ.get("dss_wall_s", float("nan")))
            gnn_wall = float(summ.get("gnn_deployment_wall_s") or summ.get("gnn_wall_s") or float("nan"))
            speedup = (
                od_ms / gnn_ms if np.isfinite(od_ms) and np.isfinite(gnn_ms) and gnn_ms > 0 else float("nan")
            )
            row = {
                "device": str(device),
                "step_min": int(step_min),
                "npts": int(npts),
                "opendss_ms_per_eval": od_ms,
                "dagps_ms_per_eval": gnn_ms,
                "opendss_batch_wall_s": od_wall,
                "dagps_deploy_wall_s": gnn_wall,
                "speedup_ms_per_eval": speedup,
                "wall_speedup": float(summ.get("wall_speedup", float("nan"))),
                "mae_pu": float(summ.get("overall_mae_pu", float("nan"))),
                "timing_ms_per_ok_step": tms,
                "speedup_detail": summ.get("speedup") or {},
            }
            rows.append(row)
            print(
                f"[qsts_runtime] RESULT device={device} npts={npts} step={step_min}min  "
                f"OD={od_ms:.3f} ms/eval  DA-GPS={gnn_ms:.3f} ms/eval  "
                f"speedup={speedup:.2f}x  MAE={row['mae_pu']:.6f} pu",
                flush=True,
            )

    summary = {
        "feeder": "8500",
        "mode": "warm_started_daily_qsts",
        "run_dir": str(run_dir),
        "cache_pt": str(cache_pt),
        "checkpoint": str(checkpoint),
        "load_profile": str(load_profile),
        "irr_profile": str(irr_profile),
        "devices": devices,
        "resolutions": [{"step_min": a, "npts": b} for a, b in resolutions],
        "rows": rows,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    jp = out_dir / "qsts_runtime_scaling_8500.json"
    jp.write_text(
        json.dumps(summary, indent=2, allow_nan=True, default=str),
        encoding="utf-8",
    )

    # CSV
    csv_path = out_dir / "qsts_runtime_scaling_8500.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "device,step_min,npts,opendss_ms_per_eval,dagps_ms_per_eval,"
            "speedup_ms_per_eval,wall_speedup,mae_pu,opendss_batch_wall_s,dagps_deploy_wall_s\n"
        )
        for r in rows:
            f.write(
                f"{r['device']},{r['step_min']},{r['npts']},"
                f"{r['opendss_ms_per_eval']:.6f},{r['dagps_ms_per_eval']:.6f},"
                f"{r['speedup_ms_per_eval']:.6f},{r['wall_speedup']:.6f},{r['mae_pu']:.8f},"
                f"{r['opendss_batch_wall_s']:.6f},{r['dagps_deploy_wall_s']:.6f}\n"
            )

    pdf = out_dir / "qsts_runtime_scaling.pdf"
    png = out_dir / "qsts_runtime_scaling.png"
    try:
        _plot_scaling(rows, pdf, png)
        print(f"[qsts_runtime] wrote figure {pdf}", flush=True)
    except Exception as e:
        print(f"[qsts_runtime] figure skipped: {e}", flush=True)

    print("\n=== QSTS warm-start scaling table ===", flush=True)
    print(
        f"{'dev':>5} | {'step':>4} | {'npts':>5} | {'OD ms':>10} | {'GNN ms':>10} | "
        f"{'speedup':>8} | {'MAE pu':>10}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['device']:>5} | {r['step_min']:4d} | {r['npts']:5d} | "
            f"{r['opendss_ms_per_eval']:10.3f} | {r['dagps_ms_per_eval']:10.3f} | "
            f"{r['speedup_ms_per_eval']:8.2f} | {r['mae_pu']:10.6f}",
            flush=True,
        )
    print(f"\n[qsts_runtime] wrote {jp}", flush=True)
    print(f"[qsts_runtime] wrote {csv_path}", flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Warm-started QSTS runtime scaling (IEEE 8500)")
    p.add_argument("--repo", type=str, default="")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--cache-pt", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--load-profile", type=str, default="")
    p.add_argument("--irr-profile", type=str, default="")
    p.add_argument("--device", type=str, default="cuda,cpu", help="Comma list: cuda,cpu")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--smoke", action="store_true", help="Only 60min/24 and 5min/288 on first device")
    p.add_argument("--show-plots", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo = Path(args.repo).expanduser().resolve() if str(args.repo).strip() else _repo_root()
    os.environ.setdefault("GNN2_REPO_ROOT", str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from nonunique_notebook_bootstrap import (
        resolve_cache_pt,
        resolve_feeder_checkpoint,
        resolve_feeder_run_dir,
    )

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if str(args.run_dir).strip()
        else resolve_feeder_run_dir(repo, "8500")
    )
    cache_pt = (
        Path(args.cache_pt).expanduser().resolve()
        if str(args.cache_pt).strip()
        else resolve_cache_pt(repo, "8500")
    )
    ckpt = (
        Path(args.checkpoint).expanduser().resolve()
        if str(args.checkpoint).strip()
        else resolve_feeder_checkpoint(run_dir)
    )

    day1 = repo / "a representativ days"
    load_p = (
        Path(args.load_profile).expanduser().resolve()
        if str(args.load_profile).strip()
        else day1 / "load_day_004.csv"
    )
    irr_p = (
        Path(args.irr_profile).expanduser().resolve()
        if str(args.irr_profile).strip()
        else day1 / "irr_day_004.csv"
    )

    devices = [d.strip().lower() for d in str(args.device).split(",") if d.strip()]
    devices = [("cuda" if d in ("gpu", "cuda") else d) for d in devices]
    if args.smoke:
        devices = devices[:1] or ["cuda"]
        resolutions = [(60, 24), (5, 288)]
    else:
        resolutions = list(RESOLUTIONS)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else repo / "qsts_runtime_scaling_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    run_sweep(
        repo=repo,
        run_dir=run_dir,
        cache_pt=cache_pt,
        checkpoint=ckpt,
        load_profile=load_p,
        irr_profile=irr_p,
        devices=devices,
        resolutions=resolutions,
        out_dir=out_dir,
        show_plots=bool(args.show_plots),
    )


if __name__ == "__main__":
    main()
