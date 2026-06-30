"""
Unified wall-clock timing report for ``compare_homo_mv_daily`` and ``compare_hetero_mv_daily``.

Separates OpenDSS **reassert** (snapshot bookkeeping) from **Solve() only**, labels
voltage collection as benchmarking-only, and prints deployment / net speedup lines with
explicit formulas (shared apply cost; collect V excluded from speedup comparisons).

---------------------------------------------------------------------------
Inference device (CPU vs GPU)
---------------------------------------------------------------------------

Use :func:`resolve_inference_device` so runs can target **CPU** or **CUDA** explicitly.

- **Environment:** ``GNN_COMPARE_DEVICE`` = ``auto`` (default), ``cpu``, or ``cuda``
  (``gpu`` is accepted as an alias for ``cuda``). When unset, ``auto`` is used.
- **Python:** pass ``device="cpu"``, ``device="cuda"``, or ``device="auto"`` into
  ``run_compare`` / ``run_compare_homo`` / ``run_compare_juxtapose``; this overrides
  the environment variable when not ``None``.

``auto`` selects CUDA if :func:`torch.cuda.is_available` else CPU. Requesting ``cuda``
when CUDA is unavailable emits a warning and falls back to CPU.

---------------------------------------------------------------------------
Methodology: what each timer measures
---------------------------------------------------------------------------

All times use ``time.perf_counter()`` (monotonic wall time). Per-bucket totals are
summed over **converged** timesteps only for the **mean ms/ok-step** line; non-converged
steps skip OpenDSS voltage read and all GNN work, so they do not contribute to GNN or
post-solve OpenDSS buckets.

**OpenDSS — apply loads (DSS API only)**  
``set hour=… sec=…`` plus per-load ``kW`` / ``kvar`` writes. No Python aggregation of
loads onto buses (that was moved to feature build for a fair split).

**OpenDSS — snapshot reassert**  
``reassert_snapshot_before_each_solve()``: snapshot mode and related bookkeeping **before**
``Solve()``. This is benchmark overhead so the circuit stays in snapshot mode every
step; it is **not** part of raw solver time.

**OpenDSS — Solve() only**  
``dss.Solution.Solve()`` only — the cost a neural surrogate replaces in deployment
comparisons that exclude reassert.

**OpenDSS — collect V (benchmarking only)**  
Read solved voltages into NumPy for MAE vs GNN. Not part of a pure surrogate deployment.

**Feature build**  
Bus-phase P/Q dicts, hetero or homo feature arrays, capacitor reads from OpenDSS, and
(regulator tap reads for GINE) **before** tensor pack / ``model(…)``.

**GNN — forward only**  
The ``model(…)`` call(s) only, inside ``torch.no_grad()``.

**GNN — bucket (prep + …)**  
From building GPU/CPU tensors through scatter back to NumPy node voltages. On **CUDA**,
:func:`sync_inference_device` runs ``torch.cuda.synchronize()`` **after** the forward
so asynchronous kernel time is included in the bucket; on **CPU** there is no extra sync.

**Deployment vs net speedup (printed summary)**  
- *Deployment* includes shared **apply** on both DSS and GNN sides, excludes **collect V**.  
- *Net* excludes **apply** from both sides: compares **Solve() only** vs **feature + forward-only**.

See :func:`print_mv_daily_timing_summary` for the exact printed formulas.
"""

from __future__ import annotations

import os
import warnings


def per_ok_ms(total_s: float, n_ok: int) -> float:
    """Mean milliseconds per converged timestep."""
    return 1000.0 * total_s / max(n_ok, 1)


def resolve_inference_device(
    explicit: object | None = None,
    *,
    env_var: str = "GNN_COMPARE_DEVICE",
):
    """
    Choose ``torch.device`` for GNN inference: CPU or CUDA.

    Precedence: ``explicit`` (if not ``None``/empty) > ``os.environ[env_var]`` > ``"auto"``.

    ``explicit`` may be a ``torch.device`` or a string: ``auto``, ``cpu``, ``cuda``, ``gpu`` (alias for cuda).
    """
    import torch

    if isinstance(explicit, torch.device):
        if explicit.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                f"{env_var}: torch.device('cuda') requested but CUDA is not available; using CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return explicit

    raw: str | None = None
    if explicit is not None and str(explicit).strip():
        raw = str(explicit).strip()
    if not raw:
        raw = os.environ.get(env_var, "").strip()
    choice = (raw or "auto").lower()
    if choice in ("", "auto", "default"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"{env_var} or explicit device requested 'cuda' but CUDA is not available; using CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device("cuda")
    raise ValueError(
        f"Unknown device {raw!r}; use 'auto', 'cpu', or 'cuda' (env {env_var} or explicit string)."
    )


def sync_inference_device(device) -> None:
    """After GNN forward, wait for GPU work so wall-clock timers include kernel time (no-op on CPU)."""
    import torch

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()


def print_mv_daily_timing_summary(
    *,
    n_ok: int,
    npts: int,
    n_nonconv: int,
    open_apply_s_total: float,
    open_reassert_s_total: float,
    open_solve_only_s_total: float,
    open_get_s_total: float,
    feature_build_s_total: float,
    gnn_forward_only_s_total: float,
    gnn_bucket_s_total: float,
    device: str,
    title: str = "Daily Timing Summary",
    feature_label: str = "Feature build",
    log_prefix: str = "[compare_mv_daily_timing]",
    gnn_forward_only_parts: tuple[float, float] | None = None,
    gnn_forward_only_part_labels: tuple[str, str] = ("model A", "model B"),
    gnn_setup_once_s: float | None = None,
    gnn_per_step_s: float | None = None,
    gnn_total_wall_s: float | None = None,
) -> None:
    """
    Print full breakdown + deployment / net speedup sections.

    **Buckets**
    - *apply*: DSS ``set hour/sec`` + per-load ``kW``/``kvar`` only (no Python bus aggregation).
    - *reassert*: ``reassert_snapshot_before_each_solve()`` (mode / caps bookkeeping for this benchmark).
    - *solve only*: ``dss.Solution.Solve()`` only.
    - *collect V*: ground-truth voltages for MAE — benchmarking only, not deployment.
    - *feature*: bus-phase aggregation + hetero/homo feature tensors (includes work moved out of apply).
    - *GNN forward-only*: ``model(...)`` only; *GNN bucket* adds tensor prep, norm, numpy, scatter.
    """
    def _po(total: float) -> float:
        return per_ok_ms(total, n_ok)

    print(f"\n{log_prefix} Wall-clock breakdown:", flush=True)
    print(f"=== {title} ===", flush=True)
    print(
        "OpenDSS apply loads (DSS API only): total "
        f"{open_apply_s_total:.4f}s | mean {_po(open_apply_s_total):.3f} ms/ok-step",
        flush=True,
    )
    print(
        "OpenDSS snapshot reassert overhead: total "
        f"{open_reassert_s_total:.4f}s | mean {_po(open_reassert_s_total):.3f} ms/ok-step  "
        "(``set mode=snapshot``, caps, ``Solution.Mode`` — benchmark bookkeeping, not ``Solve()``)",
        flush=True,
    )
    print(
        "OpenDSS Solve() only: total "
        f"{open_solve_only_s_total:.4f}s | mean {_po(open_solve_only_s_total):.3f} ms/ok-step  "
        "(what a surrogate replaces in deployment)",
        flush=True,
    )
    print(
        "OpenDSS collect V mag (benchmarking only): total "
        f"{open_get_s_total:.4f}s | mean {_po(open_get_s_total):.3f} ms/ok-step  "
        "(ground truth for MAE — not run in a pure GNN deployment)",
        flush=True,
    )
    print(
        f"{feature_label}: total "
        f"{feature_build_s_total:.4f}s | mean {_po(feature_build_s_total):.3f} ms/ok-step  "
        "(includes bus-phase aggregation + model inputs)",
        flush=True,
    )
    print(
        "GNN model forward only: total "
        f"{gnn_forward_only_s_total:.4f}s | mean {_po(gnn_forward_only_s_total):.3f} ms/ok-step  "
        "(CUDA: ``torch.cuda.synchronize()`` immediately after ``model(...)``)",
        flush=True,
    )
    if gnn_forward_only_parts is not None:
        pa, pb = gnn_forward_only_parts
        la, lb = gnn_forward_only_part_labels
        print(
            f"  split ({la} + {lb}): {_po(pa):.3f} + {_po(pb):.3f} = {_po(gnn_forward_only_s_total):.3f} ms/ok-step",
            flush=True,
        )
    print(
        "GNN bucket (prep+norm+to_numpy+scatter): total "
        f"{gnn_bucket_s_total:.4f}s | mean {_po(gnn_bucket_s_total):.3f} ms/ok-step",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    print(
        "GNN bucket note: includes H2D feature copy, norm, forward, denorm, scatter/D2H; on CUDA the bucket "
        "timer also syncs after the step unless ``GNN_DEFER_D2H=1`` (single bulk copy at end).",
        flush=True,
    )
    if gnn_setup_once_s is not None:
        print(
            f"GNN setup once (model load + static tables + graph capture): {gnn_setup_once_s:.4f}s",
            flush=True,
        )
    if gnn_per_step_s is not None:
        print(
            f"GNN per-step wall (feature apply + infer bucket, amortized): "
            f"{1000.0 * gnn_per_step_s:.3f} ms/ok-step",
            flush=True,
        )
    if gnn_total_wall_s is not None:
        print(
            f"GNN total wall (setup once + all steps): {gnn_total_wall_s:.4f}s",
            flush=True,
        )
    print(f"Timesteps converged: {n_ok}/{npts}  (nonconv={n_nonconv})", flush=True)

    # --- Deployment comparison (shared apply on both sides; collect V excluded) ---
    dss_deploy = _po(open_apply_s_total + open_solve_only_s_total)
    gnn_deploy = _po(open_apply_s_total + feature_build_s_total + gnn_forward_only_s_total)
    if gnn_setup_once_s is not None and gnn_total_wall_s is not None and n_ok > 0:
        gnn_setup = float(gnn_setup_once_s)
        gnn_per = float(gnn_per_step_s or 0.0)
        gnn_total = float(gnn_total_wall_s)
        gnn_full_day_ms = 1000.0 * (gnn_setup + gnn_per * n_ok) / max(n_ok, 1)
        dss_solve_total = float(open_solve_only_s_total)
        dss_solve_per = dss_solve_total / max(n_ok, 1)
        print("\n=== Deployment wall (setup once + converged steps) ===", flush=True)
        print(
            f"{log_prefix} DA-GPS deployment wall: "
            f"{gnn_setup:.4f}s + {n_ok} × {gnn_per:.4f}s = {gnn_total:.4f}s  "
            f"(gnn_setup_once_s + n_ok × gnn_per_step_s = gnn_total_wall_s)",
            flush=True,
        )
        print(
            f"{log_prefix} OpenDSS Solve() wall: "
            f"{n_ok} × {dss_solve_per:.4f}s = {dss_solve_total:.4f}s  "
            f"(n_ok × mean_Solve_per_step = dss_solve_only_s_total; compile-once not timed)",
            flush=True,
        )
        print(
            f"{log_prefix} DA-GPS amortized mean incl. setup: {gnn_full_day_ms:.3f} ms/ok-step "
            f"(setup/{n_ok} steps spread over converged steps)",
            flush=True,
        )
    print("\n=== Deployment comparison (shared apply included; collect V excluded) ===", flush=True)
    print(
        f"DSS deployment cost (apply + Solve() only):     {dss_deploy:.3f} ms/ok-step",
        flush=True,
    )
    print(
        f"GNN deployment cost (apply + feature + fwd-only): {gnn_deploy:.3f} ms/ok-step",
        flush=True,
    )
    if gnn_deploy > 0:
        print(f"Speedup (DSS_deploy / GNN_deploy):               {dss_deploy / gnn_deploy:.2f}×", flush=True)
    print(
        f"Note: collect V mean {_po(open_get_s_total):.1f} ms/ok-step is excluded — benchmarking only.",
        flush=True,
    )

    # --- Net speedup (apply excluded from both sides) ---
    dss_net = _po(open_solve_only_s_total)
    gnn_net = _po(feature_build_s_total + gnn_forward_only_s_total)
    print("\n=== Net speedup (shared apply excluded from both sides) ===", flush=True)
    print(
        "Formula:  DSS_net = Solve() only;  GNN_net = feature build + GNN forward-only.",
        flush=True,
    )
    print(f"DSS net (Solve() only):           {dss_net:.3f} ms/ok-step", flush=True)
    print(f"GNN net (feature + fwd-only):     {gnn_net:.3f} ms/ok-step", flush=True)
    if gnn_net > 0:
        print(f"Net speedup (DSS_net / GNN_net):  {dss_net / gnn_net:.2f}×", flush=True)


def amortize_gnn_timing_to_display_grid(
    *,
    display_npts: int,
    display_step_min: int,
    internal_npts: int,
    internal_step_min: int,
    gnn_setup_once_s: float | None,
    gnn_per_step_s: float | None,
    gnn_total_wall_s: float | None,
    gnn_n_ok: int | None,
) -> dict[str, float | int | bool]:
    """Map native GNN step timers onto the user-facing ``(npts, step_min)`` grid.

    The DA-GPS model is trained at 288×5 min; coarser OpenDSS / plot grids keep the
    same total wall time but spread step work over fewer displayed timesteps.
    """
    reported_npts = max(1, int(display_npts))
    internal_n = max(1, int(internal_npts))
    setup = float(gnn_setup_once_s or 0.0)
    internal_n_ok = max(1, int(gnn_n_ok if gnn_n_ok is not None else internal_n))
    if gnn_total_wall_s is not None:
        total = float(gnn_total_wall_s)
    elif gnn_per_step_s is not None:
        total = setup + float(gnn_per_step_s) * internal_n_ok
    else:
        total = setup
    step_wall_total = max(0.0, total - setup)
    per_step_reported = step_wall_total / reported_npts
    resampled = not (
        int(display_step_min) == int(internal_step_min) and reported_npts == internal_n
    )
    return {
        "display_npts": reported_npts,
        "display_step_min": int(display_step_min),
        "internal_npts": internal_n,
        "internal_step_min": int(internal_step_min),
        "resampled": resampled,
        "gnn_setup_once_s": setup,
        "gnn_per_step_s": per_step_reported,
        "gnn_total_wall_s": total,
        "n_ok": reported_npts,
        "internal_n_ok": internal_n_ok,
        "internal_per_step_s": step_wall_total / internal_n_ok,
    }


def format_gnn_grid_log(
    timing: dict[str, float | int | bool],
    *,
    prefix: str = "[da_gps_daily_compare]",
) -> str:
    """One-line description of native vs displayed GNN timestep grids."""
    if timing.get("resampled"):
        return (
            f"{prefix} DA-GPS GNN: internal {timing['internal_npts']} @ "
            f"{timing['internal_step_min']} min, overlay resampled to "
            f"{timing['display_npts']} @ {timing['display_step_min']} min"
        )
    if int(timing.get("internal_npts", 0)) == int(timing.get("display_npts", 0)) and int(
        timing.get("internal_step_min", 0)
    ) == int(timing.get("display_step_min", 0)):
        return (
            f"{prefix} DA-GPS GNN: {timing['display_npts']} forwards @ "
            f"{timing['display_step_min']} min (display grid; matches OpenDSS)"
        )
    return (
        f"{prefix} DA-GPS GNN: {timing['display_npts']} steps @ "
        f"{timing['display_step_min']} min (native grid)"
    )
