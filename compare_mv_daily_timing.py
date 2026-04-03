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
        f"{gnn_forward_only_s_total:.4f}s | mean {_po(gnn_forward_only_s_total):.3f} ms/ok-step",
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
        "GNN timing note: on CUDA, ``torch.cuda.synchronize()`` runs before stopping the GNN bucket timer "
        "so async kernel time is included; on CPU there is no GPU sync.",
        flush=True,
    )
    print(f"Timesteps converged: {n_ok}/{npts}  (nonconv={n_nonconv})", flush=True)

    # --- Deployment comparison (shared apply on both sides; collect V excluded) ---
    dss_deploy = _po(open_apply_s_total + open_solve_only_s_total)
    gnn_deploy = _po(open_apply_s_total + feature_build_s_total + gnn_forward_only_s_total)
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
