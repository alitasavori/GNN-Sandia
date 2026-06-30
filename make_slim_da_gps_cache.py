#!/usr/bin/env python3
"""Build a SLIM single-sample DA-GPS tensor cache from a full training cache.

The DA-GPS daily-compare inference path
(``run_da_gps_daily_opendss_compare.run`` /
``run_da_gps_daily_voltages``) loads the *entire* chunk tensor cache via
``torch.load(cache_pt, weights_only=False)`` but then only ever consumes:

  * ``x[ref_sample_index]``  -> the (N, n_feat) feature *template* for one
    snapshot (dynamic P/Q/PV/BESS columns are overwritten per timestep from the
    load/PV/DER profiles; the static columns are kept as-is).
  * ``node_to_local``        -> busname.phase -> row index mapping.
  * ``meta_aux_cols``        -> PV aux-column name sanity check (``z.get(...)``).

It never reads ``y_ri`` / ``y_reg`` / ``y_cap`` / ``y_pv`` / ``sample_ids`` or
any other sample at inference. So a ~358 MB file is loaded just to slice out
~0.19 MB.

This util writes a slim cache that keeps the EXACT same dict keys and a leading
sample dimension of length 1 (design (a) in the task), so the existing loader
works unchanged with ``ref_sample_index=0``:

    x      -> (1, N, n_feat)
    y_ri   -> (1, N, 2)
    y_cap  -> (1, n_cap)
    y_reg  -> (1, n_reg)
    y_pv   -> (1, n_pv)
    sample_ids   -> [orig_sample_ids[ref]]
    node_to_local, reg_target_mode, meta_aux_cols  -> copied unchanged
    (any other top-level keys: tensors with matching leading sample dim are
     sliced to [ref:ref+1]; everything else is copied verbatim)

Usage::

    python make_slim_da_gps_cache.py \
        --in  "datasets_gnn2_from pc/run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt" \
        --ref 0 \
        --out "datasets_gnn2_from pc/run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = (
    REPO_ROOT
    / "datasets_gnn2_from pc"
    / "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "datasets_gnn2_from pc"
    / "run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt"
)


def build_slim_cache(in_pt: Path, out_pt: Path, ref: int) -> dict[str, object]:
    """Load ``in_pt`` (CPU), extract sample ``ref`` keeping a leading dim of 1, save to ``out_pt``."""
    in_pt = Path(in_pt).expanduser().resolve()
    out_pt = Path(out_pt).expanduser().resolve()
    if not in_pt.is_file():
        raise FileNotFoundError(f"input cache not found: {in_pt}")

    # Same load call as the loader: full pickle, CPU, weights_only=False.
    z = torch.load(in_pt, map_location="cpu", weights_only=False)
    if "x" not in z or "node_to_local" not in z:
        raise KeyError(f"{in_pt} is not a chunk tensor cache (missing 'x'/'node_to_local').")

    x = z["x"]
    n_samples = int(x.shape[0])
    if ref < 0 or ref >= n_samples:
        raise IndexError(f"ref={ref} out of range for x.shape[0]={n_samples}")

    slim: dict[str, object] = {}
    for k, v in z.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == n_samples:
            # Per-sample tensor: keep leading dim = 1 (clone so we don't hold the big buffer).
            slim[k] = v[ref : ref + 1].clone()
        elif k == "sample_ids" and isinstance(v, (list, tuple)) and len(v) == n_samples:
            slim[k] = [v[ref]]
        else:
            # node_to_local, reg_target_mode, meta_aux_cols, scalars, etc. -> copy verbatim.
            slim[k] = v

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_pt)

    in_mb = in_pt.stat().st_size / (1024 * 1024)
    out_mb = out_pt.stat().st_size / (1024 * 1024)
    print(f"[slim] source : {in_pt}  ({in_mb:.2f} MB, {n_samples} samples)")
    print(f"[slim] ref index extracted: {ref}  (sample_id={slim.get('sample_ids')})")
    print(f"[slim] x slice shape: {tuple(slim['x'].shape)}")
    for k in ("y_ri", "y_cap", "y_reg", "y_pv"):
        if k in slim and isinstance(slim[k], torch.Tensor):
            print(f"[slim] {k} slice shape: {tuple(slim[k].shape)}")
    print(f"[slim] meta_aux_cols: {slim.get('meta_aux_cols')!r}")
    print(f"[slim] reg_target_mode: {slim.get('reg_target_mode')!r}")
    print(f"[slim] output: {out_pt}  ({out_mb:.3f} MB)")
    return slim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_pt", type=str, default=str(DEFAULT_IN), help="full cache .pt")
    ap.add_argument("--out", dest="out_pt", type=str, default=str(DEFAULT_OUT), help="slim cache .pt to write")
    ap.add_argument("--ref", type=int, default=0, help="reference sample index to extract (default 0)")
    args = ap.parse_args()
    build_slim_cache(Path(args.in_pt), Path(args.out_pt), int(args.ref))


if __name__ == "__main__":
    main()
