"""
# Load-transformer rows × (regulator distance × tap)

For every row in `hetero_mv_nodes_load_transformer.csv` (each **sample_id** × **load node**),
append **12 columns** named like **FEEDER_REGA** … **VREG4_C** with:

`distance[node, regulator] × tap_pu[sample_id, regulator]`

- **distance** comes from `edges/load_electrical_distance_to_each_regulator.csv` (Ω; 0 if not downstream).
  If that file has **duplicate** `node` rows (same script that generated it may emit one row per
  load-transformer row), we **dedupe by `node`** (`keep="first"`) before joining.
- **tap** comes from `gnn_sample_meta.csv` (columns `reg_feeder_rega_tap_pu`, …).

```bash
python merge_load_transformer_reg_dist_times_tap.py ^
  --dataset-root "datasets_gnn2/loadtype_8500_dailyagg"
```

Default output: `Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_dist_x_tap.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

# Regulator display name -> gnn_sample_meta column (same as build_hetero_mv_edge_dataset.py)
REGULATOR_TO_TAP_COL: dict[str, str] = {
    "FEEDER_REGA": "reg_feeder_rega_tap_pu",
    "FEEDER_REGB": "reg_feeder_regb_tap_pu",
    "FEEDER_REGC": "reg_feeder_regc_tap_pu",
    "VREG2_A": "reg_vreg2_a_tap_pu",
    "VREG2_B": "reg_vreg2_b_tap_pu",
    "VREG2_C": "reg_vreg2_c_tap_pu",
    "VREG3_A": "reg_vreg3_a_tap_pu",
    "VREG3_B": "reg_vreg3_b_tap_pu",
    "VREG3_C": "reg_vreg3_c_tap_pu",
    "VREG4_A": "reg_vreg4_a_tap_pu",
    "VREG4_B": "reg_vreg4_b_tap_pu",
    "VREG4_C": "reg_vreg4_c_tap_pu",
}

REGULATOR_ORDER: tuple[str, ...] = tuple(REGULATOR_TO_TAP_COL.keys())


def _norm_sample_id(s: object) -> str:
    if s is None or s == "":
        return ""
    try:
        x = float(s)
        if x == int(x):
            return str(int(x))
        return str(x)
    except (TypeError, ValueError):
        return str(s).strip()


def run(
    load_csv: Path,
    dist_csv: Path,
    meta_csv: Path,
    out_csv: Path,
) -> Path:
    load_df = pd.read_csv(load_csv)
    dist_df = pd.read_csv(dist_csv)
    meta = pd.read_csv(meta_csv)

    for c in REGULATOR_ORDER:
        if c not in dist_df.columns:
            raise ValueError(f"Distance CSV missing column {c!r}: have {list(dist_df.columns)}")
    if "node" not in dist_df.columns:
        raise ValueError("Distance CSV must have a 'node' column")

    tap_cols = list(REGULATOR_TO_TAP_COL.values())
    missing_tap = [c for c in tap_cols if c not in meta.columns]
    if missing_tap:
        raise ValueError(f"gnn_sample_meta.csv missing tap columns: {missing_tap[:5]}...")

    if "sample_id" not in load_df.columns or "node" not in load_df.columns:
        raise ValueError("load transformer CSV must have sample_id and node")

    dist_df = dist_df.copy()
    dist_df["_nk"] = dist_df["node"].astype(str).str.strip().str.lower()
    n_before = len(dist_df)
    dist_df = dist_df.drop_duplicates(subset=["_nk"], keep="first")
    if len(dist_df) < n_before:
        print(
            f"[merge_load_transformer_reg_dist_times_tap] distance CSV: dropped "
            f"{n_before - len(dist_df)} duplicate node rows ({n_before} -> {len(dist_df)} unique nodes)"
        )
    dist_idx = dist_df.set_index("_nk")

    out = load_df.copy()
    nk = out["node"].astype(str).str.strip().str.lower()

    meta_k = meta.copy()
    meta_k["_sid"] = meta_k["sample_id"].map(_norm_sample_id)
    out["_sid"] = out["sample_id"].map(_norm_sample_id)

    tap_block = meta_k[["_sid"] + tap_cols].drop_duplicates(subset=["_sid"])
    out = out.merge(tap_block, on="_sid", how="left")

    for reg in REGULATOR_ORDER:
        tap_col = REGULATOR_TO_TAP_COL[reg]
        d = nk.map(dist_idx[reg]).fillna(0.0).astype(np.float64)
        t = pd.to_numeric(out[tap_col], errors="coerce").fillna(0.0)
        out[reg] = d * t

    drop_cols = [c for c in tap_cols if c in out.columns]
    out = out.drop(columns=drop_cols + ["_sid"], errors="ignore")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.resolve()}  rows={len(out)}  cols={len(out.columns)}")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge load transformer with reg_distance × tap from meta (12 FEEDER/VREG columns)."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO / "datasets_gnn2/loadtype_8500_dailyagg",
        help="Folder containing gnn_sample_meta.csv and Heterogenous GNN dataset/",
    )
    p.add_argument(
        "--load-csv",
        type=Path,
        default=None,
        help="Default: .../Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer.csv",
    )
    p.add_argument(
        "--dist-csv",
        type=Path,
        default=None,
        help="Default: .../edges/load_electrical_distance_to_each_regulator.csv",
    )
    p.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Default: dataset-root/gnn_sample_meta.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: .../nodes/hetero_mv_nodes_load_transformer_reg_dist_x_tap.csv",
    )
    args = p.parse_args()
    root = args.dataset_root.resolve()
    bundle = root / "Heterogenous GNN dataset"
    load_csv = (args.load_csv or bundle / "nodes" / "hetero_mv_nodes_load_transformer.csv").resolve()
    dist_csv = (args.dist_csv or bundle / "edges" / "load_electrical_distance_to_each_regulator.csv").resolve()
    meta_csv = (args.meta_csv or root / "gnn_sample_meta.csv").resolve()
    out_csv = (args.out_csv or bundle / "nodes" / "hetero_mv_nodes_load_transformer_reg_dist_x_tap.csv").resolve()

    run(load_csv, dist_csv, meta_csv, out_csv)


if __name__ == "__main__":
    main()
