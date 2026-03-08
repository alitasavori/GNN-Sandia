"""
Summarize ORIGINAL dataset averages.

Computes:
- Per-node means of p_load_kw, q_load_kvar, p_pv_kw over all rows.
- Per-sample totals (sum over nodes) and their averages across samples.

Usage (from repo root):

  python summarize_original_dataset.py

Outputs:
- Prints global per-node means.
- Prints mean / std / min / max of per-sample totals.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    node_csv = os.path.join(base_dir, "datasets_gnn2", "original", "gnn_node_features_and_targets.csv")
    if not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {node_csv}. Run run_original_dataset.py first.")

    print(f"Reading ORIGINAL node table: {node_csv}")
    df = pd.read_csv(node_csv)

    # Ensure required columns exist and are numeric.
    cols = ["sample_id", "node_idx", "p_load_kw", "q_load_kvar", "p_pv_kw"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in ORIGINAL dataset: {missing}")

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).copy()

    print(f"Rows after cleaning: {len(df)}")
    n_samples = df["sample_id"].nunique()
    print(f"Unique samples: {n_samples}")

    # 1) Per-node global means (over all node rows in the dataset).
    print("\nPer-node means over all rows:")
    for c in ["p_load_kw", "q_load_kvar", "p_pv_kw"]:
        vals = df[c].to_numpy(dtype=float)
        print(f"  {c:11s}: mean={vals.mean():.6f}  std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}")

    # 2) Per-sample totals: sum over nodes for each sample_id.
    grouped = df.groupby("sample_id")[["p_load_kw", "q_load_kvar", "p_pv_kw"]].sum()
    print("\nPer-sample total P/Q (sum over nodes):")
    for c in ["p_load_kw", "q_load_kvar", "p_pv_kw"]:
        vals = grouped[c].to_numpy(dtype=float)
        print(f"  total_{c:11s}: mean={vals.mean():.6f}  std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}")


if __name__ == "__main__":
    main()

