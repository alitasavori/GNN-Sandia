#!/usr/bin/env python3
"""Regenerate colab_pf_data physics topology CSVs from full dailyagg.

Updates:
  - Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer.csv
    (adds ``node`` phase names for chunk-safe hetero mapping)
  - pf_balance_nodes_explicit.csv
    (all hetero_mv load_transformer nodes minus slack/interface; was 185 with
     Y-neighbor-only refinement through 2026-07, now ~1177)

Run from repo root after pulling dailyagg data locally:
  python regenerate_colab_pf_catalog.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import train_da_gps_multitask_complex_voltage_gine as pfmod

REPO = Path(__file__).resolve().parent
DATA_DAILYAGG = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
COLAB_PF = REPO / "colab_pf_data"
HETERO_REL = Path("Heterogenous GNN dataset") / "nodes" / "hetero_mv_nodes_load_transformer.csv"


def _master_index(colab_pf: Path) -> tuple[dict[str, int], int]:
    idx_path = colab_pf / "gnn_node_index_master.csv"
    if not idx_path.is_file():
        idx_path = DATA_DAILYAGG / "gnn_node_index_master.csv"
    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1
    return ntl, n_nodes


def regenerate_hetero_mv_nodes(*, dailyagg: Path, colab_pf: Path) -> int:
    src = dailyagg / HETERO_REL
    if not src.is_file():
        raise FileNotFoundError(f"missing dailyagg hetero catalog: {src}")
    het = pd.read_csv(src, usecols=["node", "node_idx"])
    het_u = het.drop_duplicates(subset=["node"]).sort_values("node_idx")
    out = colab_pf / HETERO_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    het_u.to_csv(out, index=False)
    return len(het_u)


def regenerate_balance_nodes_explicit(*, dailyagg: Path, colab_pf: Path) -> int:
    from build_pf_balance_nodes_explicit import build_expanded_balance_list, build_refined_balance_list

    expanded = build_expanded_balance_list(dailyagg)
    out = colab_pf / "pf_balance_nodes_explicit.csv"
    expanded.to_csv(out, index=False)
    refined = build_refined_balance_list(dailyagg=dailyagg, pf_root=colab_pf, expanded=expanded)
    refined_out = colab_pf / "pf_balance_nodes_refined.csv"
    refined.to_csv(refined_out, index=False)
    print(f"wrote {refined_out} ({len(refined)} refined balance nodes)")
    return len(expanded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dailyagg", type=Path, default=DATA_DAILYAGG)
    parser.add_argument("--colab-pf", type=Path, default=COLAB_PF)
    args = parser.parse_args()

    n_het = regenerate_hetero_mv_nodes(dailyagg=args.dailyagg, colab_pf=args.colab_pf)
    n_bal = regenerate_balance_nodes_explicit(dailyagg=args.dailyagg, colab_pf=args.colab_pf)
    print(f"wrote {args.colab_pf / HETERO_REL} ({n_het} nodes with node+node_idx)")
    print(f"wrote {args.colab_pf / 'pf_balance_nodes_explicit.csv'} ({n_bal} balance nodes)")


if __name__ == "__main__":
    main()
