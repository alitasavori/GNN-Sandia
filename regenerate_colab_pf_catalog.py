#!/usr/bin/env python3
"""Regenerate colab_pf_data physics topology CSVs from full dailyagg.

Updates:
  - Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer.csv
    (adds ``node`` phase names for chunk-safe hetero mapping)
  - pf_balance_nodes_explicit.csv
    (MV distance mask + interface/hetero/Y-neighbor refinement)

Run from repo root after pulling dailyagg data locally:
  python regenerate_colab_pf_catalog.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import train_da_gps_multitask_complex_voltage_gine as pfmod

REPO = Path(__file__).resolve().parent
DATA_DAILYAGG = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
COLAB_PF = REPO / "colab_pf_data"
HETERO_REL = Path("Heterogenous GNN dataset") / "nodes" / "hetero_mv_nodes_load_transformer.csv"
KV_BASE = 12.47
S_BASE_KVA = 5000.0
Z_BASE = (KV_BASE * 1000.0) ** 2 / (S_BASE_KVA * 1000.0)


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
    ntl, n_nodes = _master_index(colab_pf)
    dist_path = colab_pf / "electrical_distance_from_substation.csv"
    if not dist_path.is_file():
        dist_path = dailyagg / "electrical_distance_from_substation.csv"
    dist = pd.read_csv(dist_path)

    mask = torch.zeros(n_nodes, dtype=torch.bool)
    for _, row in dist.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl or pfmod._is_pf_slack_source_node(node):
            continue
        if float(row["electrical_distance_ohm"]) > 1e-9:
            mask[int(ntl[node])] = True

    edges = dailyagg / "gnn_edges_phase_static.csv"
    reg_cat = dailyagg / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
    reg_edges = pfmod._load_regulator_edges_for_pf(
        reg_cat, ntl, list(pfmod.TARGET_REG_COLS), Z_BASE
    )
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re, y_im = pfmod._build_ybus_siemens_from_edge_csv(
        edges, ntl, n_nodes, skip_undirected=skip
    )
    hetero_nodes = pfmod._load_pf_hetero_node_indices(colab_pf, ntl)
    if not hetero_nodes:
        hetero_nodes = pfmod._load_pf_hetero_node_indices(dailyagg, ntl)
    refined = pfmod._refine_pf_mv_balance_mask(
        mask,
        ntl,
        hetero_nodes,
        y_re,
        y_im,
        exclude_interface=True,
        hetero_y_neighbors_only=True,
    )

    idx_to_node = {int(li): str(node) for node, li in ntl.items()}
    rows: list[dict[str, str | int]] = []
    for li in range(n_nodes):
        if not bool(refined[li].item()):
            continue
        node = idx_to_node[int(li)]
        rows.append(
            {
                "node": node,
                "bus": str(node).strip().lower().split(".")[0],
                "node_idx": int(li),
            }
        )
    out = colab_pf / "pf_balance_nodes_explicit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


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
