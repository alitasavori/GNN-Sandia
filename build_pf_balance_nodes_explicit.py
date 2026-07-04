"""Export refined MV balance nodes for physics loss (offline CSV for Colab smoke).

Writes ``colab_pf_data/pf_balance_nodes_explicit.csv`` with hetero load nodes only
(excludes slack, interface buses, and Y couplings to non-hetero neighbors).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import train_da_gps_multitask_complex_voltage_gine as pfmod

REPO = Path(__file__).resolve().parent
DEFAULT_DATA = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
DEFAULT_OUT = REPO / "colab_pf_data" / "pf_balance_nodes_explicit.csv"


def build_refined_balance_list(
    data_root: Path,
    *,
    s_base_kva: float = 5000.0,
) -> pd.DataFrame:
    idx_path = data_root / "gnn_node_index_master.csv"
    edges_path = data_root / "gnn_edges_phase_static.csv"
    dist_path = data_root / "electrical_distance_from_substation.csv"
    for p in (idx_path, edges_path, dist_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1

    reg_edges = pfmod._load_regulator_edges_for_pf(
        data_root / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv",
        ntl,
        list(pfmod.TARGET_REG_COLS),
        None,
    )
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re, y_im = pfmod._build_ybus_siemens_from_edge_csv(
        edges_path, ntl, n_nodes, skip_undirected=skip
    )

    mask = torch.zeros(n_nodes, dtype=torch.bool)
    dist = pd.read_csv(dist_path)
    for _, row in dist.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl or pfmod._is_pf_slack_source_node(node):
            continue
        if float(row["electrical_distance_ohm"]) > 1e-9:
            mask[int(ntl[node])] = True

    hetero = pfmod._load_pf_hetero_node_indices(data_root, ntl)
    refined = pfmod._refine_pf_mv_balance_mask(
        mask,
        ntl,
        hetero,
        y_re,
        y_im,
        exclude_interface=True,
        hetero_y_neighbors_only=True,
    )

    idx_to_node = {int(v): k for k, v in ntl.items()}
    rows: list[dict[str, object]] = []
    for li in range(n_nodes):
        if not bool(refined[li].item()):
            continue
        node = idx_to_node[int(li)]
        rows.append(
            {
                "node_idx": int(li),
                "node": node,
                "bus": node.split(".")[0],
            }
        )
    out = pd.DataFrame(rows).sort_values(["bus", "node"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Refined balance mask is empty")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    df = build_refined_balance_list(args.data_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} balance nodes -> {args.out}")


if __name__ == "__main__":
    main()
