"""Export MV balance nodes for physics loss (offline CSV for Colab smoke).

Writes:
  - ``colab_pf_data/pf_balance_nodes_explicit.csv`` — full hetero MV load list (~1177)
  - ``colab_pf_data/pf_balance_nodes_refined.csv`` — Y-neighbor refined smoke list (~185)

Selection (expanded / explicit):
  All ``hetero_mv_nodes_load_transformer`` nodes on the master index, excluding
  slack/substation buses and MV interface buses (``regxfmr*``, ``190-*``, ``m/p/n*``).

Refined list (``--refined``):
  Expanded list filtered by ``electrical_distance_ohm > 0`` plus offline
  ``hetero_y_neighbors_only`` + interface exclusion (matches runtime refinement).
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
DEFAULT_OUT_REFINED = REPO / "colab_pf_data" / "pf_balance_nodes_refined.csv"
DEFAULT_PF_ROOT = REPO / "colab_pf_data"


def build_expanded_balance_list(
    data_root: Path,
) -> pd.DataFrame:
    idx_path = data_root / "gnn_node_index_master.csv"
    if not idx_path.is_file():
        idx_path = DEFAULT_PF_ROOT / "gnn_node_index_master.csv"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing gnn_node_index_master.csv under {data_root} or {DEFAULT_PF_ROOT}")

    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}

    hetero = pfmod._load_pf_hetero_node_indices(data_root, ntl)
    if not hetero:
        hetero = pfmod._load_pf_hetero_node_indices(DEFAULT_PF_ROOT, ntl)
    if not hetero:
        raise RuntimeError(
            f"No hetero_mv_nodes_load_transformer nodes under {data_root / pfmod._PF_HETERO_MV_NODES_REL}"
        )

    idx_to_node = {int(v): k for k, v in ntl.items()}
    rows: list[dict[str, object]] = []
    for li in sorted(hetero):
        node = idx_to_node[int(li)]
        if pfmod._is_pf_slack_source_node(node):
            continue
        if pfmod._is_pf_interface_node(node):
            continue
        rows.append(
            {
                "node_idx": int(li),
                "node": node,
                "bus": node.split(".")[0],
            }
        )
    out = pd.DataFrame(rows).sort_values(["bus", "node"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Expanded balance list is empty")
    return out


def build_refined_balance_list(
    *,
    data_root: Path,
    pf_root: Path,
    expanded: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Offline Y-neighbor + distance refinement (~185 nodes on full MV index)."""
    idx_path = pf_root / "gnn_node_index_master.csv"
    if not idx_path.is_file():
        idx_path = data_root / "gnn_node_index_master.csv"
    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1

    if expanded is None:
        expanded = build_expanded_balance_list(data_root)

    dist_path = pf_root / "electrical_distance_from_substation.csv"
    if not dist_path.is_file():
        dist_path = data_root / "electrical_distance_from_substation.csv"
    if not dist_path.is_file():
        raise FileNotFoundError(f"Missing electrical_distance_from_substation.csv under {pf_root}")

    dist = pd.read_csv(dist_path)
    dist_ok: set[int] = set()
    for _, row in dist.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl or pfmod._is_pf_slack_source_node(node):
            continue
        if float(row["electrical_distance_ohm"]) > 1e-9:
            dist_ok.add(int(ntl[node]))

    mask = torch.zeros(n_nodes, dtype=torch.bool)
    for _, row in expanded.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl:
            continue
        li = int(ntl[node])
        if li in dist_ok:
            mask[li] = True

    edges_path = data_root / "gnn_edges_phase_static.csv"
    if not edges_path.is_file():
        chunk_edges = sorted(data_root.glob("run_*/gnn_edges_phase_static.csv"))
        if chunk_edges:
            edges_path = chunk_edges[0]
    if not edges_path.is_file():
        chunk_edges = sorted((REPO / "datasets_gnn2_from pc").glob("**/gnn_edges_phase_static.csv"))
        edges_path = chunk_edges[0] if chunk_edges else edges_path

    reg_catalog = pf_root / "Heterogenous GNN dataset/edges/hetero_mv_edge_catalog.csv"
    reg_edges = pfmod._load_regulator_edges_for_pf(
        reg_catalog,
        ntl,
        list(pfmod.TARGET_REG_COLS),
        None,
    )
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re, y_im = pfmod._build_ybus_siemens_from_edge_csv(edges_path, ntl, n_nodes, skip_undirected=skip)
    args = argparse.Namespace(pf_exclude_interface_buses=True, pf_hetero_y_neighbors_only=True)
    mask = pfmod._refine_pf_mv_balance_mask(
        mask,
        ntl,
        pfmod._load_pf_hetero_node_indices(pf_root, ntl),
        y_re,
        y_im,
        exclude_interface=True,
        hetero_y_neighbors_only=True,
    )

    idx_to_node = {int(v): k for k, v in ntl.items()}
    rows: list[dict[str, object]] = []
    for li in sorted(int(i) for i in mask.nonzero(as_tuple=False).view(-1).tolist()):
        node = idx_to_node[int(li)]
        rows.append({"node_idx": int(li), "node": node, "bus": node.split(".")[0]})
    out = pd.DataFrame(rows).sort_values(["bus", "node"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Refined balance list is empty")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--pf_root", type=Path, default=DEFAULT_PF_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--refined",
        action="store_true",
        help="Write Y-neighbor refined list (~185 nodes) instead of full hetero list.",
    )
    p.add_argument(
        "--out_refined",
        type=Path,
        default=DEFAULT_OUT_REFINED,
        help="Output path when --refined (default: colab_pf_data/pf_balance_nodes_refined.csv).",
    )
    p.add_argument(
        "--write_both",
        action="store_true",
        help="Write both explicit (~1177) and refined (~185) CSVs.",
    )
    args = p.parse_args()
    data_root = args.data_root.resolve()
    pf_root = args.pf_root.resolve()

    expanded = build_expanded_balance_list(data_root)
    if args.write_both:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        expanded.to_csv(args.out, index=False)
        print(f"Wrote {len(expanded)} balance nodes -> {args.out}")
        refined = build_refined_balance_list(data_root=data_root, pf_root=pf_root, expanded=expanded)
        args.out_refined.parent.mkdir(parents=True, exist_ok=True)
        refined.to_csv(args.out_refined, index=False)
        print(f"Wrote {len(refined)} refined balance nodes -> {args.out_refined}")
        return

    if args.refined:
        df = build_refined_balance_list(data_root=data_root, pf_root=pf_root, expanded=expanded)
        out = args.out_refined
    else:
        df = expanded
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} balance nodes -> {out}")


if __name__ == "__main__":
    main()
