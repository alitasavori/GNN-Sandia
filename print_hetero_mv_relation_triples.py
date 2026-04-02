"""
Print all (src_type, relation, dst_type) triples in the MV hetero graph, with directed edge counts.

Why more than 2 keys?
  - Physically there are 2 relations: "line" and "reg".
  - HeteroConv keys are (src_storage, relation, dst_storage). With 4 node storages,
    endpoints can be load↔load, load↔downstream, upstream↔downstream, etc.
  - Each undirected physical edge becomes two directed triples unless tu==tv and symmetric.

Usage (from repo root):
  python print_hetero_mv_relation_triples.py
  python print_hetero_mv_relation_triples.py --dataset-dir "path/to/Heterogenous GNN dataset"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

# Topology helpers do not need torch_geometric; the search module imports it at load time.
_tg_nn = ModuleType("torch_geometric.nn")
for _name in ("GATConv", "GINEConv", "HeteroConv", "SAGEConv"):
    setattr(_tg_nn, _name, type(_name, (), {}))
sys.modules.setdefault("torch_geometric.nn", _tg_nn)
sys.modules.setdefault("torch_geometric", ModuleType("torch_geometric"))

import search_hetero_mv_gnn_architectures as hm

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

DEFAULT_DATASET = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg" / "Heterogenous GNN dataset"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--node-index",
        type=Path,
        default=REPO / "datasets_gnn2" / "loadtype_8500_dailyagg" / "gnn_node_index_master.csv",
    )
    args = p.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    nodes_dir = dataset_dir / "nodes"
    edges_dir = dataset_dir / "edges"

    name_to_gidx = hm._read_node_idx_master(args.node_index.resolve())
    extra_names: set[str] = set()
    for fn in hm.NODE_FILES.values():
        df = pd.read_csv(nodes_dir / fn, usecols=["node"])
        extra_names.update(df["node"].astype(str).str.strip().tolist())

    catalog = pd.read_csv(edges_dir / "hetero_mv_edge_catalog.csv")
    line_attr = pd.read_csv(edges_dir / "hetero_mv_line_edge_attr.csv")
    g_list = hm._collect_global_node_indices(catalog, name_to_gidx, extra_names)
    membership = hm._membership_by_csv(nodes_dir)
    _g2l, _gt, counts, edge_index_dict_cpu, _ea = hm._build_typed_topology(
        catalog, line_attr, g_list, membership
    )

    print("node_type counts:", dict(counts))
    print(f"total distinct triple keys (directed): {len(edge_index_dict_cpu)}")
    print()
    print(f"{'src':<12} {'rel':<6} {'dst':<12} {'#edges (directed)':>18}")
    print("-" * 52)
    for key in sorted(edge_index_dict_cpu.keys(), key=lambda k: (k[1], k[0], k[2])):
        s, r, t = key
        n_e = int(edge_index_dict_cpu[key].shape[1])
        print(f"{s:<12} {r:<6} {t:<12} {n_e:>18}")
    print()
    n_line = sum(edge_index_dict_cpu[k].shape[1] for k in edge_index_dict_cpu if k[1] == "line")
    n_reg = sum(edge_index_dict_cpu[k].shape[1] for k in edge_index_dict_cpu if k[1] == "reg")
    print(f"sum of directed line edges (all triples): {n_line}")
    print(f"sum of directed reg edges (all triples):   {n_reg}")
    print("(undirected pair counts are roughly half if every edge is bidirected.)")


if __name__ == "__main__":
    main()
