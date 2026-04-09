"""
Build line edges that connect only load-type MV nodes by compacting series paths.

Use case:
  - Start from hetero_mv_edge_catalog.csv (line + regulator rows).
  - Keep only line rows.
  - Keep only load-type nodes from hetero_mv_nodes_load_transformer_reg_tap_only.csv.
  - If two load nodes are connected through intermediate non-load nodes in a simple
    series path (degree-2 chain), collapse that path into one edge and sum R_full/X_full.

This preserves direct load-load line edges and contracts "in-between" non-load nodes
such as capacitor/regulator-related nodes when they sit on a simple chain.

Output:
  CSV with one row per compacted undirected load-load edge:
    from_node,to_node,edge_type,R_full,X_full,u_idx,v_idx,
    num_original_segments,internal_nodes,path_nodes,path_edges
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _node_to_idx_from_load_csv(load_nodes_csv: Path) -> dict[str, int]:
    df = pd.read_csv(load_nodes_csv, usecols=["sample_id", "node", "node_idx"])
    first_sid = df["sample_id"].iloc[0]
    first = df[df["sample_id"] == first_sid][["node", "node_idx"]].drop_duplicates()
    out: dict[str, int] = {}
    for _, r in first.iterrows():
        out[str(r["node"]).strip()] = int(r["node_idx"])
    return out


def _load_load_nodes(load_nodes_csv: Path) -> set[str]:
    df = pd.read_csv(load_nodes_csv, usecols=["node"])
    return set(df["node"].astype(str).str.strip().unique().tolist())


def _build_line_graph(catalog_csv: Path) -> tuple[dict[int, dict], dict[str, list[int]]]:
    df = pd.read_csv(catalog_csv)
    if "edge_type" not in df.columns:
        raise ValueError(f"{catalog_csv} missing edge_type column")
    df = df[df["edge_type"].astype(str).str.lower() == "line"].copy()
    if df.empty:
        raise ValueError(f"{catalog_csv} has no line edges")

    if "edge_id" not in df.columns:
        df["edge_id"] = np.arange(len(df), dtype=np.int64)

    edges: dict[int, dict] = {}
    adj: dict[str, list[int]] = defaultdict(list)
    for _, r in df.iterrows():
        eid = int(r["edge_id"])
        u = str(r["from_node"]).strip()
        v = str(r["to_node"]).strip()
        if not u or not v or u == v:
            continue
        edge = {
            "edge_id": eid,
            "u": u,
            "v": v,
            "R_full": float(r.get("R_full", 0.0) or 0.0),
            "X_full": float(r.get("X_full", 0.0) or 0.0),
        }
        edges[eid] = edge
        adj[u].append(eid)
        adj[v].append(eid)
    return edges, adj


def _other(edge: dict, node: str) -> str:
    if edge["u"] == node:
        return edge["v"]
    if edge["v"] == node:
        return edge["u"]
    raise ValueError(f"node {node} not incident to edge {edge['edge_id']}")


def build_load_only_compacted_edges(
    *,
    edge_catalog_csv: Path,
    load_nodes_csv: Path,
    out_csv: Path,
) -> None:
    load_nodes = _load_load_nodes(load_nodes_csv)
    node_to_idx = _node_to_idx_from_load_csv(load_nodes_csv)
    edges, adj = _build_line_graph(edge_catalog_csv)

    visited_halfedge: set[tuple[int, str]] = set()
    candidates: list[dict] = []
    skipped_nonseries = 0

    for start in sorted(load_nodes):
        if start not in adj:
            continue
        for eid0 in adj[start]:
            if (eid0, start) in visited_halfedge:
                continue

            e0 = edges[eid0]
            nxt = _other(e0, start)
            r_sum = float(e0["R_full"])
            x_sum = float(e0["X_full"])
            path_nodes = [start, nxt]
            path_edges = [eid0]

            visited_halfedge.add((eid0, start))
            visited_halfedge.add((eid0, nxt))

            prev = start
            cur = nxt
            ok = True

            while cur not in load_nodes:
                deg = len(adj.get(cur, []))
                if deg != 2:
                    # Not a simple series interior; skip this contraction candidate.
                    skipped_nonseries += 1
                    ok = False
                    break
                eids = adj[cur]
                e_next = eids[0] if _other(edges[eids[0]], cur) != prev else eids[1]
                nxt2 = _other(edges[e_next], cur)

                if (e_next, cur) in visited_halfedge and nxt2 != start:
                    # Already traversed from this side in another walk.
                    ok = False
                    break

                ee = edges[e_next]
                r_sum += float(ee["R_full"])
                x_sum += float(ee["X_full"])
                path_edges.append(e_next)
                path_nodes.append(nxt2)

                visited_halfedge.add((e_next, cur))
                visited_halfedge.add((e_next, nxt2))

                prev, cur = cur, nxt2
                if cur == start:
                    ok = False
                    break

            if not ok or cur == start or cur not in load_nodes:
                continue

            u, v = _canon_pair(start, cur)
            candidates.append(
                {
                    "from_node": u,
                    "to_node": v,
                    "edge_type": "line",
                    "R_full": float(r_sum),
                    "X_full": float(x_sum),
                    "u_idx": int(node_to_idx[u]) if u in node_to_idx else np.nan,
                    "v_idx": int(node_to_idx[v]) if v in node_to_idx else np.nan,
                    "num_original_segments": int(len(path_edges)),
                    "internal_nodes": json.dumps(path_nodes[1:-1]),
                    "path_nodes": json.dumps(path_nodes),
                    "path_edges": json.dumps(path_edges),
                    "z_mag": float(np.sqrt(r_sum * r_sum + x_sum * x_sum)),
                }
            )

    if not candidates:
        raise RuntimeError("No compacted load-load edges built; check input files.")

    # If multiple candidates connect the same load-node pair, keep minimum |Z|.
    cdf = pd.DataFrame(candidates)
    cdf = cdf.sort_values(["from_node", "to_node", "z_mag"], ascending=[True, True, True])
    grouped = cdf.groupby(["from_node", "to_node"], as_index=False, sort=True).first()

    out = grouped[
        [
            "from_node",
            "to_node",
            "edge_type",
            "R_full",
            "X_full",
            "u_idx",
            "v_idx",
            "num_original_segments",
            "internal_nodes",
            "path_nodes",
            "path_edges",
        ]
    ].copy()
    out.to_csv(out_csv, index=False)

    print("[build_load_only_compacted_edges]")
    print(f"  load nodes: {len(load_nodes)}")
    print(f"  line edges input: {len(edges)}")
    print(f"  candidates built: {len(cdf)}")
    print(f"  output unique load-load edges: {len(out)}")
    print(f"  skipped non-series interiors: {skipped_nonseries}")
    print(f"  wrote: {out_csv}")


def main() -> None:
    d = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg" / "Heterogenous GNN dataset"
    p = argparse.ArgumentParser(
        description=(
            "Build load-only line-edge CSV by compacting series paths through non-load nodes "
            "and summing R_full/X_full."
        )
    )
    p.add_argument(
        "--edge-catalog",
        type=Path,
        default=d / "edges" / "hetero_mv_edge_catalog.csv",
        help="Input hetero edge catalog with edge_type and R_full/X_full.",
    )
    p.add_argument(
        "--load-nodes-csv",
        type=Path,
        default=d / "nodes" / "hetero_mv_nodes_load_transformer_reg_tap_only.csv",
        help="Load-type node CSV used to define kept nodes (1177 set).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=d / "edges" / "hetero_mv_line_edges_load_only_compacted.csv",
        help="Output compacted load-only line edge CSV.",
    )
    args = p.parse_args()

    build_load_only_compacted_edges(
        edge_catalog_csv=args.edge_catalog.resolve(),
        load_nodes_csv=args.load_nodes_csv.resolve(),
        out_csv=args.out_csv.resolve(),
    )


if __name__ == "__main__":
    main()
