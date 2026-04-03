"""
# Load nodes vs regulators: downstream electrical distance

For each **load** node (from `hetero_mv_nodes_load_transformer.csv`), this script fills one column
per **regulator** with the **cumulative electrical distance** (Ω) from that regulator’s **downstream**
terminal (`terminal_2` in `regulator_involved_nodes.csv`) along **line** segments only, within the
subtree **downstream** of that regulator. If a load is **not** electrically downstream of a given
regulator, the cell is **0**.

**Distance definition:** for each catalog edge, segment length = √(R_full² + X_full²) (ohms). Missing
R/X on an edge is treated as 0. Missing regulator terminals in the graph also yield **0**.

**Requires:** `hetero_mv_edge_catalog.csv` (+ optional merge with `hetero_mv_line_edge_attr.csv` for
line rows), `regulator_involved_nodes.csv`, `hetero_mv_nodes_load_transformer.csv`, and
`gnn_node_index_master.csv` (for canonical bus names).

```bash
python compute_load_electrical_distance_to_regulators.py ^
  --dataset-dir "datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"
```

Output: `edges/load_electrical_distance_to_each_regulator.csv` (default path under `--dataset-dir`).
Uses **0** when there is no downstream path to that regulator.
"""

from __future__ import annotations

import argparse
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()


def _canon_map(node_index_csv: Path) -> dict[str, str]:
    df = pd.read_csv(node_index_csv)
    return {str(n).strip().lower(): str(n).strip() for n in df["node"].astype(str)}


def _z_mag_ohm(r: float | None, x: float | None) -> float:
    if r is None and x is None:
        return 0.0
    rr = 0.0 if r is None or (isinstance(r, float) and np.isnan(r)) else float(r)
    xx = 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    return float(np.sqrt(rr * rr + xx * xx))


def _build_weighted_graph(catalog: pd.DataFrame, line_attr: pd.DataFrame | None) -> dict[str, list[tuple[str, float]]]:
    """Undirected adjacency: node -> list of (neighbor, weight_ohm)."""
    rmap: dict[int, tuple[float, float]] = {}
    if line_attr is not None and len(line_attr):
        for _, row in line_attr.iterrows():
            eid = int(row["edge_id"])
            rmap[eid] = (float(row["R_full"]), float(row["X_full"]))

    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for _, row in catalog.iterrows():
        eid = int(row["edge_id"])
        a = str(row["from_node"]).strip().lower()
        b = str(row["to_node"]).strip().lower()
        if line_attr is not None and eid in rmap:
            rr, xx = rmap[eid]
        else:
            rr = row.get("R_full")
            xx = row.get("X_full")
            if rr is not None and pd.notna(rr):
                rr = float(rr)
            else:
                rr = None
            if xx is not None and pd.notna(xx):
                xx = float(xx)
            else:
                xx = None
        w = _z_mag_ohm(rr, xx)
        adj[a].append((b, w))
        adj[b].append((a, w))

    return adj


def _is_regulator_chord(u: str, v: str, t1: str, t2: str) -> bool:
    return (u == t1 and v == t2) or (u == t2 and v == t1)


def _downstream_component(
    adj: dict[str, list[tuple[str, float]]],
    t1: str,
    t2: str,
) -> set[str]:
    """
    Nodes reachable from t2 without traversing the regulator edge (t1—t2).
    t1 = upstream terminal, t2 = downstream terminal (see regulator_involved_nodes.csv).
    """
    t1, t2 = t1.lower(), t2.lower()
    seen: set[str] = {t2}
    stack = [t2]
    while stack:
        u = stack.pop()
        for v, _w in adj.get(u, []):
            if _is_regulator_chord(u, v, t1, t2):
                continue
            if v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return seen


def _subgraph_edges(adj: dict[str, list[tuple[str, float]]], nodes: set[str]) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for u in nodes:
        for v, w in adj.get(u, []):
            if v in nodes:
                out[u].append((v, w))
    return out


def _dijkstra(
    adj: dict[str, list[tuple[str, float]]],
    source: str,
) -> dict[str, float]:
    """Shortest-path distances from source (non-negative weights)."""
    source = source.lower()
    dist: dict[str, float] = {source: 0.0}
    pq: list[tuple[float, str]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def run(
    edges_dir: Path,
    regulator_csv: Path,
    load_nodes_csv: Path,
    node_index_csv: Path,
    out_csv: Path,
) -> Path:
    canon = _canon_map(node_index_csv)
    catalog_path = edges_dir / "hetero_mv_edge_catalog.csv"
    line_path = edges_dir / "hetero_mv_line_edge_attr.csv"
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    catalog = pd.read_csv(catalog_path)
    line_attr = pd.read_csv(line_path) if line_path.is_file() else None

    adj = _build_weighted_graph(catalog, line_attr)

    reg_df = pd.read_csv(regulator_csv)
    load_df = pd.read_csv(load_nodes_csv, usecols=lambda c: c.lower() in ("node",))
    load_nodes = [canon.get(str(x).strip().lower(), str(x).strip()) for x in load_df["node"].astype(str)]

    reg_rows: list[tuple[str, str, str]] = []
    for _, row in reg_df.iterrows():
        rname = str(row.get("Regulator", "")).strip()
        t1 = str(row.get("terminal_1 node", "")).strip()
        t2 = str(row.get("terminal_2 node", "")).strip()
        if not rname or not t1 or not t2 or t1.lower() == "nan" or t2.lower() == "nan":
            continue
        t1k = canon.get(t1.lower(), t1).lower()
        t2k = canon.get(t2.lower(), t2).lower()
        reg_rows.append((rname, t1k, t2k))

    reg_names = [r[0] for r in reg_rows]
    if len(reg_names) != len(set(reg_names)):
        raise ValueError("Duplicate Regulator names in regulator_involved_nodes.csv")

    out = pd.DataFrame({"node": load_nodes})
    for rname, t1k, t2k in reg_rows:
        col: list[float] = []
        if t1k not in adj or t2k not in adj:
            for _ in load_nodes:
                col.append(0.0)
            out[rname] = col
            continue
        down = _downstream_component(adj, t1k, t2k)
        sub = _subgraph_edges(adj, down)
        dist = _dijkstra(sub, t2k)
        for ln in load_nodes:
            lk = ln.lower()
            if lk not in down:
                col.append(0.0)
            else:
                d = dist.get(lk)
                col.append(float(d) if d is not None else 0.0)
        out[rname] = col

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.resolve()}  rows={len(out)}  regulators={len(reg_rows)}")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description="CSV: each load node vs electrical distance (ohm) to each regulator downstream subtree; 0 if not downstream."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO / "datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset",
        help="Folder containing nodes/ and edges/",
    )
    p.add_argument(
        "--regulator-csv",
        type=Path,
        default=None,
        help="Default: <parent of dataset-dir>/regulator_involved_nodes.csv",
    )
    p.add_argument(
        "--node-index",
        type=Path,
        default=None,
        help="Default: <parent of dataset-dir>/gnn_node_index_master.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: dataset-dir/edges/load_electrical_distance_to_each_regulator.csv",
    )
    args = p.parse_args()
    ds = args.dataset_dir.resolve()
    parent = ds.parent
    edges_dir = ds / "edges"
    nodes_dir = ds / "nodes"
    reg_csv = (args.regulator_csv or parent / "regulator_involved_nodes.csv").resolve()
    idx_csv = (args.node_index or parent / "gnn_node_index_master.csv").resolve()
    load_csv = nodes_dir / "hetero_mv_nodes_load_transformer.csv"
    out = (args.out_csv or edges_dir / "load_electrical_distance_to_each_regulator.csv").resolve()

    run(edges_dir, reg_csv, load_csv, idx_csv, out)


if __name__ == "__main__":
    main()
