"""
# Load nodes vs regulators: downstream **hop** count (graph edges)

Same topology as `compute_load_electrical_distance_to_regulators.py`: for each regulator,
only the subtree **downstream** of `terminal_2` (excluding the regulator chord) counts.
Each **line** edge in the hetero catalog counts as **one hop**. The downstream terminal
itself is **0** hops from that regulator’s reference point.

Output: `edges/load_hop_distance_to_each_regulator.csv` — integer hop counts; **0** if the
load is not downstream of that regulator (same convention as the Ω table).

```bash
python compute_load_hop_distance_to_regulators.py ^
  --dataset-dir "datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"
```
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from compute_load_electrical_distance_to_regulators import (
    _build_weighted_graph,
    _canon_map,
    _downstream_component,
    _subgraph_edges,
)

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()


def _bfs_hops(
    adj: dict[str, list[tuple[str, float]]],
    source: str,
) -> dict[str, int]:
    """Unweighted shortest-path length (edge count) from source."""
    source = source.lower()
    dist: dict[str, int] = {source: 0}
    q: deque[str] = deque([source])
    while q:
        u = q.popleft()
        du = dist[u]
        for v, _w in adj.get(u, []):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
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
        col: list[int] = []
        if t1k not in adj or t2k not in adj:
            for _ in load_nodes:
                col.append(0)
            out[rname] = col
            continue
        down = _downstream_component(adj, t1k, t2k)
        sub = _subgraph_edges(adj, down)
        hops = _bfs_hops(sub, t2k)
        for ln in load_nodes:
            lk = ln.lower()
            if lk not in down:
                col.append(0)
            else:
                h = hops.get(lk)
                col.append(int(h) if h is not None else 0)
        out[rname] = col

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.resolve()}  rows={len(out)}  regulators={len(reg_rows)}")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description="CSV: each load node vs hop count to each regulator downstream subtree; 0 if not downstream."
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
        help="Default: dataset-dir/edges/load_hop_distance_to_each_regulator.csv",
    )
    args = p.parse_args()
    ds = args.dataset_dir.resolve()
    parent = ds.parent
    edges_dir = ds / "edges"
    nodes_dir = ds / "nodes"
    reg_csv = (args.regulator_csv or parent / "regulator_involved_nodes.csv").resolve()
    idx_csv = (args.node_index or parent / "gnn_node_index_master.csv").resolve()
    load_csv = nodes_dir / "hetero_mv_nodes_load_transformer.csv"
    out = (args.out_csv or edges_dir / "load_hop_distance_to_each_regulator.csv").resolve()

    run(edges_dir, reg_csv, load_csv, idx_csv, out)


if __name__ == "__main__":
    main()
