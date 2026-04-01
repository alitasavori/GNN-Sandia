"""
Electrical distance from the substation (slack) for each phase-node in the GNN graph.

Distance = sum of sqrt(R^2 + X^2) along a minimum-impedance path in the *static phase-edge
graph* (same definition as run_loadtype_dataset._compute_electrical_distance_from_source).

Why a node can look "not connected" in this graph (unreachable / inf distance):

1. **Reduced graph**: Edges incident to upstream buses (e.g. sourcebus) are often *dropped*
   when building gnn_edges_phase_static.csv. The slack is then not a vertex; Dijkstra starts
   from *boundary* nodes inferred from DSS (see run_loadtype_dataset._infer_reduced_graph_roots).

2. **Incomplete phase edges**: Line/transformer export uses one edge per phase. If parsing or
   the node list omits one terminal (e.g. only sx....1 in edges, not sx....2), that node has
   no path in the adjacency list even though OpenDSS has a physical connection.

3. **3-winding service transformers**: extract_static_phase_edges_to_csv only connects the first
   two transformer bus specs per element; part of the secondary can be missing from the graph.

4. **Name mismatch**: Node list says "sx123a.1" but edges use "SX123A.1" (if ever inconsistent).

Unreachable nodes are reported with electrical_distance_ohm = NaN (default) so they are not
confused with true zero-ohm distance. Legacy dataset code maps inf -> 0.0 instead.

**IEEE 8500:** use ``datasets_gnn2/loadtype_8500_dailyagg/`` (``Run_8500Node_Daily_5min.dss``) or
``datasets_gnn2/loadtype_8500/`` (``Master.dss`` snapshot). Run ``--build`` to create the two CSVs
without generating a full dataset.
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# Preset: (node_index_relative, edges_relative)
CASE_PATHS: dict[str, tuple[str, str]] = {
    "34": (
        "datasets_gnn2/loadtype/gnn_node_index_master.csv",
        "datasets_gnn2/loadtype/gnn_edges_phase_static.csv",
    ),
    "8500-daily": (
        "datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv",
        "datasets_gnn2/loadtype_8500_dailyagg/gnn_edges_phase_static.csv",
    ),
    "8500-snapshot": (
        "datasets_gnn2/loadtype_8500/gnn_node_index_master.csv",
        "datasets_gnn2/loadtype_8500/gnn_edges_phase_static.csv",
    ),
}
DEFAULT_CASE = "8500-daily"

__all__ = [
    "REPO_ROOT",
    "CASE_PATHS",
    "DEFAULT_CASE",
    "build_8500_static_csvs",
    "compute_electrical_distances_dataframe",
    "compute_electrical_distances_legacy_zero",
    "default_results_csv_path",
    "save_electrical_distances_csv",
]


def _is_grid_source_bus(bus: str) -> bool:
    b = str(bus).strip().lower()
    return b in ("sourcebus", "800") or b.startswith("_hvmv_sub")


def _adjacency_from_edges_csv(edge_csv_path: str | Path) -> dict[str, list[tuple[str, float]]]:
    df_e = pd.read_csv(edge_csv_path)
    adj: dict[str, list[tuple[str, float]]] = {}
    for _, row in df_e.iterrows():
        a, b = str(row["from_node"]), str(row["to_node"])
        r = float(row.get("R_full", 0))
        x = float(row.get("X_full", 0))
        z_mag = float(np.sqrt(r * r + x * x))
        adj.setdefault(a, []).append((b, z_mag))
    return adj


def _dijkstra_min_impedance(
    node_names: list[str],
    adj: dict[str, list[tuple[str, float]]],
    root_dist: dict[str, float],
) -> dict[str, float]:
    """Return shortest-path sum of |Z| from roots; unreachable nodes omitted from dict."""
    dist: dict[str, float] = {n: float("inf") for n in node_names}
    for src, d0 in root_dist.items():
        if src in dist:
            dist[src] = min(dist[src], float(d0))

    heap = [(float(d0), src) for src, d0 in root_dist.items() if src in dist]
    heapq.heapify(heap)

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            if v not in dist:
                continue
            d_v = d_u + w
            if d_v < dist[v]:
                dist[v] = d_v
                heapq.heappush(heap, (d_v, v))

    return dist


def _infer_reduced_graph_roots(node_names_master: list[str]):
    """Delegate to run_loadtype_dataset (needs matching OpenDSS circuit on inj)."""
    import run_loadtype_dataset as lt

    return lt._infer_reduced_graph_roots(node_names_master)


def _prepare_dss_for_infer(case: Literal["34", "8500-daily", "8500-snapshot"]) -> None:
    """Compile the feeder so _infer_reduced_graph_roots sees the same topology as the CSVs."""
    import run_injection_dataset as inj

    importlib.reload(inj)
    if case == "34":
        inj.compile_once()
        inj.setup_daily()
    elif case == "8500-daily":
        import run_daily_aggregate_dataset_8500 as da

        da._compile_8500_daily_setup()
        da._detach_daily_loadshape_from_loads()
    else:
        import run_loadtype_dataset_8500 as lt8500

        lt8500.compile_8500()


def build_8500_static_csvs(
    mode: Literal["daily", "snapshot"] = "daily",
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Compile IEEE 8500, enumerate phase nodes, write ``gnn_node_index_master.csv`` and
    ``gnn_edges_phase_static.csv``. Matches ``run_daily_aggregate_dataset_8500`` (daily) or
    ``run_loadtype_dataset_8500`` (snapshot) static export.
    """
    import run_injection_dataset as inj

    importlib.reload(inj)
    if out_dir is None:
        out_dir = REPO_ROOT / "datasets_gnn2" / (
            "loadtype_8500_dailyagg" if mode == "daily" else "loadtype_8500"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_csv = out_dir / "gnn_node_index_master.csv"
    edge_csv = out_dir / "gnn_edges_phase_static.csv"

    if mode == "daily":
        import run_daily_aggregate_dataset_8500 as da

        da._compile_8500_daily_setup()
        da._detach_daily_loadshape_from_loads()
        excluded: tuple[str, ...] = ()
    else:
        import run_loadtype_dataset_8500 as lt8500

        lt8500.compile_8500()
        excluded = tuple(lt8500.EXCLUDED_UPSTREAM_BUSES)

    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    pd.DataFrame(
        {"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}
    ).to_csv(node_csv, index=False)
    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_master,
        edge_csv_path=str(edge_csv),
        excluded_buses=excluded,
    )
    print(f"[build_8500_static_csvs] {node_csv} | N={len(node_names_master)}")
    print(f"[build_8500_static_csvs] {edge_csv}")
    return node_csv, edge_csv


def compute_electrical_distances_dataframe(
    node_index_csv: str | Path,
    edges_csv: str | Path,
    *,
    unreachable: Literal["nan", "inf", "zero"] = "nan",
    infer_case: Literal["34", "8500-daily", "8500-snapshot"] | None = None,
) -> pd.DataFrame:
    """
    Build a table with one row per node in gnn_node_index_master.csv.

    Columns:
      - node, bus, phase
      - electrical_distance_ohm: float; NaN / inf / 0 for unreachable per `unreachable`
      - reachable: bool
    """
    node_index_csv = Path(node_index_csv)
    edges_csv = Path(edges_csv)

    nodes = pd.read_csv(node_index_csv)["node"].astype(str).tolist()
    adj = _adjacency_from_edges_csv(edges_csv)

    source_nodes = [n for n in nodes if _is_grid_source_bus(n.split(".")[0]) and n in adj]
    if source_nodes:
        root_nodes = {src: 0.0 for src in source_nodes}
    else:
        if infer_case is not None:
            _prepare_dss_for_infer(infer_case)
        root_nodes = _infer_reduced_graph_roots(nodes)
    if not root_nodes and nodes:
        root_nodes = {nodes[0]: 0.0}

    dist_inf = _dijkstra_min_impedance(nodes, adj, root_nodes)

    rows = []
    for n in nodes:
        d = dist_inf.get(n, float("inf"))
        ok = d != float("inf")
        if ok:
            d_out = float(d)
        elif unreachable == "nan":
            d_out = float("nan")
        elif unreachable == "inf":
            d_out = float("inf")
        else:
            d_out = 0.0

        bus, _, ph = n.partition(".")
        phase = int(ph) if ph.isdigit() else -1
        rows.append(
            {
                "node": n,
                "bus": bus,
                "phase": phase,
                "electrical_distance_ohm": d_out,
                "reachable": ok,
            }
        )

    return pd.DataFrame(rows)


def default_results_csv_path(node_index_csv: str | Path) -> Path:
    """Default CSV path next to ``gnn_node_index_master.csv`` (same folder as the graph inputs)."""
    return Path(node_index_csv).expanduser().resolve().parent / "electrical_distance_from_substation.csv"


def save_electrical_distances_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the electrical-distance table; creates parent dirs if needed. Returns resolved path."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep="nan")
    return path


def compute_electrical_distances_legacy_zero(
    node_index_csv: str | Path,
    edges_csv: str | Path,
) -> dict[str, float]:
    """Same as dataset pipeline: unreachable -> 0.0 (ambiguous)."""
    import run_loadtype_dataset as lt

    nodes = pd.read_csv(node_index_csv)["node"].astype(str).tolist()
    return lt._compute_electrical_distance_from_source(nodes, str(edges_csv))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Electrical distance from substation for all GNN phase-nodes (IEEE 34 or 8500)."
    )
    p.add_argument(
        "--case",
        choices=sorted(CASE_PATHS.keys()),
        default=DEFAULT_CASE,
        help=f"Preset CSV paths under repo (default: {DEFAULT_CASE}).",
    )
    p.add_argument(
        "--node-index",
        default="",
        help="Override: gnn_node_index_master.csv (default: from --case)",
    )
    p.add_argument(
        "--edges",
        default="",
        help="Override: gnn_edges_phase_static.csv (default: from --case)",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="For 8500 cases only: compile feeder and write node index + edge CSVs first.",
    )
    p.add_argument(
        "--out",
        default="",
        help=(
            "CSV output path. If omitted, writes "
            "<node-index-dir>/electrical_distance_from_substation.csv next to the node index."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write any CSV (only print summary to stdout).",
    )
    p.add_argument(
        "--legacy-zero",
        action="store_true",
        help="Use unreachable=0 to match run_loadtype_dataset output",
    )
    args = p.parse_args(argv)

    repo = REPO_ROOT
    rel_ni, rel_ed = CASE_PATHS[args.case]
    if args.node_index:
        ni = Path(args.node_index)
        if not ni.is_absolute():
            ni = repo / ni
    else:
        ni = repo / rel_ni
    if args.edges:
        ed = Path(args.edges)
        if not ed.is_absolute():
            ed = repo / ed
    else:
        ed = repo / rel_ed

    infer_map = {
        "34": "34",
        "8500-daily": "8500-daily",
        "8500-snapshot": "8500-snapshot",
    }
    infer_case = infer_map[args.case]

    if args.build:
        if args.case == "34":
            raise SystemExit("--build is only for 8500; use run_loadtype_dataset for IEEE 34 static CSVs.")
        mode = "daily" if args.case == "8500-daily" else "snapshot"
        bn, be = build_8500_static_csvs(mode=mode, out_dir=ni.parent)
        ni, ed = bn, be

    if not ni.is_file() or not ed.is_file():
        raise SystemExit(
            f"Missing CSVs:\n  {ni}\n  {ed}\n"
            f"Run: python electrical_distance.py --case {args.case} --build\n"
            "or generate the full dataset (run_daily_aggregate_dataset_8500 / run_loadtype_dataset_8500)."
        )

    if args.legacy_zero:
        if infer_case != "34":
            _prepare_dss_for_infer(infer_case)
        dmap = compute_electrical_distances_legacy_zero(ni, ed)
        df = pd.DataFrame([{"node": n, "electrical_distance_ohm": d} for n, d in dmap.items()])
        unreachable = int((df["electrical_distance_ohm"] == 0.0).sum())
    else:
        df = compute_electrical_distances_dataframe(ni, ed, unreachable="nan", infer_case=infer_case)
        unreachable = int((~df["reachable"]).sum())

    print(f"case={args.case} nodes={len(df)} unreachable_in_graph={unreachable}")
    print(df["electrical_distance_ohm"].describe())

    if not args.no_save:
        if args.out:
            outp = Path(args.out).expanduser()
            if not outp.is_absolute():
                outp = (Path.cwd() / outp).resolve()
        else:
            outp = default_results_csv_path(ni)
        save_electrical_distances_csv(df, outp)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
