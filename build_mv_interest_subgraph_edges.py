"""
Anchor-only MV subgraph (synthetic series R/X edges between anchors).

1) Collects anchor phase-nodes from the same three CSVs as before.

2) On the full `gnn_edges_phase_static.csv` graph, for each ordered pair of
   distinct anchors (A, B) with A < B (lexicographic):
   - Run single-source Dijkstra from A with edge weight sqrt(R_full^2 + X_full^2).
   - Reconstruct the shortest-|Z| path from A to B.
   - **Neighbour rule:** if any **internal** node on that path (not A or B) is
     also an **anchor**, **skip** this pair (another anchor lies “between” them
     on that shortest path).
   - Otherwise add **one synthetic undirected edge** A–B with
     R_full = sum(R), X_full = sum(X), C_full = sum(C) along that path.

3) Output `gnn_edges_phase_static_mv_only.csv` contains **only these synthetic
   rows** (not raw line segments). Nodes in the subgraph are **anchors only**.

4) `mv_interest_anchor_nodes.csv` still lists anchors + electrical distance from
   substation (multi-source Dijkstra on full graph, same as before).
"""
from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()


def _is_grid_source_bus(bus: str) -> bool:
    b = str(bus).strip().lower()
    return b in ("sourcebus", "800") or b.startswith("_hvmv_sub")


def _z_weight(row: pd.Series) -> float:
    r = float(row.get("R_full", 0) or 0)
    x = float(row.get("X_full", 0) or 0)
    return float(np.sqrt(r * r + x * x))


def _canonical_map(node_index_csv: Path) -> dict[str, str]:
    df = pd.read_csv(node_index_csv)
    out: dict[str, str] = {}
    for n in df["node"].astype(str):
        out[n.strip().lower()] = n.strip()
    return out


def _collect_interest_nodes(
    mapping_csv: Path,
    regulator_csv: Path,
    capacitor_csv: Path,
    canon: dict[str, str],
) -> tuple[list[str], list[dict]]:
    raw: list[str] = []
    if mapping_csv.is_file():
        m = pd.read_csv(mapping_csv)
        raw += m["mv_node"].astype(str).str.strip().tolist()
    if regulator_csv.is_file():
        r = pd.read_csv(regulator_csv)
        for c in ("terminal_1 node", "terminal_2 node"):
            if c in r.columns:
                raw += r[c].astype(str).str.strip().tolist()
    if capacitor_csv.is_file():
        c = pd.read_csv(capacitor_csv)
        if "From node" in c.columns:
            raw += c["From node"].astype(str).str.strip().tolist()

    resolved: list[str] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    for s in raw:
        if not s or s.lower() == "nan":
            continue
        key = s.lower()
        if key in canon:
            n = canon[key]
            if n not in seen:
                seen.add(n)
                resolved.append(n)
        else:
            skipped.append({"requested": s, "reason": "not_in_node_index_master"})
    return resolved, skipped


def _build_adjacency(df_e: pd.DataFrame) -> dict[str, list[tuple[str, float, int]]]:
    adj: dict[str, list[tuple[str, float, int]]] = {}
    for idx, row in df_e.iterrows():
        u = str(row["from_node"]).strip()
        v = str(row["to_node"]).strip()
        w = _z_weight(row)
        adj.setdefault(u, []).append((v, w, int(idx)))
        adj.setdefault(v, []).append((u, w, int(idx)))
    return adj


def _dijkstra_single_source(
    adj: dict[str, list[tuple[str, float, int]]],
    source: str,
) -> tuple[dict[str, float], dict[str, str | None], dict[str, int | None]]:
    """Shortest |Z| paths from source; parent[v], pei[v] = row index for edge (parent[v], v)."""
    dist: dict[str, float] = {source: 0.0}
    parent: dict[str, str | None] = {source: None}
    pei: dict[str, int | None] = {source: None}
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float("inf")):
            continue
        for v, w, eidx in adj.get(u, []):
            if v not in dist:
                dist[v] = float("inf")
            dv = d_u + w
            if dv < dist[v]:
                dist[v] = dv
                parent[v] = u
                pei[v] = eidx
                heapq.heappush(heap, (dv, v))

    return dist, parent, pei


def _multisource_dijkstra(
    adj: dict[str, list[tuple[str, float, int]]],
    roots: list[str],
) -> tuple[dict[str, float], dict[str, str | None], dict[str, int | None]]:
    dist: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    pei: dict[str, int | None] = {}
    heap: list[tuple[float, str]] = []
    for r in roots:
        if r not in adj:
            continue
        dist[r] = 0.0
        parent[r] = None
        pei[r] = None
        heapq.heappush(heap, (0.0, r))

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float("inf")):
            continue
        for v, w, eidx in adj.get(u, []):
            if v not in dist:
                dist[v] = float("inf")
            dv = d_u + w
            if dv < dist[v]:
                dist[v] = dv
                parent[v] = u
                pei[v] = eidx
                heapq.heappush(heap, (dv, v))

    return dist, parent, pei


def _path_a_to_b(
    A: str,
    B: str,
    parent: dict[str, str | None],
) -> list[str] | None:
    """Return [A, ..., B] along shortest-path tree rooted at A, or None."""
    if B not in parent and B != A:
        return None
    rev: list[str] = []
    cur: str | None = B
    while cur is not None:
        rev.append(cur)
        if cur == A:
            break
        cur = parent.get(cur)
    if not rev or rev[-1] != A:
        return None
    rev.reverse()
    return rev


def _bus_phase(node: str) -> tuple[str, int]:
    if "." not in node:
        return node, -1
    b, _, ph = node.partition(".")
    return b, int(ph) if ph.isdigit() else -1


def run(
    *,
    node_index_csv: Path,
    edges_csv: Path,
    mapping_csv: Path,
    regulator_csv: Path,
    capacitor_csv: Path,
    out_nodes_csv: Path,
    out_edges_csv: Path,
) -> None:
    canon = _canonical_map(node_index_csv)
    anchors, skipped = _collect_interest_nodes(
        mapping_csv, regulator_csv, capacitor_csv, canon
    )
    anchors_set = set(anchors)
    anchors_sorted = sorted(anchors)

    df_e = pd.read_csv(edges_csv)
    adj = _build_adjacency(df_e)

    all_nodes = set(df_e["from_node"].astype(str)) | set(df_e["to_node"].astype(str))
    roots = [n for n in all_nodes if _is_grid_source_bus(n.split(".")[0]) and n in adj]
    if not roots:
        roots = [n for n in all_nodes if n.split(".")[0].lower() == "sourcebus" and n in adj]
    if not roots:
        raise SystemExit("No substation-like root nodes found adjacent to the graph.")

    dist_sub, _, _ = _multisource_dijkstra(adj, roots)

    synthetic_rows: list[dict] = []
    n_pairs_ok = 0
    n_pairs_skip_internal_anchor = 0
    n_pairs_unreachable = 0

    na = len(anchors_sorted)
    for ia, A in enumerate(anchors_sorted):
        if ia % 200 == 0 and ia > 0:
            print(f"  anchor Dijkstra progress {ia}/{na} ...", flush=True)
        dist_a, parent_a, pei_a = _dijkstra_single_source(adj, A)
        for B in anchors_sorted:
            if B <= A:
                continue
            if B not in dist_a or dist_a[B] == float("inf"):
                n_pairs_unreachable += 1
                continue
            path = _path_a_to_b(A, B, parent_a)
            if path is None or len(path) < 2:
                n_pairs_unreachable += 1
                continue
            internal = path[1:-1]
            if any(n in anchors_set for n in internal):
                n_pairs_skip_internal_anchor += 1
                continue

            r_sum = x_sum = c_sum = 0.0
            for i in range(1, len(path)):
                v = path[i]
                eidx = pei_a.get(v)
                if eidx is None:
                    r_sum = float("nan")
                    break
                row = df_e.iloc[int(eidx)]
                r_sum += float(row.get("R_full", 0) or 0)
                x_sum += float(row.get("X_full", 0) or 0)
                c_sum += float(row.get("C_full", 0) or 0)

            if not np.isfinite(r_sum):
                continue

            bus_a, ph_a = _bus_phase(A)
            bus_b, ph_b = _bus_phase(B)
            synthetic_rows.append(
                {
                    "from_node": A,
                    "to_node": B,
                    "from_bus": bus_a,
                    "to_bus": bus_b,
                    "phase_from": ph_a,
                    "phase_to": ph_b,
                    "line_name": "SYNTHETIC_ANCHOR_AGG",
                    "linecode": "",
                    "nph_line": 1,
                    "length_km": np.nan,
                    "path_segments": int(len(path) - 1),
                    "R_per_len": np.nan,
                    "X_per_len": np.nan,
                    "C_per_len": np.nan,
                    "R_full": r_sum,
                    "X_full": x_sum,
                    "C_full": c_sum,
                    "num_physical_segments": int(len(path) - 1),
                    "u_idx": np.nan,
                    "v_idx": np.nan,
                }
            )
            n_pairs_ok += 1

    pd.DataFrame(synthetic_rows).to_csv(out_edges_csv, index=False)

    rows_out = []
    for a in anchors_sorted:
        d = dist_sub.get(a, float("inf"))
        rows_out.append(
            {
                "node": a,
                "electrical_distance_ohm": float(d) if d != float("inf") else np.nan,
                "reachable_from_substation": d != float("inf"),
            }
        )
    pd.DataFrame(rows_out).to_csv(out_nodes_csv, index=False)

    skip_path = out_nodes_csv.parent / "mv_interest_anchor_nodes_skipped.csv"
    if skipped:
        pd.DataFrame(skipped).to_csv(skip_path, index=False)

    print(f"[build_mv_interest_subgraph_edges] anchors={len(anchors)}")
    print("  Edge model: synthetic A–B with sum(R,X,C) on min-|Z| path, skip if another anchor on path interior.")
    print(f"  synthetic anchor–anchor edges written: {n_pairs_ok}")
    print(f"  pairs skipped (interior anchor on shortest path): {n_pairs_skip_internal_anchor}")
    print(f"  pair attempts unreachable: {n_pairs_unreachable}")
    print(f"  wrote {out_edges_csv}")
    print(f"  wrote {out_nodes_csv}")
    if skipped:
        print(f"  wrote {skip_path} ({len(skipped)} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Anchor-only MV subgraph (synthetic R/X edges).")
    d = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p.add_argument("--node-index", type=Path, default=d / "gnn_node_index_master.csv")
    p.add_argument("--edges", type=Path, default=d / "gnn_edges_phase_static.csv")
    p.add_argument("--mapping", type=Path, default=d / "mv_x_sx_node_mapping_8500.csv")
    p.add_argument("--regulator", type=Path, default=d / "regulator_involved_nodes.csv")
    p.add_argument("--capacitor", type=Path, default=d / "capacitor_involved_nodes.csv")
    p.add_argument("--out-nodes", type=Path, default=d / "mv_interest_anchor_nodes.csv", dest="out_nodes")
    p.add_argument("--out-edges", type=Path, default=d / "gnn_edges_phase_static_mv_only.csv", dest="out_edges")
    args = p.parse_args(argv)

    for path in (args.node_index, args.edges, args.mapping):
        if not path.is_file():
            print(f"Missing required file: {path}", file=sys.stderr)
            sys.exit(1)

    run(
        node_index_csv=args.node_index.resolve(),
        edges_csv=args.edges.resolve(),
        mapping_csv=args.mapping.resolve(),
        regulator_csv=args.regulator.resolve(),
        capacitor_csv=args.capacitor.resolve(),
        out_nodes_csv=args.out_nodes.resolve(),
        out_edges_csv=args.out_edges.resolve(),
    )


if __name__ == "__main__":
    main()
