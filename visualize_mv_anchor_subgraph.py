"""
Visualize the anchor-only subgraph with three anchor types in different colors.

Reads:
  - gnn_edges_phase_static_mv_only.csv (from_node, to_node)
  - mv_x_sx_node_mapping_8500.csv (mv_node)
  - regulator_involved_nodes.csv (terminal_1 node, terminal_2 node)
  - capacitor_involved_nodes.csv (From node)

Node colors (combinations if a node appears in multiple lists):
  MV only | Reg only | Cap only | MV+Reg | MV+Cap | Reg+Cap | all three
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()


def _canonical_map(node_index_csv: Path) -> dict[str, str]:
    df = pd.read_csv(node_index_csv)
    return {str(n).strip().lower(): str(n).strip() for n in df["node"].astype(str)}


def _collect_sets(
    canon: dict[str, str],
    mapping_csv: Path,
    regulator_csv: Path,
    capacitor_csv: Path,
) -> tuple[set[str], set[str], set[str]]:
    s_mv: set[str] = set()
    s_reg: set[str] = set()
    s_cap: set[str] = set()

    def add_raw(raw: list[str], target: set[str]) -> None:
        for x in raw:
            x = str(x).strip()
            if not x or x.lower() == "nan":
                continue
            k = x.lower()
            if k in canon:
                target.add(canon[k])

    if mapping_csv.is_file():
        m = pd.read_csv(mapping_csv)
        add_raw(m["mv_node"].astype(str).str.strip().tolist(), s_mv)
    if regulator_csv.is_file():
        r = pd.read_csv(regulator_csv)
        for c in ("terminal_1 node", "terminal_2 node"):
            if c in r.columns:
                add_raw(r[c].astype(str).str.strip().tolist(), s_reg)
    if capacitor_csv.is_file():
        c = pd.read_csv(capacitor_csv)
        if "From node" in c.columns:
            add_raw(c["From node"].astype(str).str.strip().tolist(), s_cap)

    return s_mv, s_reg, s_cap


def _color_and_size(n: str, s_mv: set[str], s_reg: set[str], s_cap: set[str]) -> tuple[str, float]:
    mv = n in s_mv
    rg = n in s_reg
    cp = n in s_cap
    key = (mv, rg, cp)
    colors: dict[tuple[bool, bool, bool], str] = {
        (True, False, False): "#d62728",
        (False, True, False): "#1f77b4",
        (False, False, True): "#2ca02c",
        (True, True, False): "#9467bd",
        (True, False, True): "#e377c2",
        (False, True, True): "#ff7f0e",
        (True, True, True): "#000000",
        (False, False, False): "#bcbd22",
    }
    sz = 36 if key.count(True) > 1 else 22
    return colors[key], float(sz)


def run(
    *,
    edge_csv: Path,
    node_index_csv: Path,
    mapping_csv: Path,
    regulator_csv: Path,
    capacitor_csv: Path,
    out_png: Path,
) -> None:
    try:
        import networkx as nx
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "-q"])
        import networkx as nx

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    canon = _canonical_map(node_index_csv)
    s_mv, s_reg, s_cap = _collect_sets(canon, mapping_csv, regulator_csv, capacitor_csv)

    df = pd.read_csv(edge_csv)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row["from_node"]).strip(), str(row["to_node"]).strip())

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"Graph |V|={n} |E|={m}")

    k = 2.0 / np.sqrt(max(n, 1))
    it = 40 if n < 3000 else 25
    pos = nx.spring_layout(G, seed=42, k=k, iterations=it, threshold=1e-3)

    colors = []
    sizes = []
    for node in G.nodes():
        c, s = _color_and_size(node, s_mv, s_reg, s_cap)
        colors.append(c)
        sizes.append(s)

    fig, ax = plt.subplots(figsize=(14, 11), dpi=120)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.35, edge_color="#444444")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.9, linewidths=0)
    ax.axis("off")
    ax.set_title("Anchor subgraph by type (larger = multiple types)", fontsize=12)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9, label="MV (mv_node)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=9, label="Regulator terminals"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label="Capacitor From node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467bd", markersize=9, label="MV + Reg"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e377c2", markersize=9, label="MV + Cap"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=9, label="Reg + Cap"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#000000", markersize=9, label="All three"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bcbd22", markersize=9, label="Unknown (not in lists)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved {out_png}")
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    d = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p.add_argument("--edges", type=Path, default=d / "gnn_edges_phase_static_mv_only.csv")
    p.add_argument("--node-index", type=Path, default=d / "gnn_node_index_master.csv")
    p.add_argument("--mapping", type=Path, default=d / "mv_x_sx_node_mapping_8500.csv")
    p.add_argument("--regulator", type=Path, default=d / "regulator_involved_nodes.csv")
    p.add_argument("--capacitor", type=Path, default=d / "capacitor_involved_nodes.csv")
    p.add_argument("--out", type=Path, default=d / "mv_subgraph_layout_by_type.png")
    args = p.parse_args()

    if not args.edges.is_file():
        raise SystemExit(f"Missing {args.edges}")

    run(
        edge_csv=args.edges.resolve(),
        node_index_csv=args.node_index.resolve(),
        mapping_csv=args.mapping.resolve(),
        regulator_csv=args.regulator.resolve(),
        capacitor_csv=args.capacitor.resolve(),
        out_png=args.out.resolve(),
    )


if __name__ == "__main__":
    main()
