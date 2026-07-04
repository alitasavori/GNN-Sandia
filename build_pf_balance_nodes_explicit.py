"""Export MV balance nodes for physics loss (offline CSV for Colab smoke).

Writes ``colab_pf_data/pf_balance_nodes_explicit.csv``.

Selection (expanded):
  All ``hetero_mv_nodes_load_transformer`` nodes on the master index, excluding
  slack/substation buses and MV interface buses (``regxfmr*``, ``190-*``, ``m/p/n*``).

Prior list (185 nodes, through 2026-07) additionally required
``electrical_distance_ohm > 0`` and ``hetero_y_neighbors_only`` (every Y-bus
neighbor also hetero / non-interface). That Y-neighbor filter is dropped here so
physics loss covers all principled MV load-transformer buses (~1177 nodes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import train_da_gps_multitask_complex_voltage_gine as pfmod

REPO = Path(__file__).resolve().parent
DEFAULT_DATA = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
DEFAULT_OUT = REPO / "colab_pf_data" / "pf_balance_nodes_explicit.csv"


def build_expanded_balance_list(
    data_root: Path,
) -> pd.DataFrame:
    idx_path = data_root / "gnn_node_index_master.csv"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing {idx_path}")

    idx = pd.read_csv(idx_path)
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}

    hetero = pfmod._load_pf_hetero_node_indices(data_root, ntl)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    df = build_expanded_balance_list(args.data_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} balance nodes -> {args.out}")


if __name__ == "__main__":
    main()
