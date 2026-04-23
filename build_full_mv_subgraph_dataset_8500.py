"""
Build an MV-only homogenous GNN dataset from an existing full daily-aggregate export.

- Nodes: all phase nodes whose bus kVBase is in [min_kv_base, max_kv_base] (default 2–20 kV,
  which selects 7.2 kV distribution buses on IEEE 8500 and excludes 0.12 kV and 66 kV).
- Edges: rows from gnn_edges_phase_static.csv where both endpoints are MV nodes — one row per
  Line/Transformer phase branch (same R_full/X_full as the full export; no load-only compaction).
- Node P/Q: for the 1,177 mv_node entries in mv_x_sx_node_mapping_8500.csv, set p_load_kw /
  q_load_kvar to the sum of the two mapped LV/SX load nodes (same rule as aggregate_mv_node_dataset_8500).
  All other MV nodes get P=Q=0. Voltages / electrical_distance_ohm are copied from the full node CSV.

Requires OpenDSS once at startup to read kVBase per bus.

Typical outputs (under --out-dir):
  gnn_node_index_full_mv.csv
  gnn_edges_phase_static_full_mv.csv
  gnn_node_features_and_targets_full_mv.csv
  full_mv_subgraph_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

NODE_FIELDS = [
    "sample_id",
    "node",
    "node_idx",
    "bus",
    "phase",
    "electrical_distance_ohm",
    "p_load_kw",
    "q_load_kvar",
    "vmag_pu",
    "vang_deg",
]


def _load_mapping(mapping_path: Path) -> tuple[list[dict], set[str]]:
    rules: list[dict] = []
    needed: set[str] = set()
    with open(mapping_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mv = (row.get("mv_node") or "").strip()
            lv1 = (row.get("lv_x_node_1") or "").strip()
            lv2 = (row.get("lv_x_node_2") or "").strip()
            sx1 = (row.get("sx_node_1") or "").strip()
            sx2 = (row.get("sx_node_2") or "").strip()
            if not mv or not lv1 or not lv2:
                continue
            if sx1 and sx2:
                la, lb = sx1, sx2
            else:
                la, lb = lv1, lv2
            mv_k = mv.lower()
            la_k, lb_k = la.lower(), lb.lower()
            rules.append({"mv_key": mv_k, "load_a": la_k, "load_b": lb_k})
            needed.add(mv_k)
            needed.add(la_k)
            needed.add(lb_k)
    return rules, needed


def _float_cell(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _bus_kv_map(dss_master: Path) -> dict[str, float]:
    if not dss_master.is_file():
        raise FileNotFoundError(f"Missing DSS master: {dss_master}")
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{dss_master.resolve()}"')
    dss.Text.Command("set mode=daily")
    out: dict[str, float] = {}
    for b in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(b)
        out[str(b).strip().lower()] = float(dss.Bus.kVBase())
    return out


def _mv_phase_nodes(
    node_index_csv: Path,
    bus_kv: dict[str, float],
    *,
    lo_kv: float,
    hi_kv: float,
) -> list[str]:
    df = pd.read_csv(node_index_csv, usecols=["node"])
    out: list[str] = []
    for n in df["node"].astype(str):
        bus = n.split(".")[0].strip().lower()
        kv = bus_kv.get(bus)
        if kv is None:
            continue
        if lo_kv <= kv <= hi_kv:
            out.append(n.strip())
    return sorted(out, key=lambda s: s.lower())


def _write_edges_full_mv(
    edges_csv: Path,
    mv_nodes: list[str],
    out_csv: Path,
) -> int:
    mv_set = set(mv_nodes)
    old_to_new = {n: i for i, n in enumerate(mv_nodes)}
    df = pd.read_csv(edges_csv)
    need = {"from_node", "to_node", "u_idx", "v_idx", "R_full", "X_full"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{edges_csv} missing columns {miss}")
    m = df["from_node"].astype(str).isin(mv_set) & df["to_node"].astype(str).isin(mv_set)
    df2 = df.loc[m].copy()
    df2["u_idx"] = df2["from_node"].map(lambda s: old_to_new[str(s)])
    df2["v_idx"] = df2["to_node"].map(lambda s: old_to_new[str(s)])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False)
    return int(len(df2))


def _stream_full_mv_nodes(
    *,
    full_node_csv: Path,
    out_csv: Path,
    mv_nodes: list[str],
    injection_by_mv: dict[str, dict],
    mapping_needed: set[str],
    max_samples: int | None,
) -> dict:
    mv_set_lower = {n.lower() for n in mv_nodes}
    needed = mv_set_lower | mapping_needed

    old_to_new = {n.lower(): i for i, n in enumerate(mv_nodes)}
    stats = {
        "rows_out": 0,
        "samples_flushed": 0,
        "missing_mv": 0,
        "missing_load_node": 0,
    }

    def flush(buffer: dict[str, dict], sample_id: str, writer: csv.DictWriter) -> None:
        for canon in mv_nodes:
            key = canon.lower()
            if key not in buffer:
                stats["missing_mv"] += 1
                continue
            base = buffer[key]
            bus, phs = canon.split(".", 1)
            ph = int(phs)
            rec = injection_by_mv.get(key)
            if rec is not None:
                la, lb = rec["load_a"], rec["load_b"]
                pa = _float_cell(buffer[la], "p_load_kw") if la in buffer else 0.0
                pb = _float_cell(buffer[lb], "p_load_kw") if lb in buffer else 0.0
                qa = _float_cell(buffer[la], "q_load_kvar") if la in buffer else 0.0
                qb = _float_cell(buffer[lb], "q_load_kvar") if lb in buffer else 0.0
                if la not in buffer:
                    stats["missing_load_node"] += 1
                if lb not in buffer:
                    stats["missing_load_node"] += 1
                p_load = pa + pb
                q_load = qa + qb
            else:
                p_load = 0.0
                q_load = 0.0

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "node": base["node"],
                    "node_idx": old_to_new[key],
                    "bus": bus,
                    "phase": ph,
                    "electrical_distance_ohm": _float_cell(base, "electrical_distance_ohm"),
                    "p_load_kw": float(p_load),
                    "q_load_kvar": float(q_load),
                    "vmag_pu": _float_cell(base, "vmag_pu"),
                    "vang_deg": _float_cell(base, "vang_deg"),
                }
            )
            stats["rows_out"] += 1
        stats["samples_flushed"] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(full_node_csv, newline="", encoding="utf-8", errors="replace") as fin, open(
        out_csv, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        cols = set(reader.fieldnames or [])
        if set(NODE_FIELDS) != cols:
            raise SystemExit(
                f"Unexpected columns in {full_node_csv}. Want {NODE_FIELDS}, got {reader.fieldnames}"
            )
        writer = csv.DictWriter(fout, fieldnames=NODE_FIELDS)
        writer.writeheader()

        buffer: dict[str, dict] = {}
        current_sid: str | None = None

        for row in reader:
            node = (row.get("node") or "").strip()
            node_key = node.lower()
            sid = row.get("sample_id")
            if sid is None:
                continue
            sid_str = str(sid)

            if current_sid is not None and sid_str != current_sid:
                flush(buffer, current_sid, writer)
                buffer = {}
                if max_samples is not None and stats["samples_flushed"] >= max_samples:
                    break

            current_sid = sid_str

            if node_key not in needed:
                continue
            buffer[node_key] = row

        if buffer and current_sid is not None:
            flush(buffer, current_sid, writer)

    return stats


def main(argv: list[str] | None = None) -> None:
    default_ds = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p = argparse.ArgumentParser(description="MV subgraph node+edge CSVs from full daily 8500 export.")
    p.add_argument("--dss-master", type=Path, default=REPO / "8500-node" / "Run_8500Node_Daily_5min.dss")
    p.add_argument("--node-index", type=Path, default=default_ds / "gnn_node_index_master.csv")
    p.add_argument("--full-node-csv", type=Path, default=default_ds / "gnn_node_features_and_targets.csv")
    p.add_argument("--full-edge-csv", type=Path, default=default_ds / "gnn_edges_phase_static.csv")
    p.add_argument(
        "--mapping-csv",
        type=Path,
        default=default_ds / "mv_x_sx_node_mapping_8500.csv",
        help="Fallback: 8500-node/mv_x_sx_node_mapping_8500.csv if default missing",
    )
    p.add_argument("--out-dir", type=Path, default=default_ds.parent / "loadtype_8500_dailyagg_full_mv")
    p.add_argument("--min-kv-base", type=float, default=2.0)
    p.add_argument("--max-kv-base", type=float, default=20.0)
    p.add_argument("--max-samples", type=int, default=None, help="Stop after N samples (test)")
    args = p.parse_args(argv)

    mapping_path = args.mapping_csv
    if not mapping_path.is_file():
        alt = REPO / "8500-node" / "mv_x_sx_node_mapping_8500.csv"
        if alt.is_file():
            mapping_path = alt
        else:
            print(f"Missing mapping CSV: {args.mapping_csv}", file=sys.stderr)
            sys.exit(1)

    for label, path in (
        ("node_index", args.node_index),
        ("full_node_csv", args.full_node_csv),
        ("full_edge_csv", args.full_edge_csv),
    ):
        if not path.is_file():
            print(f"Missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    rules, mapping_needed = _load_mapping(mapping_path)
    if not rules:
        raise SystemExit("No valid mapping rows.")
    injection_by_mv: dict[str, dict] = {}
    for r in rules:
        injection_by_mv[r["mv_key"]] = r

    print("[full_mv_subgraph] compiling OpenDSS for kVBase...", flush=True)
    bus_kv = _bus_kv_map(args.dss_master)
    mv_nodes = _mv_phase_nodes(
        args.node_index,
        bus_kv,
        lo_kv=float(args.min_kv_base),
        hi_kv=float(args.max_kv_base),
    )
    if not mv_nodes:
        raise SystemExit("No MV phase nodes in kV band; check min/max kV.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_out = args.out_dir / "gnn_node_index_full_mv.csv"
    edges_out = args.out_dir / "gnn_edges_phase_static_full_mv.csv"
    nodes_out = args.out_dir / "gnn_node_features_and_targets_full_mv.csv"
    manifest_out = args.out_dir / "full_mv_subgraph_manifest.json"

    pd.DataFrame({"node": mv_nodes, "node_idx": np.arange(len(mv_nodes), dtype=int)}).to_csv(
        index_out, index=False
    )
    n_e = _write_edges_full_mv(args.full_edge_csv, mv_nodes, edges_out)
    stats = _stream_full_mv_nodes(
        full_node_csv=args.full_node_csv,
        out_csv=nodes_out,
        mv_nodes=mv_nodes,
        injection_by_mv=injection_by_mv,
        mapping_needed=mapping_needed,
        max_samples=args.max_samples,
    )

    manifest = {
        "n_mv_phase_nodes": len(mv_nodes),
        "n_edges_full_mv_csv": n_e,
        "n_mapping_injection_nodes": len({r["mv_key"] for r in rules}),
        "kv_base_band": [float(args.min_kv_base), float(args.max_kv_base)],
        "dss_master": str(args.dss_master.resolve()),
        "sources": {
            "node_index": str(args.node_index.resolve()),
            "full_node_csv": str(args.full_node_csv.resolve()),
            "full_edge_csv": str(args.full_edge_csv.resolve()),
            "mapping_csv": str(mapping_path.resolve()),
        },
        "outputs": {
            "node_index": str(index_out.resolve()),
            "edges": str(edges_out.resolve()),
            "node_features": str(nodes_out.resolve()),
        },
        "stream_stats": stats,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[full_mv_subgraph] MV phase nodes N={len(mv_nodes)}  edges={n_e}")
    print(f"  wrote {index_out}")
    print(f"  wrote {edges_out}")
    print(f"  wrote {nodes_out}  ({stats})")
    print(f"  manifest -> {manifest_out}")


if __name__ == "__main__":
    main()
