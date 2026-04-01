"""
Build a reduced node dataset: one row per (sample_id, mv_node) from the mapping.

Reads:
  - mv_x_sx_node_mapping_8500.csv  (mv_node + SX/LV node names)
  - gnn_node_features_and_targets.csv  (full daily-agg run; streamed by sample_id)

For each mapping row and each sample:
  - p_load_kw / q_load_kvar = sum of the two load nodes: prefer (sx_node_1, sx_node_2)
    when both are non-empty in the mapping; otherwise (lv_x_node_1, lv_x_node_2).
  - vmag_pu / vang_deg / electrical_distance_ohm / node_idx / bus / phase come from
    the mv_node row in the source file.

Output CSV uses the same columns as gnn_node_features_and_targets.csv.

Node names in the mapping may use any case; matching against the large CSV is
case-insensitive (OpenDSS exports often use lowercase node strings).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

FIELDNAMES = [
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


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _load_mapping(mapping_path: Path) -> tuple[list[dict], set[str]]:
    """Return list of rules and set of all node names we must read from the big CSV.

    Keys are lowercased so they match OpenDSS node strings in the dataset (often lowercase).
    """
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


def _flush_sample(
    buffer: dict[str, dict],
    sample_id: str,
    rules: list[dict],
    writer: csv.DictWriter,
    stats: dict,
) -> None:
    """Write one output row per mapping rule for this sample."""
    for rec in rules:
        mv = rec["mv_key"]
        la, lb = rec["load_a"], rec["load_b"]
        if mv not in buffer:
            stats["missing_mv"] += 1
            continue
        row_mv = buffer[mv]
        pa = _float_cell(buffer[la], "p_load_kw") if la in buffer else 0.0
        pb = _float_cell(buffer[lb], "p_load_kw") if lb in buffer else 0.0
        qa = _float_cell(buffer[la], "q_load_kvar") if la in buffer else 0.0
        qb = _float_cell(buffer[lb], "q_load_kvar") if lb in buffer else 0.0
        if la not in buffer:
            stats["missing_load_node"] += 1
        if lb not in buffer:
            stats["missing_load_node"] += 1

        writer.writerow(
            {
                "sample_id": sample_id,
                "node": mv,
                "node_idx": row_mv["node_idx"],
                "bus": row_mv["bus"],
                "phase": row_mv["phase"],
                "electrical_distance_ohm": row_mv["electrical_distance_ohm"],
                "p_load_kw": pa + pb,
                "q_load_kvar": qa + qb,
                "vmag_pu": row_mv["vmag_pu"],
                "vang_deg": row_mv["vang_deg"],
            }
        )
        stats["rows_out"] += 1
    stats["samples_flushed"] += 1


def run(
    mapping_csv: Path,
    node_features_csv: Path,
    output_csv: Path,
    max_samples: int | None = None,
) -> None:
    rules, needed_nodes = _load_mapping(mapping_csv)
    if not rules:
        raise SystemExit("No valid rows in mapping CSV.")

    stats = {
        "rows_out": 0,
        "missing_mv": 0,
        "missing_load_node": 0,
        "samples_flushed": 0,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(node_features_csv, newline="", encoding="utf-8", errors="replace") as fin, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        cols = set(reader.fieldnames or [])
        if set(FIELDNAMES) != cols:
            missing = set(FIELDNAMES) - cols
            extra = cols - set(FIELDNAMES)
            raise SystemExit(
                f"Unexpected node CSV columns. Missing {missing}. Extra {extra}. Got: {reader.fieldnames}"
            )

        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
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
                _flush_sample(buffer, current_sid, rules, writer, stats)
                buffer = {}
                if max_samples is not None and stats["samples_flushed"] >= max_samples:
                    break

            current_sid = sid_str

            if node_key not in needed_nodes:
                continue

            buffer[node_key] = row

        if buffer and current_sid is not None:
            _flush_sample(buffer, current_sid, rules, writer, stats)

    print(f"[aggregate_mv_node_dataset_8500] wrote {stats['rows_out']} rows -> {output_csv}")
    print(f"  samples (power-flow cases) written: {stats['samples_flushed']}")
    print(f"  missing_mv (no mv_node in buffer for that sample): {stats['missing_mv']}")
    print(f"  missing load-node increments (node not in buffer): {stats['missing_load_node']}")


def main(argv: list[str] | None = None) -> None:
    repo = _repo_root()
    default_dir = repo / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p = argparse.ArgumentParser(description="MV-only node dataset from mapping + full node CSV.")
    p.add_argument(
        "--mapping-csv",
        type=Path,
        default=default_dir / "mv_x_sx_node_mapping_8500.csv",
        help="Path to mv_x_sx_node_mapping_8500.csv",
    )
    p.add_argument(
        "--node-csv",
        type=Path,
        default=default_dir / "gnn_node_features_and_targets.csv",
        help="Full gnn_node_features_and_targets.csv",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=default_dir / "gnn_node_features_and_targets_mv_only.csv",
        help="Output path (same schema as node CSV)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after this many completed samples (for testing)",
    )
    args = p.parse_args(argv)

    if not args.mapping_csv.is_file():
        print(f"Missing mapping: {args.mapping_csv}", file=sys.stderr)
        sys.exit(1)
    if not args.node_csv.is_file():
        print(f"Missing node CSV: {args.node_csv}", file=sys.stderr)
        sys.exit(1)

    run(
        mapping_csv=args.mapping_csv.resolve(),
        node_features_csv=args.node_csv.resolve(),
        output_csv=args.output_csv.resolve(),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
