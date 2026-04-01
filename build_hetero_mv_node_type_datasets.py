"""
Build four CSVs for heterogeneous GNN node types.

Reads gnn_node_features_and_targets.csv (full node table, all buses) and streams rows.

Node sets:
  1) regulator_upstream — rows only for terminal_1 node (regxfmr side) from regulator_involved_nodes.csv
  2) regulator_downstream — rows only for terminal_2 node (bus side) from regulator_involved_nodes.csv
     (not the full upstream/downstream graph partitions)
  3) capacitor — only the fixed From nodes listed below (not path anchors, not Feeder-side)
  4) load_transformer — same nodes as mv_only; row features from full gnn_node_features_and_targets.csv
     except p_load_kw / q_load_kvar, which are always taken from gnn_node_features_and_targets_mv_only
     (aggregated LV/SX loads; full CSV MV rows often have 0 here)

q_capacitor_bank: per sample_id, from gnn_sample_meta cap_*_q_post_kvar via capacitor_involved CAP → column,
only for the fixed cap From nodes. If several nodes share the same meta column (e.g. CAPBank3 → one
cap_capbank3_q_post_kvar for three phases), that column’s value is divided equally among those nodes
(one third each for three nodes).

Outputs (default: datasets_gnn2/loadtype_8500_dailyagg/):
  hetero_mv_nodes_regulator_upstream.csv
  hetero_mv_nodes_regulator_downstream.csv
  hetero_mv_nodes_capacitor_related.csv
  hetero_mv_nodes_load_transformer.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

OUT_NAMES = (
    "hetero_mv_nodes_regulator_upstream.csv",
    "hetero_mv_nodes_regulator_downstream.csv",
    "hetero_mv_nodes_capacitor_related.csv",
    "hetero_mv_nodes_load_transformer.csv",
)

FIELDNAMES = [
    "sample_id",
    "node",
    "node_idx",
    "electrical_distance_ohm",
    "p_load_kw",
    "q_load_kvar",
    "q_capacitor_bank",
    "vmag_pu",
    "vang_deg",
]

# Required columns in gnn_node_features_and_targets.csv
NODE_CSV_COLS = [
    "sample_id",
    "node",
    "node_idx",
    "electrical_distance_ohm",
    "p_load_kw",
    "q_load_kvar",
    "vmag_pu",
    "vang_deg",
]

# Capacitor From nodes only (capacitor-related CSV is restricted to these rows).
CAP_FROM_NODES: tuple[str, ...] = (
    "L2823592.1",
    "L2823592.2",
    "L2823592.3",
    "Q16483.1",
    "Q16483.2",
    "Q16483.3",
    "Q16642.1",
    "Q16642.2",
    "Q16642.3",
    "R18242.1",
    "R18242.2",
    "R18242.3",
)

CAP_TO_Q_POST: dict[str, str] = {
    "CAP_1A": "cap_capbank0a_q_post_kvar",
    "CAP_1B": "cap_capbank0b_q_post_kvar",
    "CAP_1C": "cap_capbank0c_q_post_kvar",
    "CAP_2A": "cap_capbank1a_q_post_kvar",
    "CAP_2B": "cap_capbank1b_q_post_kvar",
    "CAP_2C": "cap_capbank1c_q_post_kvar",
    "CAP_3A": "cap_capbank2a_q_post_kvar",
    "CAP_3B": "cap_capbank2b_q_post_kvar",
    "CAP_3C": "cap_capbank2c_q_post_kvar",
    "CAPBank3": "cap_capbank3_q_post_kvar",
}


def _canonical_map(node_index_csv: Path) -> dict[str, str]:
    df = pd.read_csv(node_index_csv)
    return {str(n).strip().lower(): str(n).strip() for n in df["node"].astype(str)}


def _norm_sample_id(s: object) -> str:
    """Stable string for sample_id keys (avoids '0' vs '0.0' mismatch between CSVs)."""
    if s is None or s == "":
        return ""
    try:
        x = float(s)
        if x == int(x):
            return str(int(x))
        return str(x)
    except (TypeError, ValueError):
        return str(s).strip()


def _regulator_terminal_node_sets(regulator_csv: Path, canon: dict[str, str]) -> tuple[set[str], set[str]]:
    """terminal_1 nodes (upstream file) and terminal_2 nodes (downstream file), canonical names."""
    df = pd.read_csv(regulator_csv)
    t1_set: set[str] = set()
    t2_set: set[str] = set()
    for _, row in df.iterrows():
        t1 = str(row.get("terminal_1 node", "")).strip()
        t2 = str(row.get("terminal_2 node", "")).strip()
        if not t1 or not t2 or t1.lower() == "nan" or t2.lower() == "nan":
            continue
        t1_set.add(canon.get(t1.lower(), t1))
        t2_set.add(canon.get(t2.lower(), t2))
    return t1_set, t2_set


def _cap_q_cols_for_fixed_from_nodes(
    capacitor_csv: Path,
    canon: dict[str, str],
    cap_fixed: set[str],
) -> dict[str, list[str]]:
    """From node only; only rows whose From node is in cap_fixed."""
    df = pd.read_csv(capacitor_csv)
    acc: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        cap_name = str(row.get("CAP", "")).strip()
        col = CAP_TO_Q_POST.get(cap_name)
        if col is None:
            raise ValueError(f"Unknown CAP={cap_name!r}; add to CAP_TO_Q_POST")
        fn = str(row.get("From node", "")).strip()
        if not fn or fn.lower() == "nan":
            continue
        n = canon.get(fn.lower(), fn)
        if n not in cap_fixed:
            continue
        acc.setdefault(n, []).append(col)
    return acc


def _meta_col_share_counts(node_cap_cols: dict[str, list[str]]) -> dict[str, int]:
    """How many cap From nodes map to each meta q_post column (for equal split of bank-total Q)."""
    counts: dict[str, int] = {}
    for _, cols in node_cap_cols.items():
        for c in set(cols):
            counts[c] = counts.get(c, 0) + 1
    return counts


def _compute_type_sets(
    regulator_csv: Path,
    capacitor_csv: Path,
    node_index_csv: Path,
) -> tuple[set[str], set[str], set[str], dict[str, list[str]]]:
    canon = _canonical_map(node_index_csv)
    reg_t1_nodes, reg_t2_nodes = _regulator_terminal_node_sets(regulator_csv, canon)

    cap_fixed = {canon.get(x.lower(), x.strip()) for x in CAP_FROM_NODES}
    node_cap_cols = _cap_q_cols_for_fixed_from_nodes(capacitor_csv, canon, cap_fixed)

    return reg_t1_nodes, reg_t2_nodes, cap_fixed, node_cap_cols


def run(
    node_features_csv: Path,
    mv_only_csv: Path,
    sample_meta_csv: Path,
    regulator_csv: Path,
    capacitor_csv: Path,
    node_index_csv: Path,
    out_dir: Path,
    max_samples: int | None = None,
) -> None:
    canon = _canonical_map(node_index_csv)
    reg_t1_nodes, reg_t2_nodes, cap_fixed, node_cap_cols = _compute_type_sets(
        regulator_csv, capacitor_csv, node_index_csv
    )

    meta = pd.read_csv(sample_meta_csv)
    if "sample_id" not in meta.columns:
        raise ValueError("gnn_sample_meta.csv must have sample_id")
    needed_q_cols = sorted({c for cols in node_cap_cols.values() for c in cols})
    missing = [c for c in needed_q_cols if c not in meta.columns]
    if missing:
        raise ValueError(f"gnn_sample_meta.csv missing q_post columns: {missing}")
    meta_idx = meta.set_index("sample_id", drop=False)
    cap_col_shares = _meta_col_share_counts(node_cap_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / n for n in OUT_NAMES]
    files = [open(p, "w", newline="", encoding="utf-8") for p in paths]

    writers = [csv.DictWriter(f, fieldnames=FIELDNAMES) for f in files]
    for w in writers:
        w.writeheader()

    def q_cap_for_node(sample_id: int | str, node_canon: str) -> float:
        cols = node_cap_cols.get(node_canon, [])
        if not cols:
            return 0.0
        try:
            row = meta_idx.loc[int(sample_id)]
        except (KeyError, TypeError, ValueError):
            try:
                row = meta_idx.loc[sample_id]
            except Exception:
                return 0.0
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        total = 0.0
        seen: set[str] = set()
        for c in cols:
            if c in seen:
                continue
            seen.add(c)
            n_share = max(1, cap_col_shares.get(c, 1))
            total += float(row.get(c, 0) or 0) / n_share
        return total

    stats = {n: 0 for n in OUT_NAMES}
    samples_done = 0
    prev_sid: str | None = None

    # Load-transformer: MV nodes from mv_only + (sample_id, node) -> P,Q from mv_only (not full CSV — MV buses
    # often have zero load there; mv_only aggregates from SX/LV load nodes per aggregate_mv_node_dataset_8500.)
    mv_nodes_in_dataset: set[str] = set()
    mv_pq_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    with open(mv_only_csv, newline="", encoding="utf-8", errors="replace") as fmv:
        r = csv.DictReader(fmv)
        fn = r.fieldnames or []
        if "node" not in fn or "sample_id" not in fn:
            raise SystemExit(f"{mv_only_csv} must have sample_id and node columns")
        if "p_load_kw" not in fn or "q_load_kvar" not in fn:
            raise SystemExit(f"{mv_only_csv} must have p_load_kw and q_load_kvar columns")

        def _f(row: dict, key: str) -> float:
            v = row.get(key)
            if v is None or v == "":
                return 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        for row in r:
            n = (row.get("node") or "").strip()
            if not n:
                continue
            n_canon = canon.get(n.lower(), n)
            mv_nodes_in_dataset.add(n_canon)
            sid_key = _norm_sample_id(row.get("sample_id"))
            mv_pq_lookup[(sid_key, n_canon)] = (_f(row, "p_load_kw"), _f(row, "q_load_kvar"))

    load_set = mv_nodes_in_dataset

    with open(node_features_csv, newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        if set(NODE_CSV_COLS) - set(reader.fieldnames or []):
            missing_f = set(NODE_CSV_COLS) - set(reader.fieldnames or [])
            raise SystemExit(f"node features CSV missing columns: {missing_f}")

        for row in reader:
            sid = row.get("sample_id")
            if sid is None:
                continue
            sid_str = _norm_sample_id(sid)

            if prev_sid is not None and sid_str != prev_sid:
                samples_done += 1
                if max_samples is not None and samples_done >= max_samples:
                    break

            prev_sid = sid_str

            node_raw = (row.get("node") or "").strip()
            n_canon = canon.get(node_raw.lower(), node_raw)
            qcb = q_cap_for_node(sid_str, n_canon)
            out_row = {
                "sample_id": sid_str,
                "node": n_canon,
                "node_idx": row.get("node_idx", ""),
                "electrical_distance_ohm": row.get("electrical_distance_ohm", ""),
                "p_load_kw": row.get("p_load_kw", ""),
                "q_load_kvar": row.get("q_load_kvar", ""),
                "q_capacitor_bank": f"{qcb:.12g}" if qcb != 0.0 else "0",
                "vmag_pu": row.get("vmag_pu", ""),
                "vang_deg": row.get("vang_deg", ""),
            }

            if n_canon in reg_t1_nodes:
                writers[0].writerow(out_row)
                stats[OUT_NAMES[0]] += 1
            if n_canon in reg_t2_nodes:
                writers[1].writerow(out_row)
                stats[OUT_NAMES[1]] += 1
            if n_canon in cap_fixed:
                writers[2].writerow(out_row)
                stats[OUT_NAMES[2]] += 1
            if n_canon in load_set:
                # P/Q always from mv_only (see module docstring); default (0,0) if key missing
                pq = mv_pq_lookup.get((sid_str, n_canon), (0.0, 0.0))
                load_row = dict(out_row)
                load_row["p_load_kw"] = f"{pq[0]:.12g}" if pq[0] != 0.0 else "0"
                load_row["q_load_kvar"] = f"{pq[1]:.12g}" if pq[1] != 0.0 else "0"
                writers[3].writerow(load_row)
                stats[OUT_NAMES[3]] += 1

    for f in files:
        f.close()

    print(f"[build_hetero_mv_node_type_datasets] out_dir={out_dir.resolve()}")
    for name, p in zip(OUT_NAMES, paths, strict=True):
        print(f"  {name}: {stats[name]} rows")
    print(
        f"  sets: regulator terminal_1={len(reg_t1_nodes)} terminal_2={len(reg_t2_nodes)} "
        f"cap_from_fixed={len(cap_fixed)} load_xfmr(nodes_in_mv_only)={len(load_set)}"
    )


def main(argv: list[str] | None = None) -> None:
    d = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p = argparse.ArgumentParser(description="Four hetero node-type CSVs from full node features + mv_only load filter.")
    p.add_argument("--node-csv", type=Path, default=d / "gnn_node_features_and_targets.csv", help="Full node table")
    p.add_argument(
        "--mv-only-csv",
        type=Path,
        default=d / "gnn_node_features_and_targets_mv_only.csv",
        help="Node list + P/Q for load_transformer rows (aggregated loads; full node CSV often has 0 on MV buses)",
    )
    p.add_argument("--sample-meta", type=Path, default=d / "gnn_sample_meta.csv")
    p.add_argument("--regulator", type=Path, default=d / "regulator_involved_nodes.csv")
    p.add_argument("--capacitor", type=Path, default=d / "capacitor_involved_nodes.csv")
    p.add_argument("--node-index", type=Path, default=d / "gnn_node_index_master.csv")
    p.add_argument("--out-dir", type=Path, default=d)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)
    run(
        node_features_csv=args.node_csv.resolve(),
        mv_only_csv=args.mv_only_csv.resolve(),
        sample_meta_csv=args.sample_meta.resolve(),
        regulator_csv=args.regulator.resolve(),
        capacitor_csv=args.capacitor.resolve(),
        node_index_csv=args.node_index.resolve(),
        out_dir=args.out_dir.resolve(),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
