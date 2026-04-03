"""
Build heterogeneous edge dataset from gnn_edges_phase_static_mv_only.csv.

Edge types:
  - line: static features R_full, X_full from the MV-only CSV.
  - regulator: per-sample single tap reg_tap_pu — the column chosen from
    gnn_sample_meta.csv using regulator_involved_nodes.csv (Regulator name →
    one tap column per phase device).

Outputs (default: same directory as --mv-edges, i.e. datasets_gnn2/loadtype_8500_dailyagg/ unless --out-dir is set):
  - hetero_mv_edge_catalog.csv
      edge_id, from_node, to_node, edge_type, Regulator, tap_column, R_full, X_full, u_idx, v_idx
  - hetero_mv_line_edge_attr.csv
      edge_id, R_full, X_full   (one row per line edge; static)
  - hetero_mv_regulator_edge_features.csv
      sample_id, from_node, to_node, edge_id, Regulator, reg_tap_pu
      (one row per regulator edge × sample; tap matches that edge's regulator only)

Inference consistency (`compare_hetero_mv_daily.py`):
  - Loads the same three files from ``<dataset_dir>/edges/`` (default dataset_dir ends with
    ``Heterogenous GNN dataset``). REGULATOR_TO_TAP_COL below is imported by that script for
    tap lookup; topology uses ``search_hetero_mv_gnn_architectures._build_typed_topology`` with
    this catalog + line_attr — same as training.
  - At daily inference, regulator taps come from OpenDSS ``_read_reg_control_state`` (keys
    ``reg_<RegControlName>_tap_pu``) matched to each catalog row via REGULATOR_TO_TAP_COL +
    ``Regulator`` label — same meta column names as ``tap_column`` here.
  - If you use the bundled hetero dataset path for compare, rebuild into that ``edges`` folder, e.g.:
    ``--out-dir datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset/edges``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    REPO = Path(__file__).resolve().parent
except NameError:
    REPO = Path.cwd()

REGULATOR_TO_TAP_COL: dict[str, str] = {
    "FEEDER_REGA": "reg_feeder_rega_tap_pu",
    "FEEDER_REGB": "reg_feeder_regb_tap_pu",
    "FEEDER_REGC": "reg_feeder_regc_tap_pu",
    "VREG2_A": "reg_vreg2_a_tap_pu",
    "VREG2_B": "reg_vreg2_b_tap_pu",
    "VREG2_C": "reg_vreg2_c_tap_pu",
    "VREG3_A": "reg_vreg3_a_tap_pu",
    "VREG3_B": "reg_vreg3_b_tap_pu",
    "VREG3_C": "reg_vreg3_c_tap_pu",
    "VREG4_A": "reg_vreg4_a_tap_pu",
    "VREG4_B": "reg_vreg4_b_tap_pu",
    "VREG4_C": "reg_vreg4_c_tap_pu",
}


def _canonical_node_map(node_index_csv: Path) -> dict[str, str]:
    df = pd.read_csv(node_index_csv)
    return {str(n).strip().lower(): str(n).strip() for n in df["node"].astype(str)}


def _node_to_idx(node_index_csv: Path) -> dict[str, int]:
    df = pd.read_csv(node_index_csv)
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        k = str(row["node"]).strip().lower()
        out[k] = int(row["node_idx"])
    return out


def _pair_key(c1: str, c2: str) -> tuple[str, str]:
    a, b = (c1, c2) if c1 <= c2 else (c2, c1)
    return (a, b)


def _regulator_pair_to_name(regulator_csv: Path, canon: dict[str, str]) -> dict[tuple[str, str], str]:
    """Map undirected terminal pair → Regulator name from regulator_involved_nodes.csv."""
    df = pd.read_csv(regulator_csv)
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        reg_name = str(row.get("Regulator", "")).strip()
        t1 = str(row.get("terminal_1 node", "")).strip()
        t2 = str(row.get("terminal_2 node", "")).strip()
        if not reg_name or not t1 or not t2 or t1.lower() == "nan" or t2.lower() == "nan":
            continue
        c1 = canon.get(t1.lower(), t1)
        c2 = canon.get(t2.lower(), t2)
        pk = _pair_key(c1, c2)
        if pk in out and out[pk] != reg_name:
            raise ValueError(f"Duplicate pair {pk} with conflicting Regulator: {out[pk]} vs {reg_name}")
        out[pk] = reg_name
    return out


def build(
    mv_edges_csv: Path,
    regulator_csv: Path,
    node_index_csv: Path,
    sample_meta_csv: Path,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or mv_edges_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    canon = _canonical_node_map(node_index_csv)
    n2i = _node_to_idx(node_index_csv)
    pair_to_regulator = _regulator_pair_to_name(regulator_csv, canon)
    reg_pair_set = set(pair_to_regulator.keys())

    df_e = pd.read_csv(mv_edges_csv)
    if "from_node" not in df_e.columns or "to_node" not in df_e.columns:
        raise ValueError(f"{mv_edges_csv} must have from_node, to_node")

    edge_types: list[str] = []
    regulator_names: list[str | float] = []
    tap_columns: list[str | float] = []
    for _, row in df_e.iterrows():
        f = canon.get(str(row["from_node"]).strip().lower(), str(row["from_node"]).strip())
        t = canon.get(str(row["to_node"]).strip().lower(), str(row["to_node"]).strip())
        pk = _pair_key(f, t)
        if pk in reg_pair_set:
            edge_types.append("regulator")
            rname = pair_to_regulator[pk]
            regulator_names.append(rname)
            col = REGULATOR_TO_TAP_COL.get(rname)
            if col is None:
                raise ValueError(f"No tap column mapping for Regulator={rname!r}; extend REGULATOR_TO_TAP_COL")
            tap_columns.append(col)
        else:
            edge_types.append("line")
            regulator_names.append(np.nan)
            tap_columns.append(np.nan)

    df_e = df_e.copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=np.int64)
    df_e["edge_type"] = edge_types
    df_e["Regulator"] = regulator_names
    df_e["tap_column"] = tap_columns

    u_idx: list[float] = []
    v_idx: list[float] = []
    for _, row in df_e.iterrows():
        fu = str(row["from_node"]).strip().lower()
        tv = str(row["to_node"]).strip().lower()
        u_idx.append(float(n2i[fu]) if fu in n2i else np.nan)
        v_idx.append(float(n2i[tv]) if tv in n2i else np.nan)

    catalog = pd.DataFrame(
        {
            "edge_id": df_e["edge_id"],
            "from_node": df_e["from_node"].astype(str).str.strip(),
            "to_node": df_e["to_node"].astype(str).str.strip(),
            "edge_type": df_e["edge_type"],
            "Regulator": df_e["Regulator"],
            "tap_column": df_e["tap_column"],
            "R_full": pd.to_numeric(df_e.get("R_full"), errors="coerce"),
            "X_full": pd.to_numeric(df_e.get("X_full"), errors="coerce"),
            "u_idx": u_idx,
            "v_idx": v_idx,
        }
    )
    path_catalog = out_dir / "hetero_mv_edge_catalog.csv"
    catalog.to_csv(path_catalog, index=False)

    line_mask = df_e["edge_type"] == "line"
    line_only = df_e.loc[line_mask, ["edge_id", "R_full", "X_full"]].copy()
    path_line = out_dir / "hetero_mv_line_edge_attr.csv"
    line_only.to_csv(path_line, index=False)

    meta = pd.read_csv(sample_meta_csv)
    needed_cols = set(REGULATOR_TO_TAP_COL.values())
    missing = [c for c in sorted(needed_cols) if c not in meta.columns]
    if missing:
        raise ValueError(f"gnn_sample_meta.csv missing columns: {missing}")

    reg_rows = df_e[df_e["edge_type"] == "regulator"][
        ["edge_id", "from_node", "to_node", "Regulator", "tap_column"]
    ].copy()
    n_s = len(meta)
    n_re = len(reg_rows)
    if n_re == 0:
        reg_feat = pd.DataFrame(
            columns=["sample_id", "from_node", "to_node", "edge_id", "Regulator", "reg_tap_pu"]
        )
    else:
        parts: list[pd.DataFrame] = []
        for _, er in reg_rows.iterrows():
            col = str(er["tap_column"])
            block = pd.DataFrame(
                {
                    "sample_id": meta["sample_id"].values,
                    "from_node": str(er["from_node"]).strip(),
                    "to_node": str(er["to_node"]).strip(),
                    "edge_id": int(er["edge_id"]),
                    "Regulator": str(er["Regulator"]).strip(),
                    "reg_tap_pu": pd.to_numeric(meta[col], errors="coerce").values,
                }
            )
            parts.append(block)
        reg_feat = pd.concat(parts, ignore_index=True)

    path_reg = out_dir / "hetero_mv_regulator_edge_features.csv"
    reg_feat.to_csv(path_reg, index=False)

    n_line = int(line_mask.sum())
    n_reg = int((df_e["edge_type"] == "regulator").sum())
    print(f"[build_hetero_mv_edge_dataset] wrote {out_dir}/")
    print(f"  edges total={len(df_e)}  line={n_line}  regulator={n_reg}")
    print(f"  samples={n_s}  regulator-edge feature rows={len(reg_feat)} (= {n_s} * {n_re})")
    print(f"  {path_catalog.name}")
    print(f"  {path_line.name}")
    print(f"  {path_reg.name}")
    if "heterogenous gnn dataset" not in str(out_dir.resolve()).lower():
        print(
            "[build_hetero_mv_edge_dataset] hint: compare_hetero_mv_daily uses "
            "<dataset_dir>/edges/ (often .../Heterogenous GNN dataset/edges). "
            "Copy these three files there or rerun with --out-dir pointing to that edges folder."
        )

    return {
        "catalog": path_catalog,
        "line_attr": path_line,
        "regulator_features": path_reg,
    }


def main() -> None:
    d = REPO / "datasets_gnn2" / "loadtype_8500_dailyagg"
    p = argparse.ArgumentParser(description="Hetero MV edge catalog + line static + regulator tap features per sample.")
    p.add_argument("--mv-edges", type=Path, default=d / "gnn_edges_phase_static_mv_only.csv")
    p.add_argument("--regulator", type=Path, default=d / "regulator_involved_nodes.csv")
    p.add_argument("--node-index", type=Path, default=d / "gnn_node_index_master.csv")
    p.add_argument("--sample-meta", type=Path, default=d / "gnn_sample_meta.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: same folder as --mv-edges")
    args = p.parse_args()
    build(
        mv_edges_csv=args.mv_edges.resolve(),
        regulator_csv=args.regulator.resolve(),
        node_index_csv=args.node_index.resolve(),
        sample_meta_csv=args.sample_meta.resolve(),
        out_dir=(args.out_dir.resolve() if args.out_dir else None),
    )


if __name__ == "__main__":
    main()
