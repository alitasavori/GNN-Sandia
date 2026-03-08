r"""
Graph sanity check for GNN2 datasets.

Usage (from notebook or shell, with cwd at repo root):

  import os
  os.chdir(r"C:\Users\alita\OneDrive\Desktop\GNN2")
  exec(open("graph_sanity_check.py", encoding="utf-8").read())

The script will prompt for which dataset to inspect:
  1 = original   (datasets_gnn2/original)
  2 = injection  (datasets_gnn2/injection)
  3 = load-type  (datasets_gnn2/loadtype)
  4 = delta-V    (datasets_gnn2/deltav)

For the selected dataset it performs:

1) Bidirectionality check on gnn_edges_phase_static.csv
2) Transformer / regulator edge checks (XFM1 832↔888, RegA 814↔814r, RegB 852↔852r)
3) Prints node features for the first sample (all nodes)
4) Aggregated P/Q sanity checks (datasets 2–4)
5) Prints edge static features for all edges (R_full, X_full, C_full in ohms)
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def _repo_root() -> str:
    """Best-effort repo root (directory containing this file)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _choose_dataset() -> Tuple[int, str, str]:
    """Prompt user for dataset id and return (id, name, out_dir)."""
    root = _repo_root()
    mapping = {
        1: ("original", os.path.join(root, "datasets_gnn2", "original")),
        2: ("injection", os.path.join(root, "datasets_gnn2", "injection")),
        3: ("loadtype", os.path.join(root, "datasets_gnn2", "loadtype")),
        4: ("deltav", os.path.join(root, "datasets_gnn2", "deltav")),
    }
    print("Select dataset to check:")
    print("  1 = original (gnn_samples_out)")
    print("  2 = injection (p_inj/q_inj)")
    print("  3 = load-type")
    print("  4 = delta-V")
    while True:
        try:
            raw = input("Dataset id [1–4]: ").strip()
        except EOFError:
            raw = ""
        if not raw:
            did = 1
            break
        try:
            did = int(raw)
        except ValueError:
            print("Please enter 1, 2, 3, or 4.")
            continue
        if did in mapping:
            break
        print("Please enter 1, 2, 3, or 4.")
    name, out_dir = mapping[did]
    print(f"\nDataset {did}: {name}  (OUT_DIR={out_dir})")
    return did, name, out_dir


def _paths_for_dataset(did: int, out_dir: str) -> Tuple[str, str, str]:
    """Return (edge_csv, node_csv, sample_csv) paths for a dataset id."""
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(out_dir, "gnn_sample_meta.csv")
    # Injection dataset uses slightly different names
    if did == 2:
        edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
        node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
        sample_csv = os.path.join(out_dir, "gnn_sample_meta.csv")
    return edge_csv, node_csv, sample_csv


def _load_edges(edge_csv: str) -> pd.DataFrame:
    print(f"\n[load] edges: {edge_csv}")
    df = pd.read_csv(edge_csv)
    print(f"  edges: {len(df)} rows, columns={list(df.columns)}")
    return df


def _load_nodes(node_csv: str) -> pd.DataFrame:
    print(f"\n[load] node features: {node_csv}")
    df = pd.read_csv(node_csv)
    print(f"  node rows: {len(df)} rows, columns={list(df.columns)}")
    return df


def _load_samples(sample_csv: str) -> pd.DataFrame:
    print(f"\n[load] sample meta: {sample_csv}")
    df = pd.read_csv(sample_csv)
    print(f"  samples: {df['sample_id'].nunique()} (rows={len(df)})")
    return df


def check_bidirectionality(df_e: pd.DataFrame) -> None:
    print("\n[check] 1) Bidirectionality")
    # Key on (u_idx, v_idx, phase)
    required_cols = {"u_idx", "v_idx", "phase"}
    if not required_cols.issubset(df_e.columns):
        print("  ! Missing u_idx/v_idx/phase columns; cannot check bidirectionality.")
        return
    keys = set(zip(df_e["u_idx"], df_e["v_idx"], df_e["phase"]))
    missing = []
    for u, v, ph in keys:
        if (v, u, ph) not in keys:
            missing.append((u, v, ph))
    if not missing:
        print("  OK: every edge has a reverse edge for the same phase.")
    else:
        print(f"  ! Found {len(missing)} edges missing a reverse; first few:")
        for (u, v, ph) in missing[:10]:
            print(f"    u={u} -> v={v}, phase={ph}")


def check_transformer_reg_edges(df_e: pd.DataFrame) -> None:
    print("\n[check] 2) Transformer / regulator edges")
    if not {"from_bus", "to_bus", "phase"}.issubset(df_e.columns):
        print("  ! Missing from_bus/to_bus/phase columns; cannot check transformer edges.")
        return

    def _has_pair(bus_a: str, bus_b: str) -> bool:
        phs = (1, 2, 3)
        for ph in phs:
            fwd = ((df_e["from_bus"] == bus_a) & (df_e["to_bus"] == bus_b) & (df_e["phase"] == ph)).any()
            rev = ((df_e["from_bus"] == bus_b) & (df_e["to_bus"] == bus_a) & (df_e["phase"] == ph)).any()
            if not (fwd and rev):
                return False
        return True

    pairs = [
        ("XFM1 832↔888", "832", "888"),
        ("RegA 814↔814r", "814", "814r"),
        ("RegB 852↔852r", "852", "852r"),
    ]
    for label, a, b in pairs:
        ok = _has_pair(a, b)
        print(f"  {label}: {'OK' if ok else 'MISSING'}")


def show_first_sample_nodes(df_nodes: pd.DataFrame) -> int:
    print("\n[check] 3) First-sample node features (all nodes)")
    if "sample_id" not in df_nodes.columns:
        print("  ! No sample_id column in node file; cannot slice by sample.")
        print("  Showing all node rows:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print(df_nodes)
        return 0
    first_sample = int(df_nodes["sample_id"].min())
    df_one = df_nodes[df_nodes["sample_id"] == first_sample].copy()
    print(f"  sample_id={first_sample} -> {len(df_one)} node rows")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df_one)
    return first_sample


def aggregated_checks_injection(df_nodes: pd.DataFrame, df_samples: pd.DataFrame, sample_id: int) -> None:
    print("\n[check] 4) Aggregated P/Q (injection dataset)")
    cols_needed = {"p_inj_kw", "q_inj_kvar"}
    if not cols_needed.issubset(df_nodes.columns):
        print("  ! p_inj_kw/q_inj_kvar not present; skipping aggregated P/Q check.")
        return
    df_n = df_nodes[df_nodes["sample_id"] == sample_id].copy()
    df_s = df_samples[df_samples["sample_id"] == sample_id].copy()
    if df_s.empty:
        print(f"  ! No sample meta row for sample_id={sample_id}")
        return
    srow = df_s.iloc[0]
    p_sum = df_n["p_inj_kw"].sum()
    q_sum = df_n["q_inj_kvar"].sum()
    p_grid = float(srow.get("P_grid_kw", np.nan))
    q_grid = float(srow.get("Q_grid_kvar", np.nan))
    print(f"  Σ p_inj_kw (nodes) = {p_sum:.3f} kW")
    print(f"  Σ q_inj_kvar (nodes) = {q_sum:.3f} kVAR")
    print(f"  P_grid_kw (sample meta) = {p_grid:.3f} kW")
    print(f"  Q_grid_kvar (sample meta) = {q_grid:.3f} kVAR")
    # For an exact balance (ignoring losses) we would have Σ P_inj + P_grid ≈ 0.
    res_p = p_sum + p_grid
    res_q = q_sum + q_grid
    print(f"  Residual P (Σ p_inj + P_grid) = {res_p:.3f} kW")
    print(f"  Residual Q (Σ q_inj + Q_grid) = {res_q:.3f} kVAR  (≈ losses)")


def aggregated_checks_loadtype(df_nodes: pd.DataFrame, df_samples: pd.DataFrame, sample_id: int) -> None:
    print("\n[check] 4) Aggregated P/Q (load-type dataset)")
    required = {"m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar", "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar",
                "p_pv_kw", "q_pv_kvar", "q_cap_kvar", "bus"}
    if not required.issubset(df_nodes.columns):
        print("  ! Missing some load-type feature columns; skipping aggregated P/Q check.")
        return
    if "p_sys_balance_kw" not in df_samples.columns:
        print("  ! p_sys_balance_kw/q_sys_balance_kvar not present in sample meta; skipping.")
        return
    df_n = df_nodes[df_nodes["sample_id"] == sample_id].copy()
    df_s = df_samples[df_samples["sample_id"] == sample_id].copy()
    if df_s.empty:
        print(f"  ! No sample meta row for sample_id={sample_id}")
        return
    srow = df_s.iloc[0]

    # Aggregate over nodes
    m_p = df_n[["m1_p_kw", "m2_p_kw", "m4_p_kw", "m5_p_kw"]].sum().sum()
    m_q = df_n[["m1_q_kvar", "m2_q_kvar", "m4_q_kvar", "m5_q_kvar"]].sum().sum()
    p_pv = df_n["p_pv_kw"].sum()
    q_pv = df_n["q_pv_kvar"].sum()
    # Capacitor Q: sum once per bus (not per phase)
    cap_by_bus = df_n.groupby("bus")["q_cap_kvar"].max()
    q_cap = cap_by_bus.sum()

    p_sys = float(srow["p_sys_balance_kw"])
    q_sys = float(srow["q_sys_balance_kvar"])

    p_pred = m_p - p_pv
    q_pred = m_q - q_pv - q_cap

    print(f"  Σ load P (m1+m2+m4+m5) = {m_p:.3f} kW")
    print(f"  Σ load Q (m1+m2+m4+m5) = {m_q:.3f} kVAR")
    print(f"  Σ PV P (actual)        = {p_pv:.3f} kW")
    print(f"  Σ PV Q (actual)        = {q_pv:.3f} kVAR")
    print(f"  Σ cap Q (per bus)      = {q_cap:.3f} kVAR")
    print(f"  p_sys_balance_kw       = {p_sys:.3f} kW")
    print(f"  q_sys_balance_kvar     = {q_sys:.3f} kVAR")
    print(f"  Predicted P_sys        = {p_pred:.3f} kW  (ΣloadP - ΣpvP)")
    print(f"  Predicted Q_sys        = {q_pred:.3f} kVAR (ΣloadQ - ΣpvQ - ΣcapQ)")
    print(f"  Residual P_sys         = {p_sys - p_pred:.3f} kW")
    print(f"  Residual Q_sys         = {q_sys - q_pred:.3f} kVAR")


def aggregated_checks_deltav(df_nodes: pd.DataFrame, df_samples: pd.DataFrame, sample_id: int) -> None:
    print("\n[check] 4) Aggregated P/Q + ΔV (delta-V dataset)")
    # Reuse the load-type P/Q logic (columns have same names for load/PV/cap/system-balance)
    required = {"m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar", "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar",
                "p_pv_kw", "q_pv_kvar", "q_cap_kvar", "bus",
                "vmag_zero_pv_pu", "vmag_delta_pu"}
    if not required.issubset(df_nodes.columns):
        print("  ! Missing some delta-V feature columns; skipping aggregated P/Q check.")
        return
    if "p_sys_balance_kw" not in df_samples.columns:
        print("  ! p_sys_balance_kw/q_sys_balance_kvar not present in sample meta; skipping.")
        return
    df_n = df_nodes[df_nodes["sample_id"] == sample_id].copy()
    df_s = df_samples[df_samples["sample_id"] == sample_id].copy()
    if df_s.empty:
        print(f"  ! No sample meta row for sample_id={sample_id}")
        return
    srow = df_s.iloc[0]

    m_p = df_n[["m1_p_kw", "m2_p_kw", "m4_p_kw", "m5_p_kw"]].sum().sum()
    m_q = df_n[["m1_q_kvar", "m2_q_kvar", "m4_q_kvar", "m5_q_kvar"]].sum().sum()
    p_pv = df_n["p_pv_kw"].sum()
    q_pv = df_n["q_pv_kvar"].sum()
    cap_by_bus = df_n.groupby("bus")["q_cap_kvar"].max()
    q_cap = cap_by_bus.sum()

    p_sys = float(srow["p_sys_balance_kw"])
    q_sys = float(srow["q_sys_balance_kvar"])

    p_pred = m_p - p_pv  # zero-PV vs with-PV uses same totals; check against stored p_sys_balance_kw
    q_pred = m_q - q_pv - q_cap

    print(f"  Σ load P (m1+m2+m4+m5) = {m_p:.3f} kW")
    print(f"  Σ load Q (m1+m2+m4+m5) = {m_q:.3f} kVAR")
    print(f"  Σ PV P (actual)        = {p_pv:.3f} kW")
    print(f"  Σ PV Q (actual)        = {q_pv:.3f} kVAR")
    print(f"  Σ cap Q (per bus)      = {q_cap:.3f} kVAR")
    print(f"  p_sys_balance_kw       = {p_sys:.3f} kW")
    print(f"  q_sys_balance_kvar     = {q_sys:.3f} kVAR")
    print(f"  Predicted P_sys        = {p_pred:.3f} kW")
    print(f"  Predicted Q_sys        = {q_pred:.3f} kVAR")
    print(f"  Residual P_sys         = {p_sys - p_pred:.3f} kW")
    print(f"  Residual Q_sys         = {q_sys - q_pred:.3f} kVAR")

    # ΔV summary
    v0 = df_n["vmag_zero_pv_pu"]
    dv = df_n["vmag_delta_pu"]
    print("\n  ΔV summary over nodes (vmag_zero_pv_pu, vmag_delta_pu):")
    print(f"    vmag_zero_pv_pu: min={v0.min():.4f}, max={v0.max():.4f}, mean={v0.mean():.4f}")
    print(f"    vmag_delta_pu  : min={dv.min():.4f}, max={dv.max():.4f}, mean={dv.mean():.4f}")


def show_edge_static_features(df_e: pd.DataFrame) -> None:
    print("\n[check] 5) Edge static features (all edges)")
    cols = ["from_node", "to_node", "line_name", "R_full", "X_full", "C_full"]
    existing = [c for c in cols if c in df_e.columns]
    if not existing:
        existing = list(df_e.columns)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df_e[existing])


def main() -> None:
    did, name, out_dir = _choose_dataset()
    edge_csv, node_csv, sample_csv = _paths_for_dataset(did, out_dir)

    for path in (edge_csv, node_csv, sample_csv):
        if not os.path.exists(path):
            print(f"\n! File not found: {path}")
            print("  Make sure you have run the dataset generation script first.")
            return

    df_e = _load_edges(edge_csv)
    df_nodes = _load_nodes(node_csv)
    df_samples = _load_samples(sample_csv)

    # 1) Bidirectionality
    check_bidirectionality(df_e)

    # 2) Transformer / regulator edges
    check_transformer_reg_edges(df_e)

    # 3) First-sample node features
    sample_id = show_first_sample_nodes(df_nodes)

    # 4) Aggregated P/Q checks (skip for dataset 1)
    if did == 2:
        aggregated_checks_injection(df_nodes, df_samples, sample_id)
    elif did == 3:
        aggregated_checks_loadtype(df_nodes, df_samples, sample_id)
    elif did == 4:
        aggregated_checks_deltav(df_nodes, df_samples, sample_id)
    else:
        print("\n[check] 4) Aggregated P/Q: skipped for original dataset.")

    # 5) Edge static features sample
    show_edge_static_features(df_e)


if __name__ == "__main__":
    main()

