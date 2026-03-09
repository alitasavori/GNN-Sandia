"""
Debug script: run one timestep (t=0) with the same inputs as the compare script and as the
loadtype dataset generator; diff features and voltages to find mismatches.

Usage (from repo root):
  python debug_compare_inputs.py

No torch required. Prints per-feature max abs diff and highlights any mismatch.
"""
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj
import run_loadtype_dataset as lt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def _build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv, node_to_electrical_dist,
                          p_sys_balance, q_sys_balance, busphQ_pv=None):
    """Copy of strict overlay build_gnn_x_loadtype (14 feat)."""
    if busphQ_pv is None:
        raise ValueError("Strict load-type debug path requires busphQ_pv.")
    X = np.zeros((len(node_names_master), 14), dtype=np.float32)
    for i, n in enumerate(node_names_master):
        bus, phs = n.split(".")
        ph = int(phs)
        m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
        m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
        m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
        m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
        m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
        m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
        m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
        m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))
        q_cap = inj.cap_q_kvar_per_node(bus, ph)
        p_pv = float(busphP_pv.get((bus, ph), 0.0))
        q_pv = float(busphQ_pv.get((bus, ph), 0.0))
        X[i, 0] = float(node_to_electrical_dist.get(n, 0.0))
        X[i, 1], X[i, 2] = m1_p, m1_q
        X[i, 3], X[i, 4] = m2_p, m2_q
        X[i, 5], X[i, 6] = m4_p, m4_q
        X[i, 7], X[i, 8] = m5_p, m5_q
        X[i, 9] = q_cap
        X[i, 10] = p_pv
        X[i, 11] = q_pv
        X[i, 12] = p_sys_balance
        X[i, 13] = q_sys_balance
    return X


def _build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid, busphQ_pv):
    """Copy of overlay build_gnn_x_injection (2 feat) using inj.CAP_Q_KVAR."""
    P_grid_per_ph = P_grid / 3.0
    Q_grid_per_ph = Q_grid / 3.0
    X = np.zeros((len(node_names_master), 2), dtype=np.float32)
    for i, n in enumerate(node_names_master):
        bus, phs = n.split(".")
        ph = int(phs)
        p_load = float(busphP_load.get((bus, ph), 0.0))
        q_load = float(busphQ_load.get((bus, ph), 0.0))
        p_pv = float(busphP_pv.get((bus, ph), 0.0))
        q_pv = float(busphQ_pv.get((bus, ph), 0.0))
        if bus == "sourcebus":
            p_inj = P_grid_per_ph
            q_inj = Q_grid_per_ph
        else:
            p_inj = p_pv - p_load
            q_inj = -q_pv - q_load + float(inj.CAP_Q_KVAR.get(bus, 0.0))
        X[i, 0] = p_inj
        X[i, 1] = q_inj
    return X


# Same as compare script and dataset generation
P_BASE = inj.BASELINE["P_load_total_kw"]
Q_BASE = inj.BASELINE["Q_load_total_kvar"]
PV_BASE = inj.BASELINE["P_pv_total_kw"]
DIR_LOADTYPE = os.path.join("datasets_gnn2", "loadtype")
EDGE_CSV = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
MASTER_CSV = os.path.join(DIR_LOADTYPE, "gnn_node_index_master.csv")
EXCLUDED_UPSTREAM_BUSES = ("sourcebus", "800")

LOADTYPE_FEAT_ORDER = [
    "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw", "q_pv_kvar",
    "p_sys_balance_kw", "q_sys_balance_kvar",
]


def get_89_node_list():
    """Same logic as compare script _resolve_node_list (without torch)."""
    if not os.path.exists(NODE_CSV) or not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"Need {NODE_CSV} and {MASTER_CSV}")
    df_n = pd.read_csv(NODE_CSV)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    kept_node_ids = sorted(df_n["node_idx"].unique())
    master_df = pd.read_csv(MASTER_CSV)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    old_to_name = master_df.set_index("node_idx")["node"].astype(str).to_dict()
    return [old_to_name[old] for old in kept_node_ids]


def get_full_95_node_list():
    """Same as compare script _get_full_master_node_list."""
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(MASTER_CSV)
    master_df = pd.read_csv(MASTER_CSV)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx")
    return master_df["node"].astype(str).tolist()


def run_one_timestep_and_build_features():
    """Run t=0: compile, snapshot, solve, build overlay X and dataset-style rows. Returns (X_loadtype, X_injection, R_loadtype_89x13, R_injection_89x2, vmag_list, node_names_89)."""
    node_names_89 = get_89_node_list()
    full_95 = get_full_95_node_list()
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_95, EDGE_CSV)

    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    t = 0
    inj.set_time_index(t)
    _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
        P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
        mL_t=float(mL[t]), mPV_t=float(mPV[t]),
        loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
        sigma_load=0.0, sigma_pv=0.0, rng=rng,
    )
    inj.dss.Solution.Solve()
    if not inj.dss.Solution.Converged():
        print("[WARN] t=0 did not converge; continuing anyway for feature diff.")
    busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
    vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_89)

    sum_p_load = float(sum(busphP_load.values()))
    sum_q_load = float(sum(busphQ_load.values()))
    sum_p_pv_actual = float(sum(busphP_pv_actual.values()))
    sum_q_pv_actual = float(sum(busphQ_pv_actual.values()))
    sum_q_cap = float(inj.total_cap_q_kvar(node_names_89))
    pwr = inj.dss.Circuit.TotalPower()
    P_grid = -float(pwr[0])
    Q_grid = -float(pwr[1])
    p_sys_balance = sum_p_load - sum_p_pv_actual
    q_sys_balance = sum_q_load + sum_q_pv_actual - sum_q_cap

    # Overlay-style X (what compare script builds)
    X_loadtype = _build_gnn_x_loadtype(
        node_names_89, busph_per_type, busphP_pv_actual,
        node_to_electrical_dist, p_sys_balance, q_sys_balance,
        busphQ_pv=busphQ_pv_actual,
    )
    X_injection = _build_gnn_x_injection(
        node_names_89, busphP_load, busphQ_load, busphP_pv_actual, P_grid, Q_grid, busphQ_pv_actual,
    )

    # Dataset-style rows: loadtype (exact logic from run_loadtype_dataset loop)
    dataset_rows = []
    for n in node_names_89:
        bus, phs = n.split(".")
        ph = int(phs)
        m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
        m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
        m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
        m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
        m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
        m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
        m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
        m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))
        q_cap_node = float(inj.CAP_Q_KVAR.get(bus, 0.0))
        p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
        elec_dist = float(node_to_electrical_dist.get(n, 0.0))
        row = [
            elec_dist, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q,
            q_cap_node, p_pv_node, p_sys_balance, q_sys_balance,
        ]
        dataset_rows.append(row)

    R_lt = np.array(dataset_rows, dtype=np.float32)

    # Dataset-style injection rows (exact logic from run_injection_dataset: p_inj, q_inj; 89 nodes, no sourcebus)
    inj_rows = []
    P_grid_per_ph = P_grid / 3.0
    Q_grid_per_ph = Q_grid / 3.0
    for n in node_names_89:
        bus, phs = n.split(".")
        ph = int(phs)
        p_load_node = float(busphP_load.get((bus, ph), 0.0))
        q_load_node = float(busphQ_load.get((bus, ph), 0.0))
        p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
        q_pv_node = float(busphQ_pv_actual.get((bus, ph), 0.0))
        if bus == "sourcebus":
            p_inj = P_grid_per_ph
            q_inj = Q_grid_per_ph
        else:
            p_inj = p_pv_node - p_load_node
            q_inj = -q_pv_node - q_load_node + float(inj.CAP_Q_KVAR.get(bus, 0.0))
        inj_rows.append([p_inj, q_inj])
    R_inj = np.array(inj_rows, dtype=np.float32)

    return X_loadtype, X_injection, R_lt, R_inj, vmag_m, node_names_89


def main():
    print("Debug: comparing compare-script inputs vs dataset-generation logic at t=0")
    print(f"  Baseline: P={P_BASE} Q={Q_BASE} PV={PV_BASE}")
    print()

    X_lt, X_inj, dataset_89x13, dataset_89x2, vmag_list, node_names_89 = run_one_timestep_and_build_features()
    N = len(node_names_89)

    # Compare injection 2 features
    diff_inj = np.abs(X_inj - dataset_89x2)
    max_inj = np.max(diff_inj)
    print("Injection (2 feat) — max |compare - dataset|:")
    print(f"  p_inj_kw  max={np.max(diff_inj[:, 0]):.6f}")
    print(f"  q_inj_kvar max={np.max(diff_inj[:, 1]):.6f}")
    if max_inj < 1e-5:
        print("  Injection features match.")
    else:
        print("  >>> MISMATCH in injection features.")
    print()

    # Compare loadtype 13 features
    diff_lt = np.abs(X_lt - dataset_89x13)
    max_per_feat = np.max(diff_lt, axis=0)
    mean_per_feat = np.mean(diff_lt, axis=0)
    print("Loadtype (13 feat) — max |compare - dataset| per feature:")
    for j, name in enumerate(LOADTYPE_FEAT_ORDER):
        ok = "OK" if max_per_feat[j] < 1e-5 else "MISMATCH"
        print(f"  {name:25s} max={max_per_feat[j]:.6f} mean={mean_per_feat[j]:.6f}  {ok}")
    if np.any(max_per_feat >= 1e-5):
        print("\n  >>> MISMATCH in loadtype features; check formulas/order.")
        # Show worst node
        worst_node_idx = np.unravel_index(np.argmax(diff_lt), diff_lt.shape)[0]
        worst_feat_idx = np.unravel_index(np.argmax(diff_lt), diff_lt.shape)[1]
        print(f"  Worst: node {node_names_89[worst_node_idx]} feat {LOADTYPE_FEAT_ORDER[worst_feat_idx]}")
        print(f"    compare={X_lt[worst_node_idx, worst_feat_idx]:.6f} dataset={dataset_89x13[worst_node_idx, worst_feat_idx]:.6f}")
    else:
        print("  Loadtype features match.")

    # Sanity: compare first sample from CSV if it exists (same t=0, same baseline is unlikely)
    # So we only compare logic vs logic above.
    print()
    valid_v = [float(v) for v in vmag_list if np.isfinite(v)]
    if valid_v:
        print("Voltage at t=0: min={:.4f} max={:.4f} (pu)".format(min(valid_v), max(valid_v)))
    else:
        print("Voltage at t=0: no valid values (solve may not have converged)")
    print("Done.")


if __name__ == "__main__":
    main()
