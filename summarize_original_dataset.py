"""
Summarize ORIGINAL dataset: scenario-wise stats and baseline daily profile.

1) Across all 200 scenarios (TRAINING data):
   For each of p_pv, q_pv (post-solve), p_load, q_load:
   - For each scenario, take the MAX of the daily profile (over its samples).
   - Report min, max, and mean of those 200 per-scenario maxima.

2) Baseline daily profile (used for VOLTAGE PROFILE INFERENCE):
   Run the full 24h at BASELINE totals (same as daily voltage profile scripts).
   At each timestep: solve, get post-solve P/Q totals; then over the day report
   daily MAX, daily MIN, daily MEAN for each quantity.
   So we compare apples-to-apples: baseline's daily max should fall within
   the training (200-scenario) daily-max range.

Usage (from repo root):

  python summarize_original_dataset.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj


def run_baseline_daily_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run full 24h at BASELINE (sigma=0). Return 4 arrays of length NPTS: P_load, Q_load, P_pv, Q_pv (post-solve totals per timestep)."""
    dss_path = inj.compile_once()
    inj.setup_daily()
    try:
        dss.Text.Command("set maxcontroliter=20000")
    except Exception:
        pass

    node_names_master, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    P_load = float(inj.BASELINE["P_load_total_kw"])
    Q_load = float(inj.BASELINE["Q_load_total_kvar"])
    P_pv = float(inj.BASELINE["P_pv_total_kw"])
    rng = np.random.default_rng(0)

    out_P_load = np.full(inj.NPTS, np.nan)
    out_Q_load = np.full(inj.NPTS, np.nan)
    out_P_pv = np.full(inj.NPTS, np.nan)
    out_Q_pv = np.full(inj.NPTS, np.nan)

    for t in range(inj.NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, _, _ = inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load,
            Q_load_total_kvar=Q_load,
            P_pv_total_kw=P_pv,
            mL_t=float(mL[t]),
            mPV_t=float(mPV[t]),
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=0.0,
            sigma_pv=0.0,
            rng=rng,
        )
        try:
            dss.Solution.Solve()
        except Exception:
            pass
        if not dss.Solution.Converged():
            continue
        busphP_pv, busphQ_pv = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        # System totals (sum over non-upstream nodes only, to match dataset)
        pl, ql, pp, qp = 0.0, 0.0, 0.0, 0.0
        for n in node_names_master:
            bus, phs = n.split(".")
            ph = int(phs)
            if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                continue
            pl += float(busphP_load.get((bus, ph), 0.0))
            ql += float(busphQ_load.get((bus, ph), 0.0))
            pp += float(busphP_pv.get((bus, ph), 0.0))
            qp += float(busphQ_pv.get((bus, ph), 0.0))
        out_P_load[t] = pl
        out_Q_load[t] = ql
        out_P_pv[t] = pp
        out_Q_pv[t] = qp

    return out_P_load, out_Q_load, out_P_pv, out_Q_pv


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "datasets_gnn2", "original")
    node_csv = os.path.join(data_dir, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(data_dir, "gnn_sample_meta.csv")

    if not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {node_csv}. Run run_original_dataset.py first.")
    if not os.path.exists(sample_csv):
        raise FileNotFoundError(f"Missing {sample_csv}. Run run_original_dataset.py first.")

    print("Reading ORIGINAL dataset...")
    df_node = pd.read_csv(node_csv)
    df_sample = pd.read_csv(sample_csv)

    cols = ["sample_id", "node_idx", "p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar"]
    missing = [c for c in cols if c not in df_node.columns]
    if missing:
        raise ValueError(f"Missing columns in node table: {missing}")

    for c in cols:
        df_node[c] = pd.to_numeric(df_node[c], errors="coerce")
    df_node = df_node.dropna(subset=cols).copy()

    # Merge to get scenario_id per sample
    df_node = df_node.merge(
        df_sample[["sample_id", "scenario_id"]].drop_duplicates(),
        on="sample_id",
        how="left",
    )
    n_samples = df_node["sample_id"].nunique()
    n_scenarios = df_node["scenario_id"].nunique()
    print(f"  Node rows: {len(df_node)}  Samples: {n_samples}  Scenarios: {n_scenarios}")

    # Per-sample system totals (sum over nodes)
    sample_totals = (
        df_node.groupby("sample_id")
        .agg(
            P_load=("p_load_kw", "sum"),
            Q_load=("q_load_kvar", "sum"),
            P_pv=("p_pv_kw", "sum"),
            Q_pv=("q_pv_kvar", "sum"),
        )
        .reset_index()
    )
    sample_totals = sample_totals.merge(
        df_node[["sample_id", "scenario_id"]].drop_duplicates(), on="sample_id", how="left"
    )

    # Per-scenario: max (and min, mean) of daily profile for each quantity
    scenario_stats = (
        sample_totals.groupby("scenario_id")[["P_load", "Q_load", "P_pv", "Q_pv"]]
        .agg(["min", "max", "mean"])
        .reset_index()
    )

    # Flatten column names: (P_load, max) -> P_load_max (handle single-level names)
    def _flatten_col(c):
        if isinstance(c, tuple):
            return f"{c[0]}_{c[1]}" if c[1] else str(c[0])
        return str(c)

    scenario_stats.columns = [_flatten_col(c) for c in scenario_stats.columns]
    if "scenario_id_" in scenario_stats.columns:
        scenario_stats = scenario_stats.rename(columns={"scenario_id_": "scenario_id"})

    # Sanity: samples per scenario (should be constant)
    n_per_scenario = sample_totals.groupby("scenario_id").size()
    print(f"\n  Samples per scenario: min={n_per_scenario.min()} max={n_per_scenario.max()} (should be equal)")
    # Sanity: for one scenario, nominal P_pv from meta vs our daily-max P_pv (post-solve can be below nominal due to Volt-Var)
    if "P_pv_total_kw" in df_sample.columns:
        s0_samples = df_sample[df_sample["scenario_id"] == 0]
        nominal_pv_s0 = float(s0_samples["P_pv_total_kw"].iloc[0])
        our_max_pv_s0 = float(scenario_stats.loc[scenario_stats["scenario_id"] == 0, "P_pv_max"].iloc[0])
        print(f"  Scenario 0: nominal P_pv (meta)={nominal_pv_s0:.2f} kW  |  our daily-max P_pv (post-solve)={our_max_pv_s0:.2f} kW")

    print("\n" + "=" * 70)
    print("ACROSS 200 SCENARIOS: min / max / mean of (daily profile MAX per scenario)")
    print("=" * 70)

    for name, col_max in [
        ("P_pv (post-solve)", "P_pv_max"),
        ("Q_pv (post-solve)", "Q_pv_max"),
        ("P_load", "P_load_max"),
        ("Q_load", "Q_load_max"),
    ]:
        if col_max not in scenario_stats.columns:
            continue
        vals = scenario_stats[col_max].to_numpy(dtype=float)
        print(
            f"  {name:20s}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}"
        )

    # Optional: also report min-of-daily-min and mean-of-daily-mean for context
    print("\n  (Per-scenario daily MIN: min/mean across scenarios)")
    for name, col_min in [
        ("P_pv", "P_pv_min"),
        ("Q_pv", "Q_pv_min"),
        ("P_load", "P_load_min"),
        ("Q_load", "Q_load_min"),
    ]:
        if col_min not in scenario_stats.columns:
            continue
        vals = scenario_stats[col_min].to_numpy(dtype=float)
        print(f"    {name:8s}: min={vals.min():.4f}  mean={vals.mean():.4f}")

    # --- Baseline daily profile (VOLTAGE PROFILE INFERENCE: same 24h as daily voltage scripts) ---
    print("\n" + "=" * 70)
    print("BASELINE DAILY PROFILE (24h at inj.BASELINE, sigma=0) — used for voltage profile inference")
    print("=" * 70)
    # Confirm same DSS as dataset generation (run_original_dataset / run_injection_dataset)
    _dss_used = os.path.abspath(inj.DSS_FILE)
    print(f"  DSS file (same as dataset generation): {_dss_used}")

    bl_P_load, bl_Q_load, bl_P_pv, bl_Q_pv = run_baseline_daily_profile()
    n_conv = np.sum(np.isfinite(bl_P_load))
    print(f"  Converged timesteps: {n_conv}/{inj.NPTS}")

    # Daily max / min / mean (same definition as per-scenario in training)
    def safe_stats(a: np.ndarray) -> tuple[float, float, float]:
        v = a[np.isfinite(a)]
        if v.size == 0:
            return np.nan, np.nan, np.nan
        return float(np.min(v)), float(np.max(v)), float(np.mean(v))

    bl_pl_min, bl_pl_max, bl_pl_mean = safe_stats(bl_P_load)
    bl_ql_min, bl_ql_max, bl_ql_mean = safe_stats(bl_Q_load)
    bl_pp_min, bl_pp_max, bl_pp_mean = safe_stats(bl_P_pv)
    bl_qp_min, bl_qp_max, bl_qp_mean = safe_stats(bl_Q_pv)

    print("\n  Baseline day — daily MIN / MAX / MEAN (post-solve totals over 24h):")
    print(f"    P_load: min={bl_pl_min:.2f}  max={bl_pl_max:.2f}  mean={bl_pl_mean:.2f} kW")
    print(f"    Q_load: min={bl_ql_min:.2f}  max={bl_ql_max:.2f}  mean={bl_ql_mean:.2f} kVAR")
    print(f"    P_pv:   min={bl_pp_min:.2f}  max={bl_pp_max:.2f}  mean={bl_pp_mean:.2f} kW (post-solve)")
    print(f"    Q_pv:   min={bl_qp_min:.2f}  max={bl_qp_max:.2f}  mean={bl_qp_mean:.2f} kVAR (post-solve)")

    # Direct comparison: baseline daily MAX should fall within training (200-scenario) daily-max range
    tr_pl_min = float(scenario_stats["P_load_max"].min())
    tr_pl_max = float(scenario_stats["P_load_max"].max())
    tr_ql_min = float(scenario_stats["Q_load_max"].min())
    tr_ql_max = float(scenario_stats["Q_load_max"].max())
    tr_pp_min = float(scenario_stats["P_pv_max"].min())
    tr_pp_max = float(scenario_stats["P_pv_max"].max())
    tr_qp_min = float(scenario_stats["Q_pv_max"].min())
    tr_qp_max = float(scenario_stats["Q_pv_max"].max())

    print("\n  MATCH CHECK (baseline daily MAX vs TRAINING daily-max range across 200 scenarios):")
    print(f"    P_load: baseline daily max = {bl_pl_max:.2f}  |  training range = [{tr_pl_min:.2f}, {tr_pl_max:.2f}]  |  in range = {tr_pl_min <= bl_pl_max <= tr_pl_max}")
    print(f"    Q_load: baseline daily max = {bl_ql_max:.2f}  |  training range = [{tr_ql_min:.2f}, {tr_ql_max:.2f}]  |  in range = {tr_ql_min <= bl_ql_max <= tr_ql_max}")
    print(f"    P_pv:   baseline daily max = {bl_pp_max:.2f}  |  training range = [{tr_pp_min:.2f}, {tr_pp_max:.2f}]  |  in range = {tr_pp_min <= bl_pp_max <= tr_pp_max}")
    print(f"    Q_pv:   baseline daily max = {bl_qp_max:.2f}  |  training range = [{tr_qp_min:.2f}, {tr_qp_max:.2f}]  |  in range = {tr_qp_min <= bl_qp_max <= tr_qp_max}")

    # --- Model INPUT distribution: Training vs Inference (same quantities the model sees per snapshot) ---
    print("\n" + "=" * 70)
    print("MODEL INPUT DISTRIBUTION: Training vs Inference (system totals per snapshot)")
    print("  Training = 57,600 samples (original dataset). Inference = 288 timesteps (baseline day).")
    print("=" * 70)

    def p5_p95(a: np.ndarray) -> tuple[float, float]:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        return float(np.percentile(a, 5)), float(np.percentile(a, 95))

    for name, tr_vals, inf_vals in [
        ("P_load (kW)", sample_totals["P_load"].to_numpy(), bl_P_load),
        ("Q_load (kVAR)", sample_totals["Q_load"].to_numpy(), bl_Q_load),
        ("P_pv (kW)", sample_totals["P_pv"].to_numpy(), bl_P_pv),
        ("Q_pv (kVAR)", sample_totals["Q_pv"].to_numpy(), bl_Q_pv),
    ]:
        inf_vals = inf_vals[np.isfinite(inf_vals)]
        tr_min, tr_max = float(np.min(tr_vals)), float(np.max(tr_vals))
        tr_mean, tr_std = float(np.mean(tr_vals)), float(np.std(tr_vals))
        tr_p5, tr_p95 = p5_p95(tr_vals)
        if inf_vals.size == 0:
            inf_min = inf_max = inf_mean = inf_std = inf_p5 = inf_p95 = np.nan
        else:
            inf_min, inf_max = float(np.min(inf_vals)), float(np.max(inf_vals))
            inf_mean, inf_std = float(np.mean(inf_vals)), float(np.std(inf_vals))
            inf_p5, inf_p95 = p5_p95(inf_vals)
        print(f"\n  {name}:")
        print(f"    Training   (n={len(tr_vals):,}): min={tr_min:.2f}  max={tr_max:.2f}  mean={tr_mean:.2f}  std={tr_std:.2f}  p5–p95=[{tr_p5:.2f}, {tr_p95:.2f}]")
        print(f"    Inference  (n={inf_vals.size}):    min={inf_min:.2f}  max={inf_max:.2f}  mean={inf_mean:.2f}  std={inf_std:.2f}  p5–p95=[{inf_p5:.2f}, {inf_p95:.2f}]")
        in_range = tr_min <= inf_min and inf_max <= tr_max if inf_vals.size else False
        print(f"    Inference range inside training range: {in_range}")

    # Per-node input distribution (training only; inference would need saving nodal values from baseline run)
    print("\n  Per-node model inputs (training only — distribution over all node×sample rows):")
    for c in ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar"]:
        if c not in df_node.columns:
            continue
        v = df_node[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        p5, p95 = np.percentile(v, 5), np.percentile(v, 95)
        print(f"    {c:14s}: min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  std={v.std():.4f}  p5–p95=[{p5:.4f}, {p95:.4f}]")

    # Legacy: per-node global means over full dataset (keep for reference)
    print("\n" + "=" * 70)
    print("LEGACY: Per-node means over ALL rows in original dataset")
    print("=" * 70)
    for c in ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar"]:
        if c not in df_node.columns:
            continue
        vals = df_node[c].to_numpy(dtype=float)
        print(f"  {c:14s}: mean={vals.mean():.6f}  std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}")


if __name__ == "__main__":
    main()
