"""
Summarize training-vs-baseline feature regimes for all three GNN2 model families:

1) Original   (`datasets_gnn2/original`)
2) Injection  (`datasets_gnn2/injection`)
3) Load-type  (`datasets_gnn2/loadtype`)

For each model family, the script reports:
- Training dataset size and feature distributions
- Per-scenario daily-max ranges across the 200 scenarios
- Baseline 24h feature statistics at `inj.BASELINE`, sigma=0
- Training-vs-inference range checks for the quantities the model actually sees

Run from repo root or a notebook with:

  %run summarize_original_dataset.py
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj
if not hasattr(inj, "total_cap_q_kvar") or not hasattr(inj, "cap_q_kvar_per_node"):
    inj = importlib.reload(inj)
import run_loadtype_dataset as lt


ORIGINAL_NODE_COLS = ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar"]
INJECTION_NODE_COLS = ["p_inj_kw", "q_inj_kvar"]
LOADTYPE_NODE_COLS = [
    "electrical_distance_ohm",
    "m1_p_kw",
    "m1_q_kvar",
    "m2_p_kw",
    "m2_q_kvar",
    "m4_p_kw",
    "m4_q_kvar",
    "m5_p_kw",
    "m5_q_kvar",
    "q_cap_kvar",
    "p_pv_kw",
    "q_pv_kvar",
    "p_sys_balance_kw",
    "q_sys_balance_kvar",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_stats(a: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(v)), float(np.max(v)), float(np.mean(v))


def _p5_p95(a: np.ndarray) -> tuple[float, float]:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def _print_distribution(label: str, values: np.ndarray) -> None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  {label:22s}: no finite values")
        return
    p5, p95 = _p5_p95(v)
    print(
        f"  {label:22s}: min={v.min():.4f}  max={v.max():.4f}  "
        f"mean={v.mean():.4f}  std={v.std():.4f}  p5-p95=[{p5:.4f}, {p95:.4f}]"
    )


def _load_dataset(dataset_name: str, node_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(BASE_DIR, "datasets_gnn2", dataset_name)
    node_csv = os.path.join(data_dir, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(data_dir, "gnn_sample_meta.csv")
    if not os.path.exists(node_csv) or not os.path.exists(sample_csv):
        raise FileNotFoundError(f"Missing dataset files for '{dataset_name}'. Expected {data_dir}")

    df_node = pd.read_csv(node_csv)
    df_sample = pd.read_csv(sample_csv)

    required = ["sample_id", "node_idx"] + node_cols
    missing = [c for c in required if c not in df_node.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing node-table columns {missing}")

    for c in required:
        df_node[c] = pd.to_numeric(df_node[c], errors="coerce")
    df_node = df_node.dropna(subset=required).copy()

    if "sample_id" in df_sample.columns:
        df_sample["sample_id"] = pd.to_numeric(df_sample["sample_id"], errors="coerce")
    if "scenario_id" in df_sample.columns:
        df_sample["scenario_id"] = pd.to_numeric(df_sample["scenario_id"], errors="coerce")
        df_node = df_node.merge(
            df_sample[["sample_id", "scenario_id"]].drop_duplicates(),
            on="sample_id",
            how="left",
        )
    return df_node, df_sample


def _build_training_sample_frame(
    dataset_name: str, df_node: pd.DataFrame, df_sample: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    if dataset_name == "original":
        sample_df = df_node.groupby("sample_id")[ORIGINAL_NODE_COLS].sum().reset_index()
        value_cols = ORIGINAL_NODE_COLS
    elif dataset_name == "injection":
        sample_df = df_node.groupby("sample_id")[INJECTION_NODE_COLS].sum().reset_index()
        value_cols = INJECTION_NODE_COLS
    elif dataset_name == "loadtype":
        sample_df = (
            df_node.groupby("sample_id")
            .agg(
                m1_p_kw=("m1_p_kw", "sum"),
                m1_q_kvar=("m1_q_kvar", "sum"),
                m2_p_kw=("m2_p_kw", "sum"),
                m2_q_kvar=("m2_q_kvar", "sum"),
                m4_p_kw=("m4_p_kw", "sum"),
                m4_q_kvar=("m4_q_kvar", "sum"),
                m5_p_kw=("m5_p_kw", "sum"),
                m5_q_kvar=("m5_q_kvar", "sum"),
                q_cap_kvar=("q_cap_kvar", "sum"),
                p_pv_kw=("p_pv_kw", "sum"),
            q_pv_kvar=("q_pv_kvar", "sum"),
                p_sys_balance_kw=("p_sys_balance_kw", "first"),
                q_sys_balance_kvar=("q_sys_balance_kvar", "first"),
            )
            .reset_index()
        )
        value_cols = [
            "m1_p_kw",
            "m1_q_kvar",
            "m2_p_kw",
            "m2_q_kvar",
            "m4_p_kw",
            "m4_q_kvar",
            "m5_p_kw",
            "m5_q_kvar",
            "q_cap_kvar",
            "p_pv_kw",
            "q_pv_kvar",
            "p_sys_balance_kw",
            "q_sys_balance_kvar",
        ]
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    if "scenario_id" in df_sample.columns:
        sample_df = sample_df.merge(
            df_sample[["sample_id", "scenario_id"]].drop_duplicates(),
            on="sample_id",
            how="left",
        )
    return sample_df, value_cols


def _scenario_daily_max_frame(sample_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if "scenario_id" not in sample_df.columns:
        raise ValueError("sample_df must contain scenario_id")
    scenario_stats = sample_df.groupby("scenario_id")[value_cols].agg(["min", "max", "mean"]).reset_index()

    def _flatten_col(c):
        if isinstance(c, tuple):
            return f"{c[0]}_{c[1]}" if c[1] else str(c[0])
        return str(c)

    scenario_stats.columns = [_flatten_col(c) for c in scenario_stats.columns]
    if "scenario_id_" in scenario_stats.columns:
        scenario_stats = scenario_stats.rename(columns={"scenario_id_": "scenario_id"})
    return scenario_stats


def _run_baseline_feature_profiles() -> tuple[dict[str, dict[str, np.ndarray]], int]:
    dss_path = inj.compile_once()
    inj.setup_daily()
    try:
        dss.Text.Command("set maxcontroliter=20000")
    except Exception:
        pass

    full_node_list, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    kept_nodes = [n for n in full_node_list if n.split(".")[0] not in inj.EXCLUDED_UPSTREAM_BUSES]
    loadtype_edge_csv = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype", "gnn_edges_phase_static.csv")
    if not os.path.exists(loadtype_edge_csv):
        raise FileNotFoundError(f"Missing {loadtype_edge_csv}. Run run_loadtype_dataset.py first.")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_node_list, loadtype_edge_csv)

    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    p_load_total = float(inj.BASELINE["P_load_total_kw"])
    q_load_total = float(inj.BASELINE["Q_load_total_kvar"])
    p_pv_total = float(inj.BASELINE["P_pv_total_kw"])
    rng = np.random.default_rng(0)

    fam = {
        "original": {
            "series": {c: np.full(inj.NPTS, np.nan) for c in ORIGINAL_NODE_COLS},
            "node_values": {c: [] for c in ORIGINAL_NODE_COLS},
        },
        "injection": {
            "series": {c: np.full(inj.NPTS, np.nan) for c in INJECTION_NODE_COLS},
            "node_values": {c: [] for c in INJECTION_NODE_COLS},
        },
        "loadtype": {
            "series": {
                c: np.full(inj.NPTS, np.nan)
                for c in [
                    "m1_p_kw",
                    "m1_q_kvar",
                    "m2_p_kw",
                    "m2_q_kvar",
                    "m4_p_kw",
                    "m4_q_kvar",
                    "m5_p_kw",
                    "m5_q_kvar",
                    "q_cap_kvar",
                    "p_pv_kw",
                    "q_pv_kvar",
                    "p_sys_balance_kw",
                    "q_sys_balance_kvar",
                ]
            },
            "node_values": {c: [] for c in LOADTYPE_NODE_COLS},
        },
    }

    converged = 0
    for t in range(inj.NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, _, _, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=p_load_total,
            Q_load_total_kvar=q_load_total,
            P_pv_total_kw=p_pv_total,
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
        converged += 1

        busphP_pv, busphQ_pv = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        sum_p_load = sum_q_load = sum_p_pv = sum_q_pv = sum_q_cap = 0.0
        sum_p_inj = sum_q_inj = 0.0
        loadtype_sums = {
            "m1_p_kw": 0.0,
            "m1_q_kvar": 0.0,
            "m2_p_kw": 0.0,
            "m2_q_kvar": 0.0,
            "m4_p_kw": 0.0,
            "m4_q_kvar": 0.0,
            "m5_p_kw": 0.0,
            "m5_q_kvar": 0.0,
            "q_cap_kvar": 0.0,
            "p_pv_kw": 0.0,
            "q_pv_kvar": 0.0,
        }

        for n in kept_nodes:
            bus, phs = n.split(".")
            ph = int(phs)
            p_load = float(busphP_load.get((bus, ph), 0.0))
            q_load = float(busphQ_load.get((bus, ph), 0.0))
            p_pv = float(busphP_pv.get((bus, ph), 0.0))
            q_pv = float(busphQ_pv.get((bus, ph), 0.0))
            q_cap = inj.cap_q_kvar_per_node(bus, ph)
            p_inj = p_pv - p_load
            q_inj = q_pv - q_load + q_cap

            fam["original"]["node_values"]["p_load_kw"].append(p_load)
            fam["original"]["node_values"]["q_load_kvar"].append(q_load)
            fam["original"]["node_values"]["p_pv_kw"].append(p_pv)
            fam["original"]["node_values"]["q_pv_kvar"].append(q_pv)

            fam["injection"]["node_values"]["p_inj_kw"].append(p_inj)
            fam["injection"]["node_values"]["q_inj_kvar"].append(q_inj)

            m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
            m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
            m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
            m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
            m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
            m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
            m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
            m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))
            elec_dist = float(node_to_electrical_dist.get(n, 0.0))
            fam["loadtype"]["node_values"]["electrical_distance_ohm"].append(elec_dist)
            fam["loadtype"]["node_values"]["m1_p_kw"].append(m1_p)
            fam["loadtype"]["node_values"]["m1_q_kvar"].append(m1_q)
            fam["loadtype"]["node_values"]["m2_p_kw"].append(m2_p)
            fam["loadtype"]["node_values"]["m2_q_kvar"].append(m2_q)
            fam["loadtype"]["node_values"]["m4_p_kw"].append(m4_p)
            fam["loadtype"]["node_values"]["m4_q_kvar"].append(m4_q)
            fam["loadtype"]["node_values"]["m5_p_kw"].append(m5_p)
            fam["loadtype"]["node_values"]["m5_q_kvar"].append(m5_q)
            fam["loadtype"]["node_values"]["q_cap_kvar"].append(q_cap)
            fam["loadtype"]["node_values"]["p_pv_kw"].append(p_pv)
            fam["loadtype"]["node_values"]["q_pv_kvar"].append(q_pv)

            sum_p_load += p_load
            sum_q_load += q_load
            sum_p_pv += p_pv
            sum_q_pv += q_pv
            sum_q_cap += q_cap
            sum_p_inj += p_inj
            sum_q_inj += q_inj
            loadtype_sums["m1_p_kw"] += m1_p
            loadtype_sums["m1_q_kvar"] += m1_q
            loadtype_sums["m2_p_kw"] += m2_p
            loadtype_sums["m2_q_kvar"] += m2_q
            loadtype_sums["m4_p_kw"] += m4_p
            loadtype_sums["m4_q_kvar"] += m4_q
            loadtype_sums["m5_p_kw"] += m5_p
            loadtype_sums["m5_q_kvar"] += m5_q
            loadtype_sums["q_cap_kvar"] += q_cap
            loadtype_sums["p_pv_kw"] += p_pv
            loadtype_sums["q_pv_kvar"] += q_pv

        p_sys_balance = sum_p_load - sum_p_pv
        q_sys_balance = sum_q_load - sum_q_pv - sum_q_cap
        for _ in kept_nodes:
            fam["loadtype"]["node_values"]["p_sys_balance_kw"].append(p_sys_balance)
            fam["loadtype"]["node_values"]["q_sys_balance_kvar"].append(q_sys_balance)

        fam["original"]["series"]["p_load_kw"][t] = sum_p_load
        fam["original"]["series"]["q_load_kvar"][t] = sum_q_load
        fam["original"]["series"]["p_pv_kw"][t] = sum_p_pv
        fam["original"]["series"]["q_pv_kvar"][t] = sum_q_pv

        fam["injection"]["series"]["p_inj_kw"][t] = sum_p_inj
        fam["injection"]["series"]["q_inj_kvar"][t] = sum_q_inj

        for key, val in loadtype_sums.items():
            fam["loadtype"]["series"][key][t] = val
        fam["loadtype"]["series"]["p_sys_balance_kw"][t] = p_sys_balance
        fam["loadtype"]["series"]["q_sys_balance_kvar"][t] = q_sys_balance

    out = {}
    for name, payload in fam.items():
        out[name] = {
            "series": {k: np.asarray(v, dtype=float) for k, v in payload["series"].items()},
            "node_values": {k: np.asarray(v, dtype=float) for k, v in payload["node_values"].items()},
        }
    return out, converged


def _print_model_family_summary(
    dataset_name: str,
    title: str,
    df_node: pd.DataFrame,
    sample_df: pd.DataFrame,
    feature_cols: list[str],
    sample_value_cols: list[str],
    baseline: dict[str, np.ndarray],
    baseline_node_values: dict[str, np.ndarray],
    converged: int,
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"  Node rows: {len(df_node):,}  Samples: {df_node['sample_id'].nunique():,}  "
        f"Scenarios: {df_node['scenario_id'].nunique():,}"
    )

    print("\n  Training per-node feature distribution:")
    for c in feature_cols:
        _print_distribution(c, df_node[c].to_numpy(dtype=float))
    if dataset_name == "loadtype":
        tr_dist = df_node["electrical_distance_ohm"].to_numpy(dtype=float)
        inf_dist = baseline_node_values["electrical_distance_ohm"]
        tr_nonzero = int(np.sum(np.abs(tr_dist) > 1e-12))
        inf_nonzero = int(np.sum(np.abs(inf_dist) > 1e-12))
        if tr_nonzero == 0 and inf_nonzero > 0:
            print(
                "  [WARN] Stored loadtype dataset has all-zero `electrical_distance_ohm`, "
                "but current inference-time recomputation is nonzero. "
                "Regenerate `datasets_gnn2/loadtype` to train with the corrected distance feature."
            )

    n_per_scenario = sample_df.groupby("scenario_id").size()
    print(
        f"\n  Samples per scenario: min={int(n_per_scenario.min())} "
        f"max={int(n_per_scenario.max())}"
    )
    scenario_stats = _scenario_daily_max_frame(sample_df, sample_value_cols)
    print("\n  Training daily-max range across scenarios:")
    for c in sample_value_cols:
        vals = scenario_stats[f"{c}_max"].to_numpy(dtype=float)
        print(f"    {c:20s}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")

    print("\n  Baseline 24h daily MIN / MAX / MEAN (sigma=0, inj.BASELINE):")
    for c in sample_value_cols:
        vmin, vmax, vmean = _safe_stats(baseline[c])
        print(f"    {c:20s}: min={vmin:.4f}  max={vmax:.4f}  mean={vmean:.4f}")
    print(f"  Converged timesteps: {converged}/{inj.NPTS}")

    print("\n  Match check: baseline daily MAX vs training daily-max range")
    for c in sample_value_cols:
        tr_min = float(scenario_stats[f"{c}_max"].min())
        tr_max = float(scenario_stats[f"{c}_max"].max())
        _, bl_max, _ = _safe_stats(baseline[c])
        print(
            f"    {c:20s}: baseline max={bl_max:.4f}  "
            f"training range=[{tr_min:.4f}, {tr_max:.4f}]  in range={tr_min <= bl_max <= tr_max}"
        )

    print("\n  Model input distribution: training vs inference (node-level rows)")
    for c in feature_cols:
        tr_vals = df_node[c].to_numpy(dtype=float)
        inf_vals = baseline_node_values[c]
        tr_min, tr_max, tr_mean = _safe_stats(tr_vals)
        inf_min, inf_max, inf_mean = _safe_stats(inf_vals)
        tr_p5, tr_p95 = _p5_p95(tr_vals)
        inf_p5, inf_p95 = _p5_p95(inf_vals)
        tr_std = float(np.nanstd(tr_vals))
        inf_std = float(np.nanstd(inf_vals))
        print(f"\n    {c}:")
        print(
            f"      Training  (n={np.isfinite(tr_vals).sum():,}): "
            f"min={tr_min:.4f}  max={tr_max:.4f}  mean={tr_mean:.4f}  "
            f"std={tr_std:.4f}  p5-p95=[{tr_p5:.4f}, {tr_p95:.4f}]"
        )
        print(
            f"      Inference (n={np.isfinite(inf_vals).sum():,}): "
            f"min={inf_min:.4f}  max={inf_max:.4f}  mean={inf_mean:.4f}  "
            f"std={inf_std:.4f}  p5-p95=[{inf_p5:.4f}, {inf_p95:.4f}]"
        )
        in_range = tr_min <= inf_min and inf_max <= tr_max if np.isfinite(inf_vals).any() else False
        print(f"      Inference range inside training range: {in_range}")


def main() -> None:
    print("=" * 80)
    print("DATASET / INFERENCE SUMMARY FOR ORIGINAL, INJECTION, AND LOAD-TYPE MODELS")
    print("=" * 80)
    print(f"  DSS file used for baseline inference: {os.path.abspath(inj.DSS_FILE)}")
    print(
        "  Baseline totals: "
        f"P_load={float(inj.BASELINE['P_load_total_kw']):.2f} kW, "
        f"Q_load={float(inj.BASELINE['Q_load_total_kvar']):.2f} kVAR, "
        f"P_pv={float(inj.BASELINE['P_pv_total_kw']):.2f} kW"
    )

    df_orig_node, df_orig_sample = _load_dataset("original", ORIGINAL_NODE_COLS)
    df_inj_node, df_inj_sample = _load_dataset("injection", INJECTION_NODE_COLS)
    df_lt_node, df_lt_sample = _load_dataset("loadtype", LOADTYPE_NODE_COLS)

    orig_sample_df, orig_sample_cols = _build_training_sample_frame("original", df_orig_node, df_orig_sample)
    inj_sample_df, inj_sample_cols = _build_training_sample_frame("injection", df_inj_node, df_inj_sample)
    lt_sample_df, lt_sample_cols = _build_training_sample_frame("loadtype", df_lt_node, df_lt_sample)

    baseline_payload, n_converged = _run_baseline_feature_profiles()

    _print_model_family_summary(
        dataset_name="original",
        title="ORIGINAL MODEL FAMILY",
        df_node=df_orig_node,
        sample_df=orig_sample_df,
        feature_cols=ORIGINAL_NODE_COLS,
        sample_value_cols=orig_sample_cols,
        baseline=baseline_payload["original"]["series"],
        baseline_node_values=baseline_payload["original"]["node_values"],
        converged=n_converged,
    )
    _print_model_family_summary(
        dataset_name="injection",
        title="INJECTION MODEL FAMILY",
        df_node=df_inj_node,
        sample_df=inj_sample_df,
        feature_cols=INJECTION_NODE_COLS,
        sample_value_cols=inj_sample_cols,
        baseline=baseline_payload["injection"]["series"],
        baseline_node_values=baseline_payload["injection"]["node_values"],
        converged=n_converged,
    )
    _print_model_family_summary(
        dataset_name="loadtype",
        title="LOAD-TYPE MODEL FAMILY",
        df_node=df_lt_node,
        sample_df=lt_sample_df,
        feature_cols=LOADTYPE_NODE_COLS,
        sample_value_cols=lt_sample_cols,
        baseline=baseline_payload["loadtype"]["series"],
        baseline_node_values=baseline_payload["loadtype"]["node_values"],
        converged=n_converged,
    )


if __name__ == "__main__":
    main()
