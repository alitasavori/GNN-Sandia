"""
Original dataset generation: per-node P/Q load and PV P features.

This script reproduces the first GNN2 dataset (gnn_samples_out) using the shared
helpers from run_injection_dataset.py, and applies the same upstream-node exclusions
and static-edge extraction used by the other datasets.

Per-node features:
  - p_load_kw  : aggregated load active power at node i
  - q_load_kvar: aggregated load reactive power at node i
  - p_pv_kw    : aggregated PV active power at node i

Targets:
  - vmag_pu, vang_deg from AC power flow

Static graph:
  - gnn_edges_phase_static.csv produced by inj.extract_static_phase_edges_to_csv,
    which skips edges incident to buses in EXCLUDED_UPSTREAM_BUSES.
"""

import os
import numpy as np
import pandas as pd

import opendssdirect as dss
import run_injection_dataset as inj

# Repo root (so paths work when cwd changes, e.g. in notebook/Colab)
_REPO_ROOT = os.path.dirname(os.path.abspath(inj.__file__))
# Unified dataset directory: datasets_gnn2/original
OUT_DIR = os.path.join(_REPO_ROOT, "datasets_gnn2", "original")
os.makedirs(OUT_DIR, exist_ok=True)

EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")


def generate_gnn_snapshot_dataset_original(
    n_scenarios: int = 200,
    k_snapshots_per_scenario_total: int = 960,
    bins_by_profile: dict | None = None,
    include_anchors: bool = True,
    master_seed: int = 20260130,
    loadshape_name: str = "5minDayShape",
    irradshape_name: str = "IrradShape",
):
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

    os.makedirs(OUT_DIR, exist_ok=True)
    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}

    pd.DataFrame(
        {"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}
    ).to_csv(NODE_INDEX_CSV, index=False)
    print(
        f"[saved] master node index -> {NODE_INDEX_CSV} | N_nodes={len(node_names_master)}"
    )

    # Static edges (lines + transformers, bidirectional), with upstream buses excluded
    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_master, edge_csv_path=EDGE_CSV
    )

    csvL_token, lineL = inj.find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvPV_token, linePV = inj.find_loadshape_csv_in_dss(dss_path, irradshape_name)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    print("Loadshape line:", lineL)
    print("Irradshape line:", linePV)
    print("Resolved load CSV:", csvL)
    print("Resolved irrad CSV:", csvPV)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[dict] = []
    rows_node: list[dict] = []
    sample_id = 0
    kept = 0
    skipped_nonconv = 0
    skipped_badV = 0

    for s in range(n_scenarios):
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()

        node_names_s, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
        if len(node_names_s) != len(node_names_master):
            raise RuntimeError(f"Scenario {s}: node count mismatch")
        loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(
            bus_to_phases
        )
        pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load = sc["P_load_total_kw"]
        Q_load = sc["Q_load_total_kvar"]
        P_pv = sc["P_pv_total_kw"]
        sigL = sc["sigma_load"]
        sigPV = sc["sigma_pv"]
        prof_load, prof_pv = mL, mPV
        prof_net = (P_load * mL) - (P_pv * mPV)

        rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        times = inj.select_times_three_profiles(
            prof_load=prof_load,
            prof_pv=prof_pv,
            prof_net=prof_net,
            K_total=k_snapshots_per_scenario_total,
            bins_by_profile=bins_by_profile,
            include_anchors=include_anchors,
            rng=rng_times,
        )
        times = [int(t) for t in times]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        control_iters_converged_this_scenario = []

        for t in times:
            inj.set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = (
                inj.apply_snapshot_timeconditioned(
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
                    sigma_load=sigL,
                    sigma_pv=sigPV,
                    rng=rng_solve,
                )
            )

            try:
                dss.Solution.Solve()
            except Exception:
                pass  # e.g. #485 Max Control Iterations Exceeded (InvControl); solution may still be valid
            if not dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            try:
                val = getattr(dss.Solution, "ControlIterations", None)
                n_ctrl = val() if callable(val) else val
                if n_ctrl is not None:
                    control_iters_converged_this_scenario.append(n_ctrl)
            except Exception:
                pass

            busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(
                pv_to_dss, pv_to_busph
            )
            vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(
                node_names_master
            )
            vmag_arr = np.asarray(vmag_m, dtype=float)
            if (
                (not np.isfinite(vmag_arr).all())
                or (vmag_arr.min() < inj.VMAG_PU_MIN)
                or (vmag_arr.max() > inj.VMAG_PU_MAX)
            ):
                skipped_badV += 1
                continue

            vdict_m = {
                n: (float(vm), float(va))
                for n, vm, va in zip(node_names_master, vmag_m, vang_m)
            }

            rows_sample.append(
                {
                    "sample_id": sample_id,
                    "scenario_id": s,
                    "t_index": t,
                    "t_minutes": t * inj.STEP_MIN,
                    "P_load_total_kw": float(P_load),
                    "Q_load_total_kvar": float(Q_load),
                    "P_pv_total_kw": float(P_pv),
                    "sigma_load": float(sigL),
                    "sigma_pv": float(sigPV),
                    "m_loadshape": float(mL[t]),
                    "m_irradshape": float(mPV[t]),
                    "P_load_time_kw": float(totals["P_load_time_kw"]),
                    "Q_load_time_kvar": float(totals["Q_load_time_kvar"]),
                    "P_pv_time_kw": float(totals["P_pv_time_kw"]),
                    "p_load_kw_set_total": float(totals["p_load_kw_set_total"]),
                    "q_load_kvar_set_total": float(totals["q_load_kvar_set_total"]),
                    "p_pv_pmpp_kw_set_total": float(totals["p_pv_pmpp_kw_set_total"]),
                    "prof_load": float(prof_load[t]),
                    "prof_pv": float(prof_pv[t]),
                    "prof_net": float(prof_net[t]),
                }
            )

            for n in node_names_master:
                bus, phs = n.split(".")
                ph = int(phs)
                if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                    continue
                p_load_node = float(busphP_load.get((bus, ph), 0.0))
                q_load_node = float(busphQ_load.get((bus, ph), 0.0))
                p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
                q_pv_node = float(busphQ_pv_actual.get((bus, ph), 0.0))
                vm, va = vdict_m.get(n, (np.nan, np.nan))

                rows_node.append(
                    {
                        "sample_id": sample_id,
                        "node": n,
                        "node_idx": int(node_to_idx_master[n]),
                        "bus": bus,
                        "phase": int(ph),
                        "p_load_kw": p_load_node,
                        "q_load_kvar": q_load_node,
                        "p_pv_kw": p_pv_node,
                        "q_pv_kvar": q_pv_node,
                        "vmag_pu": float(vm),
                        "vang_deg": float(va),
                    }
                )

            sample_id += 1
            kept += 1

        ctrl_summary = ""
        if control_iters_converged_this_scenario:
            arr = np.array(control_iters_converged_this_scenario, dtype=float)
            ctrl_summary = (
                f" ctrl_iter: n={len(arr)} min={int(arr.min())} max={int(arr.max())} mean={float(arr.mean()):.1f}"
            )
        print(
            f"[scenario {s+1}/{n_scenarios}] kept={kept} "
            f"skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} "
            f"Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}{ctrl_summary}"
        )

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[ORIGINAL DATASET] Saved to {OUT_DIR}/")
    print(
        f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | "
        f"node-rows={len(df_node)}"
    )
    print(
        "  Features per node: p_load_kw, q_load_kvar, p_pv_kw, q_pv_kvar; "
        "upstream buses sourcebus/800 excluded"
    )
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_gnn_snapshot_dataset_original(
        n_scenarios=200,
        k_snapshots_per_scenario_total=960,
        bins_by_profile={"load": 10, "pv": 10, "net": 10},
        include_anchors=True,
        master_seed=20260130,
    )

