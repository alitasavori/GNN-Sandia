"""
Fourth dataset generation: delta-V prediction (voltage change due to PV).
Scenario structure: constant loads per scenario; one zero-PV solve stores vmag_zero_pv as node feature;
samples vary only PV; target = vmag_with_pv - vmag_zero_pv (delta V).
Output: gnn_samples_deltav_full/
"""
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj
from run_loadtype_dataset import (
    DEVICE_TO_MODEL,
    _apply_snapshot_with_per_type,
    _compute_electrical_distance_from_source,
    SOURCE_BUSES,
)

OUT_DIR = "gnn_samples_deltav_full"
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")


def _apply_snapshot_zero_pv(
    P_load_total_kw, Q_load_total_kvar,
    mL_t,
    loads_dss, dev_to_dss_load, dev_to_busph_load,
    pv_dss, pv_to_dss, pv_to_busph,
    sigma_load,
    rng,
):
    """Apply loads with mL_t, PV=0. Sets Pmpp=0 for all PV systems."""
    return _apply_snapshot_with_per_type(
        P_load_total_kw=P_load_total_kw,
        Q_load_total_kvar=Q_load_total_kvar,
        P_pv_total_kw=0.0,
        mL_t=mL_t,
        mPV_t=0.0,
        loads_dss=loads_dss,
        dev_to_dss_load=dev_to_dss_load,
        dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss,
        pv_to_dss=pv_to_dss,
        pv_to_busph=pv_to_busph,
        sigma_load=sigma_load,
        sigma_pv=0.0,
        rng=rng,
    )


def generate_gnn_snapshot_dataset_deltav(
    n_scenarios=400,
    k_samples_per_scenario=480,
    master_seed=20260130,
):
    """
    Generate delta-V dataset:
    - Each scenario: constant loads (P_load, Q_load); solve with PV=0 -> vmag_zero_pv per node
    - For each scenario: k_samples with varying PV only; target = vmag_with_pv - vmag_zero_pv
    - Features: dataset-3 load-type features + vmag_zero_pv_pu
    - Edges: same as dataset 3 (R_full, X_full)
    - Total samples ≈ n_scenarios * k_samples_per_scenario (~192k)
    """
    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}

    pd.DataFrame({"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}).to_csv(
        NODE_INDEX_CSV, index=False
    )
    print(f"[saved] master node index -> {NODE_INDEX_CSV} | N_nodes={len(node_names_master)}")

    inj.extract_static_phase_edges_to_csv(node_names_master=node_names_master, edge_csv_path=EDGE_CSV)
    node_to_electrical_dist = _compute_electrical_distance_from_source(node_names_master, EDGE_CSV)

    rng_master = np.random.default_rng(master_seed)
    rows_sample = []
    rows_node = []
    sample_id = 0
    kept = 0
    skipped_nonconv = 0
    skipped_badV = 0

    # PV multiplier range for sampling (e.g. 0.1 to 1.0 to avoid pure zero)
    mPV_min, mPV_max = 0.1, 1.0

    for s in range(n_scenarios):
        inj.dss.Basic.ClearAll()
        inj.dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()

        node_names_s, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
        if len(node_names_s) != len(node_names_master):
            raise RuntimeError(f"Scenario {s}: node count mismatch")
        loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
        pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

        # Sample scenario: constant loads (P_load, Q_load, P_pv baseline for scaling)
        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load = sc["P_load_total_kw"]
        Q_load = sc["Q_load_total_kvar"]
        P_pv_baseline = sc["P_pv_total_kw"]
        sigL = sc["sigma_load"]
        sigPV = sc["sigma_pv"]

        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        # Step 1: Zero-PV solve — constant loads (mL=1.0), PV=0
        totals_z, busphP_load, busphQ_load, busphP_pv_z, busphQ_pv_z, busph_per_type = _apply_snapshot_zero_pv(
            P_load_total_kw=P_load,
            Q_load_total_kvar=Q_load,
            mL_t=1.0,
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=sigL,
            rng=rng_solve,
        )
        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            print(f"[scenario {s+1}/{n_scenarios}] ZERO-PV solve did not converge, skipping scenario")
            continue

        vmag_zero_m, vang_zero_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        vmag_zero_arr = np.asarray(vmag_zero_m, dtype=float)
        if (not np.isfinite(vmag_zero_arr).all()) or (vmag_zero_arr.min() < inj.VMAG_PU_MIN) or (vmag_zero_arr.max() > inj.VMAG_PU_MAX):
            print(f"[scenario {s+1}/{n_scenarios}] Zero-PV voltages out of range, skipping scenario")
            continue

        vmag_zero_pu = {n: float(v) for n, v in zip(node_names_master, vmag_zero_m)}

        # Load features are constant for the scenario (from zero-PV apply)
        sum_p_load = float(sum(busphP_load.values()))
        sum_q_load = float(sum(busphQ_load.values()))
        sum_q_cap = sum(inj.CAP_Q_KVAR.values())
        q_sys_balance = sum_q_load - sum_q_cap

        # Step 2: Sample k_samples with varying PV only
        mPV_samples = rng_solve.uniform(mPV_min, mPV_max, size=k_samples_per_scenario)

        for k in range(k_samples_per_scenario):
            mPV_t = float(mPV_samples[k])
            totals, busphP_load_k, busphQ_load_k, busphP_pv, busphQ_pv, busph_per_type = _apply_snapshot_with_per_type(
                P_load_total_kw=P_load,
                Q_load_total_kvar=Q_load,
                P_pv_total_kw=P_pv_baseline,
                mL_t=1.0,
                mPV_t=mPV_t,
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

            inj.dss.Solution.Solve()
            if not inj.dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            vmag_arr = np.asarray(vmag_m, dtype=float)
            if (not np.isfinite(vmag_arr).all()) or (vmag_arr.min() < inj.VMAG_PU_MIN) or (vmag_arr.max() > inj.VMAG_PU_MAX):
                skipped_badV += 1
                continue

            vmag_with_pv = {n: float(vmag_m[i]) for i, n in enumerate(node_names_master)}
            sum_p_pv = float(sum(busphP_pv.values()))
            p_sys_balance = sum_p_load - sum_p_pv

            rows_sample.append({
                "sample_id": sample_id,
                "scenario_id": s,
                "m_pv": mPV_t,
                "P_load_total_kw": float(P_load),
                "Q_load_total_kvar": float(Q_load),
                "P_pv_total_kw": float(P_pv_baseline * mPV_t),
                "sigma_load": float(sigL),
                "sigma_pv": float(sigPV),
                "p_sys_balance_kw": p_sys_balance,
                "q_sys_balance_kvar": q_sys_balance,
            })

            for n in node_names_master:
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
                p_pv_node = float(busphP_pv.get((bus, ph), 0.0))
                elec_dist = float(node_to_electrical_dist.get(n, 0.0))

                vm_with_pv = vmag_with_pv[n]
                vmag_zero = vmag_zero_pu[n]
                vmag_delta = vm_with_pv - vmag_zero

                rows_node.append({
                    "sample_id": sample_id,
                    "node": n,
                    "node_idx": int(node_to_idx_master[n]),
                    "bus": bus,
                    "phase": int(ph),
                    "electrical_distance_ohm": elec_dist,
                    "m1_p_kw": m1_p,
                    "m1_q_kvar": m1_q,
                    "m2_p_kw": m2_p,
                    "m2_q_kvar": m2_q,
                    "m4_p_kw": m4_p,
                    "m4_q_kvar": m4_q,
                    "m5_p_kw": m5_p,
                    "m5_q_kvar": m5_q,
                    "q_cap_kvar": q_cap_node,
                    "p_pv_kw": p_pv_node,
                    "p_sys_balance_kw": p_sys_balance,
                    "q_sys_balance_kvar": q_sys_balance,
                    "vmag_zero_pv_pu": vmag_zero,
                    "vmag_delta_pu": vmag_delta,
                })

            sample_id += 1
            kept += 1

        print(f"[scenario {s+1}/{n_scenarios}] kept={kept} skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} Pload={P_load:.1f} Qload={Q_load:.1f} Ppv_base={P_pv_baseline:.1f}")

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[DELTA-V DATASET] Saved to {OUT_DIR}/")
    print(f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  Features per node: electrical_distance_ohm, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv, p_sys_balance, q_sys_balance, vmag_zero_pv_pu")
    print(f"  Target: vmag_delta_pu (= vmag_with_pv - vmag_zero_pv)")
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_gnn_snapshot_dataset_deltav(
        n_scenarios=400,
        k_samples_per_scenario=480,
        master_seed=20260130,
    )
