"""
Fifth dataset generation: delta-V prediction with 5× PV scaling.
Same as fourth dataset (run_deltav_dataset.py) but PV power is scaled by 5×
to produce larger delta-V values. Uses 24h profile (mL[t], mPV[t]) like run_deltav_dataset.py.
Output: gnn_samples_deltav_5x_full/
"""
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj
from run_loadtype_dataset import (
    _apply_snapshot_with_per_type,
    _compute_electrical_distance_from_source,
)

OUT_DIR = "gnn_samples_deltav_5x_full"
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")

PV_SCALE = 5.0


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


def generate_gnn_snapshot_dataset_deltav_5x(
    n_scenarios=1000,
    total_samples=57600,
    master_seed=20260130,
    pv_scale=PV_SCALE,
    loadshape_name="5minDayShape",
    irradshape_name="IrradShape",
    bins_by_profile=None,
    include_anchors=True,
):
    """
    Generate delta-V dataset with PV scaled by pv_scale (default 5×).
    Uses 24h profile (mL[t], mPV[t]) like run_deltav_dataset.py for train-test alignment.
    """
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

    k_base = total_samples // n_scenarios
    n_extra = total_samples % n_scenarios
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

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, irradshape_name)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    rng_master = np.random.default_rng(master_seed)
    rows_sample = []
    rows_node = []
    sample_id = 0
    kept = 0
    skipped_nonconv = 0
    skipped_badV = 0

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

        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load = sc["P_load_total_kw"]
        Q_load = sc["Q_load_total_kvar"]
        P_pv_baseline = sc["P_pv_total_kw"]
        sigL = sc["sigma_load"]
        sigPV = sc["sigma_pv"]
        P_pv_scaled = P_pv_baseline * pv_scale

        prof_load = mL
        prof_pv = mPV
        prof_net = (P_load * mL) - (P_pv_scaled * mPV)
        k_this = k_base + (1 if s < n_extra else 0)
        rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        times = inj.select_times_three_profiles(
            prof_load=prof_load, prof_pv=prof_pv, prof_net=prof_net,
            K_total=k_this, bins_by_profile=bins_by_profile,
            include_anchors=include_anchors, rng=rng_times,
        )
        times = [int(t) for t in times]

        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        for t in times:
            mL_t = float(mL[t])
            mPV_t = float(mPV[t])
            if mPV_t < 0.01:
                mPV_t = 0.01

            inj.set_time_index(t)

            totals_z, busphP_load, busphQ_load, busphP_pv_z, busphQ_pv_z, busph_per_type = _apply_snapshot_zero_pv(
                P_load_total_kw=P_load,
                Q_load_total_kvar=Q_load,
                mL_t=mL_t,
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
                skipped_nonconv += 1
                continue

            vmag_zero_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            vmag_zero_arr = np.asarray(vmag_zero_m, dtype=float)
            if (not np.isfinite(vmag_zero_arr).all()) or (vmag_zero_arr.min() < inj.VMAG_PU_MIN) or (vmag_zero_arr.max() > inj.VMAG_PU_MAX):
                skipped_badV += 1
                continue

            vmag_zero_pu = {n: float(v) for n, v in zip(node_names_master, vmag_zero_m)}

            totals, busphP_load_k, busphQ_load_k, busphP_pv, busphQ_pv, busph_per_type = _apply_snapshot_with_per_type(
                P_load_total_kw=P_load,
                Q_load_total_kvar=Q_load,
                P_pv_total_kw=P_pv_scaled,
                mL_t=mL_t,
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
            sum_p_load = float(sum(busphP_load_k.values()))
            sum_p_pv = float(sum(busphP_pv.values()))
            sum_q_load = float(sum(busphQ_load_k.values()))
            sum_q_cap = sum(inj.CAP_Q_KVAR.values())
            q_sys_balance = sum_q_load - sum_q_cap
            p_sys_balance = sum_p_load - sum_p_pv

            rows_sample.append({
                "sample_id": sample_id,
                "scenario_id": s,
                "time_idx": t,
                "m_load": mL_t,
                "m_pv": mPV_t,
                "P_load_total_kw": float(P_load),
                "Q_load_total_kvar": float(Q_load),
                "P_pv_total_kw": float(P_pv_scaled * mPV_t),
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

        print(f"[scenario {s+1}/{n_scenarios}] kept={kept} skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} Pload={P_load:.1f} Qload={Q_load:.1f} Ppv_base={P_pv_baseline:.1f} (×{pv_scale}) times={len(times)}")

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[DELTA-V 5× DATASET] Saved to {OUT_DIR}/")
    print(f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  PV scale: {pv_scale}× (larger delta-V than baseline)")
    print(f"  Target: vmag_delta_pu (= vmag_with_pv - vmag_zero_pv)")
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_gnn_snapshot_dataset_deltav_5x(
        n_scenarios=1000,
        total_samples=57600,
        master_seed=20260130,
        pv_scale=5.0,
    )
