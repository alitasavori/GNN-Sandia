"""
Search for baseline (P_load, Q_load, P_pv) multipliers such that the
daily voltage profile (no noise, using 5minDayShape + IrradShape) never
drops below 0.95 pu on any non-upstream node.

We sweep a small grid of scaling factors around the current BASELINE and
report the minimum voltage for each combination. This is a diagnostic
script only; after picking a combination, we will manually update
inj.BASELINE in run_injection_dataset.py.
"""

import os
import numpy as np

import opendssdirect as dss
import run_injection_dataset as inj


def min_v_for_scales(p_load_scale, q_load_scale, p_pv_scale):
    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names_master, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    P_load_base = inj.BASELINE["P_load_total_kw"] * p_load_scale
    Q_load_base = inj.BASELINE["Q_load_total_kvar"] * q_load_scale
    P_pv_base = inj.BASELINE["P_pv_total_kw"] * p_pv_scale

    sigma_load = 0.0
    sigma_pv = 0.0

    global_min_v = float("inf")

    rng = np.random.default_rng(0)

    for t in range(inj.NPTS):
        inj.set_time_index(t)
        inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load_base,
            Q_load_total_kvar=Q_load_base,
            P_pv_total_kw=P_pv_base,
            mL_t=float(mL[t]),
            mPV_t=float(mPV[t]),
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=sigma_load,
            sigma_pv=sigma_pv,
            rng=rng,
        )

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return np.nan

        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        for n, vm in zip(node_names_master, vmag_m):
            bus = n.split(".")[0]
            if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                continue
            vmf = float(vm)
            if vmf < global_min_v:
                global_min_v = vmf

    return global_min_v


def main():
    # Coarse grid of scales to try
    p_load_scales = [0.6, 0.7, 0.8]
    p_pv_scales = [1.0, 1.2, 1.4]

    print("Baseline totals (from inj.BASELINE):")
    print(f"  P_load_total_kw = {inj.BASELINE['P_load_total_kw']:.2f}")
    print(f"  Q_load_total_kvar = {inj.BASELINE['Q_load_total_kvar']:.2f}")
    print(f"  P_pv_total_kw = {inj.BASELINE['P_pv_total_kw']:.2f}")
    print()

    results = []
    for pl in p_load_scales:
        for ppv in p_pv_scales:
            print(f"Testing P_load_scale={pl:.2f}, Q_load_scale={pl:.2f}, P_pv_scale={ppv:.2f} ...")
            vmin = min_v_for_scales(pl, pl, ppv)
            print(f"  -> vmin = {vmin:.6f} pu")
            results.append((pl, ppv, vmin))

    print("\nSummary (sorted by vmin descending):")
    results = sorted(results, key=lambda x: x[2], reverse=True)
    for pl, ppv, vmin in results:
        print(
            f"P_load_scale={pl:.2f}, Q_load_scale={pl:.2f}, "
            f"P_pv_scale={ppv:.2f} -> vmin={vmin:.6f} pu"
        )


if __name__ == "__main__":
    main()

