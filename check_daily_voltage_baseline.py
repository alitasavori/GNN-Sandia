"""
Check baseline daily voltage profile:
- Uses BASELINE values and current RANGES/flags from run_injection_dataset.py
- Applies loads/PV with sigma_load = sigma_pv = 0 (no noise)
- Sweeps all 288 timesteps and reports the minimum voltage magnitude (pu)
  over all non-upstream nodes (excluding sourcebus and 800) and all times.
"""

import os
import numpy as np

import opendssdirect as dss
import run_injection_dataset as inj


def main():
    dss_path = inj.compile_once()
    inj.setup_daily()

    # Master node list
    node_names_master, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()

    # Device maps
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    # Load and PV profiles
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    # Use BASELINE totals, but zero noise for this diagnostic
    P_load = inj.BASELINE["P_load_total_kw"]
    Q_load = inj.BASELINE["Q_load_total_kvar"]
    P_pv = inj.BASELINE["P_pv_total_kw"]
    sigma_load = 0.0
    sigma_pv = 0.0

    global_min_v = float("inf")
    global_min_t = None
    global_min_node = None

    for t in range(inj.NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
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
            sigma_load=sigma_load,
            sigma_pv=sigma_pv,
            rng=np.random.default_rng(0),
        )

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(f"[WARN] Power flow did not converge at t={t}")
            continue

        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        # Exclude upstream buses, same as datasets
        for n, vm in zip(node_names_master, vmag_m):
            bus = n.split(".")[0]
            if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                continue
            vmf = float(vm)
            if vmf < global_min_v:
                global_min_v = vmf
                global_min_t = t
                global_min_node = n

    if global_min_node is None:
        print("No valid voltages found.")
        return

    print(f"Minimum vmag over baseline day (excluding upstream buses):")
    print(f"  vmin = {global_min_v:.6f} pu at node {global_min_node} at timestep t={global_min_t} (minute={global_min_t * inj.STEP_MIN})")
    if global_min_v < 0.95:
        print("  --> There ARE points below 0.95 pu.")
    else:
        print("  --> All non-upstream nodes stay above 0.95 pu over the day.")


if __name__ == "__main__":
    main()

