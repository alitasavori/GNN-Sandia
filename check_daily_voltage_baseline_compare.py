"""
Compare daily voltage extremes (min and max) for two baselines:

1) Original DSS nominal totals:
     P_load_total_kw = 1769.0
     Q_load_total_kvar = 1044.0
     P_pv_total_kw   = 1000.0

2) Current dataset baseline from run_injection_dataset.BASELINE
   (the already-scaled values used in all dataset generators).

For each baseline we:
  - Set sigma_load = sigma_pv = 0 (no noise)
  - Sweep all 288 timesteps (5 min resolution)
  - Apply snapshot using the same DEVICE_P_SHARE / PV shares as datasets
  - Solve AC power flow
  - Track global min and max |V_i| over all non-upstream nodes and times.
"""

import numpy as np
import opendssdirect as dss

import run_injection_dataset as inj


def daily_vmin_vmax(P_load_kw, Q_load_kvar, P_pv_kw, max_control_iter=None):
    dss_path = inj.compile_once()
    inj.setup_daily()
    if max_control_iter is not None:
        try:
            dss.Text.Command(f"set maxcontroliter={max_control_iter}")
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

    sigma_load = 0.0
    sigma_pv = 0.0
    rng = np.random.default_rng(0)

    vmin = float("inf")
    vmax = float("-inf")
    node_vmin = None
    node_vmax = None
    t_vmin = None
    phase_voltages_at_vmin = None  # dict phase -> pu for bus where vmin occurred

    for t in range(inj.NPTS):
        inj.set_time_index(t)
        inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load_kw,
            Q_load_total_kvar=Q_load_kvar,
            P_pv_total_kw=P_pv_kw,
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

        try:
            dss.Solution.Solve()
        except Exception:
            pass  # e.g. Max Control Iterations Exceeded warning raised as exception
        if not dss.Solution.Converged():
            continue

        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        node_to_v = {n: float(v) for n, v in zip(node_names_master, vmag_m)}
        for n, vm in zip(node_names_master, vmag_m):
            bus = n.split(".")[0]
            if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                continue
            vmf = float(vm)
            if vmf < vmin:
                vmin = vmf
                node_vmin = n  # format "bus.phase"
                t_vmin = t
                # All phase voltages for this bus at this time
                phase_voltages_at_vmin = {}
                for nn, vv in node_to_v.items():
                    if nn.startswith(bus + "."):
                        try:
                            ph = nn.split(".")[1]
                            phase_voltages_at_vmin[ph] = vv
                        except IndexError:
                            pass
            if vmf > vmax:
                vmax = vmf
                node_vmax = n

    return vmin, vmax, node_vmin, node_vmax, t_vmin, phase_voltages_at_vmin


def main():
    # Original DSS nominal totals
    P_load_orig = 1769.0
    Q_load_orig = 1044.0
    P_pv_orig = 1000.0

    print("=== Original DSS nominal totals ===")
    print(f"P_load_total_kw = {P_load_orig:.1f}")
    print(f"Q_load_total_kvar = {Q_load_orig:.1f}")
    print(f"P_pv_total_kw = {P_pv_orig:.1f}")
    vmin_orig, vmax_orig, node_min_orig, node_max_orig, t_min_orig, pv_min_orig = daily_vmin_vmax(P_load_orig, Q_load_orig, P_pv_orig)
    print(f"  vmin = {vmin_orig:.6f} pu  (bus {node_min_orig.split('.')[0]}, phase {node_min_orig.split('.')[1]})" if node_min_orig else f"  vmin = {vmin_orig:.6f} pu")
    print(f"  vmax = {vmax_orig:.6f} pu  (bus {node_max_orig.split('.')[0]}, phase {node_max_orig.split('.')[1]})" if node_max_orig else f"  vmax = {vmax_orig:.6f} pu\n")

    # Current dataset baseline (already scaled) from inj.BASELINE
    P_load_new = float(inj.BASELINE["P_load_total_kw"])
    Q_load_new = float(inj.BASELINE["Q_load_total_kvar"])
    P_pv_new = float(inj.BASELINE["P_pv_total_kw"])

    print("=== Current dataset BASELINE totals ===")
    print(f"P_load_total_kw = {P_load_new:.2f}")
    print(f"Q_load_total_kvar = {Q_load_new:.2f}")
    print(f"P_pv_total_kw = {P_pv_new:.2f}")
    vmin_new, vmax_new, node_min_new, node_max_new, t_min_new, pv_min_new = daily_vmin_vmax(P_load_new, Q_load_new, P_pv_new)
    print(f"  vmin = {vmin_new:.6f} pu  (bus {node_min_new.split('.')[0]}, phase {node_min_new.split('.')[1]})" if node_min_new else f"  vmin = {vmin_new:.6f} pu")
    print(f"  vmax = {vmax_new:.6f} pu  (bus {node_max_new.split('.')[0]}, phase {node_max_new.split('.')[1]})" if node_max_new else f"  vmax = {vmax_new:.6f} pu")


if __name__ == "__main__":
    main()

