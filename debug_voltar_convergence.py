"""
Debug Volt-Var convergence: run one or a few (scenario, timestep) solves and print
ControlIterations, Converged, and voltage at PV bus and PV Q.

Usage:
  python debug_voltar_convergence.py
  DEBUG_VOLTVAR=1 python run_original_dataset.py   # log nonconv in main pipeline

Single-snapshot: set SCENARIO_ID and T_INDEX below (0-based), then run this script.
"""
import os
import sys
import numpy as np

import opendssdirect as dss
import run_injection_dataset as inj

# ---------- Configure one snapshot to debug (0-based) ----------
SCENARIO_ID = 0
T_INDEX = 100
MAX_CONTROL_ITER = 20000
LOADSHAPE_NAME = "5minDayShape"
IRRADSHAPE_NAME = "IrradShape"
# Set to True to run multiple maxcontroliter values and see when it converges
SWEEP_ITER = False
ITER_LIST = [100, 500, 1000, 5000, 10000, 20000]
MASTER_SEED = 20260130


def get_control_iterations():
    try:
        return dss.Solution.ControlIterations()
    except Exception:
        return None


def get_v_at_bus(bus_name: str):
    """Return list of (phase, Vmag pu) at bus."""
    dss.Circuit.SetActiveBus(bus_name)
    n = dss.Bus.NumNodes()
    if n == 0:
        return []
    nodes = dss.Bus.Nodes()
    volts = dss.Bus.VMagAngle()
    base_kv = dss.Bus.kVBase()
    if base_kv and base_kv > 0:
        v_pu = [volts[2 * i] / (base_kv * 1000) for i in range(len(volts) // 2)]
    else:
        v_pu = [volts[2 * i] for i in range(len(volts) // 2)]
    return list(zip(nodes, v_pu))


def get_pv_q(pv_name: str = "PV850"):
    """Return (P, Q) in kW, kVAR for PVSystem."""
    dss.Circuit.SetActiveElement(f"PVSystem.{pv_name}")
    powers = dss.CktElement.Powers()
    P = sum(powers[i] / 1000.0 for i in range(0, len(powers), 2))
    Q = sum(powers[i + 1] / 1000.0 for i in range(0, len(powers), 2))
    return P, Q


def run_one_snapshot(s_id: int, t_idx: int, max_ctrl: int, mL, mPV):
    """Compile, set scenario and time, solve once; return converged, control_iters, V/Q."""
    dss_path = os.path.abspath(inj.DSS_FILE)
    if not os.path.exists(dss_path):
        print(f"DSS not found: {dss_path}")
        return None
    dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{dss_path}"')
    inj._apply_voltage_bases()
    inj.setup_daily()

    node_names_s, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    rng = np.random.default_rng(MASTER_SEED)
    for _ in range(s_id + 1):
        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng)
    P_load = sc["P_load_total_kw"]
    Q_load = sc["Q_load_total_kvar"]
    P_pv = sc["P_pv_total_kw"]
    sigL = sc["sigma_load"]
    sigPV = sc["sigma_pv"]

    inj.set_time_index(t_idx)
    rng_solve = np.random.default_rng(MASTER_SEED + 1)
    inj.apply_snapshot_timeconditioned(
        P_load_total_kw=P_load,
        Q_load_total_kvar=Q_load,
        P_pv_total_kw=P_pv,
        mL_t=float(mL[t_idx]),
        mPV_t=float(mPV[t_idx]),
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

    dss.Text.Command(f"set maxcontroliter={max_ctrl}")
    try:
        dss.Solution.Solve()
    except Exception as e:
        print(f"  Solve() raised: {e}")

    converged = dss.Solution.Converged()
    n_ctrl = get_control_iterations()
    v850 = get_v_at_bus("850")
    p_pv, q_pv = get_pv_q("PV850")
    return {
        "converged": converged,
        "control_iterations": n_ctrl,
        "V_850": v850,
        "P_PV850_kW": p_pv,
        "Q_PV850_kvar": q_pv,
    }


def main():
    dss_path = os.path.abspath(inj.DSS_FILE)
    if not os.path.exists(dss_path):
        print(f"DSS not found: {dss_path}")
        sys.exit(1)
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, LOADSHAPE_NAME)
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, IRRADSHAPE_NAME)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    print("Debug Volt-Var convergence (single snapshot)")
    print(f"  Scenario={SCENARIO_ID}, t_index={T_INDEX}, maxcontroliter={MAX_CONTROL_ITER}")
    print()

    if SWEEP_ITER:
        print("Sweeping maxcontroliter to see when solution converges:")
        for max_iter in ITER_LIST:
            res = run_one_snapshot(SCENARIO_ID, T_INDEX, max_iter, mL, mPV)
            if res is None:
                sys.exit(1)
            c = "OK" if res["converged"] else "FAIL"
            print(f"  maxcontroliter={max_iter:5d} -> Converged={res['converged']} ({c}) ControlIterations={res['control_iterations']}")
            if not res["converged"] and res["V_850"]:
                print(f"    V at 850 (pu): {res['V_850']}  Q_PV850={res['Q_PV850_kvar']:.2f} kVAR")
        return

    res = run_one_snapshot(SCENARIO_ID, T_INDEX, MAX_CONTROL_ITER, mL, mPV)
    if res is None:
        sys.exit(1)
    print(f"Converged: {res['converged']}")
    print(f"ControlIterations: {res['control_iterations']}")
    print(f"V at bus 850 (pu): {res['V_850']}")
    print(f"PV850 P={res['P_PV850_kW']:.2f} kW  Q={res['Q_PV850_kvar']:.2f} kVAR")
    if not res["converged"] and res["control_iterations"] == MAX_CONTROL_ITER:
        print("\n-> Hit max control iterations; inverter Q may be oscillating. Try relaxing VoltVar curve or VoltageChangeTolerance in DSS.")


if __name__ == "__main__":
    main()
