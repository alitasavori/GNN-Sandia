"""
Run the baseline daily voltage check (288 steps) with Volt-Var *disabled* for PV850
by using a temporary DSS that has the InvControl line commented out.
Compares converged vs non-converged to confirm the problem is with PV Volt-Var.
"""
import os
import sys
import tempfile

import run_injection_dataset as inj
from check_daily_voltage_baseline_compare import daily_vmin_vmax, _print_control_iters

# Original DSS path
ORIG_DSS = os.path.join(os.path.dirname(os.path.abspath(inj.__file__)), "new dss from dr mirzaei", "IEEE34_PV.dss")
INVCONTROL_LINE_START = "New InvControl.InvCtrl_PV850"


def make_dss_no_voltvar():
    """Create a temp DSS with InvControl commented out. Returns path to temp file."""
    with open(ORIG_DSS, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.strip().startswith(INVCONTROL_LINE_START):
            out_lines.append("! " + line)  # comment out
        else:
            out_lines.append(line)
    fd, path = tempfile.mkstemp(suffix=".dss", prefix="IEEE34_PV_no_voltvar_", dir=os.path.dirname(ORIG_DSS))
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return path


def main():
    if not os.path.isfile(ORIG_DSS):
        print("DSS not found:", ORIG_DSS)
        sys.exit(1)
    temp_dss = make_dss_no_voltvar()
    try:
        old_dss = inj.DSS_FILE
        inj.DSS_FILE = temp_dss
        try:
            print("=" * 60)
            print("Daily voltage: Volt-Var DISABLED (InvControl commented out)")
            print("DSS: (temp copy of IEEE34_PV.dss without InvControl.InvCtrl_PV850)")
            print("Baseline: P_load=1769 kW, Q_load=1044 kVAR, P_pv=1000 kW")
            print("288 steps, sigma=0. Same profiles (5minDayShape, 5MinuteIrradiance).")
            print("=" * 60)
            vmin, vmax, node_vmin, node_vmax, t_vmin, phase_voltages_at_vmin, n_nonconv, control_iters_converged = daily_vmin_vmax(
                inj.BASELINE["P_load_total_kw"],
                inj.BASELINE["Q_load_total_kvar"],
                inj.BASELINE["P_pv_total_kw"],
                max_control_iter=20000,
            )
            n_conv = inj.NPTS - n_nonconv
            print("  daily profile: {}/{} converged, {} non-converged".format(n_conv, inj.NPTS, n_nonconv))
            _print_control_iters(control_iters_converged)
            print("  vmin = {:.6f} pu at {}  vmax = {:.6f} pu at {}".format(
                vmin, node_vmin or "?", vmax, node_vmax or "?"))
        finally:
            inj.DSS_FILE = old_dss
    finally:
        try:
            os.remove(temp_dss)
        except Exception:
            pass
    print("")
    print("Compare with Volt-Var ENABLED: 86/288 converged, 202 non-converged.")
    print("If with Volt-Var disabled you get 288/288 (or nearly all), the problem is PV Volt-Var.")


if __name__ == "__main__":
    main()
