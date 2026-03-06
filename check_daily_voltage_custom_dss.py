"""
Run daily voltage min/max for one or more DSS files (e.g. in 'new dss from dr mirzaei').
Uses the same baseline totals and device shares as the main dataset; 288 steps, no noise.
Reports vmin and vmax over all non-upstream nodes for the daily profile.
"""
import os
import sys
import run_injection_dataset as inj
from check_daily_voltage_baseline_compare import daily_vmin_vmax, _print_control_iters


def run_one(dss_path, P_load_kw=None, Q_load_kvar=None, P_pv_kw=None):
    """Run daily vmin/vmax for a single DSS file. Uses inj.BASELINE if P/Q/PV not given."""
    if P_load_kw is None:
        P_load_kw = float(inj.BASELINE["P_load_total_kw"])
    if Q_load_kvar is None:
        Q_load_kvar = float(inj.BASELINE["Q_load_total_kvar"])
    if P_pv_kw is None:
        P_pv_kw = float(inj.BASELINE["P_pv_total_kw"])
    dss_path = os.path.abspath(dss_path)
    if not os.path.exists(dss_path):
        raise FileNotFoundError(f"DSS file not found: {dss_path}")
    # Temporarily override the DSS file used by compile_once()
    old_dss = inj.DSS_FILE
    # Old model (ieee34Mod1_with_loadshape) used PV840 and PV860; current default is IEEE34_PV (pv850/pv860)
    path_lower = dss_path.lower().replace("\\", "/")
    use_old_pv = "ieee34mod1_with_loadshape" in path_lower and "mirzaei" not in path_lower
    old_pv_share = old_pv_busph = None
    if use_old_pv:
        old_pv_share = inj.PV_PMMP_SHARE
        old_pv_busph = inj.PV_TO_BUSPH
        inj.PV_PMMP_SHARE = {"pv840": 0.5, "pv860": 0.5}
        inj.PV_TO_BUSPH = {
            "pv840": [("840", 1, 1/3), ("840", 2, 1/3), ("840", 3, 1/3)],
            "pv860": [("860", 1, 1/3), ("860", 2, 1/3), ("860", 3, 1/3)],
        }
    try:
        inj.DSS_FILE = dss_path
        vmin, vmax, node_vmin, node_vmax, t_vmin, phase_voltages_at_vmin, n_nonconv, control_iters_converged = daily_vmin_vmax(
            P_load_kw, Q_load_kvar, P_pv_kw, max_control_iter=5
        )
        return vmin, vmax, node_vmin, node_vmax, t_vmin, phase_voltages_at_vmin, n_nonconv, control_iters_converged
    finally:
        inj.DSS_FILE = old_dss
        if old_pv_share is not None:
            inj.PV_PMMP_SHARE = old_pv_share
        if old_pv_busph is not None:
            inj.PV_TO_BUSPH = old_pv_busph


def main():
    # Default: run for all .dss in "new dss from dr mirzaei" (skip line-code-only files if desired)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "new dss from dr mirzaei")
    if not os.path.isdir(folder):
        folder = script_dir
    dss_files = []
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            if os.path.isfile(a):
                dss_files.append(os.path.abspath(a))
            elif os.path.isdir(a):
                dss_files.extend([
                    os.path.join(a, f) for f in os.listdir(a)
                    if f.lower().endswith(".dss")
                ])
    else:
        dss_files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".dss") and "linecode" not in f.lower()
        ]
    dss_files = sorted(set(dss_files))
    P = float(inj.BASELINE["P_load_total_kw"])
    Q = float(inj.BASELINE["Q_load_total_kvar"])
    PV = float(inj.BASELINE["P_pv_total_kw"])
    print("Baseline: P_load={:.2f} kW, Q_load={:.2f} kVAR, P_pv={:.2f} kW".format(P, Q, PV))
    print("Daily profile: 288 steps (5 min), sigma=0. Excluded buses: sourcebus, 800.\n")
    for path in dss_files:
        name = os.path.basename(path)
        print("--- {} ---".format(name))
        try:
            vmin, vmax, node_vmin, node_vmax, t_vmin, phase_voltages_at_vmin, n_nonconv, control_iters_converged = run_one(
                path, P_load_kw=P, Q_load_kvar=Q, P_pv_kw=PV
            )
            n_conv = inj.NPTS - n_nonconv
            print("  daily profile: {}/{} converged, {} non-converged".format(n_conv, inj.NPTS, n_nonconv))
            _print_control_iters(control_iters_converged)
            print("  vmin = {:.6f} pu".format(vmin))
            if node_vmin:
                parts = node_vmin.split(".")
                bus_min = parts[0]
                print("        at bus {}, phase {}".format(bus_min, parts[1]))
                if t_vmin is not None:
                    # t is 5-min step: 0 = 00:00, 1 = 00:05, ... 288 not used (0..287)
                    min_from_midnight = t_vmin * 5
                    hour = min_from_midnight // 60
                    minute = min_from_midnight % 60
                    print("        time of day: {:02d}:{:02d}".format(hour, minute))
                if phase_voltages_at_vmin and bus_min:
                    print("        bus {} voltages at that time (pu):".format(bus_min))
                    for ph in sorted(phase_voltages_at_vmin.keys(), key=lambda x: (len(x), x)):
                        print("          phase {}: {:.6f}".format(ph, phase_voltages_at_vmin[ph]))
            print("  vmax = {:.6f} pu".format(vmax))
            if node_vmax:
                parts = node_vmax.split(".")
                print("        at bus {}, phase {}".format(parts[0], parts[1]))
            print("")
        except Exception as e:
            print("  Error: {}\n".format(e))


if __name__ == "__main__":
    main()
