"""
Draw daily voltage profile (magnitude pu vs time) for selected nodes using the
new dss from dr mirzaei (IEEE34_PV.dss, 5minDayShape.csv, 5MinuteIrradiance.csv).

Volt-Var can be disabled via a temporary DSS so all 288 steps converge and the
profile is smooth for the Volt-Var OFF case. Baseline totals are taken from
run_injection_dataset.BASELINE (P_load_total_kw / Q_load_total_kvar / P_pv_total_kw),
with sigma_load = sigma_pv = 0 for this diagnostic.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import opendssdirect as dss
import run_injection_dataset as inj

ORIG_DSS = os.path.join(os.path.dirname(os.path.abspath(inj.__file__)), "new dss from dr mirzaei", "IEEE34_PV.dss")
INVCONTROL_LINE_START = "New InvControl."  # comment out all InvControl lines for Volt-Var OFF

# Nodes to plot (bus.phase format). 890.1 = vmin bus; 802.3 = vmax; 848.3 = capacitor area
NODES_TO_PLOT = ["890.1", "802.3", "848.3"]
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_voltage_profile_new_dss.png")


def make_dss_no_voltvar():
    with open(ORIG_DSS, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.strip().startswith(INVCONTROL_LINE_START) and "InvControl" in line:
            out_lines.append("! " + line)
        else:
            out_lines.append(line)
    fd, path = tempfile.mkstemp(suffix=".dss", prefix="IEEE34_PV_no_voltvar_", dir=os.path.dirname(ORIG_DSS))
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return path


def run_daily_profile_collect_voltages(nodes_to_collect, use_voltvar=False):
    """
    Run 288 steps; return (time_hours, {node: vmag_array}).
    vmag_array has length 288; use np.nan for non-converged steps if use_voltvar=True.
    """
    temp_dss = None
    if not use_voltvar:
        temp_dss = make_dss_no_voltvar()
        old_dss = inj.DSS_FILE
        inj.DSS_FILE = temp_dss
    try:
        dss_path_abs = os.path.abspath(inj.DSS_FILE)
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path_abs}"')
        inj._apply_voltage_bases()
        inj.setup_daily()
        dss.Text.Command("set maxcontroliter=20000")

        node_names_master, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
        loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
        pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

        csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path_abs, "5minDayShape")
        csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path_abs, "IrradShape")
        csvL = inj.resolve_csv_path(csvL_token, dss_path_abs)
        csvPV = inj.resolve_csv_path(csvPV_token, dss_path_abs)
        mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
        mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

        P_load = inj.BASELINE["P_load_total_kw"]
        Q_load = inj.BASELINE["Q_load_total_kvar"]
        P_pv = inj.BASELINE["P_pv_total_kw"]
        sigma_load = 0.0
        sigma_pv = 0.0
        rng = np.random.default_rng(0)

        node_set = set(nodes_to_collect)
        node_to_idx = {n: i for i, n in enumerate(node_names_master) if n in node_set}
        if not node_to_idx:
            raise ValueError("None of the requested nodes found in circuit: " + str(nodes_to_collect))

        # vmag[node][t] = voltage at node at timestep t (nan if not converged)
        vmag = {n: np.full(inj.NPTS, np.nan, dtype=float) for n in nodes_to_collect if n in node_to_idx}
        # PV850 reactive power (kVAR) and voltage at 850 when Volt-Var enabled
        pv850_q_kvar = np.full(inj.NPTS, np.nan, dtype=float)
        v850_pu = np.full(inj.NPTS, np.nan, dtype=float)

        for t in range(inj.NPTS):
            inj.set_time_index(t)
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
                sigma_load=sigma_load,
                sigma_pv=sigma_pv,
                rng=rng,
            )
            try:
                dss.Solution.Solve()
            except Exception:
                pass
            if not dss.Solution.Converged():
                continue
            vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            for n in vmag:
                idx = node_names_master.index(n) if n in node_names_master else None
                if idx is not None:
                    vmag[n][t] = float(vmag_m[idx])
            # PV actual P/Q (Volt-Var Q for PV850) and voltage at PV bus 850
            if use_voltvar:
                busphP_pv, busphQ_pv = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
                q850 = sum(busphQ_pv.get(("850", ph), 0.0) for ph in (1, 2, 3))
                pv850_q_kvar[t] = q850
                if "850.1" in node_names_master:
                    idx850 = node_names_master.index("850.1")
                    v850_pu[t] = float(vmag_m[idx850])

        time_hours = np.arange(inj.NPTS, dtype=float) * inj.STEP_MIN / 60.0
        pv_series = {"Q_PV850_kvar": pv850_q_kvar, "V_850_pu": v850_pu} if use_voltvar else None
        return time_hours, vmag, pv_series
    finally:
        if not use_voltvar and temp_dss is not None:
            inj.DSS_FILE = old_dss
            try:
                os.remove(temp_dss)
            except Exception:
                pass


def main():
    if not os.path.isfile(ORIG_DSS):
        print("DSS not found:", ORIG_DSS)
        return
    baseline_str = "{:.2f}/{:.2f}/{:.2f}".format(
        float(inj.BASELINE["P_load_total_kw"]),
        float(inj.BASELINE["Q_load_total_kvar"]),
        float(inj.BASELINE["P_pv_total_kw"]),
    )
    print("Running 288-step daily profile with new DSS...")
    print("DSS: new dss from dr mirzaei/IEEE34_PV.dss")
    print("Profiles: 5minDayShape.csv, 5MinuteIrradiance.csv")
    print("Nodes:", NODES_TO_PLOT)

    # Volt-Var OFF (temp DSS without InvControl)
    time_hours_off, vmag_off, _ = run_daily_profile_collect_voltages(NODES_TO_PLOT, use_voltvar=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for n in NODES_TO_PLOT:
        if n not in vmag_off:
            continue
        v = vmag_off[n]
        valid = np.isfinite(v)
        if np.any(valid):
            ax.plot(time_hours_off[valid], v[valid], "-", label=f"{n} (no Volt-Var)", lw=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"Daily voltage profile (IEEE34_PV, baseline {baseline_str}, Volt-Var OFF)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(1.05, color="gray", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print("Saved (Volt-Var OFF):", OUTPUT_PATH)

    # Volt-Var ON (original DSS with InvControl): voltage at nodes + PV850 Q and voltage at 850
    time_hours_on, vmag_on, pv_series = run_daily_profile_collect_voltages(NODES_TO_PLOT, use_voltvar=True)
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for n in NODES_TO_PLOT:
        if n not in vmag_on:
            continue
        v = vmag_on[n]
        valid = np.isfinite(v)
        if np.any(valid):
            ax2a.plot(time_hours_on[valid], v[valid], "-", label=f"{n}", lw=1.5)
    if pv_series is not None and np.any(np.isfinite(pv_series["V_850_pu"])):
        ax2a.plot(time_hours_on, pv_series["V_850_pu"], "-", label="850.1 (PV bus)", lw=1.5)
    ax2a.set_ylabel("Voltage magnitude (pu)")
    ax2a.set_title(f"Daily voltage profile (IEEE34_PV, baseline {baseline_str}, Volt-Var ON)")
    ax2a.legend(loc="best")
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlim(0, 24)
    ax2a.axhline(0.95, color="gray", linestyle="--", alpha=0.7)
    ax2a.axhline(1.05, color="gray", linestyle="--", alpha=0.7)
    if pv_series is not None and np.any(np.isfinite(pv_series["Q_PV850_kvar"])):
        valid_q = np.isfinite(pv_series["Q_PV850_kvar"])
        ax2b.plot(time_hours_on[valid_q], pv_series["Q_PV850_kvar"][valid_q], "-", color="C3", lw=1.5, label="PV850 Q (kVAR)")
        ax2b.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2b.set_xlabel("Time (hours)")
    ax2b.set_ylabel("Reactive power (kVAR)")
    ax2b.set_title("PV850 reactive power injection (Volt-Var enabled)")
    ax2b.legend(loc="best")
    ax2b.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_on = os.path.join(os.path.dirname(OUTPUT_PATH), "daily_voltage_profile_new_dss_voltvar_on.png")
    fig2.savefig(out_on, dpi=150)
    plt.close(fig2)
    print("Saved (Volt-Var ON):", out_on)


if __name__ == "__main__":
    main()
