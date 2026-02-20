"""
Nominal vs Realized power comparison for 10 OpenDSS samples.
Compares set/nominal values with actual power-flow results.
Saves to nominal_vs_realized_comparison/ and displays summary.
"""
import os
import numpy as np
import pandas as pd
import opendssdirect as dss

# Import helpers from run_injection_dataset (run from GNN2 dir)
import run_injection_dataset as inj

def _normalize_name(s):
    return str(s).strip().lower()

LOAD_MODEL_NAMES = {1: "M1", 2: "M2", 3: "M3", 4: "M4", 5: "M5", -1: "?"}

def parse_bus_spec(bus_full):
    parts = bus_full.split(".")
    bus = parts[0]
    phs = []
    for p in parts[1:]:
        try:
            ip = int(p)
            if ip in (1, 2, 3):
                phs.append(ip)
        except Exception:
            pass
    return bus, sorted(list(set(phs))) if phs else [1, 2, 3]

def get_load_nominal():
    """Get nominal (set) P, Q for each Load from DSS (read after apply)."""
    out = {}
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        try:
            p = float(dss.Loads.kW())
            q = float(dss.Loads.kvar())
        except Exception:
            p, q = 0.0, 0.0
        out[_normalize_name(name)] = (p, q)
        if not dss.Loads.Next():
            break
    return out

def get_load_realized_and_model():
    """Get realized P, Q and model for each Load. Returns dict: load_name -> (P_kw, Q_kvar, model)."""
    out = {}
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        pwr = dss.CktElement.TotalPowers()
        if len(pwr) >= 2:
            P = float(pwr[0])
            Q = float(pwr[1])
        else:
            P, Q = 0.0, 0.0
        try:
            model = int(dss.Loads.Model())
        except Exception:
            model = -1
        out[_normalize_name(name)] = (P, Q, model)
        if not dss.Loads.Next():
            break
    return out

def get_pv_realized():
    """Get realized P, Q for each PVSystem. OpenDSS returns negative for generation; flip to injection convention (+ = into bus)."""
    out = {}
    dss.PVsystems.First()
    while True:
        name = dss.PVsystems.Name()
        dss.Circuit.SetActiveElement(f"PVSystem.{name}")
        pwr = dss.CktElement.TotalPowers()
        if len(pwr) >= 2:
            P = -float(pwr[0])  # negate: OpenDSS negative = generation = injection
            Q = -float(pwr[1])
        else:
            P, Q = 0.0, 0.0
        out[_normalize_name(name)] = (P, Q)
        if not dss.PVsystems.Next():
            break
    return out

def get_capacitor_realized():
    """Get realized Q for each Capacitor. OpenDSS returns negative for injection; flip to injection convention (+ = into bus)."""
    out = {}
    dss.Capacitors.First()
    while True:
        name = dss.Capacitors.Name()
        dss.Circuit.SetActiveElement(f"Capacitor.{name}")
        pwr = dss.CktElement.TotalPowers()
        if len(pwr) >= 2:
            P = float(pwr[0])
            Q = -float(pwr[1])  # negate: OpenDSS negative = injection
        else:
            P, Q = 0.0, 0.0
        bus_full = dss.CktElement.BusNames()[0] if dss.CktElement.BusNames() else ""
        bus, phs = parse_bus_spec(bus_full)
        out[name] = (P, Q, bus, phs)
        if not dss.Capacitors.Next():
            break
    return out

def get_grid_realized():
    """Get realized P, Q from upstream grid (TotalPower)."""
    pwr = dss.Circuit.TotalPower()
    P = -float(pwr[0])
    Q = -float(pwr[1])
    return P, Q

def get_circuit_losses():
    """Get total realized losses (P, Q) from OpenDSS Circuit.Losses. Returns (P_kw, Q_kvar)."""
    loss = dss.Circuit.Losses()
    P = float(loss[0])
    Q = float(loss[1])
    # OpenDSS returns Watts/VAR; convert to kW/kVAR for consistency with other power values
    if abs(P) > 1000 or abs(Q) > 1000:
        P, Q = P / 1000.0, Q / 1000.0
    return P, Q

def aggregate_load_to_busph(load_realized, dev_to_busph):
    """Map per-load realized to (bus, ph) using DEVICE_TO_BUSPH weights."""
    busphP = {}
    busphQ = {}
    for dev_key_raw, (P, Q, _) in load_realized.items():
        if dev_key_raw not in dev_to_busph:
            continue
        for (bus, ph, w) in dev_to_busph[dev_key_raw]:
            busphP[(bus, ph)] = busphP.get((bus, ph), 0.0) + P * w
            busphQ[(bus, ph)] = busphQ.get((bus, ph), 0.0) + Q * w
    return busphP, busphQ

def aggregate_cap_to_busph(cap_realized):
    """Map capacitor realized Q to (bus, ph)."""
    busphQ = {}
    for capname, (_, Q, bus, phs) in cap_realized.items():
        nph = len(phs) if phs else 3
        q_per_ph = Q / nph if nph > 0 else 0
        for ph in (phs if phs else [1, 2, 3]):
            busphQ[(bus, ph)] = busphQ.get((bus, ph), 0.0) + q_per_ph
    return busphQ

def run_comparison(n_samples=10, seed=42, save_csv=True):
    out_dir = os.path.abspath("nominal_vs_realized_comparison")
    os.makedirs(out_dir, exist_ok=True)

    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps({})
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV)

    rng_master = np.random.default_rng(seed)
    rng_times = np.random.default_rng(seed + 1)
    prof_load = inj.BASELINE["P_load_total_kw"] * np.array(mL)
    prof_pv = inj.BASELINE["P_pv_total_kw"] * np.array(mPV)
    prof_net = prof_load - inj.BASELINE["P_pv_total_kw"] * np.array(mPV)
    times = inj.select_times_three_profiles(
        prof_load, prof_pv, prof_net, K_total=min(n_samples * 3, 288),
        bins_by_profile={"load": 5, "pv": 5, "net": 5}, include_anchors=True, rng=rng_times
    )
    times = times[:n_samples]

    rows_load = []
    rows_node_load = []
    rows_node_pv = []
    rows_node_cap = []
    rows_grid = []
    rows_sum_check = []

    for idx, t in enumerate(times):
        inj.set_time_index(t)
        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load, Q_load, P_pv = sc["P_load_total_kw"], sc["Q_load_total_kvar"], sc["P_pv_total_kw"]
        sigL, sigPV = sc["sigma_load"], sc["sigma_pv"]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load, Q_load_total_kvar=Q_load, P_pv_total_kw=P_pv,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=sigL, sigma_pv=sigPV, rng=rng_solve
        )

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(f"[sample {idx}] Did not converge, skipping")
            continue

        # --- Nominal (set) values read back from DSS ---
        load_nominal = get_load_nominal()

        # --- Realized values ---
        load_realized = get_load_realized_and_model()
        pv_realized = get_pv_realized()
        cap_realized = get_capacitor_realized()
        P_grid_real, Q_grid_real = get_grid_realized()
        P_loss_kw, Q_loss_kvar = get_circuit_losses()

        busphP_load_real, busphQ_load_real = aggregate_load_to_busph(load_realized, dev_to_busph_load)
        busphQ_cap_real = aggregate_cap_to_busph(cap_realized)

        # --- Per-load table (nominal = set values from DSS, realized = power-flow result) ---
        for dev_key_raw in inj.DEVICE_P_SHARE.keys():
            dev_key = _normalize_name(dev_key_raw)
            if dev_key not in dev_to_dss_load or dev_key not in load_realized:
                continue
            P_real, Q_real, model = load_realized[dev_key]
            p_nom, q_nom = load_nominal.get(dev_key, (0.0, 0.0))
            load_type = LOAD_MODEL_NAMES.get(model, f"M{model}")
            rows_load.append({
                "sample_id": idx, "load_name": dev_to_dss_load[dev_key],
                "load_model": model, "load_type": load_type,
                "p_nominal_kw": p_nom, "q_nominal_kvar": q_nom,
                "p_realized_kw": P_real, "q_realized_kvar": Q_real,
            })

        # --- Per-node loads (aggregated nominal vs realized) ---
        for (bus, ph) in set(busphP_load.keys()) | set(busphP_load_real.keys()):
            p_nom = busphP_load.get((bus, ph), 0)
            q_nom = busphQ_load.get((bus, ph), 0)
            p_real = busphP_load_real.get((bus, ph), 0)
            q_real = busphQ_load_real.get((bus, ph), 0)
            rows_node_load.append({
                "sample_id": idx, "node": f"{bus}.{ph}",
                "p_nominal_kw": p_nom, "q_nominal_kvar": q_nom,
                "p_realized_kw": p_real, "q_realized_kvar": q_real,
            })

        # --- Per-node PV ---
        for (bus, ph) in set(busphP_pv.keys()):
            p_nom = busphP_pv.get((bus, ph), 0)
            q_nom = busphQ_pv.get((bus, ph), 0)
            p_real = 0.0
            q_real = 0.0
            for pv_key, lst in pv_to_busph.items():
                for (b, ph2, w) in lst:
                    if str(b) == bus and int(ph2) == ph:
                        pvname_dss = pv_to_dss.get(pv_key)
                        pv_key_norm = _normalize_name(pvname_dss) if pvname_dss else None
                        if pv_key_norm and pv_key_norm in pv_realized:
                            pr, qr = pv_realized[pv_key_norm]
                            p_real += pr * w
                            q_real += qr * w
            rows_node_pv.append({
                "sample_id": idx, "node": f"{bus}.{ph}",
                "p_nominal_kw": p_nom, "q_nominal_kvar": q_nom,
                "p_realized_kw": p_real, "q_realized_kvar": q_real,
            })

        # --- Per-node capacitors (nominal = rated Q per phase; CAP_Q_KVAR is already per-phase) ---
        for bus, q_per_ph in inj.CAP_Q_KVAR.items():
            for ph in [1, 2, 3]:
                q_nom = q_per_ph  # CAP_Q_KVAR is per-phase (kVAR)
                q_real = busphQ_cap_real.get((bus, ph), 0)
                rows_node_cap.append({
                    "sample_id": idx, "node": f"{bus}.{ph}",
                    "q_nominal_kvar": q_nom, "q_realized_kvar": q_real,
                })

        # --- Grid: use realized from TotalPower (same as dataset generation) ---
        rows_grid.append({
            "sample_id": idx,
            "p_grid_kw": P_grid_real, "q_grid_kvar": Q_grid_real,
        })

        # --- Sum check: nominal (balance = 0 by construction) vs realized (≈ losses) ---
        sum_p_load_real = sum(busphP_load_real.values())
        sum_q_load_real = sum(busphQ_load_real.values())
        sum_p_load_nom = sum(busphP_load.values())
        sum_q_load_nom = sum(busphQ_load.values())
        sum_p_pv_real = sum(pv_realized[n][0] for n in pv_realized)
        sum_q_pv_real = sum(pv_realized[n][1] for n in pv_realized)
        sum_p_pv_nom = sum(busphP_pv.values())
        sum_q_pv_nom = sum(busphQ_pv.values())
        sum_q_cap_real = sum(busphQ_cap_real.values())
        sum_q_cap_nom = sum(v * 3 for v in inj.CAP_Q_KVAR.values())
        sum_p_real = P_grid_real + sum_p_pv_real - sum_p_load_real
        sum_q_real = Q_grid_real + sum_q_pv_real + sum_q_cap_real - sum_q_load_real
        P_grid_nom_bal = sum_p_load_nom - sum_p_pv_nom  # nominal grid to balance load - pv
        Q_grid_nom_bal = sum_q_load_nom - sum_q_pv_nom - sum_q_cap_nom
        sum_p_nom = P_grid_nom_bal + sum_p_pv_nom - sum_p_load_nom  # = 0 by construction
        sum_q_nom = Q_grid_nom_bal + sum_q_pv_nom + sum_q_cap_nom - sum_q_load_nom  # = 0
        # Balance: sum_p_real = P_loss, sum_q_real = Q_loss (supply - demand = losses)
        # Residual = our sum minus OpenDSS losses; should be ~0 if consistent
        res_p = sum_p_real - P_loss_kw
        res_q = sum_q_real - Q_loss_kvar
        rows_sum_check.append({
            "sample_id": idx,
            "sum_p_nominal_kw": sum_p_nom, "sum_q_nominal_kvar": sum_q_nom,
            "sum_p_realized_kw": sum_p_real, "sum_q_realized_kvar": sum_q_real,
            "P_loss_kw": P_loss_kw, "Q_loss_kvar": Q_loss_kvar,
            "res_p_kw": res_p, "res_q_kvar": res_q,
            "P_grid_nom": P_grid_nom_bal, "P_pv_nom": sum_p_pv_nom, "P_load_nom": sum_p_load_nom,
            "Q_grid_nom": Q_grid_nom_bal, "Q_pv_nom": sum_q_pv_nom, "Q_load_nom": sum_q_load_nom, "Q_cap_nom": sum_q_cap_nom,
            "P_grid": P_grid_real, "P_pv": sum_p_pv_real, "P_load": sum_p_load_real,
            "Q_grid": Q_grid_real, "Q_pv": sum_q_pv_real, "Q_load": sum_q_load_real, "Q_cap": sum_q_cap_real,
        })

    # --- Save CSVs ---
    if save_csv:
        if rows_load:
            pd.DataFrame(rows_load).to_csv(os.path.join(out_dir, "per_load_nominal_vs_realized.csv"), index=False)
        if rows_node_load:
            pd.DataFrame(rows_node_load).to_csv(os.path.join(out_dir, "per_node_load_nominal_vs_realized.csv"), index=False)
        if rows_node_pv:
            pd.DataFrame(rows_node_pv).to_csv(os.path.join(out_dir, "per_node_pv_nominal_vs_realized.csv"), index=False)
        if rows_node_cap:
            pd.DataFrame(rows_node_cap).to_csv(os.path.join(out_dir, "per_node_cap_nominal_vs_realized.csv"), index=False)
        if rows_grid:
            pd.DataFrame(rows_grid).to_csv(os.path.join(out_dir, "grid_pq.csv"), index=False)
        if rows_sum_check:
            path = os.path.join(out_dir, "sum_check_realized.csv")
            try:
                pd.DataFrame(rows_sum_check).to_csv(path, index=False)
                print(f"[saved] {path}")
            except PermissionError:
                alt = os.path.join(out_dir, "sum_check_realized_new.csv")
                pd.DataFrame(rows_sum_check).to_csv(alt, index=False)
                print(f"[saved] {path} (original locked, wrote {alt})")

    return rows_load, rows_node_load, rows_node_pv, rows_node_cap, rows_grid, rows_sum_check

if __name__ == "__main__":
    rows_load, rows_node_load, rows_node_pv, rows_node_cap, rows_grid, rows_sum = run_comparison(n_samples=10)
    print("\n=== Saved to nominal_vs_realized_comparison/ ===")
    if rows_sum:
        df = pd.DataFrame(rows_sum)
        print("\n--- Sum check (nominal=0 by construction; realized = losses) ---")
        cols = ["sample_id", "sum_p_realized_kw", "P_loss_kw", "res_p_kw", "sum_q_realized_kvar", "Q_loss_kvar", "res_q_kvar"]
        print(df[[c for c in cols if c in df.columns]].to_string())
        print("\nBalance: sum_p_real = P_grid + P_pv - P_load (= P_loss); sum_q_real = Q_grid + Q_pv + Q_cap - Q_load (= Q_loss)")
        print("res_p = sum_p_real - P_loss, res_q = sum_q_real - Q_loss (should be ~0)")
        if "res_p_kw" in df.columns:
            print(f"\nMean |res_p|: {df['res_p_kw'].abs().mean():.6f} kW")
            print(f"Mean |res_q|: {df['res_q_kvar'].abs().mean():.6f} kVAR")
