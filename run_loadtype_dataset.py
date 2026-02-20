"""
Third dataset generation: per-node load-type features (M1/M2/M4/M5 P&Q, cap Q, PV P, grid P/Q).
Standalone script — same scenario/snapshot flow as run_injection_dataset.py, but self-contained.
Output: gnn_samples_loadtype_full/
"""
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj

# Load model per device (OpenDSS Model 1-5: 1=ConstPQ, 2=ConstZ, 4=CVR, 5=ConstI)
DEVICE_TO_MODEL = {
    "S860": 1, "S848": 1, "D802_806sb": 1, "D802_806rb": 1, "D802_806sc": 1, "D802_806rc": 1,
    "D820_822sa": 1, "D820_822ra": 1, "D828_830ra": 1, "D828_830sa": 1, "D854_856sb": 1, "D854_856rb": 1,
    "D858_864sb": 1, "D858_864rb": 1, "D858_834sa": 1, "D858_834ra": 1, "D858_834sb": 1, "D858_834rb": 1,
    "D858_834sc": 1, "D858_834rc": 1, "D860_836sa": 1, "D860_836ra": 1, "D860_836sb": 1, "D860_836rb": 1,
    "D860_836sc": 1, "D860_836rc": 1, "D862_838sb": 1, "D862_838rb": 1, "D842_844sa": 1, "D842_844ra": 1,
    "D844_846sb": 1, "D844_846rb": 1, "D844_846sc": 1, "D844_846rc": 1, "D846_848sb": 1, "D846_848rb": 1,
    "D824_828sc": 1, "D824_828rc": 1,
    "S844": 2, "S830a": 2, "S830b": 2, "S830c": 2, "D818_820ra": 2, "D818_820sa": 2,
    "D834_860ra": 2, "D834_860rb": 2, "D834_860rc": 2, "D834_860sa": 2, "D834_860sb": 2, "D834_860sc": 2,
    "D858_834ra": 2, "D858_834rb": 2, "D858_834rc": 2, "D832_858ra": 2, "D832_858rb": 2, "D832_858rc": 2,
    "D832_858sa": 2, "D832_858sb": 2, "D832_858sc": 2,
    "D808_810rb": 4, "D808_810sb": 4,
    "S890": 5, "S840": 5, "D816_824sb": 5, "D816_824rb": 5, "D824_826sb": 5, "D824_826rb": 5,
    "D836_840sa": 5, "D836_840ra": 5, "D836_840sb": 5, "D836_840rb": 5,
}


def _apply_snapshot_with_per_type(
    P_load_total_kw, Q_load_total_kvar, P_pv_total_kw,
    mL_t, mPV_t,
    loads_dss, dev_to_dss_load, dev_to_busph_load,
    pv_dss, pv_to_dss, pv_to_busph,
    sigma_load, sigma_pv,
    rng,
):
    """Same as inj.apply_snapshot_timeconditioned but also returns busph_per_type (M1/M2/M4/M5 P&Q)."""
    P_load_t = float(P_load_total_kw) * float(mL_t)
    Q_load_t = float(Q_load_total_kvar) * float(mL_t)
    P_pv_t = float(P_pv_total_kw) * float(mPV_t)

    busphP_load = {}
    busphQ_load = {}
    busph_per_type = {m: (dict(), dict()) for m in (1, 2, 4, 5)}

    for dev_key_raw in inj.DEVICE_P_SHARE.keys():
        dev_key = inj._normalize_name(dev_key_raw)
        if dev_key not in dev_to_dss_load or dev_key not in dev_to_busph_load:
            continue
        p0 = P_load_t * float(inj.DEVICE_P_SHARE.get(dev_key_raw, 0.0))
        q0 = Q_load_t * float(inj.DEVICE_Q_SHARE.get(dev_key_raw, 0.0))
        fp = inj._noise_factor(rng, sigma_load)
        fq = inj._noise_factor(rng, sigma_load)
        p_set = float(p0 * fp)
        q_set = float(q0 * fq)
        ln = dev_to_dss_load[dev_key]
        inj.dss.Loads.Name(ln)
        inj.dss.Loads.kW(p_set)
        inj.dss.Loads.kvar(q_set)
        model = DEVICE_TO_MODEL.get(dev_key_raw, 1)
        for (bus, ph, w) in dev_to_busph_load[dev_key]:
            busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + p_set * w
            busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + q_set * w
            if model in busph_per_type:
                bp, bq = busph_per_type[model]
                bp[(bus, ph)] = bp.get((bus, ph), 0.0) + p_set * w
                bq[(bus, ph)] = bq.get((bus, ph), 0.0) + q_set * w

    busphP_pv = {}
    busphQ_pv = {}
    for pv_key_raw in inj.PV_PMMP_SHARE.keys():
        pv_key = inj._normalize_name(pv_key_raw)
        if pv_key not in pv_to_dss or pv_key not in pv_to_busph:
            continue
        pmpp0 = P_pv_total_kw * float(inj.PV_PMMP_SHARE.get(pv_key_raw, 0.0))
        f = inj._noise_factor(rng, sigma_pv)
        pmpp_set = float(pmpp0 * f)
        pvname = pv_to_dss[pv_key]
        inj.dss.PVsystems.Name(pvname)
        inj.dss.PVsystems.Pmpp(pmpp_set)
        p_nominal = pmpp_set * float(mPV_t)
        for (bus, ph, w) in pv_to_busph[pv_key]:
            busphP_pv[(bus, ph)] = busphP_pv.get((bus, ph), 0.0) + p_nominal * w
            busphQ_pv[(bus, ph)] = busphQ_pv.get((bus, ph), 0.0) + 0.0

    totals = dict(
        P_load_time_kw=P_load_t,
        Q_load_time_kvar=Q_load_t,
        P_pv_time_kw=P_pv_t,
        p_load_kw_set_total=float(sum(busphP_load.values())),
        q_load_kvar_set_total=float(sum(busphQ_load.values())),
        p_pv_pmpp_kw_set_total=float(sum(busphP_pv.values())),
    )
    return totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type


# Reuse config and mappings from injection dataset
OUT_DIR = "gnn_samples_loadtype_full"
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")

# Source bus for grid P/Q (nominal balance)
# IEEE 34: grid feeds sourcebus; transformer secondary is 800. Use 800 if sourcebus not in node list.
SOURCE_BUSES = ("sourcebus", "800")


def _compute_electrical_distance_from_source(node_names_master, edge_csv_path):
    """
    Compute electrical distance from each node to the source bus.
    Uses path impedance magnitude |Z| = sqrt(R² + X²) summed along the minimum-impedance path.
    Best single metric: captures both resistance and reactance; voltage drop ∝ |I|·|Z|.
    Returns dict: node -> float (ohms).
    """
    import heapq

    df_e = pd.read_csv(edge_csv_path)
    # Build weighted adjacency: (neighbor, edge_weight) where weight = |Z| = sqrt(R² + X²)
    adj = {}
    for _, row in df_e.iterrows():
        a, b = str(row["from_node"]), str(row["to_node"])
        r = float(row.get("R_full", 0))
        x = float(row.get("X_full", 0))
        z_mag = np.sqrt(r * r + x * x)
        adj.setdefault(a, []).append((b, z_mag))

    source_nodes = [n for n in node_names_master if n.split(".")[0] in SOURCE_BUSES]
    if not source_nodes:
        source_nodes = [node_names_master[0]] if node_names_master else []

    dist = {n: float("inf") for n in node_names_master}
    for src in source_nodes:
        if src in node_names_master:
            dist[src] = 0.0

    # Dijkstra: min-impedance path from all source nodes
    heap = [(0.0, src) for src in source_nodes if src in node_names_master]
    heapq.heapify(heap)
    seen = set(source_nodes)

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            d_v = d_u + w
            if d_v < dist.get(v, float("inf")):
                dist[v] = d_v
                heapq.heappush(heap, (d_v, v))
                seen.add(v)

    return {n: float(dist[n]) if dist[n] != float("inf") else 0.0 for n in node_names_master}


def generate_gnn_snapshot_dataset_loadtype(
    n_scenarios=200,
    k_snapshots_per_scenario_total=960,
    bins_by_profile=None,
    include_anchors=True,
    master_seed=20260130,
    loadshape_name="5minDayShape",
    irradshape_name="IrradShape",
):
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

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
        P_load, Q_load, P_pv = sc["P_load_total_kw"], sc["Q_load_total_kvar"], sc["P_pv_total_kw"]
        sigL, sigPV = sc["sigma_load"], sc["sigma_pv"]
        prof_load, prof_pv = mL, mPV
        prof_net = (P_load * mL) - (P_pv * mPV)

        rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        times = inj.select_times_three_profiles(
            prof_load=prof_load, prof_pv=prof_pv, prof_net=prof_net,
            K_total=k_snapshots_per_scenario_total, bins_by_profile=bins_by_profile,
            include_anchors=include_anchors, rng=rng_times
        )
        times = [int(t) for t in times]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        for t in times:
            inj.set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = _apply_snapshot_with_per_type(
                P_load_total_kw=P_load, Q_load_total_kvar=Q_load, P_pv_total_kw=P_pv,
                mL_t=float(mL[t]), mPV_t=float(mPV[t]),
                loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
                sigma_load=sigL, sigma_pv=sigPV, rng=rng_solve,
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

            vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

            # System-wide balance (same for all nodes; pre-solve estimate of grid injection)
            sum_p_load = float(sum(busphP_load.values()))
            sum_q_load = float(sum(busphQ_load.values()))
            sum_p_pv = float(sum(busphP_pv.values()))
            sum_q_cap = sum(inj.CAP_Q_KVAR.values())
            p_sys_balance = sum_p_load - sum_p_pv
            q_sys_balance = sum_q_load - sum_q_cap  # PV Q ≈ 0

            rows_sample.append({
                "sample_id": sample_id, "scenario_id": s, "t_index": t, "t_minutes": t * inj.STEP_MIN,
                "P_load_total_kw": float(P_load), "Q_load_total_kvar": float(Q_load), "P_pv_total_kw": float(P_pv),
                "sigma_load": float(sigL), "sigma_pv": float(sigPV),
                "m_loadshape": float(mL[t]), "m_irradshape": float(mPV[t]),
                "p_sys_balance_kw": p_sys_balance, "q_sys_balance_kvar": q_sys_balance,
            })

            for n in node_names_master:
                bus, phs = n.split(".")
                ph = int(phs)

                # Load by type (8 features)
                m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
                m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
                m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
                m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
                m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
                m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
                m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
                m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))

                # Capacitor Q (nominal per phase at this bus)
                # CAP_Q_KVAR is already per-phase (100 for 844, 150 for 848); no extra division
                q_cap_node = float(inj.CAP_Q_KVAR.get(bus, 0.0))

                # PV P (nominal)
                p_pv_node = float(busphP_pv.get((bus, ph), 0.0))

                # System balance P, Q (same for all nodes — global context of grid injection)
                p_sys_node = p_sys_balance
                q_sys_node = q_sys_balance

                vm, va = vdict_m.get(n, (np.nan, np.nan))
                elec_dist = float(node_to_electrical_dist.get(n, 0.0))

                rows_node.append({
                    "sample_id": sample_id, "node": n, "node_idx": int(node_to_idx_master[n]),
                    "bus": bus, "phase": int(ph),
                    "electrical_distance_ohm": elec_dist,
                    "m1_p_kw": m1_p, "m1_q_kvar": m1_q,
                    "m2_p_kw": m2_p, "m2_q_kvar": m2_q,
                    "m4_p_kw": m4_p, "m4_q_kvar": m4_q,
                    "m5_p_kw": m5_p, "m5_q_kvar": m5_q,
                    "q_cap_kvar": q_cap_node,
                    "p_pv_kw": p_pv_node,
                    "p_sys_balance_kw": p_sys_node, "q_sys_balance_kvar": q_sys_node,
                    "vmag_pu": float(vm), "vang_deg": float(va),
                })

            sample_id += 1
            kept += 1

        print(f"[scenario {s+1}/{n_scenarios}] kept={kept} skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}")

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[LOADTYPE DATASET] Saved to {OUT_DIR}/")
    print(f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  Features per node: electrical_distance_ohm, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv, p_sys_balance, q_sys_balance")
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_gnn_snapshot_dataset_loadtype(
        n_scenarios=200,
        k_snapshots_per_scenario_total=960,
        bins_by_profile={"load": 10, "pv": 10, "net": 10},
        include_anchors=True,
        master_seed=20260130,
    )
