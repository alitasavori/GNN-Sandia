"""
Third dataset generation: per-node load-type features (M1/M2/M4/M5 P&Q, cap Q, PV P, grid P/Q).
Standalone script — same scenario/snapshot flow as run_injection_dataset.py, but self-contained.
Output: datasets_gnn2/loadtype/
"""
import importlib
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj
inj = importlib.reload(inj)

_REPO_ROOT = os.path.dirname(os.path.abspath(inj.__file__))
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

# Buses whose node-level rows are excluded from training/evaluation datasets
EXCLUDED_UPSTREAM_BUSES = ("sourcebus", "800")

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


# Unified dataset directory: datasets_gnn2/loadtype (absolute so cwd changes don't break saves)
OUT_DIR = os.path.join(_REPO_ROOT, "datasets_gnn2", "loadtype")
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")

# Source bus for grid P/Q (nominal balance)
# IEEE 34: grid feeds sourcebus; transformer secondary is 800. Use 800 if sourcebus not in node list.
SOURCE_BUSES = ("sourcebus", "800")


def _ensure_active_circuit():
    try:
        _ = list(inj.dss.Circuit.AllNodeNames())
        return
    except Exception:
        pass
    inj.compile_once()
    inj.setup_daily()


def _select_shared_phases(phs1, phs2, nph):
    if len(phs1) == 0 or len(phs2) == 0:
        phs = list(range(1, min(3, nph) + 1))
    else:
        phs = sorted(list(set(phs1).intersection(set(phs2))))
        if len(phs) == 0:
            phs = sorted(list(set(phs1).union(set(phs2))))
    phs = [ph for ph in phs if ph in (1, 2, 3)]
    if len(phs) == 0:
        phs = list(range(1, min(3, nph) + 1))
    return phs


def _infer_reduced_graph_roots(node_names_master):
    """Infer kept-node roots for the reduced graph after upstream buses are excluded.

    The static edge CSV intentionally omits edges incident to `SOURCE_BUSES`, so the
    distance graph has no explicit source nodes. Here we recover the first downstream
    kept nodes directly from the DSS circuit and use them as Dijkstra seeds.
    """
    _ensure_active_circuit()
    node_set = set(node_names_master)
    roots = {}

    def _register_boundary_edge(bus_a, phs_a, bus_b, phs_b, nph, r_full, x_full):
        z_mag = float(np.sqrt(float(r_full) * float(r_full) + float(x_full) * float(x_full)))
        if bus_a in SOURCE_BUSES and bus_b not in SOURCE_BUSES:
            for ph in _select_shared_phases(phs_a, phs_b, nph):
                node = f"{bus_b}.{ph}"
                if node in node_set:
                    roots[node] = min(float(roots.get(node, float("inf"))), z_mag)
        if bus_b in SOURCE_BUSES and bus_a not in SOURCE_BUSES:
            for ph in _select_shared_phases(phs_a, phs_b, nph):
                node = f"{bus_a}.{ph}"
                if node in node_set:
                    roots[node] = min(float(roots.get(node, float("inf"))), z_mag)

    inj.dss.Lines.First()
    while True:
        busnames = inj.dss.CktElement.BusNames()
        if len(busnames) >= 2:
            b1, phs1 = inj.parse_bus_spec(busnames[0])
            b2, phs2 = inj.parse_bus_spec(busnames[1])
            length = float(inj.dss.Lines.Length())
            nph_line = int(inj.dss.Lines.Phases())
            linecode = str(inj.dss.Lines.LineCode()).strip()
            use_linecode = linecode != ""
            if use_linecode:
                try:
                    inj.dss.LineCodes.Name(linecode)
                    Rm = inj.dss.LineCodes.Rmatrix()
                    Xm = inj.dss.LineCodes.Xmatrix()
                    use_linecode = len(Rm) > 0 and len(Xm) > 0
                except Exception:
                    use_linecode = False
            if use_linecode:
                Rraw = inj.list_to_sq(Rm)
                Xraw = inj.list_to_sq(Xm)
                kmat = Rraw.shape[0]
            else:
                r1 = float(inj.dss.Lines.R1())
                x1 = float(inj.dss.Lines.X1())
                Rraw = np.diag([r1, r1, r1])
                Xraw = np.diag([x1, x1, x1])
                kmat = 3
            phs = _select_shared_phases(phs1, phs2, nph_line)
            for ph in phs:
                pos_local = (ph - 1) if kmat >= 3 else phs.index(ph)
                r_full = float(Rraw[pos_local, pos_local]) * length
                x_full = float(Xraw[pos_local, pos_local]) * length
                _register_boundary_edge(b1, phs1, b2, phs2, nph_line, r_full, x_full)
        if not inj.dss.Lines.Next():
            break

    inj.dss.Transformers.First()
    while True:
        busnames = inj.dss.CktElement.BusNames()
        if len(busnames) >= 2:
            b1, phs1 = inj.parse_bus_spec(busnames[0])
            b2, phs2 = inj.parse_bus_spec(busnames[1])
            nph = int(inj.dss.CktElement.NumPhases())
            xhl = float(inj.dss.Transformers.Xhl())
            inj.dss.Transformers.Wdg(1)
            kv1 = float(inj.dss.Transformers.kV())
            kva1 = float(inj.dss.Transformers.kVA())
            r1_pct = float(inj.dss.Transformers.R())
            r2_pct = 0.0
            if inj.dss.Transformers.NumWindings() >= 2:
                inj.dss.Transformers.Wdg(2)
                r2_pct = float(inj.dss.Transformers.R())
            if kv1 > 0 and kva1 > 0:
                z_base = (kv1 ** 2) / kva1
                r_full = (r1_pct + r2_pct) / 100.0 * z_base
                x_full = xhl / 100.0 * z_base
                _register_boundary_edge(b1, phs1, b2, phs2, nph, r_full, x_full)
        if not inj.dss.Transformers.Next():
            break

    return roots


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

    source_nodes = [n for n in node_names_master if n.split(".")[0] in SOURCE_BUSES and n in adj]
    root_nodes = {}
    if source_nodes:
        root_nodes = {src: 0.0 for src in source_nodes}
    else:
        root_nodes = _infer_reduced_graph_roots(node_names_master)
    if not root_nodes and node_names_master:
        root_nodes = {node_names_master[0]: 0.0}

    dist = {n: float("inf") for n in node_names_master}
    for src, d0 in root_nodes.items():
        if src in node_names_master:
            dist[src] = float(d0)

    # Dijkstra: min-impedance path from all source nodes
    heap = [(float(d0), src) for src, d0 in root_nodes.items() if src in node_names_master]
    heapq.heapify(heap)
    seen = set(root_nodes)

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

    os.makedirs(OUT_DIR, exist_ok=True)
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
        control_iters_converged_this_scenario: list[float] = []

        for t in times:
            inj.set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = _apply_snapshot_with_per_type(
                P_load_total_kw=P_load, Q_load_total_kvar=Q_load, P_pv_total_kw=P_pv,
                mL_t=float(mL[t]), mPV_t=float(mPV[t]),
                loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
                sigma_load=sigL, sigma_pv=sigPV, rng=rng_solve,
            )

            try:
                inj.dss.Solution.Solve()
            except Exception:
                pass  # e.g. #485 Max Control Iterations Exceeded; solution may still be valid
            if not inj.dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            try:
                val = getattr(inj.dss.Solution, "ControlIterations", None)
                n_ctrl = val() if callable(val) else val
                if n_ctrl is not None:
                    control_iters_converged_this_scenario.append(float(n_ctrl))
            except Exception:
                pass

            busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(
                pv_to_dss, pv_to_busph
            )
            vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            vmag_arr = np.asarray(vmag_m, dtype=float)
            if (not np.isfinite(vmag_arr).all()) or (vmag_arr.min() < inj.VMAG_PU_MIN) or (vmag_arr.max() > inj.VMAG_PU_MAX):
                skipped_badV += 1
                continue

            vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

            # System-wide balance (same for all nodes; actual P/Q from PV after solve)
            sum_p_load = float(sum(busphP_load.values()))
            sum_q_load = float(sum(busphQ_load.values()))
            sum_p_pv = float(sum(busphP_pv_actual.values()))
            sum_q_pv = float(sum(busphQ_pv_actual.values()))
            kept_nodes = [n for n in node_names_master if n.split(".")[0] not in EXCLUDED_UPSTREAM_BUSES]
            sum_q_cap = inj.total_cap_q_kvar(kept_nodes)
            p_sys_balance = sum_p_load - sum_p_pv
            q_sys_balance = sum_q_load + sum_q_pv - sum_q_cap

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
                if bus in EXCLUDED_UPSTREAM_BUSES:
                    continue

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
                q_cap_node = inj.cap_q_kvar_per_node(bus, ph)

                # PV P and Q (actual after solve, includes Volt-Var Q)
                p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
                q_pv_node = float(busphQ_pv_actual.get((bus, ph), 0.0))

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
                    "q_pv_kvar": q_pv_node,
                    "p_sys_balance_kw": p_sys_node, "q_sys_balance_kvar": q_sys_node,
                    "vmag_pu": float(vm), "vang_deg": float(va),
                })

            sample_id += 1
            kept += 1

        ctrl_summary = ""
        if control_iters_converged_this_scenario:
            arr = np.array(control_iters_converged_this_scenario, dtype=float)
            ctrl_summary = (
                f" ctrl_iter: n={len(arr)} min={int(arr.min())} max={int(arr.max())} mean={float(arr.mean()):.1f}"
            )
        print(
            f"[scenario {s+1}/{n_scenarios}] kept={kept} skip_nonconv={skipped_nonconv} "
            f"skip_badV={skipped_badV} Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}{ctrl_summary}"
        )

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[LOADTYPE DATASET] Saved to {OUT_DIR}/")
    print(f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  Features per node: electrical_distance_ohm, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv, q_pv_kvar, p_sys_balance, q_sys_balance")
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
