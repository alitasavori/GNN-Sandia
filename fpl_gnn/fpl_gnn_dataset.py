import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj
from run_loadtype_dataset import _compute_electrical_distance_from_source


def build_zbus_and_baseline(
    t_index: int = 0,
    dP_kw: float = 1.0,
    dQ_kvar: float = 1.0,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate an FPL-style linearization around a single baseline operating point.

    Returns
    -------
    node_names : list of str
        Monitored bus.phase node names in a fixed order.
    v_base : np.ndarray, shape (N,)
        Baseline voltage magnitudes in per unit.
    x_base : np.ndarray, shape (2N,)
        Baseline stacked injections [P_1..P_N, Q_1..Q_N] in kW/kVAr (PV minus load).
    J : np.ndarray, shape (N, 2N)
        Finite-difference sensitivity mapping Delta x -> Delta v_mag.
    """
    # Compile feeder and set baseline time index
    dss_path = inj.compile_once()
    inj.setup_daily()
    inj.set_time_index(t_index)

    baseline = dict(inj.BASELINE)
    P_load = baseline["P_load_total_kw"]
    Q_load = baseline["Q_load_total_kvar"]
    P_pv = baseline["P_pv_total_kw"]
    sigL = baseline["sigma_load"]
    sigPV = baseline["sigma_pv"]

    node_names, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    N = len(node_names)
    node_to_idx = {n: i for i, n in enumerate(node_names)}

    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    # Baseline PF
    _, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
        P_load_total_kw=P_load,
        Q_load_total_kvar=Q_load,
        P_pv_total_kw=P_pv,
        mL_t=1.0,
        mPV_t=1.0,
        loads_dss=loads_dss,
        dev_to_dss_load=dev_to_dss_load,
        dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss,
        pv_to_dss=pv_to_dss,
        pv_to_busph=pv_to_busph,
        sigma_load=sigL,
        sigma_pv=sigPV,
        rng=rng,
    )
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError("Baseline PF did not converge.")

    vmag_base, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
    v_base = np.asarray(vmag_base, dtype=np.float64)

    P_node = np.zeros(N, dtype=np.float64)
    Q_node = np.zeros(N, dtype=np.float64)
    for i, n in enumerate(node_names):
        bus, phs = n.split(".")
        ph = int(phs)
        p_load = float(busphP_load.get((bus, ph), 0.0))
        q_load = float(busphQ_load.get((bus, ph), 0.0))
        p_pv = float(busphP_pv.get((bus, ph), 0.0))
        q_pv = float(busphQ_pv.get((bus, ph), 0.0))
        P_node[i] = p_pv - p_load
        Q_node[i] = q_pv - q_load
    x_base = np.concatenate([P_node, Q_node], axis=0)

    # Finite-difference sensitivities
    J = np.zeros((N, 2 * N), dtype=np.float64)

    def _solve_with_perturb(idx_node: int, dP: float, dQ: float) -> np.ndarray | None:
        inj.dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()
        inj.set_time_index(t_index)

        node_names_s, _, _, bus_to_phases_s = inj.get_all_bus_phase_nodes()
        if node_names_s != node_names:
            raise RuntimeError("Node list changed between baseline and perturbation.")
        loads_dss_s, dev_to_dss_load_s, dev_to_busph_load_s = inj.build_load_device_maps(bus_to_phases_s)
        pv_dss_s, pv_to_dss_s, pv_to_busph_s = inj.build_pv_device_maps()
        rng_s = np.random.default_rng(0)

        inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load,
            Q_load_total_kvar=Q_load,
            P_pv_total_kw=P_pv,
            mL_t=1.0,
            mPV_t=1.0,
            loads_dss=loads_dss_s,
            dev_to_dss_load=dev_to_dss_load_s,
            dev_to_busph_load=dev_to_busph_load_s,
            pv_dss=pv_dss_s,
            pv_to_dss=pv_to_dss_s,
            pv_to_busph=pv_to_busph_s,
            sigma_load=sigL,
            sigma_pv=sigPV,
            rng=rng_s,
        )

        node = node_names[idx_node]
        bus, phs = node.split(".")
        ph = int(phs)
        # Extra PQ load of -dP - j dQ -> positive injection at that node
        dss.Loads.New(
            name="fpl_sens_load",
            bus=f"{bus}.{ph}",
            phases=1,
            conn="Wye",
            model=1,
            kv=4.16,
            kW=-dP,
            kvar=-dQ,
        )

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return None
        vmag_new, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
        return np.asarray(vmag_new, dtype=np.float64)

    for i in range(N):
        v_plus = _solve_with_perturb(i, dP_kw, 0.0)
        if v_plus is not None:
            J[:, i] = (v_plus - v_base) / dP_kw
        v_plus = _solve_with_perturb(i, 0.0, dQ_kvar)
        if v_plus is not None:
            J[:, N + i] = (v_plus - v_base) / dQ_kvar

    return node_names, v_base, x_base, J


def generate_fpl_residual_dataset(
    out_dir: str = os.path.join("fpl_gnn", "gnn_samples_fpl_residual_full"),
    n_scenarios: int = 200,
    k_snapshots_per_scenario_total: int = 960,
    master_seed: int = 20260303,
):
    """
    Generate a dataset for FPL+GNN residual learning.

    Per-node features:
        - electrical_distance_ohm
        - P_node_kw, Q_node_kvar

    Targets:
        - vmag_pu_true, vmag_pu_fpl, vmag_pu_resid
    """
    os.makedirs(out_dir, exist_ok=True)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(out_dir, "gnn_sample_meta.csv")
    node_index_csv = os.path.join(out_dir, "gnn_node_index_master.csv")

    node_names, v_base, x_base, J = build_zbus_and_baseline(t_index=0)
    N = len(node_names)
    node_to_idx = {n: i for i, n in enumerate(node_names)}

    pd.DataFrame({"node": node_names, "node_idx": np.arange(N, dtype=int)}).to_csv(
        node_index_csv, index=False
    )

    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names, edge_csv_path=edge_csv
    )
    node_to_electrical_dist = _compute_electrical_distance_from_source(
        node_names, edge_csv
    )

    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[Dict] = []
    rows_node: list[Dict] = []
    sample_id = 0
    kept = skipped_nonconv = skipped_badV = 0

    for s in range(n_scenarios):
        inj.dss.Basic.ClearAll()
        inj.dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()

        node_names_s, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
        if node_names_s != node_names:
            raise RuntimeError("Node list changed across scenarios.")

        loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(
            bus_to_phases
        )
        pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load = sc["P_load_total_kw"]
        Q_load = sc["Q_load_total_kvar"]
        P_pv = sc["P_pv_total_kw"]
        sigL = sc["sigma_load"]
        sigPV = sc["sigma_pv"]

        prof_load = mL
        prof_pv = mPV
        prof_net = (P_load * mL) - (P_pv * mPV)

        rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        times = inj.select_times_three_profiles(
            prof_load=prof_load,
            prof_pv=prof_pv,
            prof_net=prof_net,
            K_total=k_snapshots_per_scenario_total,
            bins_by_profile={"load": 10, "pv": 10, "net": 10},
            include_anchors=True,
            rng=rng_times,
        )
        times = [int(t) for t in times]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        for t in times:
            inj.set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
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
                sigma_load=sigL,
                sigma_pv=sigPV,
                rng=rng_solve,
            )

            inj.dss.Solution.Solve()
            if not inj.dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(
                node_names
            )
            vmag_arr = np.asarray(vmag_m, dtype=np.float64)
            if (
                (not np.isfinite(vmag_arr).all())
                or (vmag_arr.min() < inj.VMAG_PU_MIN)
                or (vmag_arr.max() > inj.VMAG_PU_MAX)
            ):
                skipped_badV += 1
                continue

            P_node = np.zeros(N, dtype=np.float64)
            Q_node = np.zeros(N, dtype=np.float64)
            for i, n in enumerate(node_names):
                bus, phs = n.split(".")
                ph = int(phs)
                p_load = float(busphP_load.get((bus, ph), 0.0))
                q_load = float(busphQ_load.get((bus, ph), 0.0))
                p_pv = float(busphP_pv.get((bus, ph), 0.0))
                q_pv = float(busphQ_pv.get((bus, ph), 0.0))
                P_node[i] = p_pv - p_load
                Q_node[i] = q_pv - q_load

            x = np.concatenate([P_node, Q_node], axis=0)
            dx = x - x_base
            v_fpl = v_base + J @ dx
            eps = vmag_arr - v_fpl

            rows_sample.append(
                {
                    "sample_id": sample_id,
                    "scenario_id": s,
                    "t_index": t,
                    "t_minutes": t * inj.STEP_MIN,
                }
            )

            for i, n in enumerate(node_names):
                bus, phs = n.split(".")
                ph = int(phs)
                elec_dist = float(node_to_electrical_dist.get(n, 0.0))
                vm, va = float(vmag_arr[i]), float(vang_m[i])

                rows_node.append(
                    {
                        "sample_id": sample_id,
                        "node": n,
                        "node_idx": int(node_to_idx[n]),
                        "bus": bus,
                        "phase": int(ph),
                        "electrical_distance_ohm": elec_dist,
                        "P_node_kw": float(P_node[i]),
                        "Q_node_kvar": float(Q_node[i]),
                        "vmag_pu_true": vm,
                        "vmag_pu_fpl": float(v_fpl[i]),
                        "vmag_pu_resid": float(eps[i]),
                    }
                )

            sample_id += 1
            kept += 1

        print(
            f"[scenario {s+1}/{n_scenarios}] kept={kept} "
            f"skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} "
            f"Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}"
        )

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(sample_csv, index=False)
    df_node.to_csv(node_csv, index=False)

    print(f"\n[FPL+RESIDUAL DATASET] Saved to {out_dir}/")
    print(
        f"  {node_csv} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}"
    )
    return df_sample, df_node, edge_csv


