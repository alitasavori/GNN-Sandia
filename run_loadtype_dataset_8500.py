"""
IEEE 8500-node load-type-style dataset (GNN2).

Same CSV layout as run_loadtype_dataset.py (datasets_gnn2/loadtype/):
  - gnn_node_index_master.csv
  - gnn_edges_phase_static.csv
  - gnn_sample_meta.csv
  - gnn_node_features_and_targets.csv

Differences from the IEEE 34 pipeline:
  - Uses 8500-node/Master.dss (balanced case).
  - No daily loadshape / irradiance profiles: each sample is one snapshot with
    independent random scaling of each Load kW/kvar and each PV Pmpp (pre-solve setpoints).
  - Load-type buckets M1/M2/M4/M5 come from OpenDSS Load.Model() per element.
  - Capacitor q_cap per node is set to 0 here (IEEE34-specific table not used).

Output: datasets_gnn2/loadtype_8500/
"""
from __future__ import annotations

import importlib
import os
import numpy as np
import pandas as pd

import run_injection_dataset as inj

inj = importlib.reload(inj)
import opendssdirect as dss  # noqa: E402

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()

REPO_ROOT = os.path.abspath(_SCRIPT_DIR)
MASTER_8500 = os.path.join(REPO_ROOT, "8500-node", "Master.dss")
OUT_DIR = os.path.join(REPO_ROOT, "datasets_gnn2", "loadtype_8500")
os.makedirs(OUT_DIR, exist_ok=True)
EDGE_CSV = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DIR, "gnn_node_index_master.csv")

# Exclude substation / source-style buses from per-node training rows (tune for 8500 if needed)
EXCLUDED_UPSTREAM_BUSES: tuple[str, ...] = ()


def compile_8500() -> None:
    if not os.path.isfile(MASTER_8500):
        raise FileNotFoundError(f"Missing IEEE 8500 master: {MASTER_8500}")
    dss.Basic.ClearAll()
    dss.Text.Command(f'redirect "{os.path.abspath(MASTER_8500)}"')
    dss.Solution.Mode(1)
    inj._apply_voltage_bases()


def _model_bucket(model: int) -> int:
    m = int(model)
    if m in (1, 2, 4, 5):
        return m
    return 1


def _busph_fracs_load(load_name: str) -> list[tuple[str, int, float]]:
    dss.Circuit.SetActiveElement(f"Load.{load_name}")
    buses = dss.CktElement.BusNames()
    if not buses:
        return []
    bus_full = buses[0]
    bus, phs = inj.parse_bus_spec(bus_full)
    if not phs:
        phs = [1, 2, 3]
    n = max(len(phs), 1)
    w = 1.0 / n
    return [(bus, int(ph), w) for ph in phs]


def _busph_fracs_pv(pv_name: str) -> list[tuple[str, int, float]]:
    dss.Circuit.SetActiveElement(f"PVSystem.{pv_name}")
    buses = dss.CktElement.BusNames()
    if not buses:
        return []
    bus_full = buses[0]
    bus, phs = inj.parse_bus_spec(bus_full)
    if not phs:
        phs = [1, 2, 3]
    n = max(len(phs), 1)
    w = 1.0 / n
    return [(bus, int(ph), w) for ph in phs]


def _collect_baselines() -> tuple[list[dict], list[dict]]:
    loads: list[dict] = []
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        dss.Loads.Name(name)
        loads.append(
            {
                "name": name,
                "kw": float(dss.Loads.kW()),
                "kvar": float(dss.Loads.kvar()),
                "model": int(dss.Loads.Model()),
            }
        )
        if not dss.Loads.Next():
            break
    pvs: list[dict] = []
    for name in dss.PVsystems.AllNames():
        dss.PVsystems.Name(name)
        pvs.append({"name": name, "pmpp": float(dss.PVsystems.Pmpp())})
    return loads, pvs


def _build_pv_maps(pvs: list[dict]) -> tuple[dict, dict]:
    pv_to_dss: dict[str, str] = {}
    pv_to_busph: dict[str, list[tuple[str, int, float]]] = {}
    for pv in pvs:
        name = pv["name"]
        k = str(name).strip().lower()
        pv_to_dss[k] = name
        pv_to_busph[k] = _busph_fracs_pv(name)
    return pv_to_dss, pv_to_busph


def _apply_snapshot_8500(
    base_loads: dict[str, dict],
    base_pvs: dict[str, float],
    pv_to_dss: dict[str, str],
    pv_to_busph: dict[str, list[tuple[str, int, float]]],
    rng: np.random.Generator,
    sigma_load: float,
    sigma_pv: float,
):
    """Set all loads and PVs; return busph aggregates and per-type load dicts (pre-solve)."""
    busph_per_type = {m: ({}, {}) for m in (1, 2, 4, 5)}
    busphP_load: dict = {}
    busphQ_load: dict = {}
    busphP_pv: dict = {}
    busphQ_pv: dict = {}

    for _k, row in base_loads.items():
        name = row["name"]
        fp = inj._noise_factor(rng, sigma_load)
        fq = inj._noise_factor(rng, sigma_load)
        p_set = float(row["kw"] * fp)
        q_set = float(row["kvar"] * fq)
        bkt = _model_bucket(row["model"])
        dss.Loads.Name(name)
        dss.Loads.kW(p_set)
        dss.Loads.kvar(q_set)
        for (bus, ph, w) in _busph_fracs_load(name):
            busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + p_set * w
            busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + q_set * w
            bp, bq = busph_per_type[bkt]
            bp[(bus, ph)] = bp.get((bus, ph), 0.0) + p_set * w
            bq[(bus, ph)] = bq.get((bus, ph), 0.0) + q_set * w

    for k, dss_name in pv_to_dss.items():
        pmpp0 = float(base_pvs[dss_name])
        f = inj._noise_factor(rng, sigma_pv)
        pmpp_set = pmpp0 * f
        dss.PVsystems.Name(dss_name)
        dss.PVsystems.Pmpp(pmpp_set)
        for (bus, ph, w) in pv_to_busph.get(k, []):
            busphP_pv[(bus, ph)] = busphP_pv.get((bus, ph), 0.0) + pmpp_set * w
            busphQ_pv[(bus, ph)] = busphQ_pv.get((bus, ph), 0.0) + 0.0

    sum_p_load = float(sum(busphP_load.values()))
    sum_q_load = float(sum(busphQ_load.values()))
    sum_p_pv_set = float(sum(busphP_pv.values()))
    return (
        busph_per_type,
        busphP_load,
        busphQ_load,
        busphP_pv,
        busphQ_pv,
        sum_p_load,
        sum_q_load,
        sum_p_pv_set,
    )


def generate_gnn_snapshot_dataset_loadtype_8500(
    n_samples: int = 500,
    master_seed: int = 20260322,
    sigma_load: float = 0.12,
    sigma_pv: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compile_8500()
    loads, pvs = _collect_baselines()
    if not loads:
        raise RuntimeError("No Load elements found in 8500-node circuit.")
    base_loads = {d["name"]: dict(d) for d in loads}
    base_pvs = {d["name"]: d["pmpp"] for d in pvs}

    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}
    pd.DataFrame(
        {"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}
    ).to_csv(NODE_INDEX_CSV, index=False)
    print(f"[saved] master node index -> {NODE_INDEX_CSV} | N_nodes={len(node_names_master)}")

    inj.extract_static_phase_edges_to_csv(node_names_master=node_names_master, edge_csv_path=EDGE_CSV)
    pv_to_dss, pv_to_busph = _build_pv_maps(pvs)

    node_to_electrical_dist = {n: 0.0 for n in node_names_master}

    rows_sample: list[dict] = []
    rows_node: list[dict] = []
    sample_id = 0
    skipped_nonconv = 0
    skipped_badV = 0

    for s in range(n_samples):
        compile_8500()
        rng = np.random.default_rng(int(master_seed + sample_id))

        (
            busph_per_type,
            busphP_load,
            busphQ_load,
            busphP_pv,
            busphQ_pv,
            sum_p_load,
            sum_q_load,
            sum_p_pv_set,
        ) = _apply_snapshot_8500(
            base_loads,
            base_pvs,
            pv_to_dss,
            pv_to_busph,
            rng,
            sigma_load,
            sigma_pv,
        )

        try:
            dss.Solution.Solve()
        except Exception:
            pass
        if not dss.Solution.Converged():
            skipped_nonconv += 1
            continue

        busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
        vmag_arr = np.asarray(vmag_m, dtype=float)
        if not np.isfinite(vmag_arr).all():
            skipped_badV += 1
            continue

        vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

        sum_p_pv_act = float(sum(busphP_pv_actual.values()))
        sum_q_pv_act = float(sum(busphQ_pv_actual.values()))
        sum_q_cap = 0.0
        p_sys_balance = sum_p_load - sum_p_pv_act
        q_sys_balance = sum_q_load + sum_q_pv_act - sum_q_cap

        rows_sample.append(
            {
                "sample_id": sample_id,
                "scenario_id": 0,
                "t_index": s,
                "t_minutes": 0,
                "P_load_total_kw": sum_p_load,
                "Q_load_total_kvar": sum_q_load,
                "P_pv_total_kw": sum_p_pv_set,
                "sigma_load": float(sigma_load),
                "sigma_pv": float(sigma_pv),
                "m_loadshape": 1.0,
                "m_irradshape": 1.0,
                "p_sys_balance_kw": p_sys_balance,
                "q_sys_balance_kvar": q_sys_balance,
            }
        )

        for n in node_names_master:
            bus, phs = n.split(".")
            ph = int(phs)
            if bus in EXCLUDED_UPSTREAM_BUSES:
                continue

            m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
            m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
            m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
            m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
            m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
            m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
            m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
            m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))

            q_cap_node = 0.0
            p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
            q_pv_node = float(busphQ_pv_actual.get((bus, ph), 0.0))

            vm, va = vdict_m.get(n, (np.nan, np.nan))
            elec_dist = float(node_to_electrical_dist.get(n, 0.0))

            rows_node.append(
                {
                    "sample_id": sample_id,
                    "node": n,
                    "node_idx": int(node_to_idx_master[n]),
                    "bus": bus,
                    "phase": int(ph),
                    "electrical_distance_ohm": elec_dist,
                    "m1_p_kw": m1_p,
                    "m1_q_kvar": m1_q,
                    "m2_p_kw": m2_p,
                    "m2_q_kvar": m2_q,
                    "m4_p_kw": m4_p,
                    "m4_q_kvar": m4_q,
                    "m5_p_kw": m5_p,
                    "m5_q_kvar": m5_q,
                    "q_cap_kvar": q_cap_node,
                    "p_pv_kw": p_pv_node,
                    "q_pv_kvar": q_pv_node,
                    "p_sys_balance_kw": p_sys_balance,
                    "q_sys_balance_kvar": q_sys_balance,
                    "vmag_pu": float(vm),
                    "vang_deg": float(va),
                }
            )

        sample_id += 1
        if (s + 1) % max(1, n_samples // 10) == 0 or s == n_samples - 1:
            print(
                f"[8500 loadtype] progress {s+1}/{n_samples} kept_samples={sample_id} "
                f"skip_nonconv={skipped_nonconv} skip_badV={skipped_badV}"
            )

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)

    print(f"\n[LOADTYPE 8500] Saved to {OUT_DIR}/")
    if len(df_sample) == 0:
        print("  [WARN] No successful samples — check OpenDSS convergence and voltage limits.")
    else:
        print(f"  {NODE_CSV} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_sample, df_node


def main() -> None:
    generate_gnn_snapshot_dataset_loadtype_8500(
        n_samples=500,
        master_seed=20260322,
        sigma_load=0.12,
        sigma_pv=0.12,
    )


if __name__ == "__main__":
    main()
