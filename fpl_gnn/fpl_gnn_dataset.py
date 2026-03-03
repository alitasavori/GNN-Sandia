"""
FPL residual dataset with per-scenario draws from normal distributions.

Per scenario:
- Draw (P_load_total_kw, Q_load_total_kvar, P_der_total_kw, Q_der_total_kvar)
  from Normal(means, sigma_shared) with clipping at >= 0.
- sigma_shared is constant across all scenarios and applies to all 4 draws.
- No per-timestep noise is used in OpenDSS snapshot application (sigma_load=0, sigma_pv=0).

Per scenario per hour (24 samples):
- Base case (load-only, DER off): solve PF -> v_base.
- Compute FPL coefficients J = [A|B] for that hour by finite differences
  around the load-only base (inject tiny ΔP and ΔQ at each node).
- DER case (load + DER): DER P and Q follow the same DER profile over the day.
  Since P_der and Q_der share the same profile, each scenario has a constant Q/P ratio.
- Residual label: resid = v_true - (v_base + J @ [ΔP_der; ΔQ_der]).

The downstream model (MLP) learns resid as a function of [ΔP_der; ΔQ_der].
"""
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj

# Tiny perturbations for finite-difference A/B (J)
STEPS_PER_HOUR = inj.NPTS // 24  # 288/24 = 12
dP_KW = 1.0
dQ_KVAR = 1.0


def _hour_to_step(h: int) -> int:
    """Map hour 0..23 to a representative step index (start of hour)."""
    return h * STEPS_PER_HOUR


def _apply_snapshot_load_only(
    dss_path: str,
    node_names: List[str],
    P_load: float,
    Q_load: float,
    hour: int,
    mL: np.ndarray,
    loads_dss,
    dev_to_dss_load,
    dev_to_busph_load,
    pv_dss,
    pv_to_dss,
    pv_to_busph,
    sigma_load: float = 0.0,
    sigma_der: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply load at hour h only; DER = 0. Solve and return (v_base, x_base)."""
    inj.dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{dss_path}"')
    inj._apply_voltage_bases()
    inj.setup_daily()
    step = _hour_to_step(hour)
    inj.set_time_index(step)
    rng = np.random.default_rng(0)
    _, busphP_load, busphQ_load, _, _ = inj.apply_snapshot_timeconditioned(
        P_load_total_kw=P_load,
        Q_load_total_kvar=Q_load,
        P_pv_total_kw=0.0,
        mL_t=float(mL[step]),
        mPV_t=0.0,
        loads_dss=loads_dss,
        dev_to_dss_load=dev_to_dss_load,
        dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss,
        pv_to_dss=pv_to_dss,
        pv_to_busph=pv_to_busph,
        sigma_load=sigma_load,
        sigma_pv=sigma_der,
        rng=rng,
    )
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        return None, None
    vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
    v_base = np.asarray(vmag, dtype=np.float64)
    N = len(node_names)
    P_node = np.zeros(N)
    Q_node = np.zeros(N)
    for i, n in enumerate(node_names):
        bus, phs = n.split(".")
        ph = int(phs)
        p_load = float(busphP_load.get((bus, ph), 0.0))
        q_load = float(busphQ_load.get((bus, ph), 0.0))
        P_node[i] = -p_load
        Q_node[i] = -q_load
    x_base = np.concatenate([P_node, Q_node], axis=0)
    return v_base, x_base


def _apply_snapshot_load_and_der(
    dss_path: str,
    node_names: List[str],
    P_load: float,
    Q_load: float,
    P_der: float,
    Q_der: float,
    hour: int,
    mL: np.ndarray,
    mPV: np.ndarray,
    loads_dss,
    dev_to_dss_load,
    dev_to_busph_load,
    pv_dss,
    pv_to_dss,
    pv_to_busph,
    sigma_load: float = 0.0,
    sigma_der: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply load + DER at hour h. DER P and Q follow same profile (mPV). Return (v_true, delta_P, delta_Q) per node."""
    inj.dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{dss_path}"')
    inj._apply_voltage_bases()
    inj.setup_daily()
    step = _hour_to_step(hour)
    inj.set_time_index(step)
    rng = np.random.default_rng(0)
    mL_t = float(mL[step])
    mPV_t = float(mPV[step])
    _, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
        P_load_total_kw=P_load,
        Q_load_total_kvar=Q_load,
        P_pv_total_kw=P_der,
        mL_t=mL_t,
        mPV_t=mPV_t,
        loads_dss=loads_dss,
        dev_to_dss_load=dev_to_dss_load,
        dev_to_busph_load=dev_to_busph_load,
        pv_dss=pv_dss,
        pv_to_dss=pv_to_dss,
        pv_to_busph=pv_to_busph,
        sigma_load=sigma_load,
        sigma_pv=sigma_der,
        rng=rng,
    )
    # Set PF per scenario so the DER produces the desired Q/P ratio.
    # Note: OpenDSS PVSystem uses PF to determine reactive output (subject to kVA limits).
    pf = 1.0
    if (P_der * mPV_t) > 1e-9:
        pf = float((P_der) / np.sqrt(P_der * P_der + Q_der * Q_der))
        pf = float(np.clip(pf, 0.01, 1.0))
    for pv_name in pv_dss:
        dss.PVsystems.Name(pv_name)
        try:
            dss.PVsystems.PF(pf)
        except Exception:
            # If PF is not available in this build, we still proceed; delta_Q will be synthetic.
            pass
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        return None, None, None
    vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
    v_true = np.asarray(vmag, dtype=np.float64)
    N = len(node_names)
    delta_P = np.zeros(N)
    delta_Q = np.zeros(N)
    P_der_t = P_der * mPV_t
    Q_der_t = Q_der * mPV_t
    k_qp = float(Q_der_t / P_der_t) if abs(P_der_t) > 1e-9 else 0.0
    for i, n in enumerate(node_names):
        bus, phs = n.split(".")
        ph = int(phs)
        delta_P[i] = float(busphP_pv.get((bus, ph), 0.0))
        # Enforce that DER P and Q share the same per-node distribution (same profile + same share mapping).
        delta_Q[i] = float(delta_P[i] * k_qp)
    return v_true, delta_P, delta_Q


def _compute_J_at_hour(
    dss_path: str,
    node_names: List[str],
    v_base: np.ndarray,
    x_base: np.ndarray,
    P_load: float,
    Q_load: float,
    hour: int,
    mL: np.ndarray,
    loads_dss,
    dev_to_dss_load,
    dev_to_busph_load,
    pv_dss,
    pv_to_dss,
    pv_to_busph,
    sigma_load: float = 0.0,
    sigma_der: float = 0.0,
) -> np.ndarray:
    """Finite-difference J at this hour (load-only base). J shape (N, 2N)."""
    N = len(node_names)

    def _solve_perturb(idx_node: int, dP: float, dQ: float) -> np.ndarray | None:
        inj.dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()
        step = _hour_to_step(hour)
        inj.set_time_index(step)
        rng = np.random.default_rng(0)
        inj.apply_snapshot_timeconditioned(
            P_load_total_kw=P_load,
            Q_load_total_kvar=Q_load,
            P_pv_total_kw=0.0,
            mL_t=float(mL[step]),
            mPV_t=0.0,
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=sigma_load,
            sigma_pv=sigma_der,
            rng=rng,
        )
        node = node_names[idx_node]
        bus, phs = node.split(".")
        ph = int(phs)
        load_name = f"fpl_sens_{idx_node}"
        cmd = (
            f"new Load.{load_name} bus={bus}.{ph} phases=1 conn=wye model=1 kv=4.16 "
            f"kW={-float(dP)} kvar={-float(dQ)}"
        )
        dss.Text.Command(cmd)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return None
        vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
        return np.asarray(vmag, dtype=np.float64)

    J = np.zeros((N, 2 * N), dtype=np.float64)
    for i in range(N):
        vp = _solve_perturb(i, dP_KW, 0.0)
        if vp is not None:
            J[:, i] = (vp - v_base) / dP_KW
        vp = _solve_perturb(i, 0.0, dQ_KVAR)
        if vp is not None:
            J[:, N + i] = (vp - v_base) / dQ_KVAR
    return J


def generate_fpl_residual_dataset(
    out_dir: str = os.path.join("fpl_gnn", "gnn_samples_fpl_residual_full"),
    n_scenarios: int = 500,
    master_seed: int = 20260303,
    mu_P_load_kw: float = 1415.2,
    mu_Q_load_kvar: float = 835.2,
    mu_P_der_kw: float = 1000.0,
    mu_Q_der_kvar: float = 0.0,
    sigma_shared: float = 50.0,
):
    """
    Generate FPL residual dataset:

    - 500 scenarios; per scenario (P_load, Q_load, P_der, Q_der) drawn from Normal(means, sigma_shared)
      with clipping at >= 0. sigma_shared is constant across all scenarios.
    - 24 hours per scenario; at each hour we compute A/B (J) from load-only base,
      then solve with DER (P and Q, same profile), store delta_P/delta_Q from DER,
      and residual = v_true - (v_base + J @ delta_der).
    - Output: one row per (sample, node) with delta_P_kw, delta_Q_kvar, vmag_resid;
      and sample-level metadata. For MLP: input = stacked [delta_P, delta_Q] (2N), output = residual (N).
    """
    os.makedirs(out_dir, exist_ok=True)
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(out_dir, "gnn_sample_meta.csv")
    node_index_csv = os.path.join(out_dir, "gnn_node_index_master.csv")

    dss_path = inj.compile_once()
    inj.setup_daily()
    node_names, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    N = len(node_names)
    node_to_idx = {n: i for i, n in enumerate(node_names)}

    pd.DataFrame({"node": node_names, "node_idx": np.arange(N, dtype=int)}).to_csv(
        node_index_csv, index=False
    )

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    rng = np.random.default_rng(master_seed)
    rows_sample: List[Dict] = []
    rows_node: List[Dict] = []
    sample_id = 0
    kept = 0
    skipped = 0

    for s in range(n_scenarios):
        # One draw per scenario from Normal distributions; clip at >= 0
        P_load = float(max(0.0, rng.normal(mu_P_load_kw, sigma_shared)))
        Q_load = float(max(0.0, rng.normal(mu_Q_load_kvar, sigma_shared)))
        P_der = float(max(0.0, rng.normal(mu_P_der_kw, sigma_shared)))
        Q_der = float(max(0.0, rng.normal(mu_Q_der_kvar, sigma_shared)))

        # No per-timestep noise (sigma=0) inside OpenDSS application
        sigL = 0.0
        sigD = 0.0

        for hour in range(24):
            v_base, x_base = _apply_snapshot_load_only(
                dss_path, node_names, P_load, Q_load, hour, mL,
                loads_dss, dev_to_dss_load, dev_to_busph_load,
                pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
            )
            if v_base is None:
                skipped += 1
                continue

            J = _compute_J_at_hour(
                dss_path, node_names, v_base, x_base, P_load, Q_load, hour, mL,
                loads_dss, dev_to_dss_load, dev_to_busph_load,
                pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
            )

            v_true, delta_P, delta_Q = _apply_snapshot_load_and_der(
                dss_path, node_names, P_load, Q_load, P_der, Q_der,
                hour, mL, mPV,
                loads_dss, dev_to_dss_load, dev_to_busph_load,
                pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
            )
            if v_true is None:
                skipped += 1
                continue

            delta_x = np.concatenate([np.asarray(delta_P), np.asarray(delta_Q)], axis=0)
            v_fpl = v_base + J @ delta_x
            resid = v_true - v_fpl

            if not np.isfinite(resid).all() or np.any(v_true < inj.VMAG_PU_MIN) or np.any(v_true > inj.VMAG_PU_MAX):
                skipped += 1
                continue

            rows_sample.append({
                "sample_id": sample_id,
                "scenario_id": s,
                "hour": hour,
            })

            for i, n in enumerate(node_names):
                rows_node.append({
                    "sample_id": sample_id,
                    "node": n,
                    "node_idx": node_to_idx[n],
                    "delta_P_kw": float(delta_P[i]),
                    "delta_Q_kvar": float(delta_Q[i]),
                    "vmag_resid": float(resid[i]),
                    "vmag_true": float(v_true[i]),
                    "vmag_fpl": float(v_fpl[i]),
                })

            sample_id += 1
            kept += 1

        if (s + 1) % 50 == 0 or s == 0:
            print(f"[scenario {s+1}/{n_scenarios}] kept={kept} skipped={skipped} Pload={P_load:.1f} Qload={Q_load:.1f} Pder={P_der:.1f}")

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(sample_csv, index=False)
    df_node.to_csv(node_csv, index=False)

    print(f"\n[FPL+RESIDUAL DATASET] Saved to {out_dir}/")
    print(f"  samples={len(df_sample)} node-rows={len(df_node)}")
    return df_sample, df_node, node_csv
