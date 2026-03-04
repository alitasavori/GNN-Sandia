"""
Analytical FPL Jacobian (J) via Z_LL and comparison against finite-difference J.

This module exposes a helper to:
- Compute J for one *day* (single scenario with fixed P_load/Q_load) using:
  - Finite differences (existing implementation in fpl_gnn_dataset)
  - Analytical FPL formula based on the network impedance matrix Z_LL
- Compare the two for each hour of the day.
"""

from typing import Dict

import numpy as np
import opendssdirect as dss

import run_injection_dataset as inj
from fpl_gnn.fpl_gnn_dataset import (
    _hour_to_step,
    _apply_snapshot_load_only,
    _compute_J_at_hour,
)


def _compute_J_analytic_at_hour(
    dss_path: str,
    node_names,
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
):
    """
    Compute J via analytical FPL formula (Z_LL-based) for a given hour.

    Uses:
      - Load-only operating point at this hour (DER=0)
      - Y-bus from OpenDSS (via YMatrix.getYsparse + Circuit.YNodeOrder)
      - Simplified per-node version of the A/B formulas from the draft:

        For each pair of nodes (j, i):
            A[j, i] = Re( v_j* * Z_LL[j, i] / v_i* ) / |v_j|
            B[j, i] = Im( v_j* * Z_LL[j, i] / v_i* ) / |v_j|

    Returns:
      J_analytic (N, 2N), A (N, N), B (N, N)
    """
    # Recreate load-only base for this hour (DER = 0)
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
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        return None, None, None

    # Complex bus-phase voltages at this hour
    vmag, vang = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
    vmag_arr = np.asarray(vmag, dtype=np.float64)
    vang_arr = np.asarray(vang, dtype=np.float64)
    v = vmag_arr * np.exp(1j * np.deg2rad(vang_arr))

    # Build Y-bus and extract Y_LL for our node ordering
    # Ensure Y is built for current operating point
    dss.Solution.BuildYMatrix(1, 1)
    y_nodes = list(dss.Circuit.YNodeOrder())
    data, indices, indptr = dss.YMatrix.getYsparse()
    data = np.asarray(data, dtype=np.complex128)
    indices = np.asarray(indices, dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)
    nY = len(y_nodes)
    Y = np.zeros((nY, nY), dtype=np.complex128)
    for col in range(nY):
        start = indptr[col]
        end = indptr[col + 1]
        for k in range(start, end):
            row = indices[k]
            Y[row, col] = data[k]

    # Map our node_names (bus.phase) to Y-node indices; preserve node_names order
    y_map: Dict[str, int] = {str(n).lower(): i for i, n in enumerate(y_nodes)}
    idx = []
    for n in node_names:
        key = str(n).lower()
        if key not in y_map:
            raise RuntimeError(f"Node {n} not found in Circuit.YNodeOrder()")
        idx.append(y_map[key])
    idx = np.asarray(idx, dtype=int)

    Y_LL = Y[np.ix_(idx, idx)]
    Z_LL = np.linalg.inv(Y_LL)

    N = len(node_names)
    A = np.zeros((N, N), dtype=np.float64)
    B = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        vj = v[j]
        vj_mag = float(np.abs(vj))
        if vj_mag <= 1e-9:
            continue
        for i in range(N):
            vi = v[i]
            # Simplified per-node FPL coefficient
            num = np.conj(vj) * Z_LL[j, i] / np.conj(vi)
            A[j, i] = float(np.real(num) / vj_mag)
            B[j, i] = float(np.imag(num) / vj_mag)

    J_analytic = np.concatenate([A, B], axis=1)
    return J_analytic, A, B


def compare_analytic_vs_fd_one_scenario(
    P_load_kw: float,
    Q_load_kvar: float,
    master_seed: int = 20260304,
    loadshape_name: str = "5minDayShape",
):
    """
    For a single scenario (one day, fixed P_load/Q_load), compute J analytically vs
    finite-difference and compare for each hour (0..23).

    The scenario here is:
    - Total P_load/Q_load are fixed (inputs to this function)
    - No DER is used; we look only at load-only operating points.

    Returns:
      dict: hour -> {
          "J_fd": (N, 2N) finite-difference J,
          "J_analytic": (N, 2N) analytical J,
          "max_abs_diff": max_ij |J_analytic - J_fd|,
          "fro_norm_diff": Frobenius norm of the difference
      }
    """
    del master_seed  # kept for potential future use; not needed currently

    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)

    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    # No per-timestep noise for this comparison
    sigL = 0.0
    sigD = 0.0

    results_by_hour: Dict[int, Dict] = {}

    for hour in range(24):
        # Finite-difference J around load-only base
        v_base, x_base = _apply_snapshot_load_only(
            dss_path, node_names, P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
        )
        if v_base is None:
            print(f"[hour {hour:02d}] base solve did not converge; skipping")
            continue

        J_fd = _compute_J_at_hour(
            dss_path, node_names, v_base, x_base, P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
        )

        # Analytical J from Z_LL
        J_analytic, A_analytic, B_analytic = _compute_J_analytic_at_hour(
            dss_path, node_names, P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
        )
        if J_analytic is None:
            print(f"[hour {hour:02d}] analytic solve did not converge; skipping")
            continue

        diff = J_analytic - J_fd
        max_abs = float(np.max(np.abs(diff)))
        fro_norm = float(np.linalg.norm(diff))

        print(f"[hour {hour:02d}] J analytic vs FD: max|ΔJ|={max_abs:.3e}, ||ΔJ||_F={fro_norm:.3e}")

        results_by_hour[hour] = dict(
            J_fd=J_fd,
            J_analytic=J_analytic,
            A_analytic=A_analytic,
            B_analytic=B_analytic,
            max_abs_diff=max_abs,
            fro_norm_diff=fro_norm,
        )

    return results_by_hour

