"""
Analytical FPL Jacobian (J) via Z_LL and comparison against finite-difference J.

This module exposes a helper to:
- Compute J for one *day* (single scenario with fixed P_load/Q_load) using:
  - Finite differences (existing implementation in fpl_gnn_dataset)
  - Analytical FPL formula based on the network impedance matrix Z_LL
- Compare the two for each hour of the day.
"""

from typing import Dict, List, Tuple

import numpy as np
import opendssdirect as dss

import run_injection_dataset as inj
from fpl_gnn.fpl_gnn_dataset import (
    _hour_to_step,
    _apply_snapshot_load_only,
    _compute_J_at_hour,
)


def _build_bus_phase_map(
    node_names: List[str],
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    From node_names like '800.1', '800.2', build:
      - ordered list of unique bus names
      - mapping bus -> list of node indices (phases) Φ_i
    This is used to implement the bus-level, multi-phase formulas in the draft.
    """
    bus_to_idx: Dict[str, List[int]] = {}
    for idx, n in enumerate(node_names):
        bus, _ = n.split(".")
        bus_to_idx.setdefault(bus, []).append(idx)
    bus_names = sorted(bus_to_idx.keys())
    # Sort each Φ_i by node index for reproducibility
    for b in bus_names:
        bus_to_idx[b] = sorted(bus_to_idx[b])
    return bus_names, bus_to_idx


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

    This is a faithful implementation of the draft's bus-level, multi-phase
    coefficients (A_{jφ,i,t}, B_{jφ,i,t}), adapted to our node list:

      - Each node index j corresponds to (bus j, phase φ).
      - Each bus i has a phase set Φ_i (indices of nodes at that bus).
      - For each bus i and measurement node jφ:

        A_{jφ,i,t} = (1/|Φ_i|) * sum_{ψ in Φ_i} Re[ v_j* Z_LL[jφ, iψ] / v_iψ* ] / |v_j|
        B_{jφ,i,t} = (1/|Φ_i|) * sum_{ψ in Φ_i} Im[ v_j* Z_LL[jφ, iψ] / v_iψ* ] / |v_j|

    We then convert these from sensitivities per unit of per-unit power
    (as in the draft) to sensitivities per kW/kvar to match the FD J in
    our dataset (which uses 1 kW / 1 kvar perturbations).

    Returns:
      J_analytic: (N_nodes, 2*N_buses) = [A | B] in units of pu per kW/kvar
      A_bus: (N_nodes, N_buses) active-power sensitivities (per kW)
      B_bus: (N_nodes, N_buses) reactive-power sensitivities (per kvar)
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

    # Sanity check 1: compare against dense SystemY (same YNodeOrder)
    try:
        Y_dense_flat = np.array(dss.Circuit.SystemY(), dtype=np.complex128)
        if Y_dense_flat.size == nY * nY:
            Y_dense = Y_dense_flat.reshape((nY, nY))
            Y_diff = Y - Y_dense
            max_abs_Y = float(np.max(np.abs(Y)))
            max_abs_diff_Y = float(np.max(np.abs(Y_diff)))
            fro_diff_Y = float(np.linalg.norm(Y_diff))
            print(
                f"    [Y-check] nY={nY}, max|Y|={max_abs_Y:.3e}, "
                f"max|Y_sparse-Y_dense|={max_abs_diff_Y:.3e}, "
                f"||Y_sparse-Y_dense||_F={fro_diff_Y:.3e}"
            )
        else:
            print(
                f"    [Y-check] SystemY size mismatch: "
                f"{Y_dense_flat.size} vs {nY*nY}"
            )
    except Exception as exc:
        print(f"    [Y-check] SystemY comparison failed: {exc}")

    # Map our node_names (bus.phase) to Y-node indices; preserve node_names order
    y_map: Dict[str, int] = {str(n).lower(): i for i, n in enumerate(y_nodes)}
    idx = []
    for n in node_names:
        key = str(n).lower()
        if key not in y_map:
            raise RuntimeError(f"Node {n} not found in Circuit.YNodeOrder()")
        idx.append(y_map[key])
    idx = np.asarray(idx, dtype=int)

    # Sanity check 2: show a few node-to-Y mappings
    print("    [Y-map] first 5 node_names -> YNodeOrder indices:")
    for k in range(min(5, len(node_names))):
        n = node_names[k]
        print(f"        node {k}: {n} -> Y index {idx[k]} (YNodeOrder={y_nodes[idx[k]]})")

    Y_LL = Y[np.ix_(idx, idx)]
    # Sanity check 3: condition number and basic stats of Y_LL
    try:
        cond_YLL = float(np.linalg.cond(Y_LL))
    except Exception:
        cond_YLL = float("inf")
    max_abs_YLL = float(np.max(np.abs(Y_LL)))
    print(
        f"    [Y_LL] shape={Y_LL.shape}, max|Y_LL|={max_abs_YLL:.3e}, cond(Y_LL)≈{cond_YLL:.3e}"
    )

    Z_LL = np.linalg.inv(Y_LL)

    N = len(node_names)

    # Build bus-level phase sets Φ_i for the draft formulas.
    bus_names, bus_to_idx = _build_bus_phase_map(node_names)
    M = len(bus_names)
    print(f"    [bus-map] N_nodes={N}, N_buses={M}")
    for b in bus_names[:5]:
        phases = [node_names[idx] for idx in bus_to_idx[b]]
        print(f"        bus {b}: Φ_i = {phases}")

    A_bus = np.zeros((N, M), dtype=np.float64)
    B_bus = np.zeros((N, M), dtype=np.float64)

    # Attempt to get system MVA base from OpenDSS for unit alignment.
    # FPL A,B in the draft are sensitivities per unit of *per-unit* power.
    # Our finite-difference J_fd uses kW/kvar. If S_base is in MVA and P is in kW,
    # then 1 kW = 0.001 MW, and P_pu = P_MW / S_base = (P_kw * 0.001) / S_base.
    # Therefore dV/dP_kw = (dV/dP_pu) * (0.001 / S_base).
    try:
        S_base_MVA = float(dss.Solution.MVABase())
    except Exception:
        try:
            S_base_MVA = float(dss.Circuit.MVABase())
        except Exception:
            S_base_MVA = 1.0
            print("    [units] MVABase not available; falling back to 1.0 MVA")

    scale_per_kw = 1e-3 / float(S_base_MVA)
    print(f"    [units] MVABase={S_base_MVA:.3f} MVA, scale_per_kw={scale_per_kw:.3e}")

    # Compute A,B using the bus-level, multi-phase formulas.
    max_abs_A_pu = 0.0
    max_abs_B_pu = 0.0
    for bus_idx, bus in enumerate(bus_names):
        phi_indices = bus_to_idx[bus]
        if not phi_indices:
            continue
        Phi_card = float(len(phi_indices))
        for j in range(N):
            vj = v[j]
            vj_mag = float(np.abs(vj))
            if vj_mag <= 1e-9:
                continue
            num_sum = 0.0 + 0.0j
            for idx_i in phi_indices:
                vi = v[idx_i]
                num_sum += np.conj(vj) * Z_LL[j, idx_i] / np.conj(vi)
            num_avg = num_sum / Phi_card
            a_pu = float(np.real(num_avg) / vj_mag)
            b_pu = float(np.imag(num_avg) / vj_mag)
            max_abs_A_pu = max(max_abs_A_pu, abs(a_pu))
            max_abs_B_pu = max(max_abs_B_pu, abs(b_pu))
            # Convert to sensitivity per kW/kvar (to match finite-difference J_fd)
            A_bus[j, bus_idx] = a_pu * scale_per_kw
            B_bus[j, bus_idx] = b_pu * scale_per_kw

    print(
        f"    [A,B_pu] max|A_pu|={max_abs_A_pu:.3e}, max|B_pu|={max_abs_B_pu:.3e}"
    )
    print(
        f"    [A,B_kw] max|A|={np.max(np.abs(A_bus)):.3e}, max|B|={np.max(np.abs(B_bus)):.3e}"
    )

    J_analytic = np.concatenate([A_bus, B_bus], axis=1)
    return J_analytic, A_bus, B_bus


def _compute_J_fd_bus_at_hour(
    dss_path: str,
    node_names: List[str],
    bus_names: List[str],
    bus_to_idx: Dict[str, List[int]],
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
    dP_bus_kw: float = 1.0,
    dQ_bus_kvar: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bus-level finite-difference A,B at this hour.

    For each bus i:
      - For A: inject +dP_bus_kw at bus i, equally across Φ_i phases; Q=0.
      - For B: inject +dQ_bus_kvar at bus i, equally across Φ_i phases; P=0.

    Use the same load-only operating point (DER=0) as the analytic J. All solves
    use kW/kvar injections, so the resulting A_fd,B_fd are in pu per kW/kvar,
    directly comparable to the scaled analytic A_bus,B_bus.
    """
    N = len(node_names)
    M = len(bus_names)
    A_fd = np.zeros((N, M), dtype=np.float64)
    B_fd = np.zeros((N, M), dtype=np.float64)

    # Base solve (load-only) for this hour
    v_base, _ = _apply_snapshot_load_only(
        dss_path, node_names, P_load, Q_load, hour, mL,
        loads_dss, dev_to_dss_load, dev_to_busph_load,
        pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigma_load, sigma_der=sigma_der,
    )
    if v_base is None:
        return A_fd, B_fd
    v_base = np.asarray(v_base, dtype=np.float64)

    # Helper to apply base + bus-level perturbation and return v
    def _solve_bus_perturb(bus_idx: int, dP_kw: float, dQ_kvar: float) -> np.ndarray | None:
        inj.dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()
        step = _hour_to_step(hour)
        inj.set_time_index(step)
        rng = np.random.default_rng(0)
        # Base load-only
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
        bus = bus_names[bus_idx]
        phi_indices = bus_to_idx.get(bus, [])
        if not phi_indices:
            return None
        nphi = len(phi_indices)
        dP_each = float(dP_kw) / float(nphi) if nphi > 0 else 0.0
        dQ_each = float(dQ_kvar) / float(nphi) if nphi > 0 else 0.0
        for idx_node in phi_indices:
            node = node_names[idx_node]
            bus_name, phs = node.split(".")
            ph = int(phs)
            load_name = f"fd_bus_{bus_idx}_{ph}"
            cmd = (
                f"new Load.{load_name} bus={bus_name}.{ph} phases=1 conn=wye model=1 kv=4.16 "
                f"kW={-dP_each} kvar={-dQ_each}"
            )
            dss.Text.Command(cmd)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return None
        vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
        return np.asarray(vmag, dtype=np.float64)

    # Active-power sensitivities (A_fd)
    for bi, _ in enumerate(bus_names):
        vp = _solve_bus_perturb(bi, dP_bus_kw, 0.0)
        if vp is not None:
            A_fd[:, bi] = (vp - v_base) / dP_bus_kw

    # Reactive-power sensitivities (B_fd)
    for bi, _ in enumerate(bus_names):
        vp = _solve_bus_perturb(bi, 0.0, dQ_bus_kvar)
        if vp is not None:
            B_fd[:, bi] = (vp - v_base) / dQ_bus_kvar

    return A_fd, B_fd


def compare_analytic_vs_fd_one_scenario(
    P_load_kw: float,
    Q_load_kvar: float,
    master_seed: int = 20260304,
    loadshape_name: str = "5minDayShape",
):
    """
    For a single scenario (one day, fixed P_load/Q_load), compute bus-level
    FPL coefficients analytically vs finite-difference and compare for each
    hour (0..23).

    Both sides implement the draft's bus-level, multi-phase A,B:
      - Analytic: uses Z_LL and the formulas in the LaTeX draft.
      - Finite-difference: injects 1 kW / 1 kvar at each bus (spread across
        its phases) and observes ΔV at all nodes.

    Returns:
      dict: hour -> {
          "A_fd": (N_nodes, N_buses),
          "B_fd": (N_nodes, N_buses),
          "A_analytic": (N_nodes, N_buses),
          "B_analytic": (N_nodes, N_buses),
          "max_abs_diff": max_ij |J_analytic - J_fd| where J=[A|B],
          "fro_norm_diff": Frobenius norm of J_analytic - J_fd
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

    # Build bus map once (time-invariant)
    bus_names, bus_to_idx = _build_bus_phase_map(node_names)
    print(f"[global] N_nodes={len(node_names)}, N_buses={len(bus_names)}")

    results_by_hour: Dict[int, Dict] = {}

    for hour in range(24):
        print(f"\n=== Hour {hour:02d} ===")

        # Analytic A,B from Z_LL (bus-level, multi-phase)
        J_analytic, A_analytic, B_analytic = _compute_J_analytic_at_hour(
            dss_path, node_names, P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
        )
        if J_analytic is None:
            print(f"[hour {hour:02d}] analytic solve did not converge; skipping")
            continue

        # Finite-difference A,B at bus level
        A_fd, B_fd = _compute_J_fd_bus_at_hour(
            dss_path, node_names, bus_names, bus_to_idx,
            P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
            dP_bus_kw=1.0, dQ_bus_kvar=1.0,
        )

        J_fd = np.concatenate([A_fd, B_fd], axis=1)

        # Per-hour sanity: basic norms of each J
        max_abs_fd = float(np.max(np.abs(J_fd)))
        fro_fd = float(np.linalg.norm(J_fd))
        max_abs_an = float(np.max(np.abs(J_analytic)))
        fro_an = float(np.linalg.norm(J_analytic))

        print(
            f"[hour {hour:02d}] J_fd (bus-level): max|J|={max_abs_fd:.3e}, ||J||_F={fro_fd:.3e}; "
            f"J_an: max|J|={max_abs_an:.3e}, ||J||_F={fro_an:.3e}"
        )

        # Column-wise sanity for first few buses
        for bi, bus in enumerate(bus_names[:3]):
            col_fd_A = A_fd[:, bi]
            col_an_A = A_analytic[:, bi]
            print(
                f"    [hour {hour:02d}] bus {bus}: "
                f"max|A_fd|={np.max(np.abs(col_fd_A)):.3e}, "
                f"max|A_an|={np.max(np.abs(col_an_A)):.3e}"
            )

        diff = J_analytic - J_fd
        max_abs = float(np.max(np.abs(diff)))
        fro_norm = float(np.linalg.norm(diff))

        # Best-fit scalar c such that c*J_analytic is closest to J_fd in Frobenius norm:
        #   c* = <J_fd, J_an> / ||J_an||_F^2
        num = float(np.vdot(J_analytic, J_fd).real)  # inner product
        den = float(np.vdot(J_analytic, J_analytic).real) + 1e-16
        c_star = num / den
        J_scaled = c_star * J_analytic
        diff_scaled = J_scaled - J_fd
        max_abs_scaled = float(np.max(np.abs(diff_scaled)))
        fro_norm_scaled = float(np.linalg.norm(diff_scaled))

        print(f"[hour {hour:02d}] J analytic vs FD: max|ΔJ|={max_abs:.3e}, ||ΔJ||_F={fro_norm:.3e}")
        print(
            f"    [hour {hour:02d}] best-fit c*: {c_star:.3e}; "
            f"scaled: max|ΔJ|={max_abs_scaled:.3e}, ||ΔJ||_F={fro_norm_scaled:.3e}"
        )

        results_by_hour[hour] = dict(
            A_fd=A_fd,
            B_fd=B_fd,
            A_analytic=A_analytic,
            B_analytic=B_analytic,
            max_abs_diff=max_abs,
            fro_norm_diff=fro_norm,
            best_fit_c=c_star,
            max_abs_diff_scaled=max_abs_scaled,
            fro_norm_diff_scaled=fro_norm_scaled,
        )

    return results_by_hour

