"""
Analytical FPL Jacobian (J) via Z_LL and comparison against finite-difference J.

This module exposes a helper to:
- Compute J for one *day* (single scenario with fixed P_load/Q_load) using:
  - Finite differences (existing implementation in fpl_gnn_dataset)
  - Analytical FPL formula based on the network impedance matrix Z_LL
- Compare the two for each hour of the day.

In this experiment we make BOTH methods node-level:
- Inputs: per node (bus.phase) P/Q injections
- Outputs: per node (bus.phase) voltage magnitudes
So J has shape (N_nodes, 2*N_nodes) for both analytic and FD.
"""

from typing import Dict, List

import numpy as np
import opendssdirect as dss

import run_injection_dataset as inj
from fpl_gnn.fpl_gnn_dataset import (
    _hour_to_step,
    _apply_snapshot_load_only,
    _compute_J_at_hour,
)


def _compute_J_analytic_node_at_hour(
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
):
    """
    Compute NODE-LEVEL analytic J via Z_LL for a given hour.

    Here both inputs and outputs are at the node (bus.phase) level:
      - Inputs: ΔP, ΔQ per node (no bus-level averaging over phases).
      - Outputs: Δ|V| per node.

    For each pair of nodes (j, i):
        A[j, i] = Re( v_j* * Z_LL[j, i] / v_i* ) / |v_j|
        B[j, i] = Im( v_j* * Z_LL[j, i] / v_i* ) / |v_j|

    This approximates the fixed-point linearization at the node level.

    Returns:
      J_analytic: (N_nodes, 2*N_nodes) = [A | B] in units of pu per kW/kvar
      A_node: (N_nodes, N_nodes) active-power sensitivities (per kW)
      B_node: (N_nodes, N_nodes) reactive-power sensitivities (per kvar)
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
        print(f"    [Y-check] SystemY comparison failed: {type(exc).__name__}: {exc}")
    print(
        f"    [dbg] Y: nY={nY} (full circuit nodes), N={len(node_names)} (our node set), "
        f"Y_LL shape=({len(node_names)},{len(node_names)})"
    )

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

    # Build per-node voltage and current bases to convert Y_LL to per-unit.
    N = len(node_names)
    Vbase_ln = np.zeros(N, dtype=np.float64)  # line-to-neutral base volts per node
    Ibase = np.zeros(N, dtype=np.float64)     # base amps per node

    # System base MVA (same for all nodes).
    # OpenDSS often leaves MVABase unset after compile, so get can fail or return 0.
    # Set it explicitly via the text interface if needed, then read.
    S_base_MVA = None
    _has_sol = hasattr(dss.Solution, "MVABase")
    _has_ckt = hasattr(dss.Circuit, "MVABase")
    print(f"    [dbg] MVABase: hasattr(Solution.MVABase)={_has_sol}, hasattr(Circuit.MVABase)={_has_ckt}")
    for attempt in range(2):  # try get, then set+get once
        try:
            if _has_sol:
                val = dss.Solution.MVABase()
                print(f"    [dbg] MVABase attempt={attempt} Solution.MVABase() raw={val!r} type={type(val).__name__}")
                if val is not None and float(val) > 0:
                    S_base_MVA = float(val)
                    print(f"    [dbg] MVABase using Solution.MVABase() -> {S_base_MVA}")
                    break
        except Exception as e:
            print(f"    [dbg] MVABase attempt={attempt} Solution.MVABase() exception: {type(e).__name__}: {e}")
        try:
            if S_base_MVA is None and _has_ckt:
                val = dss.Circuit.MVABase()
                print(f"    [dbg] MVABase attempt={attempt} Circuit.MVABase() raw={val!r} type={type(val).__name__}")
                if val is not None and float(val) > 0:
                    S_base_MVA = float(val)
                    print(f"    [dbg] MVABase using Circuit.MVABase() -> {S_base_MVA}")
                    break
        except Exception as e:
            print(f"    [dbg] MVABase attempt={attempt} Circuit.MVABase() exception: {type(e).__name__}: {e}")
        if S_base_MVA is None:
            try:
                dss.Text.Command("set MVABase=100")
                print("    [dbg] MVABase sent: set MVABase=100")
            except Exception as e:
                print(f"    [dbg] MVABase set command exception: {type(e).__name__}: {e}")
    if S_base_MVA is None or S_base_MVA <= 0:
        S_base_MVA = 100.0
        print("    [units] MVABase not available; using 100.0 MVA (set or fallback)")
    S_base_VA = S_base_MVA * 1e6
    print(f"    [dbg] Units: S_base_MVA={S_base_MVA}, S_base_VA={S_base_VA:.3e}")

    # For each node, get its bus kVBase (line-to-line) and derive line-to-neutral Vbase.
    buses_seen: Dict[str, float] = {}
    for k, n in enumerate(node_names):
        bus_name, _ = n.split(".")
        if bus_name not in buses_seen:
            dss.Circuit.SetActiveBus(bus_name)
            kv_ll = float(dss.Bus.kVBase())
            buses_seen[bus_name] = kv_ll
        kv_ll = buses_seen[bus_name]
        if kv_ll <= 0.0:
            kv_ll = 1.0  # fallback to avoid zero base
        Vbase_ll = kv_ll * 1000.0
        Vbase_ln[k] = Vbase_ll / np.sqrt(3.0)
        Ibase[k] = S_base_VA / (np.sqrt(3.0) * Vbase_ll)

    print(
        f"    [dbg] Vbase/Ibase sample: node0 {node_names[0]} Vbase_ln={Vbase_ln[0]:.2f} V Ibase={Ibase[0]:.4f} A; "
        f"min/max Vbase_ln={Vbase_ln.min():.2f}/{Vbase_ln.max():.2f} Ibase={Ibase.min():.4f}/{Ibase.max():.4f}"
    )

    # Convert physical Y_LL to per-unit Y_LL_pu using:
    #   I_pu_i = I_phys_i / Ibase_i
    #   V_pu_j = V_phys_j / Vbase_ln_j
    #   => Y_pu_ij = Y_phys_ij * Vbase_ln_j / Ibase_i
    Y_LL_pu = np.zeros_like(Y_LL, dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            Y_LL_pu[i, j] = Y_LL[i, j] * (Vbase_ln[j] / Ibase[i])

    # Sanity check 3: condition number and basic stats of Y_LL_pu
    try:
        cond_YLL_pu = float(np.linalg.cond(Y_LL_pu))
    except Exception:
        cond_YLL_pu = float("inf")
    max_abs_YLL_pu = float(np.max(np.abs(Y_LL_pu)))
    print(
        f"    [Y_LL_pu] shape={Y_LL_pu.shape}, max|Y_LL_pu|={max_abs_YLL_pu:.3e}, "
        f"cond(Y_LL_pu)≈{cond_YLL_pu:.3e}"
    )

    Z_LL_pu = np.linalg.inv(Y_LL_pu)

    A_node = np.zeros((N, N), dtype=np.float64)
    B_node = np.zeros((N, N), dtype=np.float64)

    # J_fd uses kW/kvar; the formulas below give dV_pu/dP_pu, so we scale by 0.001/MVABase.
    scale_per_kw = 1e-3 / float(S_base_MVA)
    print(f"    [units] MVABase={S_base_MVA:.3f} MVA, scale_per_kw={scale_per_kw:.3e}")
    v_mag = np.abs(v)
    print(
        f"    [dbg] FPL voltages: min|v|={v_mag.min():.6f} max|v|={v_mag.max():.6f} mean|v|={v_mag.mean():.6f}; "
        f"nonzero={np.count_nonzero(v_mag > 1e-9)}/{N}"
    )

    # Compute A,B using the node-level formulas with Z_LL in pu.
    max_abs_A_pu = 0.0
    max_abs_B_pu = 0.0
    for j in range(N):
        vj = v[j]
        vj_mag = float(np.abs(vj))
        if vj_mag <= 1e-9:
            continue
        for i in range(N):
            vi = v[i]
            num = np.conj(vj) * Z_LL_pu[j, i] / np.conj(vi)
            a_pu = float(np.real(num) / vj_mag)
            b_pu = float(np.imag(num) / vj_mag)
            max_abs_A_pu = max(max_abs_A_pu, abs(a_pu))
            max_abs_B_pu = max(max_abs_B_pu, abs(b_pu))
            # Convert to sensitivity per kW/kvar (to match finite-difference J_fd)
            A_node[j, i] = a_pu * scale_per_kw
            B_node[j, i] = b_pu * scale_per_kw

    print(
        f"    [A,B_pu] max|A_pu|={max_abs_A_pu:.3e}, max|B_pu|={max_abs_B_pu:.3e}"
    )
    print(
        f"    [A,B_kw] max|A|={np.max(np.abs(A_node)):.3e}, max|B|={np.max(np.abs(B_node)):.3e}"
    )

    J_analytic = np.concatenate([A_node, B_node], axis=1)
    return J_analytic, A_node, B_node


def compare_analytic_vs_fd_one_scenario(
    P_load_kw: float,
    Q_load_kvar: float,
    master_seed: int = 20260304,
    loadshape_name: str = "5minDayShape",
):
    """
    For a single scenario (one day, fixed P_load/Q_load), compute NODE-LEVEL
    FPL coefficients analytically vs finite-difference and compare for each
    hour (0..23).

    Both sides are node-level now:
      - Analytic: uses Z_LL and the node-level FPL formula (no phase averaging).
      - Finite-difference: injects 1 kW / 1 kvar at each node (bus.phase) and
        observes ΔV at all nodes.

    Returns:
      dict: hour -> {
          "J_fd": (N_nodes, 2*N_nodes),
          "J_analytic": (N_nodes, 2*N_nodes),
          "A_analytic": (N_nodes, N_nodes),
          "B_analytic": (N_nodes, N_nodes),
          "max_abs_diff": max_ij |J_analytic - J_fd|,
          "fro_norm_diff": Frobenius norm of J_analytic - J_fd,
          "best_fit_c": scalar minimizing ||c*J_analytic - J_fd||_F,
          "max_abs_diff_scaled": max_ij |c*J_analytic - J_fd|,
          "fro_norm_diff_scaled": ||c*J_analytic - J_fd||_F
      }
    """
    del master_seed  # kept for potential future use; not needed currently

    print("[dbg] compare_analytic_vs_fd_one_scenario: P_load_kw={} Q_load_kvar={} loadshape_name={}".format(
        P_load_kw, Q_load_kvar, loadshape_name
    ))
    dss_path = inj.compile_once()
    inj.setup_daily()
    print(f"[dbg] dss_path={dss_path}")

    node_names, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)

    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

    # No per-timestep noise for this comparison
    sigL = 0.0
    sigD = 0.0

    N = len(node_names)
    print(f"[global] N_nodes={N}")
    print(f"[dbg] first 3 node_names={node_names[:3]} last 3={node_names[-3:]}")

    results_by_hour: Dict[int, Dict] = {}

    for hour in range(24):
        print(f"\n=== Hour {hour:02d} ===")

        # Finite-difference J around load-only base (node-level)
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

        # Analytic J from Z_LL (node-level)
        J_analytic, A_analytic, B_analytic = _compute_J_analytic_node_at_hour(
            dss_path, node_names, P_load_kw, Q_load_kvar, hour, mL,
            loads_dss, dev_to_dss_load, dev_to_busph_load,
            pv_dss, pv_to_dss, pv_to_busph, sigma_load=sigL, sigma_der=sigD,
        )
        if J_analytic is None:
            print(f"[hour {hour:02d}] analytic solve did not converge; skipping")
            continue

        # Per-hour sanity: basic norms of each J
        max_abs_fd = float(np.max(np.abs(J_fd)))
        fro_fd = float(np.linalg.norm(J_fd))
        max_abs_an = float(np.max(np.abs(J_analytic)))
        fro_an = float(np.linalg.norm(J_analytic))

        print(
            f"[hour {hour:02d}] J_fd (node-level): max|J|={max_abs_fd:.3e}, ||J||_F={fro_fd:.3e}; "
            f"J_an: max|J|={max_abs_an:.3e}, ||J||_F={fro_an:.3e}"
        )

        # Column-wise sanity for first few input nodes (P columns)
        for i in range(min(3, N)):
            col_fd = J_fd[:, i]
            col_an = J_analytic[:, i]
            print(
                f"    [hour {hour:02d}] input node {node_names[i]} (P): "
                f"max|J_fd|={np.max(np.abs(col_fd)):.3e}, "
                f"max|J_an|={np.max(np.abs(col_an)):.3e}"
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
            J_fd=J_fd,
            J_analytic=J_analytic,
            A_analytic=A_analytic,
            B_analytic=B_analytic,
            max_abs_diff=max_abs,
            fro_norm_diff=fro_norm,
            best_fit_c=c_star,
            max_abs_diff_scaled=max_abs_scaled,
            fro_norm_diff_scaled=fro_norm_scaled,
        )

    # Ultimate debug summary
    hours_ok = list(results_by_hour.keys())
    print(
        f"[dbg] SUMMARY: hours_computed={len(hours_ok)}/{24} hours_ok={hours_ok[:5]}{'...' if len(hours_ok) > 5 else ''}"
    )
    if hours_ok:
        c_list = [results_by_hour[h]["best_fit_c"] for h in hours_ok]
        fro_list = [results_by_hour[h]["fro_norm_diff"] for h in hours_ok]
        fro_scaled_list = [results_by_hour[h]["fro_norm_diff_scaled"] for h in hours_ok]
        print(
            f"[dbg] SUMMARY: best_fit_c* min={min(c_list):.4e} max={max(c_list):.4e} mean={np.mean(c_list):.4e}"
        )
        print(
            f"[dbg] SUMMARY: ||ΔJ||_F (unscaled) min={min(fro_list):.4e} max={max(fro_list):.4e} mean={np.mean(fro_list):.4e}"
        )
        print(
            f"[dbg] SUMMARY: ||ΔJ||_F (scaled) min={min(fro_scaled_list):.4e} max={max(fro_scaled_list):.4e} mean={np.mean(fro_scaled_list):.4e}"
        )
    return results_by_hour

