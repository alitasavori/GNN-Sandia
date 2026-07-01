"""
Validate PF physics-informed loss against synthetic ground truth.

Methodology
-----------
1. Hand-build a 4-bus network (slack/reg primary, reg secondary, cap bus, load bus)
   with known line R/X, one regulator branch, one capacitor bank.
2. Absolute truth: pick complex voltages V (pu), build Ybus (lines + tap + cap shunt),
   compute nodal injections S = V * conj(Y V) in kW/kVAR.
3. Decompose S into load / PV / cap components matching ``_assemble_pf_injections``.
4. Call implementation helpers with matching tensors → residual ≈ 0.
5. Perturb V, tap, cap, or injections → residual grows.
6. Check autograd through voltage predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import train_da_gps_multitask_complex_voltage_gine as pfmod

# ---------------------------------------------------------------------------
# Network constants (4-bus toy feeder)
# ---------------------------------------------------------------------------
N_NODES = 4
S_BASE_KVA = 5000.0
KV_BASE = 12.47
Z_BASE = (KV_BASE * 1000.0) ** 2 / (S_BASE_KVA * 1000.0)

# Node roles: 0=reg primary (upstream), 1=reg secondary, 2=cap bus, 3=load bus
LINE_EDGES = [
    (1, 2, 0.02, 0.04),
    (2, 3, 0.03, 0.05),
]
# Regulator branch: (iu, iv, g_pu, b_pu, reg_col_index) — g,b match _load_regulator_edges_for_pf
_REG_R, _REG_X = 0.01, 0.02
_REG_Z2 = _REG_R * _REG_R + _REG_X * _REG_X
_REG_G = (_REG_R / _REG_Z2) * Z_BASE
_REG_B = (-_REG_X / _REG_Z2) * Z_BASE
REG_EDGE = (0, 1, _REG_G, _REG_B, 0)  # iu=secondary, iv=primary
CAP_BANK = (2, 120.0, 0)  # node, Q_nom kVAR when fully ON, cap_col_index

TAP_TRUTH = 1.025
CAP_ON_TRUTH = 0.75
CAP_Q_TRUTH = CAP_ON_TRUTH * CAP_BANK[1]  # 90 kVAR injected at cap bus

# Known complex voltages (pu) — not a full power-flow solve; self-consistent via Y@V
V_TRUTH = np.array(
    [
        [1.020, 0.005],
        [1.010, -0.008],
        [0.998, -0.012],
        [0.985, -0.018],
    ],
    dtype=np.float64,
)

# PV feature weights for proportional distribution (meta-aux totals applied by weight)
P_PV_FEAT = np.array([0.0, 0.50, 0.30, 0.20])
Q_PV_FEAT = np.array([0.0, 0.40, 0.35, 0.25])

# Chosen meta-aux PV totals; loads are back-solved so assembly reproduces Y@V exactly.
P_PV_TOTAL_TRUTH = 200.0
Q_PV_TOTAL_TRUTH = 80.0

META_AUX_COLS = ["pv_farm_p_post_kw", "pv_farm_q_post_kvar"]
NODE_FEATURE_COLS = [
    "p_load_kw",
    "q_load_kvar",
    "p_pv_kw",
    "q_pv_kvar",
]

TOL_ZERO_F32 = 0.05  # float32 training path; large kW magnitudes limit precision
TOL_ZERO_F64 = 1e-10
TOL_GROW = 1.0


def _stamp_line_ybus(
    y_re: np.ndarray, y_im: np.ndarray, iu: int, iv: int, rf: float, xf: float
) -> None:
    z2 = rf * rf + xf * xf
    g, b = rf / z2, -xf / z2
    ylr, yli = g * Z_BASE, b * Z_BASE
    y_re[iu, iv] -= ylr
    y_re[iv, iu] -= ylr
    y_im[iu, iv] -= yli
    y_im[iv, iu] -= yli
    y_re[iu, iu] += ylr
    y_re[iv, iv] += ylr
    y_im[iu, iu] += yli
    y_im[iv, iv] += yli


def _build_base_ybus() -> tuple[np.ndarray, np.ndarray]:
    y_re = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    y_im = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for iu, iv, r, x in LINE_EDGES:
        _stamp_line_ybus(y_re, y_im, iu, iv, r, x)
    return y_re, y_im


def _nodal_power_kw_kvar(
    v_ri: np.ndarray, y_re: np.ndarray, y_im: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference: same algebra as ``nodal_power_balance_residual`` (single graph)."""
    v_re, v_im = v_ri[:, 0], v_ri[:, 1]
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    return s_re * S_BASE_KVA, s_im * S_BASE_KVA


def _build_full_ybus_truth(
    y_re_base: np.ndarray, y_im_base: np.ndarray, tap: float, cap_on: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``_ybus_with_predicted_controls`` with scalar tap/cap (batch=1)."""
    y_re = y_re_base.copy()
    y_im = y_im_base.copy()
    iu, iv, g, b, _ = REG_EDGE
    a = float(np.clip(tap, 0.9, 1.1))
    a2 = a * a
    y_re[iu, iu] += g
    y_im[iu, iu] += b
    y_re[iv, iv] += g / a2
    y_im[iv, iv] += b / a2
    y_re[iu, iv] -= g / a
    y_re[iv, iu] -= g / a
    y_im[iu, iv] -= b / a
    y_im[iv, iu] -= b / a
    ni, q_nom, _ = CAP_BANK
    y_im[ni, ni] += cap_on * (q_nom / S_BASE_KVA)
    return y_re, y_im


def _loads_for_exact_assembly(
    p_net_kw: np.ndarray,
    q_net_kvar: np.ndarray,
    *,
    p_pv_total: float,
    q_pv_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Back-solve per-node loads so ``_assemble_pf_injections`` equals Y@V nodal power.

    p_inj[i] = w_p[i]*P_total - P_load[i] = p_net[i]
    q_inj[i] = -w_q[i]*Q_total - Q_load[i] + cap_q*1{i=cap} = q_net[i]
    """
    w_p = P_PV_FEAT / P_PV_FEAT.sum()
    w_q = Q_PV_FEAT / Q_PV_FEAT.sum()
    ni, q_nom, _ = CAP_BANK
    cap_q = CAP_ON_TRUTH * q_nom

    p_load = w_p * float(p_pv_total) - p_net_kw
    q_load = -w_q * float(q_pv_total) - q_net_kvar
    q_load[ni] += cap_q

    p_from_pv = w_p * float(p_pv_total)
    q_from_pv = w_q * float(q_pv_total)
    p_inj = p_from_pv - p_load
    q_inj = -q_from_pv - q_load
    q_inj[ni] += cap_q
    return p_load, q_load, p_inj, q_inj


def _make_synthetic_batch(x_denorm: torch.Tensor) -> Data:
    """Minimal PyG batch (1 graph, N_NODES nodes)."""
    return Data(
        x=x_denorm.reshape(1, -1),
        batch=torch.zeros(N_NODES, dtype=torch.long),
        num_graphs=1,
    )


def _run_impl_residual(
    v_ri: torch.Tensor,
    tap: torch.Tensor,
    cap_on: torch.Tensor,
    x_denorm: torch.Tensor,
    pv_totals: torch.Tensor,
    y_re_base: torch.Tensor,
    y_im_base: torch.Tensor,
) -> torch.Tensor:
    y_re, y_im = pfmod._ybus_with_predicted_controls(
        y_re_base,
        y_im_base,
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=1,
    )
    p_inj, q_inj = pfmod._assemble_pf_injections(
        x_denorm.unsqueeze(0),
        NODE_FEATURE_COLS,
        cap_banks=[CAP_BANK],
        cap_on=cap_on,
        meta_aux_cols=META_AUX_COLS,
        pv_pred_denorm=pv_totals,
        batch=_make_synthetic_batch(x_denorm),
        n_nodes=N_NODES,
    )
    mask = torch.ones(N_NODES, dtype=torch.bool)
    return pfmod.nodal_power_balance_residual(
        v_ri.unsqueeze(0), p_inj, q_inj, y_re, y_im, mask, S_BASE_KVA
    )


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return cond


def main() -> int:
    print("=" * 72)
    print("PF physics loss validation (synthetic 4-bus network)")
    print("=" * 72)

    y_re_b, y_im_b = _build_base_ybus()
    y_re_f, y_im_f = _build_full_ybus_truth(y_re_b, y_im_b, TAP_TRUTH, CAP_ON_TRUTH)

    p_net, q_net = _nodal_power_kw_kvar(V_TRUTH, y_re_f, y_im_f)
    p_load, q_load, p_inj_exp, q_inj_exp = _loads_for_exact_assembly(
        p_net,
        q_net,
        p_pv_total=P_PV_TOTAL_TRUTH,
        q_pv_total=Q_PV_TOTAL_TRUTH,
    )
    p_pv_total, q_pv_total = P_PV_TOTAL_TRUTH, Q_PV_TOTAL_TRUTH

    print("\n--- Network setup ---")
    print(f"  Nodes: {N_NODES}  |  S_base={S_BASE_KVA} kVA  |  Z_base={Z_BASE:.4f} ohm")
    print(f"  Lines: {LINE_EDGES}")
    print(f"  Reg edge 0-1: R={_REG_R}, X={_REG_X}, g_pu={_REG_G:.3f}, tap={TAP_TRUTH}")
    print(f"  Cap at node {CAP_BANK[0]}: Q_nom={CAP_BANK[1]} kVAR, cap_on={CAP_ON_TRUTH}")
    print(f"  Meta PV totals: P={p_pv_total:.3f} kW, Q={q_pv_total:.3f} kVAR")
    print(f"  Back-solved loads P_load={np.round(p_load, 2).tolist()}")
    print(f"  Back-solved loads Q_load={np.round(q_load, 2).tolist()}")

    print("\n--- Truth (Y@V nodal power, kW / kVAR) ---")
    for i in range(N_NODES):
        print(
            f"  bus {i}: V=({V_TRUTH[i,0]:.4f}, {V_TRUTH[i,1]:+.4f}) j  "
            f"P_net={p_net[i]:+8.3f}  Q_net={q_net[i]:+8.3f}"
        )

    print("\n--- Expected injections from assembly formula ---")
    for i in range(N_NODES):
        print(
            f"  bus {i}: P_inj={p_inj_exp[i]:+8.3f}  Q_inj={q_inj_exp[i]:+8.3f}  "
            f"(dP={p_inj_exp[i]-p_net[i]:+.2e}, dQ={q_inj_exp[i]-q_net[i]:+.2e})"
        )

    # Build tensors for implementation path
    dev = torch.device("cpu")
    y_re_base_t = torch.from_numpy(y_re_b).float().to(dev)
    y_im_base_t = torch.from_numpy(y_im_b).float().to(dev)

    x_denorm = torch.zeros(N_NODES, 4, dtype=torch.float32)
    col = {c: i for i, c in enumerate(NODE_FEATURE_COLS)}
    x_denorm[:, col["p_load_kw"]] = torch.tensor(p_load, dtype=torch.float32)
    x_denorm[:, col["q_load_kvar"]] = torch.tensor(q_load, dtype=torch.float32)
    x_denorm[:, col["p_pv_kw"]] = torch.tensor(P_PV_FEAT, dtype=torch.float32)
    x_denorm[:, col["q_pv_kvar"]] = torch.tensor(Q_PV_FEAT, dtype=torch.float32)

    pv_totals = torch.tensor([[p_pv_total, q_pv_total]], dtype=torch.float32)
    tap = torch.tensor([[TAP_TRUTH]], dtype=torch.float32)
    cap_on = torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32)
    v_ri = torch.tensor(V_TRUTH, dtype=torch.float32)

    # --- Test 1: assembly matches hand decomposition ---
    p_inj_impl, q_inj_impl = pfmod._assemble_pf_injections(
        x_denorm.unsqueeze(0),
        NODE_FEATURE_COLS,
        cap_banks=[CAP_BANK],
        cap_on=cap_on,
        meta_aux_cols=META_AUX_COLS,
        pv_pred_denorm=pv_totals,
        batch=_make_synthetic_batch(x_denorm),
        n_nodes=N_NODES,
    )
    asm_ok = torch.allclose(p_inj_impl[0], torch.tensor(p_inj_exp, dtype=torch.float32), atol=1e-5) and torch.allclose(
        q_inj_impl[0], torch.tensor(q_inj_exp, dtype=torch.float32), atol=1e-5
    )
    inj_match_net = np.allclose(p_inj_exp, p_net, atol=1e-6) and np.allclose(q_inj_exp, q_net, atol=1e-6)

    print("\n--- Test 1: _assemble_pf_injections ---")
    _check("injection assembly matches hand decomposition", asm_ok)
    _check(
        "back-solved loads reproduce Y@V nodal power",
        inj_match_net,
        f"max |dP|={np.max(np.abs(p_inj_exp - p_net)):.3e}, max |dQ|={np.max(np.abs(q_inj_exp - q_net)):.3e}",
    )
    if not asm_ok:
        print("    impl P:", p_inj_impl[0].tolist())
        print("    impl Q:", q_inj_impl[0].tolist())

    y_re_t, y_im_t = pfmod._ybus_with_predicted_controls(
        y_re_base_t,
        y_im_base_t,
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=1,
    )

    # --- Test 2: residual ~ 0 when consistent ---
    res_match = _run_impl_residual(v_ri, tap, cap_on, x_denorm, pv_totals, y_re_base_t, y_im_base_t)
    res_val = float(res_match.item())

    print("\n--- Test 2: residual at truth ---")
    _check("residual ~ 0 when V/tap/cap/injections consistent (float32)", res_val < TOL_ZERO_F32, f"residual={res_val:.6e}")

    # Direct nodal residual with truth P/Q from Y@V (implementation Ybus)
    p_net_impl, q_net_impl = _nodal_power_kw_kvar(V_TRUTH, y_re_t[0].numpy(), y_im_t[0].numpy())
    p_net_t = torch.tensor(p_net_impl, dtype=torch.float32).unsqueeze(0)
    q_net_t = torch.tensor(q_net_impl, dtype=torch.float32).unsqueeze(0)
    res_direct = pfmod.nodal_power_balance_residual(
        v_ri.unsqueeze(0), p_net_t, q_net_t, y_re_t, y_im_t, None, S_BASE_KVA
    )
    res_direct_val = float(res_direct.item())
    _check(
        "direct Y@V injections give ~0 residual (float32)",
        res_direct_val < TOL_ZERO_F32,
        f"residual={res_direct_val:.6e}",
    )

    # float64: confirms formula is exact aside from numerics
    y_re_f64 = y_re_base_t.double()
    y_im_f64 = y_im_base_t.double()
    tap_f64 = tap.double()
    cap_f64 = cap_on.double()
    y_re_f64b, y_im_f64b = pfmod._ybus_with_predicted_controls(
        y_re_f64, y_im_f64, reg_edges=[REG_EDGE], cap_banks=[CAP_BANK],
        tap_pu=tap_f64, cap_on=cap_f64, s_base_kva=S_BASE_KVA, batch_size=1,
    )
    p64, q64 = _nodal_power_kw_kvar(V_TRUTH, y_re_f64b[0].numpy(), y_im_f64b[0].numpy())
    v64 = torch.tensor(V_TRUTH, dtype=torch.float64).unsqueeze(0)
    res_f64 = pfmod.nodal_power_balance_residual(
        v64,
        torch.tensor(p64, dtype=torch.float64).unsqueeze(0),
        torch.tensor(q64, dtype=torch.float64).unsqueeze(0),
        y_re_f64b, y_im_f64b, None, S_BASE_KVA,
    )
    res_f64_val = float(res_f64.item())
    _check(
        "direct Y@V injections give ~0 residual (float64)",
        res_f64_val < TOL_ZERO_F64,
        f"residual={res_f64_val:.6e}",
    )

    # --- Test 3: perturbations increase residual ---
    print("\n--- Test 3: deliberate errors increase residual ---")
    v_bad = v_ri.clone()
    v_bad[3, 0] += 0.02
    res_v = float(_run_impl_residual(v_bad, tap, cap_on, x_denorm, pv_totals, y_re_base_t, y_im_base_t).item())
    _check("wrong voltage -> larger residual", res_v > res_val + TOL_GROW, f"{res_val:.3e} -> {res_v:.3e}")

    tap_bad = tap.clone()
    tap_bad[0, 0] = TAP_TRUTH + 0.03
    res_tap = float(
        _run_impl_residual(v_ri, tap_bad, cap_on, x_denorm, pv_totals, y_re_base_t, y_im_base_t).item()
    )
    _check("wrong tap -> larger residual", res_tap > res_val + TOL_GROW, f"{res_val:.3e} -> {res_tap:.3e}")

    cap_bad = cap_on.clone()
    cap_bad[0, 0] = 0.2
    res_cap = float(
        _run_impl_residual(v_ri, tap, cap_bad, x_denorm, pv_totals, y_re_base_t, y_im_base_t).item()
    )
    _check("wrong cap state -> larger residual", res_cap > res_val + TOL_GROW, f"{res_val:.3e} -> {res_cap:.3e}")

    pv_bad = pv_totals.clone()
    pv_bad[0, 0] += 50.0
    res_inj = float(
        _run_impl_residual(v_ri, tap, cap_on, x_denorm, pv_bad, y_re_base_t, y_im_base_t).item()
    )
    _check("wrong PV injection -> larger residual", res_inj > res_val + TOL_GROW, f"{res_val:.3e} -> {res_inj:.3e}")

    # --- Test 4: gradients through voltage ---
    print("\n--- Test 4: autograd through predicted voltage ---")
    v_var = v_ri.clone().detach().requires_grad_(True)
    res_g = _run_impl_residual(v_var, tap, cap_on, x_denorm, pv_totals, y_re_base_t, y_im_base_t)
    res_g.backward()
    grad_ok = v_var.grad is not None and float(v_var.grad.abs().sum()) > 0
    _check("d(residual)/dV is non-zero", grad_ok, f"|grad|_1={float(v_var.grad.abs().sum()) if v_var.grad is not None else 0:.4e}")

    # tap/cap without detach should also get gradients
    tap_var = torch.tensor([[TAP_TRUTH]], dtype=torch.float32, requires_grad=True)
    cap_var = torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32, requires_grad=True)
    res_tc = _run_impl_residual(v_ri.detach(), tap_var, cap_var, x_denorm, pv_totals, y_re_base_t, y_im_base_t)
    res_tc.backward()
    tap_grad = tap_var.grad is not None and float(tap_var.grad.abs()) > 0
    cap_grad = cap_var.grad is not None and float(cap_var.grad.abs()) > 0
    _check("d(residual)/dtap non-zero", tap_grad)
    _check("d(residual)/dcap_on non-zero", cap_grad)

    # --- Test 5: Ybus stamping vs hand reference ---
    print("\n--- Test 5: _ybus_with_predicted_controls vs hand stamp ---")
    y_re_impl, y_im_impl = pfmod._ybus_with_predicted_controls(
        y_re_base_t,
        y_im_base_t,
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=1,
    )
    ybus_ok = np.allclose(y_re_impl[0].numpy(), y_re_f, rtol=1e-5, atol=1e-6) and np.allclose(
        y_im_impl[0].numpy(), y_im_f, rtol=1e-5, atol=1e-6
    )
    _check("Ybus matches hand-stamped reference", ybus_ok)

    # --- Summary ---
    print("\n" + "=" * 72)
    print("Methodology summary")
    print("=" * 72)
    print(
        """
  Truth path: fixed V (pu), build Y (lines + reg tap + cap shunt), compute
    P_kw = Re(V * conj(YV)) * S_base,  Q_kvar = Im(...) * S_base.
  Implementation path: same V/tap/cap fed to _ybus_with_predicted_controls and
    _assemble_pf_injections (loads + distributed PV meta-aux + cap Q).
  Consistency requires injection assembly to equal Y@V nodal power; residual
    r_P = P_inj - P_YV, r_Q = Q_inj - Q_YV should vanish when all match.

  Note: loads are back-solved from Y@V so the assembly model can represent the
  synthetic truth exactly. This validates implementation algebra, not OpenDSS.
"""
    )

    decomp_ok = inj_match_net
    all_pass = (
        asm_ok
        and decomp_ok
        and res_val < TOL_ZERO_F32
        and res_direct_val < TOL_ZERO_F32
        and res_f64_val < TOL_ZERO_F64
        and res_v > res_val + TOL_GROW
        and res_tap > res_val + TOL_GROW
        and res_cap > res_val + TOL_GROW
        and res_inj > res_val + TOL_GROW
        and grad_ok
        and tap_grad
        and cap_grad
        and ybus_ok
    )
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — see above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
