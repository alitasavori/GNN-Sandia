"""
Validate PF physics-informed loss against synthetic ground truth.

Methodology
-----------
1. Hand-build a 4-bus network (slack/reg primary, reg secondary, cap bus, load bus)
   with known line R/X, one regulator branch, one capacitor bank.
2. Absolute truth: pick complex voltages V (pu), build Ybus (lines + tap + cap shunt),
   compute nodal injections S = V * conj(Y V) in kW/kVAR.
3. Decompose S into load / PV using OpenDSS convention (cap shunt in Y only):
   P_inj = P_pv - P_load,  Q_inj = -Q_pv - Q_load.
4. Call implementation helpers with matching tensors → residual ≈ 0.
5. Perturb V, tap, cap, or injections → residual grows.
6. Cap double-count test: adding cap Q to Q_inj while also in Y must inflate residual.
7. Optional real snapshot smoke test when mvagg CSV is available locally.
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
_REG_R, _REG_X = 0.01, 0.02
_REG_Z2 = _REG_R * _REG_R + _REG_X * _REG_X
_REG_G = (_REG_R / _REG_Z2) * Z_BASE
_REG_B = (-_REG_X / _REG_Z2) * Z_BASE
REG_EDGE = (0, 1, _REG_G, _REG_B, 0)  # iu=secondary, iv=primary
CAP_BANK = (2, 120.0, 0)  # node, Q_nom kVAR when fully ON, cap_col_index

TAP_TRUTH = 1.025
CAP_ON_TRUTH = 0.75

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

# Per-node known PV features (OpenDSS: P_inj = P_pv - P_load, Q_inj = -Q_pv - Q_load)
P_PV_NODE = np.array([0.0, 50.0, 30.0, 20.0])
Q_PV_NODE = np.array([0.0, 20.0, 15.0, 10.0])

NODE_FEATURE_COLS = [
    "p_load_kw",
    "q_load_kvar",
    "p_pv_kw",
    "q_pv_kvar",
]

TOL_ZERO_F32 = 0.05
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
    v_re, v_im = v_ri[:, 0], v_ri[:, 1]
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    return s_re * S_BASE_KVA, s_im * S_BASE_KVA


def _build_full_ybus_truth(
    y_re_base: np.ndarray, y_im_base: np.ndarray, tap: float, cap_on: float
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Back-solve loads so ``_assemble_pf_injections`` equals Y@V (cap in Y only)."""
    p_load = P_PV_NODE - p_net_kw
    q_load = -Q_PV_NODE - q_net_kvar
    p_inj = P_PV_NODE - p_load
    q_inj = -Q_PV_NODE - q_load
    return p_load, q_load, p_inj, q_inj


def _make_synthetic_batch(x_denorm: torch.Tensor) -> Data:
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
    y_re_base: torch.Tensor,
    y_im_base: torch.Tensor,
    *,
    q_inj_cap_extra: torch.Tensor | None = None,
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
        batch=_make_synthetic_batch(x_denorm),
        n_nodes=N_NODES,
    )
    if q_inj_cap_extra is not None:
        q_inj = q_inj + q_inj_cap_extra
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


def _test_cap_only_in_y(
    v_ri: torch.Tensor,
    tap: torch.Tensor,
    cap_on: torch.Tensor,
    x_denorm: torch.Tensor,
    y_re_base_t: torch.Tensor,
    y_im_base_t: torch.Tensor,
    res_correct: float,
) -> bool:
    """Cap shunt must appear in Y only; adding cap Q to Q_inj should inflate residual."""
    ni, q_nom, _ = CAP_BANK
    cap_q = cap_on[:, 0] * float(q_nom)
    extra = torch.zeros(1, N_NODES)
    extra[0, ni] = cap_q[0]
    res_double = float(
        _run_impl_residual(
            v_ri, tap, cap_on, x_denorm, y_re_base_t, y_im_base_t, q_inj_cap_extra=extra
        ).item()
    )
    return _check(
        "cap-only-in-Y (double-count inflates residual)",
        res_double > res_correct + TOL_GROW,
        f"{res_correct:.3e} -> {res_double:.3e}",
    )


def _test_ybase_skips_regulator_branch() -> bool:
    """Line CSV with xfmr row must be skipped when pair is in reg catalog skip set."""
    import pandas as pd
    import tempfile

    nodes = ["a.1", "b.1", "c.1"]
    n2l = {n: i for i, n in enumerate(nodes)}
    z_base = Z_BASE
    skip = {pfmod._undirected_node_pair(0, 1)}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "edges.csv"
        pd.DataFrame(
            [
                {"from_node": "a.1", "to_node": "b.1", "R_full": 0.01, "X_full": 0.02,
                 "line_name": "Transformer.reg1", "linecode": "xfmr"},
                {"from_node": "b.1", "to_node": "c.1", "R_full": 0.03, "X_full": 0.05,
                 "line_name": "Line.l1", "linecode": "abc"},
            ]
        ).to_csv(p, index=False)
        y_re, y_im = pfmod._build_ybus_pu_from_edge_csv(p, n2l, 3, z_base, skip_undirected=skip)
    # Only b-c line should be stamped; a-b reg branch skipped
    has_bc = abs(float(y_re[1, 2].item())) > 1e-9
    no_ab = abs(float(y_re[0, 1].item())) < 1e-9
    return _check("Y_base skips regulator xfmr branch", has_bc and no_ab)


def _test_slack_mask_exclusion() -> bool:
    ok_name = not pfmod._is_pf_slack_source_node("l1234567.1")
    slack1 = pfmod._is_pf_slack_source_node("sourcebus.1")
    slack2 = pfmod._is_pf_slack_source_node("_hvmv_sub_lsb.1")
    return _check("slack/source node name detection", ok_name and slack1 and slack2)


def _optional_real_snapshot_test() -> bool | None:
    """Smoke test on one OpenDSS snapshot row if mvagg nodes CSV exists locally."""
    candidates = list(REPO.glob("**/gnn_node_features_and_targets_mvagg.csv"))
    if not candidates:
        print("\n--- Optional real snapshot test ---")
        print("  [SKIP] no gnn_node_features_and_targets_mvagg.csv found")
        return None
    nodes_csv = candidates[0]
    chunk_dir = nodes_csv.parent
    edges_csv = chunk_dir / "gnn_edges_phase_static.csv"
    reg_csv = chunk_dir / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
    if not edges_csv.is_file() or not reg_csv.is_file():
        print("\n--- Optional real snapshot test ---")
        print(f"  [SKIP] missing edges or reg catalog beside {nodes_csv}")
        return None

    import pandas as pd

    print("\n--- Optional real snapshot test ---")
    print(f"  nodes: {nodes_csv}")
    idx_csv = chunk_dir / "gnn_node_index_master.csv"
    if not idx_csv.is_file():
        idx_csv = REPO / "datasets_gnn2_from pc" / "gnn_node_index_master.csv"
    if not idx_csv.is_file():
        print("  [SKIP] no gnn_node_index_master.csv")
        return None

    idx = pd.read_csv(idx_csv)
    node_to_local = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1

    hdr = pd.read_csv(nodes_csv, nrows=0).columns.tolist()
    need = ["sample_id", "node", "p_load_kw", "q_load_kvar", "p_pv_kw", "vmag_pu", "vang_deg"]
    if not all(c in hdr for c in need):
        print(f"  [SKIP] nodes CSV missing columns (need {need})")
        return None
    usecols = need + (["q_pv_kvar"] if "q_pv_kvar" in hdr else [])
    if "electrical_distance_ohm" in hdr:
        usecols.append("electrical_distance_ohm")
    sub = pd.read_csv(nodes_csv, usecols=usecols)
    sid = int(sub["sample_id"].iloc[0])
    sub = sub[sub["sample_id"] == sid]
    if sub.empty:
        print("  [SKIP] empty sample slice")
        return None

    z_base = Z_BASE
    reg_edges = pfmod._load_regulator_edges_for_pf(reg_csv, node_to_local, [], z_base)
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re_b, y_im_b = pfmod._build_ybus_pu_from_edge_csv(
        edges_csv, node_to_local, n_nodes, z_base, skip_undirected=skip
    )

    v_full = np.zeros((n_nodes, 2), dtype=np.float64)
    q_pv_col = "q_pv_kvar" if "q_pv_kvar" in sub.columns else None
    p_load = np.zeros(n_nodes)
    q_load = np.zeros(n_nodes)
    p_pv = np.zeros(n_nodes)
    q_pv = np.zeros(n_nodes)
    for _, row in sub.iterrows():
        ni = node_to_local[str(row["node"]).strip().lower()]
        ang = np.deg2rad(float(row["vang_deg"]))
        mag = float(row["vmag_pu"])
        v_full[ni, 0] = mag * np.cos(ang)
        v_full[ni, 1] = mag * np.sin(ang)
        p_load[ni] = float(row["p_load_kw"])
        q_load[ni] = float(row["q_load_kvar"])
        p_pv[ni] = float(row["p_pv_kw"])
        q_pv[ni] = float(row.get("q_pv_kvar", 0.0)) if q_pv_col else 0.0

    p_inj = p_pv - p_load
    q_inj = -q_pv - q_load
    y_re = y_re_b.unsqueeze(0)
    y_im = y_im_b.unsqueeze(0)
    v_t = torch.tensor(v_full, dtype=torch.float32).unsqueeze(0)
    p_t = torch.tensor(p_inj, dtype=torch.float32).unsqueeze(0)
    q_t = torch.tensor(q_inj, dtype=torch.float32).unsqueeze(0)

    if "electrical_distance_ohm" in sub.columns:
        mask = torch.zeros(n_nodes, dtype=torch.bool)
        for _, row in sub.iterrows():
            node = str(row["node"]).strip().lower()
            if pfmod._is_pf_slack_source_node(node):
                continue
            if float(row["electrical_distance_ohm"]) > 1e-9:
                mask[node_to_local[node]] = True
    else:
        mask = torch.ones(n_nodes, dtype=torch.bool)

    res = pfmod.nodal_power_balance_residual(v_t, p_t, q_t, y_re, y_im, mask, S_BASE_KVA)
    res_val = float(res.item())
    # Without true taps/caps this is a coarse bound, not exact PF
    ok = res_val < 1e6
    return _check(
        "real snapshot coarse residual (line-Y only, taps=1, caps off)",
        ok,
        f"residual={res_val:.4e} (expected large without controls; sanity: finite & <1e6)",
    )


def main() -> int:
    print("=" * 72)
    print("PF physics loss validation (synthetic 4-bus network)")
    print("=" * 72)

    y_re_b, y_im_b = _build_base_ybus()
    y_re_f, y_im_f = _build_full_ybus_truth(y_re_b, y_im_b, TAP_TRUTH, CAP_ON_TRUTH)

    p_net, q_net = _nodal_power_kw_kvar(V_TRUTH, y_re_f, y_im_f)
    p_load, q_load, p_inj_exp, q_inj_exp = _loads_for_exact_assembly(p_net, q_net)

    print("\n--- Network setup ---")
    print(f"  Nodes: {N_NODES}  |  S_base={S_BASE_KVA} kVA  |  Z_base={Z_BASE:.4f} ohm")
    print(f"  Tap={TAP_TRUTH}  cap_on={CAP_ON_TRUTH}")
    print(f"  Back-solved loads P_load={np.round(p_load, 2).tolist()}")

    dev = torch.device("cpu")
    y_re_base_t = torch.from_numpy(y_re_b).float().to(dev)
    y_im_base_t = torch.from_numpy(y_im_b).float().to(dev)

    x_denorm = torch.zeros(N_NODES, 4, dtype=torch.float32)
    col = {c: i for i, c in enumerate(NODE_FEATURE_COLS)}
    x_denorm[:, col["p_load_kw"]] = torch.tensor(p_load, dtype=torch.float32)
    x_denorm[:, col["q_load_kvar"]] = torch.tensor(q_load, dtype=torch.float32)
    x_denorm[:, col["p_pv_kw"]] = torch.tensor(P_PV_NODE, dtype=torch.float32)
    x_denorm[:, col["q_pv_kvar"]] = torch.tensor(Q_PV_NODE, dtype=torch.float32)

    tap = torch.tensor([[TAP_TRUTH]], dtype=torch.float32)
    cap_on = torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32)
    v_ri = torch.tensor(V_TRUTH, dtype=torch.float32)

    # --- Test 1: assembly matches OpenDSS convention ---
    p_inj_impl, q_inj_impl = pfmod._assemble_pf_injections(
        x_denorm.unsqueeze(0),
        NODE_FEATURE_COLS,
        batch=_make_synthetic_batch(x_denorm),
        n_nodes=N_NODES,
    )
    asm_ok = torch.allclose(p_inj_impl[0], torch.tensor(p_inj_exp, dtype=torch.float32), atol=1e-5) and torch.allclose(
        q_inj_impl[0], torch.tensor(q_inj_exp, dtype=torch.float32), atol=1e-5
    )
    inj_match_net = np.allclose(p_inj_exp, p_net, atol=1e-6) and np.allclose(q_inj_exp, q_net, atol=1e-6)

    print("\n--- Test 1: _assemble_pf_injections (known PV, cap in Y only) ---")
    t1a = _check("injection assembly matches hand decomposition", asm_ok)
    t1b = _check(
        "back-solved loads reproduce Y@V nodal power",
        inj_match_net,
        f"max |dP|={np.max(np.abs(p_inj_exp - p_net)):.3e}, max |dQ|={np.max(np.abs(q_inj_exp - q_net)):.3e}",
    )

    # --- Test 2: residual ~ 0 when consistent ---
    res_match = _run_impl_residual(v_ri, tap, cap_on, x_denorm, y_re_base_t, y_im_base_t)
    res_val = float(res_match.item())

    print("\n--- Test 2: residual at truth ---")
    t2a = _check("residual ~ 0 when V/tap/cap/injections consistent (float32)", res_val < TOL_ZERO_F32, f"residual={res_val:.6e}")

    y_re_t, y_im_t = pfmod._ybus_with_predicted_controls(
        y_re_base_t, y_im_base_t, reg_edges=[REG_EDGE], cap_banks=[CAP_BANK],
        tap_pu=tap, cap_on=cap_on, s_base_kva=S_BASE_KVA, batch_size=1,
    )
    p_net_impl, q_net_impl = _nodal_power_kw_kvar(V_TRUTH, y_re_t[0].numpy(), y_im_t[0].numpy())
    res_direct = pfmod.nodal_power_balance_residual(
        v_ri.unsqueeze(0),
        torch.tensor(p_net_impl, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q_net_impl, dtype=torch.float32).unsqueeze(0),
        y_re_t, y_im_t, None, S_BASE_KVA,
    )
    t2b = _check("direct Y@V injections give ~0 residual (float32)", float(res_direct.item()) < TOL_ZERO_F32)

    # --- Test 3: cap double-count ---
    print("\n--- Test 3: cap-only-in-Y ---")
    t3 = _test_cap_only_in_y(v_ri, tap, cap_on, x_denorm, y_re_base_t, y_im_base_t, res_val)

    # --- Test 4: perturbations ---
    print("\n--- Test 4: deliberate errors increase residual ---")
    v_bad = v_ri.clone()
    v_bad[3, 0] += 0.02
    res_v = float(_run_impl_residual(v_bad, tap, cap_on, x_denorm, y_re_base_t, y_im_base_t).item())
    t4a = _check("wrong voltage -> larger residual", res_v > res_val + TOL_GROW, f"{res_val:.3e} -> {res_v:.3e}")

    tap_bad = tap.clone()
    tap_bad[0, 0] = TAP_TRUTH + 0.03
    res_tap = float(_run_impl_residual(v_ri, tap_bad, cap_on, x_denorm, y_re_base_t, y_im_base_t).item())
    t4b = _check("wrong tap -> larger residual", res_tap > res_val + TOL_GROW)

    cap_bad = cap_on.clone()
    cap_bad[0, 0] = 0.2
    res_cap = float(_run_impl_residual(v_ri, tap, cap_bad, x_denorm, y_re_base_t, y_im_base_t).item())
    t4c = _check("wrong cap state -> larger residual", res_cap > res_val + TOL_GROW)

    x_bad = x_denorm.clone()
    x_bad[3, col["p_pv_kw"]] += 50.0
    res_inj = float(_run_impl_residual(v_ri, tap, cap_on, x_bad, y_re_base_t, y_im_base_t).item())
    t4d = _check("wrong PV feature -> larger residual", res_inj > res_val + TOL_GROW)

    # --- Test 5: autograd ---
    print("\n--- Test 5: autograd through predicted voltage ---")
    v_var = v_ri.clone().detach().requires_grad_(True)
    res_g = _run_impl_residual(v_var, tap, cap_on, x_denorm, y_re_base_t, y_im_base_t)
    res_g.backward()
    t5a = _check("d(residual)/dV is non-zero", v_var.grad is not None and float(v_var.grad.abs().sum()) > 0)

    tap_var = torch.tensor([[TAP_TRUTH]], dtype=torch.float32, requires_grad=True)
    cap_var = torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32, requires_grad=True)
    res_tc = _run_impl_residual(v_ri.detach(), tap_var, cap_var, x_denorm, y_re_base_t, y_im_base_t)
    res_tc.backward()
    t5b = _check("d(residual)/dtap non-zero", tap_var.grad is not None and float(tap_var.grad.abs()) > 0)
    t5c = _check("d(residual)/dcap_on non-zero", cap_var.grad is not None and float(cap_var.grad.abs()) > 0)

    # --- Test 6: Ybus stamping ---
    print("\n--- Test 6: Ybus helpers ---")
    ybus_ok = np.allclose(y_re_t[0].numpy(), y_re_f, rtol=1e-5, atol=1e-6) and np.allclose(
        y_im_t[0].numpy(), y_im_f, rtol=1e-5, atol=1e-6
    )
    t6a = _check("Ybus matches hand-stamped reference", ybus_ok)
    t6b = _test_ybase_skips_regulator_branch()
    t6c = _test_slack_mask_exclusion()

    real_ok = _optional_real_snapshot_test()

    all_pass = all(
        [
            t1a, t1b, t2a, t2b, t3, t4a, t4b, t4c, t4d, t5a, t5b, t5c, t6a, t6b, t6c,
        ]
    )
    if real_ok is not None:
        all_pass = all_pass and real_ok

    print("\n" + "=" * 72)
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — see above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
