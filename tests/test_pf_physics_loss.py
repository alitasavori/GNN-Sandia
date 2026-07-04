"""
Physics-informed PF loss tests for train_da_gps_multitask_complex_voltage_gine.

Categories
----------
A. Self-consistency (4-bus synthetic, residual ~ 0 at truth)
B. Negative / wrong-physics (must fail if regressions reintroduce bugs)
C. Recipe / API contract
D. Real OpenDSS snapshot (local loadtype_8500_dailyagg data when present)
E. Gradient smoke
F. Edge cases (batch B>1, mv mask, slack exclusion)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import train_da_gps_multitask_complex_voltage_gine as pfmod

# ---------------------------------------------------------------------------
# 4-bus synthetic network (shared fixtures)
# ---------------------------------------------------------------------------
N_NODES = 4
S_BASE_KVA = 5000.0
KV_BASE = 12.47
Z_BASE = (KV_BASE * 1000.0) ** 2 / (S_BASE_KVA * 1000.0)

LINE_EDGES = [(1, 2, 0.02, 0.04), (2, 3, 0.03, 0.05)]
_REG_R, _REG_X = 0.01, 0.02
_REG_Z2 = _REG_R * _REG_R + _REG_X * _REG_X
_REG_G = _REG_R / _REG_Z2
_REG_B = -_REG_X / _REG_Z2
REG_EDGE = (0, 1, _REG_G, _REG_B, 0)
KV_LN_SYN = KV_BASE / (3.0 ** 0.5)
V_SCALE_SYN = KV_LN_SYN * 1000.0
CAP_BANK = (2, 120.0, 0)

TAP_TRUTH = 1.025
CAP_ON_TRUTH = 0.75

V_TRUTH = np.array(
    [[1.020, 0.005], [1.010, -0.008], [0.998, -0.012], [0.985, -0.018]],
    dtype=np.float64,
)
P_PV_NODE = np.array([0.0, 50.0, 30.0, 20.0])
Q_PV_NODE = np.array([0.0, 20.0, 15.0, 10.0])

NODE_FEATURE_COLS = ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar"]

TOL_ZERO_F32 = 0.05
TOL_GROW = 1.0

DATA_DAILYAGG = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
NODES8500 = REPO / "datasets_gnn2_from pc" / "loadtype_8500" / "gnn_node_features_and_targets.csv"
HETERO_LOAD_NODES = (
    DATA_DAILYAGG / "Heterogenous GNN dataset" / "nodes" / "hetero_mv_nodes_load_transformer.csv"
)


def _stamp_line_ybus(y_re, y_im, iu, iv, rf, xf):
    z2 = rf * rf + xf * xf
    g, b = rf / z2, -xf / z2
    ylr, yli = g, b
    y_re[iu, iv] -= ylr
    y_re[iv, iu] -= ylr
    y_im[iu, iv] -= yli
    y_im[iv, iu] -= yli
    y_re[iu, iu] += ylr
    y_re[iv, iv] += ylr
    y_im[iu, iu] += yli
    y_im[iv, iv] += yli


def _build_base_ybus():
    y_re = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    y_im = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for iu, iv, r, x in LINE_EDGES:
        _stamp_line_ybus(y_re, y_im, iu, iv, r, x)
    return y_re, y_im


def _build_full_ybus_truth(y_re_base, y_im_base, tap, cap_on):
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
    y_im[ni, ni] += cap_on * (float(q_nom) * 1000.0) / (V_SCALE_SYN * V_SCALE_SYN)
    return y_re, y_im


def _nodal_power_kw_kvar(v_ri, y_re, y_im, v_scale: float = V_SCALE_SYN):
    v_re, v_im = v_ri[:, 0] * v_scale, v_ri[:, 1] * v_scale
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    return s_re / 1000.0, s_im / 1000.0


def _loads_for_exact_assembly(p_net_kw, q_net_kvar):
    p_load = P_PV_NODE - p_net_kw
    q_load = -Q_PV_NODE - q_net_kvar
    p_inj = P_PV_NODE - p_load
    q_inj = -Q_PV_NODE - q_load
    return p_load, q_load, p_inj, q_inj


def _make_synthetic_batch(x_denorm: torch.Tensor, *, batch_size: int = 1) -> Data:
    ntot = N_NODES * batch_size
    return Data(
        x=x_denorm.reshape(ntot, -1),
        batch=torch.arange(batch_size).repeat_interleave(N_NODES),
        num_graphs=batch_size,
    )


@pytest.fixture(scope="module")
def synthetic_truth():
    y_re_b, y_im_b = _build_base_ybus()
    y_re_f, y_im_f = _build_full_ybus_truth(y_re_b, y_im_b, TAP_TRUTH, CAP_ON_TRUTH)
    p_net, q_net = _nodal_power_kw_kvar(V_TRUTH, y_re_f, y_im_f)
    p_load, q_load, p_inj_exp, q_inj_exp = _loads_for_exact_assembly(p_net, q_net)

    col = {c: i for i, c in enumerate(NODE_FEATURE_COLS)}
    x_denorm = torch.zeros(N_NODES, 4, dtype=torch.float32)
    x_denorm[:, col["p_load_kw"]] = torch.tensor(p_load, dtype=torch.float32)
    x_denorm[:, col["q_load_kvar"]] = torch.tensor(q_load, dtype=torch.float32)
    x_denorm[:, col["p_pv_kw"]] = torch.tensor(P_PV_NODE, dtype=torch.float32)
    x_denorm[:, col["q_pv_kvar"]] = torch.tensor(Q_PV_NODE, dtype=torch.float32)

    return {
        "y_re_b": torch.from_numpy(y_re_b).float(),
        "y_im_b": torch.from_numpy(y_im_b).float(),
        "y_re_f": y_re_f,
        "y_im_f": y_im_f,
        "x_denorm": x_denorm,
        "p_inj_exp": p_inj_exp,
        "q_inj_exp": q_inj_exp,
        "p_net": p_net,
        "q_net": q_net,
        "v_ri": torch.tensor(V_TRUTH, dtype=torch.float32),
        "tap": torch.tensor([[TAP_TRUTH]], dtype=torch.float32),
        "cap_on": torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32),
        "col": col,
        "v_scale": torch.full((N_NODES,), V_SCALE_SYN, dtype=torch.float32),
    }


def _run_impl_residual(
    truth,
    *,
    v_ri=None,
    tap=None,
    cap_on=None,
    x_denorm=None,
    q_inj_cap_extra=None,
    batch_size: int = 1,
):
    v_ri = truth["v_ri"] if v_ri is None else v_ri
    tap = truth["tap"] if tap is None else tap
    cap_on = truth["cap_on"] if cap_on is None else cap_on
    x_denorm = truth["x_denorm"] if x_denorm is None else x_denorm
    if batch_size > 1:
        v_ri = v_ri.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        tap = tap.expand(batch_size, -1)
        cap_on = cap_on.expand(batch_size, -1)
        x_denorm = x_denorm.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    else:
        v_ri = v_ri if v_ri.dim() == 2 else v_ri

    v_scale = truth["v_scale"].unsqueeze(0).expand(batch_size, -1)
    y_re, y_im = pfmod._ybus_with_predicted_controls(
        truth["y_re_b"],
        truth["y_im_b"],
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=batch_size,
        v_scale_volts=v_scale,
    )
    p_inj, q_inj = pfmod._assemble_pf_injections(
        x_denorm if batch_size > 1 else x_denorm.unsqueeze(0),
        NODE_FEATURE_COLS,
        batch=_make_synthetic_batch(x_denorm if batch_size == 1 else x_denorm[0], batch_size=batch_size),
        n_nodes=N_NODES,
    )
    if q_inj_cap_extra is not None:
        q_inj = q_inj + q_inj_cap_extra
    v_batch = v_ri.unsqueeze(0) if v_ri.dim() == 2 else v_ri
    mask = torch.ones(N_NODES, dtype=torch.bool)
    v_scale = truth["v_scale"].unsqueeze(0).expand(v_batch.shape[0], -1)
    return pfmod.nodal_power_balance_residual(
        v_batch,
        p_inj,
        q_inj,
        y_re,
        y_im,
        mask,
        S_BASE_KVA,
        v_scale_volts=v_scale,
        huber_delta_kw=10.0,
    )


# ---------------------------------------------------------------------------
# A. Self-consistency
# ---------------------------------------------------------------------------
class TestSelfConsistency:
    def test_injection_assembly_matches_decomposition(self, synthetic_truth):
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        assert torch.allclose(p_inj[0], torch.tensor(synthetic_truth["p_inj_exp"], dtype=torch.float32), atol=1e-5)
        assert torch.allclose(q_inj[0], torch.tensor(synthetic_truth["q_inj_exp"], dtype=torch.float32), atol=1e-5)

    def test_back_solved_loads_reproduce_yv_power(self, synthetic_truth):
        assert np.allclose(synthetic_truth["p_inj_exp"], synthetic_truth["p_net"], atol=1e-6)
        assert np.allclose(synthetic_truth["q_inj_exp"], synthetic_truth["q_net"], atol=1e-6)

    def test_residual_near_zero_at_truth(self, synthetic_truth):
        res = float(_run_impl_residual(synthetic_truth).item())
        assert res < TOL_ZERO_F32, f"residual={res:.6e}"

    def test_direct_yv_injections_near_zero(self, synthetic_truth):
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_net, q_net = _nodal_power_kw_kvar(V_TRUTH, y_re[0].numpy(), y_im[0].numpy())
        res = pfmod.nodal_power_balance_residual(
            synthetic_truth["v_ri"].unsqueeze(0),
            torch.tensor(p_net, dtype=torch.float32).unsqueeze(0),
            torch.tensor(q_net, dtype=torch.float32).unsqueeze(0),
            y_re,
            y_im,
            None,
            S_BASE_KVA,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
            huber_delta_kw=10.0,
        )
        assert float(res.item()) < TOL_ZERO_F32


# ---------------------------------------------------------------------------
# B. Negative / wrong-physics
# ---------------------------------------------------------------------------
class TestWrongPhysics:
    def test_cap_double_count_inflates_residual(self, synthetic_truth):
        res_ok = float(_run_impl_residual(synthetic_truth).item())
        ni, q_nom, _ = CAP_BANK
        extra = torch.zeros(1, N_NODES)
        extra[0, ni] = synthetic_truth["cap_on"][0, 0] * float(q_nom)
        res_bad = float(
            _run_impl_residual(synthetic_truth, q_inj_cap_extra=extra).item()
        )
        assert res_bad > res_ok * 50.0
        assert res_bad > 1e-6

    def test_wrong_p_inj_sign_inflates_residual(self, synthetic_truth):
        res_ok = float(_run_impl_residual(synthetic_truth).item())
        x_bad = synthetic_truth["x_denorm"].clone()
        x_bad[:, synthetic_truth["col"]["p_pv_kw"]] *= -1.0
        res_bad = float(_run_impl_residual(synthetic_truth, x_denorm=x_bad).item())
        assert res_bad > res_ok * 50.0
        assert res_bad > 1e-6

    def test_wrong_q_inj_sign_inflates_residual(self, synthetic_truth):
        res_ok = float(_run_impl_residual(synthetic_truth).item())
        x_bad = synthetic_truth["x_denorm"].clone()
        x_bad[:, synthetic_truth["col"]["q_load_kvar"]] *= -1.0
        res_bad = float(_run_impl_residual(synthetic_truth, x_denorm=x_bad).item())
        assert res_bad > res_ok * 50.0
        assert res_bad > 1e-6

    def test_slack_excluded_despite_large_imbalance(self, synthetic_truth):
        """Slack (node 0) can violate balance; excluding it from mask should drop residual."""
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        v = synthetic_truth["v_ri"].unsqueeze(0)
        p_slack = p_inj.clone()
        q_slack = q_inj.clone()
        p_slack[0, 0] += 5e5
        q_slack[0, 0] += 5e5
        mask_all = torch.ones(N_NODES, dtype=torch.bool)
        mask_no_slack = torch.tensor([False, True, True, True])
        v_scale = synthetic_truth["v_scale"].unsqueeze(0)
        res_all = float(
            pfmod.nodal_power_balance_residual(
                v, p_slack, q_slack, y_re, y_im, mask_all, S_BASE_KVA,
                v_scale_volts=v_scale, huber_delta_kw=10.0,
            ).item()
        )
        res_mv = float(
            pfmod.nodal_power_balance_residual(
                v, p_slack, q_slack, y_re, y_im, mask_no_slack, S_BASE_KVA,
                v_scale_volts=v_scale, huber_delta_kw=10.0,
            ).item()
        )
        assert res_all > res_mv + 0.05
        assert res_mv < TOL_ZERO_F32 * 10


# ---------------------------------------------------------------------------
# C. Recipe / API contract
# ---------------------------------------------------------------------------
class TestApiContract:
    def test_assembly_ignores_pv_pred_and_meta_aux_columns(self, synthetic_truth):
        cols = NODE_FEATURE_COLS + ["pv_pred_p_kw", "meta_aux_foo"]
        wide = torch.zeros(N_NODES, len(cols), dtype=torch.float32)
        base_col = {c: i for i, c in enumerate(cols)}
        for c in NODE_FEATURE_COLS:
            wide[:, base_col[c]] = synthetic_truth["x_denorm"][:, NODE_FEATURE_COLS.index(c)]
        wide[:, base_col["pv_pred_p_kw"]] = 1e6
        wide[:, base_col["meta_aux_foo"]] = -1e6
        p_a, q_a = pfmod._assemble_pf_injections(
            wide.unsqueeze(0), cols, batch=_make_synthetic_batch(synthetic_truth["x_denorm"]), n_nodes=N_NODES
        )
        p_b, q_b = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        assert torch.allclose(p_a, p_b) and torch.allclose(q_a, q_b)

    def test_assembly_does_not_add_cap_q(self, synthetic_truth):
        """Cap reactive power must not appear in Q_inj (only in Y shunt)."""
        cols = NODE_FEATURE_COLS + ["q_cap_kvar"]
        wide = torch.zeros(N_NODES, len(cols), dtype=torch.float32)
        col = {c: i for i, c in enumerate(cols)}
        for c in NODE_FEATURE_COLS:
            wide[:, col[c]] = synthetic_truth["x_denorm"][:, NODE_FEATURE_COLS.index(c)]
        wide[:, col["q_cap_kvar"]] = 1e6
        p_a, q_a = pfmod._assemble_pf_injections(
            wide.unsqueeze(0),
            cols,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        p_b, q_b = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        assert torch.allclose(p_a, p_b) and torch.allclose(q_a, q_b)

    def test_assembly_requires_p_load_and_p_pv(self, synthetic_truth):
        with pytest.raises(ValueError, match="p_load_kw"):
            pfmod._assemble_pf_injections(
                synthetic_truth["x_denorm"].unsqueeze(0),
                ["q_load_kvar", "p_pv_kw"],
                batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
                n_nodes=N_NODES,
            )
        with pytest.raises(ValueError, match="p_pv_kw"):
            pfmod._assemble_pf_injections(
                synthetic_truth["x_denorm"].unsqueeze(0),
                ["p_load_kw", "q_load_kvar"],
                batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
                n_nodes=N_NODES,
            )

    def test_ybase_skips_regulator_xfmr_branch(self):
        import pandas as pd
        import tempfile

        nodes = ["a.1", "b.1", "c.1"]
        n2l = {n: i for i, n in enumerate(nodes)}
        skip = {pfmod._undirected_node_pair(0, 1)}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "edges.csv"
            pd.DataFrame(
                [
                    {
                        "from_node": "a.1",
                        "to_node": "b.1",
                        "R_full": 0.01,
                        "X_full": 0.02,
                        "line_name": "Transformer.reg1",
                        "linecode": "xfmr",
                    },
                    {
                        "from_node": "b.1",
                        "to_node": "c.1",
                        "R_full": 0.03,
                        "X_full": 0.05,
                        "line_name": "Line.l1",
                        "linecode": "abc",
                    },
                ]
            ).to_csv(p, index=False)
            y_re, y_im = pfmod._build_ybus_siemens_from_edge_csv(
                p, n2l, 3, skip_undirected=skip
            )
        assert abs(float(y_re[1, 2].item())) > 1e-9
        assert abs(float(y_re[0, 1].item())) < 1e-9

    @pytest.mark.skipif(not (DATA_DAILYAGG / "gnn_edges_phase_static.csv").is_file(), reason="no local edge CSV")
    def test_real_csv_ybase_has_no_regulator_branch_stamp(self):
        import pandas as pd

        idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        reg_csv = DATA_DAILYAGG / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
        reg_edges = pfmod._load_regulator_edges_for_pf(
            reg_csv, ntl, list(pfmod.TARGET_REG_COLS), Z_BASE
        )
        skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
        y_re, _ = pfmod._build_ybus_siemens_from_edge_csv(
            DATA_DAILYAGG / "gnn_edges_phase_static.csv",
            ntl,
            n_nodes,
            skip_undirected=skip,
        )
        iu, iv, _, _, _ = reg_edges[0]
        assert abs(float(y_re[iu, iv].item())) < 1e-6
        assert abs(float(y_re[iv, iu].item())) < 1e-6

    def test_capbank3_maps_all_three_phases(self):
        import pandas as pd

        pytest.importorskip("pandas")
        if not (DATA_DAILYAGG / "gnn_node_index_master.csv").is_file():
            pytest.skip("no local dailyagg index")
        idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        cap_cols = list(pfmod.TARGET_CAP_COLS)
        banks = pfmod._resolve_cap_bus_nodes(
            cap_cols,
            ntl,
            cap_nodes_csv=DATA_DAILYAGG / "capacitor_involved_nodes.csv",
            meta_csv=DATA_DAILYAGG / "gnn_sample_meta.csv",
            capacitors_dss=REPO / "8500-node" / "Capacitors.dss",
        )
        cap3 = [b for b in banks if b[2] == cap_cols.index("cap_capbank3_n_steps_on")]
        nodes = sorted({b[0] for b in cap3})
        assert len(nodes) == 3
        assert cap3[0][1] == pytest.approx(300.0, rel=1e-6)


# ---------------------------------------------------------------------------
# D. Real OpenDSS snapshot
# ---------------------------------------------------------------------------
def _real_snapshot_tensors(sample_id: int = 0):
    import pandas as pd

    idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
    ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
    n_nodes = int(idx["node_idx"].max()) + 1

    reg_cols = list(pfmod.TARGET_REG_COLS)
    cap_cols = list(pfmod.TARGET_CAP_COLS)
    reg_edges = pfmod._load_regulator_edges_for_pf(
        DATA_DAILYAGG / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv",
        ntl,
        reg_cols,
        None,
    )
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re_b, y_im_b = pfmod._build_ybus_siemens_from_edge_csv(
        DATA_DAILYAGG / "gnn_edges_phase_static.csv", ntl, n_nodes, skip_undirected=skip
    )

    meta = pd.read_csv(DATA_DAILYAGG / "gnn_sample_meta.csv")
    mrow = meta[meta["sample_id"] == sample_id].iloc[0]
    tap = torch.tensor([[float(mrow[c]) for c in reg_cols]], dtype=torch.float32)
    cap_on = torch.tensor([[float(mrow[c]) for c in cap_cols]], dtype=torch.float32)
    cap_banks = pfmod._resolve_cap_bus_nodes(
        cap_cols,
        ntl,
        cap_nodes_csv=DATA_DAILYAGG / "capacitor_involved_nodes.csv",
        meta_csv=DATA_DAILYAGG / "gnn_sample_meta.csv",
        capacitors_dss=REPO / "8500-node" / "Capacitors.dss",
    )
    from gnn2_pf_bus_kv import load_or_build_bus_kv_tensors

    v_scale_np, _, _ = load_or_build_bus_kv_tensors(
        repo=REPO, data_root=DATA_DAILYAGG, node_to_local=ntl, n_nodes=n_nodes
    )
    v_scale_t = torch.tensor(v_scale_np, dtype=torch.float32)
    y_re, y_im = pfmod._ybus_with_predicted_controls(
        y_re_b,
        y_im_b,
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=1,
        v_scale_volts=v_scale_t,
    )

    het = pd.read_csv(HETERO_LOAD_NODES)
    het = het[het["sample_id"] == sample_id]
    pv = pd.read_csv(NODES8500, usecols=["sample_id", "node", "p_pv_kw", "q_pv_kvar"])
    pv = pv[pv["sample_id"] == sample_id]
    pv_map = {
        str(r["node"]).strip().lower(): (float(r["p_pv_kw"]), float(r["q_pv_kvar"])) for _, r in pv.iterrows()
    }

    p_load = np.zeros(n_nodes)
    q_load = np.zeros(n_nodes)
    p_pv = np.zeros(n_nodes)
    q_pv = np.zeros(n_nodes)
    v = np.zeros((n_nodes, 2))
    for _, row in het.iterrows():
        ni = int(row["node_idx"])
        ang = np.deg2rad(float(row["vang_deg"]))
        mag = float(row["vmag_pu"])
        v[ni, 0] = mag * np.cos(ang)
        v[ni, 1] = mag * np.sin(ang)
        p_load[ni] = float(row["p_load_kw"])
        q_load[ni] = float(row["q_load_kvar"])
        key = str(row["node"]).strip().lower()
        if key in pv_map:
            p_pv[ni], q_pv[ni] = pv_map[key]

    p_inj = p_pv - p_load
    q_inj = -q_pv - q_load

    dist = pd.read_csv(DATA_DAILYAGG / "electrical_distance_from_substation.csv")
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    for _, row in dist.iterrows():
        node = str(row["node"]).strip().lower()
        if node not in ntl or pfmod._is_pf_slack_source_node(node):
            continue
        if float(row["electrical_distance_ohm"]) > 1e-9:
            mask[int(ntl[node])] = True

    hetero_nodes = pfmod._load_pf_hetero_node_indices(DATA_DAILYAGG)
    if hetero_nodes:
        mask = pfmod._refine_pf_mv_balance_mask(
            mask,
            ntl,
            hetero_nodes,
            y_re_b,
            y_im_b,
            exclude_interface=True,
            hetero_y_neighbors_only=True,
        )

    return {
        "v": torch.tensor(v, dtype=torch.float32).unsqueeze(0),
        "p_inj": torch.tensor(p_inj, dtype=torch.float32).unsqueeze(0),
        "q_inj": torch.tensor(q_inj, dtype=torch.float32).unsqueeze(0),
        "y_line": (y_re_b.unsqueeze(0), y_im_b.unsqueeze(0)),
        "y_full": (y_re, y_im),
        "mask": mask,
        "v_scale": v_scale_t,
    }


@pytest.mark.skipif(
    not all(
        p.is_file()
        for p in (
            DATA_DAILYAGG / "gnn_edges_phase_static.csv",
            HETERO_LOAD_NODES,
            NODES8500,
            DATA_DAILYAGG / "electrical_distance_from_substation.csv",
        )
    ),
    reason="local OpenDSS snapshot CSVs not available",
)
class TestRealSnapshot:
    """Tolerance notes (sample 0, MV mask from electrical_distance_from_substation):

    - Global MSE is dominated by a few interface/cap-feeder buses (|dP|~1e8 kW);
      use robust per-node medians and within-1-kW fractions instead.
    - Line-Y only: >75% of MV nodes have |dP|,|dQ| < 1 kW; median residual ~0.
    - Full line+true tap+cap Y: median |dP|,|dQ| similar; does not worsen vs line-only.
    """

    def test_line_y_robust_balance_fraction(self):
        snap = _real_snapshot_tensors(0)
        v, p_inj, q_inj, mask = snap["v"], snap["p_inj"], snap["q_inj"], snap["mask"]
        y_re, y_im = snap["y_line"]
        frac_ok = _fraction_within_kw(v, p_inj, q_inj, y_re, y_im, mask, snap["v_scale"], kw=1.0)
        assert frac_ok["p"] > 0.75
        assert frac_ok["q"] > 0.75

    def test_full_y_median_balance(self):
        snap = _real_snapshot_tensors(0)
        v, p_inj, q_inj, mask = snap["v"], snap["p_inj"], snap["q_inj"], snap["mask"]
        med_line = _median_abs_residual_kw(v, p_inj, q_inj, *snap["y_line"], mask, snap["v_scale"])
        med_full = _median_abs_residual_kw(v, p_inj, q_inj, *snap["y_full"], mask, snap["v_scale"])
        assert med_line["p"] < 50.0
        assert med_line["q"] < 50.0
        assert med_full["p"] <= med_line["p"] * 1.05 + 1.0
        assert med_full["q"] <= med_line["q"] * 1.05 + 1.0


def _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask, v_scale):
    v_np = v[0].numpy()
    vs = v_scale.numpy()
    yre, yim = y_re[0].numpy(), y_im[0].numpy()
    vre, vim = v_np[:, 0] * vs, v_np[:, 1] * vs
    ire = vre @ yre.T - vim @ yim.T
    iim = vre @ yim.T + vim @ yre.T
    p_kw = (vre * ire + vim * iim) / 1000.0
    q_kvar = (vim * ire - vre * iim) / 1000.0
    m = mask.numpy()
    dp = np.abs(p_inj[0].numpy() - p_kw)[m]
    dq = np.abs(q_inj[0].numpy() - q_kvar)[m]
    return dp, dq


def _median_abs_residual_kw(v, p_inj, q_inj, y_re, y_im, mask, v_scale):
    dp, dq = _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask, v_scale)
    return {"p": float(np.median(dp)), "q": float(np.median(dq))}


def _fraction_within_kw(v, p_inj, q_inj, y_re, y_im, mask, v_scale, *, kw: float):
    dp, dq = _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask, v_scale)
    return {"p": float((dp < kw).mean()), "q": float((dq < kw).mean())}


# ---------------------------------------------------------------------------
# E. Gradient smoke
# ---------------------------------------------------------------------------
class TestGradients:
    def test_grad_wrt_voltage(self, synthetic_truth):
        v_var = synthetic_truth["v_ri"].clone().detach().requires_grad_(True)
        res = _run_impl_residual(synthetic_truth, v_ri=v_var)
        res.backward()
        assert v_var.grad is not None and float(v_var.grad.abs().sum()) > 0

    def test_grad_wrt_tap_and_cap_when_not_detached(self, synthetic_truth):
        tap_var = torch.tensor([[TAP_TRUTH]], dtype=torch.float32, requires_grad=True)
        cap_var = torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32, requires_grad=True)
        res = _run_impl_residual(
            synthetic_truth,
            v_ri=synthetic_truth["v_ri"].detach(),
            tap=tap_var,
            cap_on=cap_var,
        )
        res.backward()
        assert tap_var.grad is not None and float(tap_var.grad.abs()) > 0
        assert cap_var.grad is not None and float(cap_var.grad.abs()) > 0


# ---------------------------------------------------------------------------
# H. Physical units (OpenDSS-faithful Siemens path)
# ---------------------------------------------------------------------------


def _build_synthetic_siemens_ybus_truth(y_re_base, y_im_base, tap, cap_on, v_scale_volts):
    """Truth Y in Siemens with per-bus cap stamping."""
    y_re = y_re_base.clone()
    y_im = y_im_base.clone()
    iu, iv = 0, 1
    g = _REG_R / _REG_Z2
    b = -_REG_X / _REG_Z2
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
    v_nom = float(v_scale_volts[ni])
    y_im[ni, ni] += cap_on * (float(q_nom) * 1000.0) / (v_nom * v_nom)
    return y_re, y_im


def _build_base_ybus_siemens():
    y_re = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    y_im = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for iu, iv, r, x in LINE_EDGES:
        z2 = r * r + x * x
        g, b = r / z2, -x / z2
        y_re[iu, iv] -= g
        y_re[iv, iu] -= g
        y_im[iu, iv] -= b
        y_im[iv, iu] -= b
        y_re[iu, iu] += g
        y_re[iv, iv] += g
        y_im[iu, iu] += b
        y_im[iv, iv] += b
    return y_re, y_im


class TestPhysicalUnits:
    def test_siemens_ybus_no_zbase_multiplier(self):
        import pandas as pd
        import tempfile

        nodes = ["a.1", "b.1"]
        n2l = {n: i for i, n in enumerate(nodes)}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "edges.csv"
            pd.DataFrame(
                [{"from_node": "a.1", "to_node": "b.1", "R_full": 0.02, "X_full": 0.04, "line_name": "L1", "linecode": "abc"}]
            ).to_csv(p, index=False)
            y_re_s, y_im_s = pfmod._build_ybus_siemens_from_edge_csv(p, n2l, 2)
        z2 = 0.02 * 0.02 + 0.04 * 0.04
        g_s = 0.02 / z2
        assert abs(float(y_re_s[0, 1].item()) + g_s) < 1e-9

    def test_v_scale_array_per_node(self):
        from gnn2_pf_bus_kv import kv_base_ln_v_array

        ntl = {"a.1": 0, "b.2": 1}
        kv = {"a.1": 7.2, "b.2": 7.1996}
        arr = kv_base_ln_v_array(ntl, kv, 2)
        assert arr[0] == pytest.approx(7200.0)
        assert arr[1] == pytest.approx(7199.6)

    def test_physical_residual_near_zero_at_truth(self):
        y_re_b = torch.from_numpy(_build_base_ybus_siemens()[0]).float()
        y_im_b = torch.from_numpy(_build_base_ybus_siemens()[1]).float()
        v_scale = torch.full((N_NODES,), V_SCALE_SYN, dtype=torch.float32)
        reg_g = _REG_R / _REG_Z2
        reg_b = -_REG_X / _REG_Z2
        reg_edge_s = (0, 1, reg_g, reg_b, 0)
        y_re_f, y_im_f = _build_synthetic_siemens_ybus_truth(
            y_re_b, y_im_b, TAP_TRUTH, CAP_ON_TRUTH, v_scale.numpy()
        )
        p_net, q_net = _nodal_power_kw_kvar_physical(V_TRUTH, y_re_f.numpy(), y_im_f.numpy(), v_scale.numpy())
        p_load, q_load, p_inj_exp, q_inj_exp = _loads_for_exact_assembly(p_net, q_net)
        col = {c: i for i, c in enumerate(NODE_FEATURE_COLS)}
        x_denorm = torch.zeros(N_NODES, 4, dtype=torch.float32)
        x_denorm[:, col["p_load_kw"]] = torch.tensor(p_load, dtype=torch.float32)
        x_denorm[:, col["q_load_kvar"]] = torch.tensor(q_load, dtype=torch.float32)
        x_denorm[:, col["p_pv_kw"]] = torch.tensor(P_PV_NODE, dtype=torch.float32)
        x_denorm[:, col["q_pv_kvar"]] = torch.tensor(Q_PV_NODE, dtype=torch.float32)
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            y_re_b,
            y_im_b,
            reg_edges=[reg_edge_s],
            cap_banks=[CAP_BANK],
            tap_pu=torch.tensor([[TAP_TRUTH]], dtype=torch.float32),
            cap_on=torch.tensor([[CAP_ON_TRUTH]], dtype=torch.float32),
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=v_scale,
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            x_denorm.unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(x_denorm),
            n_nodes=N_NODES,
        )
        res = pfmod.nodal_power_balance_residual(
            torch.tensor(V_TRUTH, dtype=torch.float32).unsqueeze(0),
            p_inj,
            q_inj,
            y_re,
            y_im,
            None,
            S_BASE_KVA,
            v_scale_volts=v_scale,
            huber_delta_kw=10.0,
        )
        assert float(res.item()) < TOL_ZERO_F32

    @pytest.mark.skipif(
        not (DATA_DAILYAGG / "gnn_node_index_master.csv").is_file(),
        reason="no local dailyagg index for kV cache",
    )
    def test_bus_kv_cache_roundtrip(self, tmp_path):
        from gnn2_pf_bus_kv import read_bus_kv_cache, write_bus_kv_cache

        ntl = {"l1234567.1": 0, "l7654321.2": 1}
        kv = {"l1234567.1": 7.2, "l7654321.2": 7.1996}
        v_scale = np.array([7200.0, 7199.6])
        cache = tmp_path / "bus_kv_base_by_node.csv"
        write_bus_kv_cache(cache, ntl, kv, v_scale)
        kv2, v2 = read_bus_kv_cache(cache)
        assert kv2["l1234567.1"] == pytest.approx(7.2)
        assert v2[0] == pytest.approx(7200.0)


def _nodal_power_kw_kvar_physical(v_ri, y_re, y_im, v_scale):
    vs = np.asarray(v_scale, dtype=np.float64).reshape(-1)
    v_re = v_ri[:, 0] * vs
    v_im = v_ri[:, 1] * vs
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    return s_re / 1000.0, s_im / 1000.0


# ---------------------------------------------------------------------------
# I. Flow-relative physics (substantive physics-informed mode)
# ---------------------------------------------------------------------------
class TestFlowRelativePhysics:
    def test_zero_when_v_pred_matches_label(self, synthetic_truth):
        v = synthetic_truth["v_ri"]
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        res = float(
            pfmod.nodal_power_balance_residual(
                v.unsqueeze(0),
                torch.zeros(1, N_NODES),
                torch.zeros(1, N_NODES),
                y_re,
                y_im,
                torch.ones(N_NODES, dtype=torch.bool),
                S_BASE_KVA,
                v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
                huber_delta_kw=10.0,
                loss_mode="flow_relative",
                label_ri=v.unsqueeze(0),
            ).item()
        )
        assert res < TOL_ZERO_F32

    def test_insensitive_to_p_inj_error_at_label_v(self, synthetic_truth):
        """flow_relative ignores feature P_inj errors; absolute does not."""
        v = synthetic_truth["v_ri"]
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        p_bad = p_inj.clone()
        p_bad[0, 1] += 500.0
        mask = torch.ones(N_NODES, dtype=torch.bool)
        v_scale = synthetic_truth["v_scale"].unsqueeze(0)
        res_rel = float(
            pfmod.nodal_power_balance_residual(
                v.unsqueeze(0),
                p_bad,
                q_inj,
                y_re,
                y_im,
                mask,
                S_BASE_KVA,
                v_scale_volts=v_scale,
                huber_delta_kw=10.0,
                loss_mode="flow_relative",
                label_ri=v.unsqueeze(0),
            ).item()
        )
        res_abs = float(
            pfmod.nodal_power_balance_residual(
                v.unsqueeze(0),
                p_bad,
                q_inj,
                y_re,
                y_im,
                mask,
                S_BASE_KVA,
                v_scale_volts=v_scale,
                huber_delta_kw=10.0,
                loss_mode="absolute",
            ).item()
        )
        assert res_rel < TOL_ZERO_F32
        assert res_abs > res_rel * 50.0

    def test_nonzero_when_v_pred_perturbed(self, synthetic_truth):
        v = synthetic_truth["v_ri"]
        v_bad = v.clone()
        v_bad[:, 0] += 0.05
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        res = float(
            pfmod.nodal_power_balance_residual(
                v_bad.unsqueeze(0),
                p_inj,
                q_inj,
                y_re,
                y_im,
                torch.ones(N_NODES, dtype=torch.bool),
                S_BASE_KVA,
                v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
                huber_delta_kw=10.0,
                loss_mode="flow_relative",
                label_ri=v.unsqueeze(0),
            ).item()
        )
        assert res > 1e-4

    def test_flow_relative_grad_aligns_with_voltage_error(self, synthetic_truth):
        v_lbl = synthetic_truth["v_ri"]
        v_var = v_lbl.clone().requires_grad_(True)
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        loss = pfmod.nodal_power_balance_residual(
            v_var.unsqueeze(0),
            p_inj,
            q_inj,
            y_re,
            y_im,
            torch.ones(N_NODES, dtype=torch.bool),
            S_BASE_KVA,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
            huber_delta_kw=10.0,
            loss_mode="flow_relative",
            label_ri=v_lbl.unsqueeze(0),
        )
        loss.backward()
        assert v_var.grad is not None and float(v_var.grad.abs().sum()) > 0


class TestPfAutoScale:
    def test_effective_weight_caps_ratio(self):
        pf = pfmod.PfPhysicsState(weight=1.0, auto_scale_volt=True, auto_scale_max=100.0)
        loss_v = torch.tensor(1.0)
        loss_pf = torch.tensor(1e-8)
        w = pfmod._pf_effective_weight(pf, loss_v=loss_v, loss_pf=loss_pf)
        assert float(w) == pytest.approx(100.0)

    def test_effective_weight_unity_when_balanced(self):
        pf = pfmod.PfPhysicsState(weight=0.5, auto_scale_volt=True, auto_scale_max=100.0)
        loss_v = torch.tensor(0.2)
        loss_pf = torch.tensor(0.2)
        w = pfmod._pf_effective_weight(pf, loss_v=loss_v, loss_pf=loss_pf)
        assert float(w) == pytest.approx(0.5)

    def test_flow_relative_volt_scale_raises_magnitude(self, synthetic_truth):
        v_lbl = synthetic_truth["v_ri"]
        v_bad = v_lbl.clone()
        v_bad[:, 0] += 0.05
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        raw = float(
            pfmod.nodal_power_balance_residual(
                v_bad.unsqueeze(0),
                p_inj,
                q_inj,
                y_re,
                y_im,
                torch.ones(N_NODES, dtype=torch.bool),
                S_BASE_KVA,
                v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
                huber_delta_kw=10.0,
                loss_mode="flow_relative",
                label_ri=v_lbl.unsqueeze(0),
            ).item()
        )
        y_std = torch.full((1, N_NODES, 2), 0.02, dtype=torch.float32)
        scaled = raw * pfmod._pf_flow_relative_volt_scale(y_std)
        assert raw < 1e-2
        assert scaled > raw * 100.0
        assert scaled > 1e-4

    def test_scaled_flow_relative_grad_nonzero_when_v_perturbed(self, synthetic_truth):
        v_lbl = synthetic_truth["v_ri"]
        v_var = v_lbl.clone().requires_grad_(True)
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        p_inj, q_inj = pfmod._assemble_pf_injections(
            synthetic_truth["x_denorm"].unsqueeze(0),
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        raw = pfmod.nodal_power_balance_residual(
            (v_var + 0.05).unsqueeze(0),
            p_inj,
            q_inj,
            y_re,
            y_im,
            torch.ones(N_NODES, dtype=torch.bool),
            S_BASE_KVA,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
            huber_delta_kw=10.0,
            loss_mode="flow_relative",
            label_ri=v_lbl.unsqueeze(0),
        )
        y_std = torch.full((1, N_NODES, 2), 0.02, dtype=torch.float32)
        loss = raw * pfmod._pf_flow_relative_volt_scale(y_std)
        loss.backward()
        assert v_var.grad is not None and float(v_var.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# J. Regulator CE tap expectation (heterogeneous n_classes)
# ---------------------------------------------------------------------------
class TestExpectedRegTapPu:
    def test_heterogeneous_n_classes_batch_matmul(self):
        """cv[j] is padded to max_classes; probs[j] width must match per-regulator slice."""
        batch_size = 4
        n_reg = 2
        n_classes = (19, 29)
        reg_class_values = torch.full((n_reg, max(n_classes)), float("nan"), dtype=torch.float32)
        for j, nc in enumerate(n_classes):
            reg_class_values[j, :nc] = torch.linspace(0.95, 1.05, nc)

        reg_logits = [torch.randn(batch_size, nc, requires_grad=True) for nc in n_classes]
        tap = pfmod._expected_reg_tap_pu(
            torch.zeros(batch_size, n_reg),
            reg_loss="ce",
            reg_mean=None,
            reg_std=None,
            reg_logits=reg_logits,
            reg_class_values=reg_class_values,
        )
        assert tap.shape == (batch_size, n_reg)
        for j, lg in enumerate(reg_logits):
            probs = torch.softmax(lg.float(), dim=-1)
            expected = probs @ reg_class_values[j, : n_classes[j]]
            assert torch.allclose(tap[:, j], expected, atol=1e-5)

        loss = tap.sum()
        loss.backward()
        assert all(lg.grad is not None and float(lg.grad.abs().sum()) > 0 for lg in reg_logits)


class TestScaleRobustness:
    def test_huber_truth_loss_not_dominated_by_outliers(self, synthetic_truth):
        """At exact balance, Huber+pu loss stays O(1); raw kW MSE would be unusable at IEEE scale."""
        res_huber = float(_run_impl_residual(synthetic_truth).item())
        assert res_huber < TOL_ZERO_F32

    @pytest.mark.skipif(
        not all(
            p.is_file()
            for p in (
                DATA_DAILYAGG / "gnn_edges_phase_static.csv",
                HETERO_LOAD_NODES,
                NODES8500,
                DATA_DAILYAGG / "electrical_distance_from_substation.csv",
            )
        ),
        reason="local OpenDSS snapshot CSVs not available",
    )
    def test_real_snapshot_refined_mask_residual_gap_p95_kw(self):
        """Refined MV mask should keep |r_py - r_dss| p95 in the few-kW band at label V."""
        pytest.importorskip("opendssdirect")
        from gnn2_pf_physics_verify import compare_physical_opendss, load_snapshot_state

        snap = load_snapshot_state(0, repo=REPO)
        cmp = compare_physical_opendss(snap, repo=REPO, run_opendss=True)
        if cmp.get("opendss_skipped"):
            pytest.skip(str(cmp["opendss_skipped"]))
        gap = cmp["residual_gap_stats_p"]
        assert gap["p95"] < 50.0, f"refined gap p95 too large: {gap['p95']:.4g} kW"
        assert gap["max"] < 100.0, f"refined gap max too large: {gap['max']:.4g} kW"

    @pytest.mark.skipif(
        not all(
            p.is_file()
            for p in (
                DATA_DAILYAGG / "gnn_edges_phase_static.csv",
                HETERO_LOAD_NODES,
                NODES8500,
                DATA_DAILYAGG / "electrical_distance_from_substation.csv",
            )
        ),
        reason="local OpenDSS snapshot CSVs not available",
    )
    def test_real_snapshot_huber_loss_bounded_at_truth(self):
        snap = _real_snapshot_tensors(0)
        v, p_inj, q_inj, mask = snap["v"], snap["p_inj"], snap["q_inj"], snap["mask"]
        y_re, y_im = snap["y_full"]
        res_huber = float(
            pfmod.nodal_power_balance_residual(
                v,
                p_inj,
                q_inj,
                y_re,
                y_im,
                mask,
                S_BASE_KVA,
                v_scale_volts=snap["v_scale"],
                huber_delta_kw=10.0,
            ).item()
        )
        # Raw kW Huber at truth is huge on this feeder; pu-scaled Huber must stay trainable-scale.
        assert res_huber < 1.0, f"huber loss at truth too large: {res_huber:.4e}"

    def test_batch_size_gt_one(self, synthetic_truth):
        res = float(_run_impl_residual(synthetic_truth, batch_size=2).item())
        assert res < TOL_ZERO_F32

    def test_slack_source_detection(self):
        assert pfmod._is_pf_slack_source_node("sourcebus.1")
        assert pfmod._is_pf_slack_source_node("_hvmv_sub_lsb.1")
        assert not pfmod._is_pf_slack_source_node("l1234567.1")

    def test_interface_node_detection(self):
        assert pfmod._is_pf_interface_node("regxfmr_190-8581.1")
        assert pfmod._is_pf_interface_node("190-8581.2")
        assert pfmod._is_pf_interface_node("m1142828.3")
        assert pfmod._is_pf_interface_node("p1121282.3")
        assert pfmod._is_pf_interface_node("n1136366.1")
        assert not pfmod._is_pf_interface_node("l2674047.3")

    @pytest.mark.skipif(not (DATA_DAILYAGG / "gnn_node_index_master.csv").is_file(), reason="no local dailyagg")
    def test_mv_mask_refinement_reduces_interface_nodes(self):
        import pandas as pd

        idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        dist = pd.read_csv(DATA_DAILYAGG / "electrical_distance_from_substation.csv")
        mask = torch.zeros(n_nodes, dtype=torch.bool)
        for _, row in dist.iterrows():
            node = str(row["node"]).strip().lower()
            if node not in ntl or pfmod._is_pf_slack_source_node(node):
                continue
            if float(row["electrical_distance_ohm"]) > 1e-9:
                mask[int(ntl[node])] = True
        reg_edges = pfmod._load_regulator_edges_for_pf(
            DATA_DAILYAGG / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv",
            ntl,
            list(pfmod.TARGET_REG_COLS),
            Z_BASE,
        )
        skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
        y_re, y_im = pfmod._build_ybus_siemens_from_edge_csv(
            DATA_DAILYAGG / "gnn_edges_phase_static.csv", ntl, n_nodes, skip_undirected=skip
        )
        hetero = pfmod._load_pf_hetero_node_indices(DATA_DAILYAGG)
        refined = pfmod._refine_pf_mv_balance_mask(
            mask, ntl, hetero, y_re, y_im, exclude_interface=True, hetero_y_neighbors_only=True
        )
        assert int(refined.sum().item()) < int(mask.sum().item())
        assert int(refined.sum().item()) >= 100
        idx_to_node = {int(v): k for k, v in ntl.items()}
        for li in range(n_nodes):
            if not bool(refined[li].item()):
                continue
            assert pfmod._is_pf_interface_node(idx_to_node[li]) is False
            assert int(li) in hetero

    @pytest.mark.skipif(
        not (REPO / "colab_pf_data" / "pf_balance_nodes_explicit.csv").is_file(),
        reason="no explicit PF balance node list CSV",
    )
    def test_explicit_balance_list_overrides_mask_count(self):
        import pandas as pd

        idx = pd.read_csv(REPO / "colab_pf_data" / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        list_csv = REPO / "colab_pf_data" / "pf_balance_nodes_explicit.csv"

        dist = pd.read_csv(REPO / "colab_pf_data" / "electrical_distance_from_substation.csv")
        dist_mask = torch.zeros(n_nodes, dtype=torch.bool)
        for _, row in dist.iterrows():
            node = str(row["node"]).strip().lower()
            if node not in ntl or pfmod._is_pf_slack_source_node(node):
                continue
            if float(row["electrical_distance_ohm"]) > 1e-9:
                dist_mask[int(ntl[node])] = True

        explicit_mask = pfmod._load_pf_balance_mask_from_explicit_list(list_csv, ntl, n_nodes)
        assert int(explicit_mask.sum().item()) == 185
        assert int(explicit_mask.sum().item()) < int(dist_mask.sum().item())

    def test_explicit_balance_list_prefers_node_over_bus_and_node_idx(self, tmp_path):
        import pandas as pd

        ntl = {"l3141395.3": 100, "l3141395.1": 101, "190-7361.1": 200}
        list_csv = tmp_path / "pf_balance_nodes_explicit.csv"
        pd.DataFrame(
            {"node_idx": [200], "bus": ["l3141395"], "node": ["l3141395.3"]}
        ).to_csv(list_csv, index=False)
        mask = pfmod._load_pf_balance_mask_from_explicit_list(list_csv, ntl, 300)
        assert bool(mask[100].item())
        assert not bool(mask[101].item())
        assert not bool(mask[200].item())

    def test_explicit_balance_list_prefers_bus_over_node_idx(self, tmp_path):
        import pandas as pd

        ntl = {"l3141395.3": 100, "190-7361.1": 200}
        list_csv = tmp_path / "pf_balance_nodes_explicit.csv"
        pd.DataFrame({"node_idx": [200], "bus": ["l3141395"]}).to_csv(list_csv, index=False)
        mask = pfmod._load_pf_balance_mask_from_explicit_list(list_csv, ntl, 300)
        assert bool(mask[100].item())
        assert not bool(mask[200].item())

    @pytest.mark.skipif(
        not (REPO / "datasets_gnn2_from pc/loadtype_8500_dailyagg_full_mv/gnn_node_index_full_mv.csv").is_file(),
        reason="no full_mv subgraph node index",
    )
    def test_explicit_balance_list_has_no_interface_nodes_on_full_mv(self):
        import pandas as pd

        data = REPO / "datasets_gnn2_from pc/loadtype_8500_dailyagg_full_mv"
        idx = pd.read_csv(data / "gnn_node_index_full_mv.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        list_csv = REPO / "colab_pf_data/pf_balance_nodes_explicit.csv"
        mask = pfmod._load_pf_balance_mask_from_explicit_list(list_csv, ntl, n_nodes)
        idx_to_node = {int(v): k for k, v in ntl.items()}
        assert int(mask.sum().item()) == 185
        for li in range(n_nodes):
            if not bool(mask[li].item()):
                continue
            assert not pfmod._is_pf_interface_node(idx_to_node[li])

    @pytest.mark.skipif(not NODES8500.is_file(), reason="no loadtype_8500 nodes CSV")
    def test_mv_mask_nonempty_excludes_slack(self):
        import pandas as pd

        idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1

        # Build mask the same way as training when nodes CSV carries electrical_distance_ohm.
        dist = pd.read_csv(DATA_DAILYAGG / "electrical_distance_from_substation.csv")
        mask = torch.zeros(n_nodes, dtype=torch.bool)
        for _, row in dist.iterrows():
            node = str(row["node"]).strip().lower()
            if node not in ntl or pfmod._is_pf_slack_source_node(node):
                continue
            if float(row["electrical_distance_ohm"]) > 1e-9:
                mask[int(ntl[node])] = True

        assert bool(mask.any())
        for slack in ("sourcebus.1", "_hvmv_sub_lsb.1", "hvmv_sub_48332.1"):
            if slack in ntl:
                assert not bool(mask[int(ntl[slack])].item())

    @pytest.mark.skipif(
        not (REPO / "datasets_gnn2_from pc" / "gnn_node_index_master.csv").is_file(),
        reason="no repo node index with electrical_distance_ohm",
    )
    def test_mv_mask_from_node_pe_when_mvagg_lacks_distance(self):
        import pandas as pd
        import tempfile

        idx = pd.read_csv(DATA_DAILYAGG / "gnn_node_index_master.csv")
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        pe_path = REPO / "datasets_gnn2_from pc" / "gnn_node_index_master.csv"
        if not pe_path.is_file():
            pytest.skip("no repo gnn_node_index_master with electrical_distance_ohm")

        with tempfile.TemporaryDirectory() as td:
            nodes_path = Path(td) / "gnn_node_features_and_targets_mvagg.csv"
            pd.DataFrame(
                {
                    "sample_id": [0, 0],
                    "node": ["l1234567.1", "sourcebus.1"],
                    "p_load_kw": [1.0, 0.0],
                    "q_load_kvar": [0.5, 0.0],
                    "p_pv_kw": [0.0, 0.0],
                }
            ).to_csv(nodes_path, index=False)

            distance_csv, _ = pfmod._resolve_pf_electrical_distance_csv(
                nodes_csv=nodes_path,
                node_pe_csv=pe_path,
                data_root=DATA_DAILYAGG,
                repo=REPO,
                mode="mv",
            )
            assert distance_csv == pe_path

            mask = pfmod._load_pf_balance_mask(
                nodes_path,
                ntl,
                n_nodes,
                "mv",
                distance_csv=pe_path,
            )
            assert bool(mask.any())
            if "sourcebus.1" in ntl:
                assert not bool(mask[int(ntl["sourcebus.1"])].item())

    def test_mv_mask_clear_error_without_distance_sources(self, tmp_path):
        import pandas as pd

        nodes_path = tmp_path / "nodes.csv"
        pd.DataFrame(
            {
                "sample_id": [0],
                "node": ["l1234567.1"],
                "p_load_kw": [1.0],
            }
        ).to_csv(nodes_path, index=False)
        ntl = {"l1234567.1": 0}
        with pytest.raises(ValueError, match="electrical_distance_ohm"):
            pfmod._load_pf_balance_mask(
                nodes_path,
                ntl,
                1,
                "mv",
                distance_tried=["nodes_csv=" + str(nodes_path)],
            )

    def test_explicit_balance_list_mask_count(self, tmp_path):
        import pandas as pd

        list_csv = tmp_path / "pf_balance_nodes_explicit.csv"
        pd.DataFrame({"node_idx": [1, 2, 3]}).to_csv(list_csv, index=False)
        ntl = {f"n{i}.1": i for i in range(5)}
        mask = pfmod._load_pf_balance_mask_from_explicit_list(list_csv, ntl, 5)
        assert int(mask.sum().item()) == 3

    @pytest.mark.skipif(
        not (REPO / "colab_pf_data/pf_balance_nodes_explicit.csv").is_file(),
        reason="no colab_pf_data/pf_balance_nodes_explicit.csv",
    )
    def test_colab_explicit_balance_list_has_expected_count(self):
        import pandas as pd

        list_csv = REPO / "colab_pf_data/pf_balance_nodes_explicit.csv"
        df = pd.read_csv(list_csv)
        assert "node" in df.columns
        assert "node_idx" in df.columns
        assert len(df) == 185

    def test_colab_hetero_catalog_maps_on_chunk_subgraph(self):
        import pandas as pd

        het_path = REPO / "colab_pf_data" / "Heterogenous GNN dataset" / "nodes" / "hetero_mv_nodes_load_transformer.csv"
        chunk_idx = (
            REPO
            / "datasets_gnn2_from pc/original_8500_unbalanced/run_001_scen_0000_0049_seed_20360133/gnn_node_index_master.csv"
        )
        if not het_path.is_file() or not chunk_idx.is_file():
            pytest.skip("no colab hetero catalog or chunk node index")
        het = pd.read_csv(het_path)
        assert "node" in het.columns, "colab hetero must include node names for chunk-safe mapping"
        idx = pd.read_csv(chunk_idx)
        ntl = {str(r["node"]).strip().lower(): int(r["node_idx"]) for _, r in idx.iterrows()}
        n_nodes = int(idx["node_idx"].max()) + 1
        het_root = REPO / "colab_pf_data"
        hetero = pfmod._load_pf_hetero_node_indices(het_root, ntl)
        explicit_mask = pfmod._load_pf_balance_mask_from_explicit_list(
            REPO / "colab_pf_data/pf_balance_nodes_explicit.csv", ntl, n_nodes
        )
        bad = sum(1 for li in range(n_nodes) if explicit_mask[li] and int(li) not in hetero)
        assert bad == 0, f"{bad} balance nodes not in hetero on chunk subgraph"

    def test_mv_mask_fallback_all_non_slack_warns(self, tmp_path, capsys):
        import pandas as pd

        nodes_path = tmp_path / "nodes.csv"
        pd.DataFrame(
            {
                "sample_id": [0, 0],
                "node": ["l1234567.1", "sourcebus.1"],
                "p_load_kw": [1.0, 0.0],
            }
        ).to_csv(nodes_path, index=False)
        ntl = {"l1234567.1": 0, "sourcebus.1": 1}
        mask = pfmod._load_pf_balance_mask(
            nodes_path,
            ntl,
            2,
            "mv",
            mv_fallback_all_non_slack=True,
            distance_tried=["nodes_csv=" + str(nodes_path)],
        )
        out = capsys.readouterr().out
        assert "falling back to all non-slack" in out
        assert bool(mask[0].item())
        assert not bool(mask[1].item())


# ---------------------------------------------------------------------------
# G. Sparse Y@V parity (O(E) vs dense)
# ---------------------------------------------------------------------------
def _run_residual_with_y_mode(truth, *, use_sparse_y: bool, batch_size: int = 1):
    v_ri = truth["v_ri"]
    tap = truth["tap"]
    cap_on = truth["cap_on"]
    x_denorm = truth["x_denorm"]
    if batch_size > 1:
        v_ri = v_ri.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        tap = tap.expand(batch_size, -1)
        cap_on = cap_on.expand(batch_size, -1)
        x_denorm = x_denorm.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    else:
        v_ri = v_ri if v_ri.dim() == 2 else v_ri
    v_batch = v_ri.unsqueeze(0) if v_ri.dim() == 2 else v_ri

    p_inj, q_inj = pfmod._assemble_pf_injections(
        x_denorm if batch_size > 1 else x_denorm.unsqueeze(0),
        NODE_FEATURE_COLS,
        batch=_make_synthetic_batch(x_denorm if batch_size == 1 else x_denorm[0], batch_size=batch_size),
        n_nodes=N_NODES,
    )
    mask = torch.ones(N_NODES, dtype=torch.bool)
    v_scale = truth["v_scale"].unsqueeze(0).expand(v_batch.shape[0], -1)

    if use_sparse_y:
        y_coo = pfmod._dense_y_to_coo(truth["y_re_b"], truth["y_im_b"])
        return pfmod.nodal_power_balance_residual(
            v_batch,
            p_inj,
            q_inj,
            None,
            None,
            mask,
            S_BASE_KVA,
            v_scale_volts=v_scale,
            huber_delta_kw=10.0,
            y_coo=y_coo,
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=tap,
            cap_on=cap_on,
            use_sparse_y=True,
        )

    v_scale = truth["v_scale"].unsqueeze(0).expand(batch_size, -1)
    y_re, y_im = pfmod._ybus_with_predicted_controls(
        truth["y_re_b"],
        truth["y_im_b"],
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=batch_size,
        v_scale_volts=v_scale,
    )
    return pfmod.nodal_power_balance_residual(
        v_batch,
        p_inj,
        q_inj,
        y_re,
        y_im,
        mask,
        S_BASE_KVA,
        v_scale_volts=v_scale,
        huber_delta_kw=10.0,
        use_sparse_y=False,
    )


class TestSparseYParity:
    def test_sparse_dense_yv_current_parity(self, synthetic_truth):
        v_re = synthetic_truth["v_ri"][:, 0].unsqueeze(0)
        v_im = synthetic_truth["v_ri"][:, 1].unsqueeze(0)
        y_re, y_im = pfmod._ybus_with_predicted_controls(
            synthetic_truth["y_re_b"],
            synthetic_truth["y_im_b"],
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            s_base_kva=S_BASE_KVA,
            batch_size=1,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        i_dense_re, i_dense_im = pfmod._compute_yv_current(
            v_re,
            v_im,
            Y_re=y_re[0],
            Y_im=y_im[0],
            use_sparse_y=False,
        )
        y_coo = pfmod._dense_y_to_coo(synthetic_truth["y_re_b"], synthetic_truth["y_im_b"])
        i_sparse_re, i_sparse_im = pfmod._compute_yv_current(
            v_re,
            v_im,
            y_coo=y_coo,
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=synthetic_truth["tap"],
            cap_on=synthetic_truth["cap_on"],
            use_sparse_y=True,
            s_base_kva=S_BASE_KVA,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
        )
        assert torch.allclose(i_dense_re, i_sparse_re, atol=1e-4, rtol=1e-5)
        assert torch.allclose(i_dense_im, i_sparse_im, atol=1e-4, rtol=1e-5)

    def test_sparse_dense_residual_parity_at_truth(self, synthetic_truth):
        res_dense = float(_run_residual_with_y_mode(synthetic_truth, use_sparse_y=False).item())
        res_sparse = float(_run_residual_with_y_mode(synthetic_truth, use_sparse_y=True).item())
        assert res_dense < TOL_ZERO_F32
        assert res_sparse < TOL_ZERO_F32
        assert abs(res_sparse - res_dense) < 0.05

    def test_sparse_dense_residual_parity_batch_gt_one(self, synthetic_truth):
        res_dense = float(_run_residual_with_y_mode(synthetic_truth, use_sparse_y=False, batch_size=2).item())
        res_sparse = float(_run_residual_with_y_mode(synthetic_truth, use_sparse_y=True, batch_size=2).item())
        assert res_dense < TOL_ZERO_F32
        assert res_sparse < TOL_ZERO_F32
        assert abs(res_sparse - res_dense) < 0.05

    def test_sparse_path_gradients_flow(self, synthetic_truth):
        v = synthetic_truth["v_ri"].clone().requires_grad_(True)
        tap = synthetic_truth["tap"].clone().requires_grad_(True)
        cap_on = synthetic_truth["cap_on"].clone().requires_grad_(True)
        x_denorm = synthetic_truth["x_denorm"].unsqueeze(0)
        y_coo = pfmod._dense_y_to_coo(synthetic_truth["y_re_b"], synthetic_truth["y_im_b"])
        p_inj, q_inj = pfmod._assemble_pf_injections(
            x_denorm,
            NODE_FEATURE_COLS,
            batch=_make_synthetic_batch(synthetic_truth["x_denorm"]),
            n_nodes=N_NODES,
        )
        loss = pfmod.nodal_power_balance_residual(
            v.unsqueeze(0),
            p_inj,
            q_inj,
            None,
            None,
            torch.ones(N_NODES, dtype=torch.bool),
            S_BASE_KVA,
            v_scale_volts=synthetic_truth["v_scale"].unsqueeze(0),
            huber_delta_kw=10.0,
            y_coo=y_coo,
            reg_edges=[REG_EDGE],
            cap_banks=[CAP_BANK],
            tap_pu=tap,
            cap_on=cap_on,
            use_sparse_y=True,
        )
        loss.backward()
        assert v.grad is not None and float(v.grad.abs().sum()) > 0
        assert tap.grad is not None and float(tap.grad.abs().sum()) > 0
        assert cap_on.grad is not None and float(cap_on.grad.abs().sum()) > 0

    @pytest.mark.skipif(
        not all(
            p.is_file()
            for p in (
                DATA_DAILYAGG / "gnn_edges_phase_static.csv",
                HETERO_LOAD_NODES,
                NODES8500,
            )
        ),
        reason="local snapshot CSVs not available",
    )
    def test_real_snapshot_sparse_dense_yv_parity(self):
        snap = _real_snapshot_tensors(0)
        v = snap["v"]
        y_re, y_im = snap["y_full"]
        v_re, v_im = v[:, :, 0], v[:, :, 1]
        i_dense_re, i_dense_im = pfmod._compute_yv_current(
            v_re,
            v_im,
            Y_re=y_re[0],
            Y_im=y_im[0],
            use_sparse_y=False,
        )
        coo = pfmod._dense_y_to_coo(y_re[0], y_im[0])
        i_sparse_re, i_sparse_im = pfmod._yv_from_line_coo(v_re, v_im, coo)
        assert torch.allclose(i_dense_re, i_sparse_re, atol=1e-3, rtol=1e-4)
        assert torch.allclose(i_dense_im, i_sparse_im, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# H. Regulator tap orientation vs OpenDSS + per-bus kVBase
# ---------------------------------------------------------------------------
_REG_CATALOG = DATA_DAILYAGG / "Heterogenous GNN dataset" / "edges" / "hetero_mv_edge_catalog.csv"
_REGULATOR_NODES_CSV = DATA_DAILYAGG / "regulator_involved_nodes.csv"
_BUS_KV_CACHE = DATA_DAILYAGG / "bus_kv_base_by_node.csv"


class TestRegulatorOrientation:
    @pytest.mark.skipif(not _REGULATOR_NODES_CSV.is_file(), reason="no regulator_involved_nodes.csv")
    @pytest.mark.skipif(not _REG_CATALOG.is_file(), reason="no hetero_mv_edge_catalog.csv")
    def test_catalog_from_node_is_opendss_downstream_terminal(self):
        """``from_node`` = terminal_2 (tap winding 2); ``to_node`` = terminal_1 (regxfmr)."""
        import pandas as pd

        reg_df = pd.read_csv(_REGULATOR_NODES_CSV)
        catalog = pd.read_csv(_REG_CATALOG)
        reg_rows = catalog[catalog["edge_type"].astype(str).str.strip().str.lower() == "regulator"]
        for _, row in reg_rows.iterrows():
            rname = str(row["Regulator"]).strip()
            rrow = reg_df[reg_df["Regulator"].astype(str).str.strip() == rname].iloc[0]
            t1 = str(rrow["terminal_1 node"]).strip().lower()
            t2 = str(rrow["terminal_2 node"]).strip().lower()
            assert str(row["from_node"]).strip().lower() == t2, (
                f"{rname}: from_node must be downstream terminal_2 ({t2}), got {row['from_node']!r}"
            )
            assert str(row["to_node"]).strip().lower() == t1, (
                f"{rname}: to_node must be upstream regxfmr terminal_1 ({t1}), got {row['to_node']!r}"
            )

    def test_regulator_stamp_tap_on_downstream_node_analytic(self):
        """Isolated 2-bus stamp: tap on downstream (iu) vs swapped changes branch current."""
        tap = torch.tensor([1.025], dtype=torch.float32)
        g, b = 1.0, -0.5
        v = np.array([1.02 + 0.01j, 1.00 + 0.0j], dtype=np.complex128)

        y_re = torch.zeros(1, 2, 2)
        y_im = torch.zeros(1, 2, 2)
        pfmod._stamp_reg_branch_ybus(y_re, y_im, 1, 0, g, b, tap)
        y_correct = y_re[0].detach().numpy() + 1j * y_im[0].detach().numpy()
        i_down_correct = y_correct[1, :] @ v

        y_re_s = torch.zeros(1, 2, 2)
        y_im_s = torch.zeros(1, 2, 2)
        pfmod._stamp_reg_branch_ybus(y_re_s, y_im_s, 0, 1, g, b, tap)
        y_swapped = y_re_s[0].detach().numpy() + 1j * y_im_s[0].detach().numpy()
        i_down_swapped = y_swapped[1, :] @ v

        assert abs(i_down_correct - i_down_swapped) > 0.01

    @pytest.mark.skipif(
        not all(
            p.is_file()
            for p in (
                DATA_DAILYAGG / "gnn_edges_phase_static.csv",
                HETERO_LOAD_NODES,
                NODES8500,
                _REG_CATALOG,
            )
        ),
        reason="local OpenDSS snapshot CSVs not available",
    )
    def test_catalog_orientation_opendss_residual_gap_small(self):
        """Integration: catalog orientation keeps |r_py - r_dss| p95 in few-kW band."""
        pytest.importorskip("opendssdirect")
        from gnn2_pf_physics_verify import compare_physical_opendss, load_snapshot_state

        snap = load_snapshot_state(0, repo=REPO)
        cmp = compare_physical_opendss(snap, repo=REPO, run_opendss=True)
        if cmp.get("opendss_skipped"):
            pytest.skip(str(cmp["opendss_skipped"]))
        gap = cmp["residual_gap_stats_p"]
        assert gap["p95"] < 50.0, f"OpenDSS gap p95 too large: {gap['p95']:.4g} kW"


class TestBusKvBase:
    @pytest.mark.skipif(not _BUS_KV_CACHE.is_file(), reason="no bus_kv_base_by_node.csv cache")
    def test_cache_has_per_bus_kv_no_global_fallback(self):
        from gnn2_pf_bus_kv import DEFAULT_KV_FALLBACK_LN, read_bus_kv_cache, summarize_kv_coverage

        kv_by, v_scale = read_bus_kv_cache(_BUS_KV_CACHE)
        summary = summarize_kv_coverage(kv_by)
        assert summary["n_fallback"] == 0
        assert int(summary["n_distinct_kv_ln"]) >= 2
        assert len(kv_by) == len(v_scale)

    @pytest.mark.skipif(not _BUS_KV_CACHE.is_file(), reason="no bus_kv_base_by_node.csv cache")
    def test_hetero_nodes_use_mv_kv_not_fallback(self):
        import pandas as pd

        from gnn2_pf_bus_kv import DEFAULT_KV_FALLBACK_LN, read_bus_kv_cache

        kv_by, _ = read_bus_kv_cache(_BUS_KV_CACHE)
        het = pd.read_csv(HETERO_LOAD_NODES)
        het_nodes = het["node"].astype(str).str.strip().str.lower().unique()
        mv_kv = 7.199558
        for node in het_nodes:
            kv = kv_by.get(str(node).lower())
            assert kv is not None, f"missing kV for hetero node {node!r}"
            assert kv > 1.0, f"hetero node {node!r} has non-MV kV LN={kv}"
            assert abs(kv - mv_kv) / mv_kv < 0.01, f"hetero node {node!r} kv={kv}"

    @pytest.mark.skipif(not _BUS_KV_CACHE.is_file(), reason="no bus_kv_base_by_node.csv cache")
    def test_secondary_nodes_not_assigned_mv_fallback(self):
        import pandas as pd

        from gnn2_pf_bus_kv import DEFAULT_KV_FALLBACK_LN, read_bus_kv_cache

        df = pd.read_csv(_BUS_KV_CACHE)
        secondary = df[df["kv_base_ln"] < 1.0]
        assert len(secondary) > 1000
        assert not np.any(np.abs(secondary["kv_base_ln"] - DEFAULT_KV_FALLBACK_LN) < 1e-7)

    @pytest.mark.skipif(not _BUS_KV_CACHE.is_file(), reason="no bus_kv_base_by_node.csv cache")
    def test_bus_kv_matches_opendss_for_sample_buses(self):
        """Spot-check cached kV LN against a fresh OpenDSS compile."""
        pytest.importorskip("opendssdirect")
        import pandas as pd

        from gnn2_pf_bus_kv import _bus_kv_map_all_buses, node_bus_name, read_bus_kv_cache, resolve_opendss_master

        kv_by, _ = read_bus_kv_cache(_BUS_KV_CACHE)
        df = pd.read_csv(_BUS_KV_CACHE)
        sample_nodes = df["node"].astype(str).str.lower().tolist()[::1700][:5]
        bus_kv = _bus_kv_map_all_buses(resolve_opendss_master(REPO))
        for node in sample_nodes:
            bus = node_bus_name(node)
            assert bus in bus_kv, f"OpenDSS missing bus {bus!r}"
            assert abs(kv_by[node] - bus_kv[bus]) / bus_kv[bus] < 1e-4, node
