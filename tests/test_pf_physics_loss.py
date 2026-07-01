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
_REG_G = (_REG_R / _REG_Z2) * Z_BASE
_REG_B = (-_REG_X / _REG_Z2) * Z_BASE
REG_EDGE = (0, 1, _REG_G, _REG_B, 0)
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
    ylr, yli = g * Z_BASE, b * Z_BASE
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
    y_im[ni, ni] += cap_on * (q_nom / S_BASE_KVA)
    return y_re, y_im


def _nodal_power_kw_kvar(v_ri, y_re, y_im):
    v_re, v_im = v_ri[:, 0], v_ri[:, 1]
    i_re = v_re @ y_re.T - v_im @ y_im.T
    i_im = v_re @ y_im.T + v_im @ y_re.T
    s_re = v_re * i_re + v_im * i_im
    s_im = v_im * i_re - v_re * i_im
    return s_re * S_BASE_KVA, s_im * S_BASE_KVA


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

    y_re, y_im = pfmod._ybus_with_predicted_controls(
        truth["y_re_b"],
        truth["y_im_b"],
        reg_edges=[REG_EDGE],
        cap_banks=[CAP_BANK],
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=batch_size,
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
    return pfmod.nodal_power_balance_residual(
        v_batch, p_inj, q_inj, y_re, y_im, mask, S_BASE_KVA
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
        assert res_bad > res_ok + TOL_GROW

    def test_wrong_p_inj_sign_inflates_residual(self, synthetic_truth):
        res_ok = float(_run_impl_residual(synthetic_truth).item())
        x_bad = synthetic_truth["x_denorm"].clone()
        x_bad[:, synthetic_truth["col"]["p_pv_kw"]] *= -1.0
        res_bad = float(_run_impl_residual(synthetic_truth, x_denorm=x_bad).item())
        assert res_bad > res_ok + TOL_GROW

    def test_wrong_q_inj_sign_inflates_residual(self, synthetic_truth):
        res_ok = float(_run_impl_residual(synthetic_truth).item())
        x_bad = synthetic_truth["x_denorm"].clone()
        x_bad[:, synthetic_truth["col"]["q_load_kvar"]] *= -1.0
        res_bad = float(_run_impl_residual(synthetic_truth, x_denorm=x_bad).item())
        assert res_bad > res_ok + TOL_GROW

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
        res_all = float(
            pfmod.nodal_power_balance_residual(v, p_slack, q_slack, y_re, y_im, mask_all, S_BASE_KVA).item()
        )
        res_mv = float(
            pfmod.nodal_power_balance_residual(v, p_slack, q_slack, y_re, y_im, mask_no_slack, S_BASE_KVA).item()
        )
        assert res_all > res_mv + TOL_GROW
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
            y_re, y_im = pfmod._build_ybus_pu_from_edge_csv(
                p, n2l, 3, Z_BASE, skip_undirected=skip
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
        y_re, _ = pfmod._build_ybus_pu_from_edge_csv(
            DATA_DAILYAGG / "gnn_edges_phase_static.csv",
            ntl,
            n_nodes,
            Z_BASE,
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
        Z_BASE,
    )
    skip = {pfmod._undirected_node_pair(iu, iv) for iu, iv, _, _, _ in reg_edges}
    y_re_b, y_im_b = pfmod._build_ybus_pu_from_edge_csv(
        DATA_DAILYAGG / "gnn_edges_phase_static.csv", ntl, n_nodes, Z_BASE, skip_undirected=skip
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
    y_re, y_im = pfmod._ybus_with_predicted_controls(
        y_re_b,
        y_im_b,
        reg_edges=reg_edges,
        cap_banks=cap_banks,
        tap_pu=tap,
        cap_on=cap_on,
        s_base_kva=S_BASE_KVA,
        batch_size=1,
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

    return {
        "v": torch.tensor(v, dtype=torch.float32).unsqueeze(0),
        "p_inj": torch.tensor(p_inj, dtype=torch.float32).unsqueeze(0),
        "q_inj": torch.tensor(q_inj, dtype=torch.float32).unsqueeze(0),
        "y_line": (y_re_b.unsqueeze(0), y_im_b.unsqueeze(0)),
        "y_full": (y_re, y_im),
        "mask": mask,
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
        frac_ok = _fraction_within_kw(v, p_inj, q_inj, y_re, y_im, mask, kw=1.0)
        assert frac_ok["p"] > 0.75
        assert frac_ok["q"] > 0.75

    def test_full_y_median_balance(self):
        snap = _real_snapshot_tensors(0)
        v, p_inj, q_inj, mask = snap["v"], snap["p_inj"], snap["q_inj"], snap["mask"]
        med_line = _median_abs_residual_kw(v, p_inj, q_inj, *snap["y_line"], mask)
        med_full = _median_abs_residual_kw(v, p_inj, q_inj, *snap["y_full"], mask)
        assert med_line["p"] < 50.0
        assert med_line["q"] < 50.0
        assert med_full["p"] <= med_line["p"] * 1.05 + 1.0
        assert med_full["q"] <= med_line["q"] * 1.05 + 1.0


def _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask):
    v_np = v[0].numpy()
    yre, yim = y_re[0].numpy(), y_im[0].numpy()
    vre, vim = v_np[:, 0], v_np[:, 1]
    ire = vre @ yre.T - vim @ yim.T
    iim = vre @ yim.T + vim @ yre.T
    p_kw = (vre * ire + vim * iim) * S_BASE_KVA
    q_kvar = (vim * ire - vre * iim) * S_BASE_KVA
    m = mask.numpy()
    dp = np.abs(p_inj[0].numpy() - p_kw)[m]
    dq = np.abs(q_inj[0].numpy() - q_kvar)[m]
    return dp, dq


def _median_abs_residual_kw(v, p_inj, q_inj, y_re, y_im, mask):
    dp, dq = _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask)
    return {"p": float(np.median(dp)), "q": float(np.median(dq))}


def _fraction_within_kw(v, p_inj, q_inj, y_re, y_im, mask, *, kw: float):
    dp, dq = _nodal_residual_kw(v, p_inj, q_inj, y_re, y_im, mask)
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
# G. Regulator CE tap expectation (heterogeneous n_classes)
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


# ---------------------------------------------------------------------------
# F. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_batch_size_gt_one(self, synthetic_truth):
        res = float(_run_impl_residual(synthetic_truth, batch_size=2).item())
        assert res < TOL_ZERO_F32

    def test_slack_source_detection(self):
        assert pfmod._is_pf_slack_source_node("sourcebus.1")
        assert pfmod._is_pf_slack_source_node("_hvmv_sub_lsb.1")
        assert not pfmod._is_pf_slack_source_node("l1234567.1")

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
