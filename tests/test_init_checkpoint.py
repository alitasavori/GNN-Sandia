"""Init-checkpoint loading and save-only-if-improved gate (Engage-style fine-tune)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from train_da_gps_multitask_complex_voltage_gine import (
    _BASELINE_REPORT_SPECS,
    _apply_init_run_norm_stats,
    _baseline_report_specs_for,
    _checkpoint_improves_over_baseline,
    _format_baseline_metric_value,
    _load_init_checkpoint,
    _metric_delta_status,
    _print_eval_metrics_line,
    _print_vs_baseline_deltas,
    _try_load_norm_stats_from_init_run_dir,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_load_init_checkpoint_strict_false_allows_extra_keys(tmp_path: Path) -> None:
    m = _TinyModel()
    ckpt_path = tmp_path / "da_gps_multitask_best.pt"
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "hidden": 64,
            "extra_meta": 1,
        },
        ckpt_path,
    )
    m2 = _TinyModel()
    m2.lin = nn.Linear(4, 3)  # different out_features -> shape mismatch on lin.weight
    with pytest.raises(RuntimeError):
        _load_init_checkpoint(m2, ckpt_path, strict=True)

    m3 = _TinyModel()
    pack = _load_init_checkpoint(m3, ckpt_path, strict=False)
    assert pack.get("hidden") == 64
    assert torch.allclose(m3.lin.weight, m.lin.weight)
    assert torch.allclose(m3.lin.bias, m.lin.bias)


def test_checkpoint_improves_primary_without_secondary_regression() -> None:
    baseline = {
        "mae_vmag_pu": 0.0100,
        "mae_angle_deg": 1.0,
        "mse_ri_normalized": 0.0020,
    }
    better_v = {
        "mae_vmag_pu": 0.0095,
        "mae_angle_deg": 1.0,
        "mse_ri_normalized": 0.0020,
    }
    ok, _ = _checkpoint_improves_over_baseline(baseline, better_v, epsilon=1e-6)
    assert ok

    worse_angle = {
        "mae_vmag_pu": 0.0095,
        "mae_angle_deg": 1.000002,
        "mse_ri_normalized": 0.0020,
    }
    ok2, reason = _checkpoint_improves_over_baseline(baseline, worse_angle, epsilon=1e-6)
    assert not ok2
    assert "mae_angle_deg" in reason

    flat_v = {
        "mae_vmag_pu": 0.0100,
        "mae_angle_deg": 0.9,
        "mse_ri_normalized": 0.0019,
    }
    ok3, reason3 = _checkpoint_improves_over_baseline(baseline, flat_v, epsilon=1e-6)
    assert not ok3
    assert "mae_vmag_pu" in reason3


def test_metric_delta_status_lower_is_better() -> None:
    assert _metric_delta_status(-0.00007) == "improved"
    assert _metric_delta_status(0.00001) == "worse"
    assert _metric_delta_status(0.0) == "unchanged"
    assert _metric_delta_status(1e-13) == "unchanged"


def test_metric_delta_status_higher_is_better() -> None:
    assert _metric_delta_status(0.0001, higher_is_better=True) == "improved"
    assert _metric_delta_status(-0.0001, higher_is_better=True) == "worse"
    assert _metric_delta_status(0.0, higher_is_better=True) == "unchanged"


def test_baseline_report_specs_include_voltage_and_loss_keys() -> None:
    keys = {k for k, _lbl, _hib in _BASELINE_REPORT_SPECS}
    assert {
        "mae_vmag_pu",
        "mae_angle_deg",
        "mse_ri_normalized",
        "r2_vmag_mean",
        "r2_vmag_min",
        "mae_vmag_worst_node",
        "loss_tot",
        "loss_volt",
        "loss_pf",
    } <= keys


def test_baseline_report_specs_omit_pf_when_nan() -> None:
    no_pf = _baseline_report_specs_for({"loss_pf": float("nan")})
    assert all(k != "loss_pf" for k, _l, _h in no_pf)
    with_pf = _baseline_report_specs_for({"loss_pf": 0.01})
    assert any(k == "loss_pf" for k, _l, _h in with_pf)


def test_format_baseline_metric_value_precision() -> None:
    assert _format_baseline_metric_value("mae_vmag_pu", 0.003889) == "0.003889"
    assert _format_baseline_metric_value("r2_vmag_mean", 0.934) == "0.9340"
    assert _format_baseline_metric_value("loss_tot", 0.12345) == "0.1235"


def test_print_vs_baseline_deltas_all_metrics(capsys) -> None:
    baseline = {
        "mae_vmag_pu": 0.01000,
        "mae_angle_deg": 1.0,
        "mse_ri_normalized": 0.0020,
        "r2_vmag_mean": 0.90,
        "r2_vmag_min": 0.20,
        "mae_vmag_worst_node": 0.05,
        "loss_tot": 0.1000,
        "loss_volt": 0.0500,
        "loss_pf": float("nan"),
    }
    current = {
        **baseline,
        "mae_vmag_pu": 0.00950,
        "r2_vmag_mean": 0.91,
        "loss_tot": 0.0950,
    }
    _print_vs_baseline_deltas(
        epoch=1,
        baseline_val=baseline,
        baseline_test=baseline,
        val_met=current,
        test_met=baseline,
    )
    out = capsys.readouterr().out
    assert "[vs baseline] epoch 1 val |V| MAE:" in out
    assert "r2_mean:" in out
    assert "worst_mae:" in out
    assert "tot:" in out
    assert "volt:" in out
    assert "loss_pf" not in out
    assert "improved" in out
    assert "unchanged" in out


def test_print_eval_metrics_line_includes_r2_and_worst(capsys) -> None:
    met = {
        "mae_vmag_pu": 0.003889,
        "mae_angle_deg": 0.196477,
        "mse_ri_normalized": 0.010858,
        "r2_vmag_mean": 0.9340,
        "r2_vmag_min": 0.1943,
        "mae_vmag_worst_node": 0.0134,
        "loss_tot": 0.0500,
        "loss_volt": 0.0300,
        "loss_pf": float("nan"),
    }
    _print_eval_metrics_line("Val", met)
    out = capsys.readouterr().out
    assert "r2_mean=0.9340" in out
    assert "r2_min=0.1943" in out
    assert "worst_mae=0.013400" in out
    assert "tot=0.0500" in out
    assert "volt=0.0300" in out


def test_try_load_norm_stats_from_init_run_dir(tmp_path: Path) -> None:
    n_nodes = 2
    n_feat = 3
    n_reg = 1
    n_pv = 2
    torch.save(torch.zeros(1, n_feat), tmp_path / "x_mean.pt")
    torch.save(torch.ones(1, n_feat), tmp_path / "x_std.pt")
    torch.save(torch.zeros(1, n_nodes * 2), tmp_path / "y_mean.pt")
    torch.save(torch.ones(1, n_nodes * 2), tmp_path / "y_std.pt")
    torch.save(torch.zeros(1, n_reg), tmp_path / "reg_mean.pt")
    torch.save(torch.ones(1, n_reg), tmp_path / "reg_std.pt")
    torch.save(torch.zeros(1, n_pv), tmp_path / "pv_mean.pt")
    torch.save(torch.ones(1, n_pv), tmp_path / "pv_std.pt")

    loaded = _try_load_norm_stats_from_init_run_dir(
        tmp_path,
        n_node_features=n_feat,
        n_nodes=n_nodes,
        n_reg=n_reg,
        n_pv_aux=n_pv,
        reg_loss="mse",
    )
    assert loaded is not None
    assert loaded["y_mean"].shape == (1, n_nodes * 2)
    assert loaded["pv_mean"] is not None
    assert int(loaded["pv_mean"].shape[1]) == n_pv


def test_apply_init_run_norm_stats_prefers_init_dir(tmp_path: Path) -> None:
    n_nodes = 2
    n_feat = 3
    n_reg = 1
    torch.save(torch.full((1, n_feat), 9.0), tmp_path / "x_mean.pt")
    torch.save(torch.ones(1, n_feat), tmp_path / "x_std.pt")
    torch.save(torch.zeros(1, n_nodes * 2), tmp_path / "y_mean.pt")
    torch.save(torch.ones(1, n_nodes * 2), tmp_path / "y_std.pt")
    torch.save(torch.zeros(1, n_reg), tmp_path / "reg_mean.pt")
    torch.save(torch.ones(1, n_reg), tmp_path / "reg_std.pt")

    class _Args:
        recompute_norm_stats = False

    ckpt = tmp_path / "da_gps_multitask_best.pt"
    ckpt.write_bytes(b"x")
    x_mean, *_rest = _apply_init_run_norm_stats(
        init_run_dir=tmp_path,
        init_ckpt_path=ckpt,
        args=_Args(),
        n_node_features=n_feat,
        n_nodes=n_nodes,
        n_reg=n_reg,
        n_pv_aux=0,
        reg_loss="mse",
        x_mean=torch.zeros(1, n_feat),
        x_std=torch.ones(1, n_feat),
        y_mean=torch.zeros(1, n_nodes * 2),
        y_std=torch.ones(1, n_nodes * 2),
        reg_mean=torch.zeros(1, n_reg),
        reg_std=torch.ones(1, n_reg),
        pv_mean=None,
        pv_std=None,
    )
    assert float(x_mean[0, 0].item()) == 9.0
