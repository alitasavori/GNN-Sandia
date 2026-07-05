"""Init-checkpoint loading and save-only-if-improved gate (Engage-style fine-tune)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from train_da_gps_multitask_complex_voltage_gine import (
    _checkpoint_improves_over_baseline,
    _load_init_checkpoint,
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
