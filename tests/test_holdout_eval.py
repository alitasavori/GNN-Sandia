"""Holdout dual-eval helpers for chunk_parent fine-tune."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from train_da_gps_multitask_complex_voltage_gine import (
    _ChunkEvalPool,
    _chunk_permutation_split,
    _eval_holdout_seed,
    _eval_pool_report_block,
    _print_eval_pool_section,
)


class _Args:
    seed = 42
    eval_holdout_seed = -1


def test_eval_holdout_seed_defaults_to_train_seed() -> None:
    assert _eval_holdout_seed(_Args()) == 42


def test_eval_holdout_seed_override() -> None:
    a = _Args()
    a.eval_holdout_seed = 99
    assert _eval_holdout_seed(a) == 99


def test_chunk_permutation_split_sizes() -> None:
    tr, va, te = _chunk_permutation_split(
        100,
        split_seed=7,
        chunk_idx=2,
        train_frac=0.8,
        val_frac=0.1,
    )
    assert len(tr) == 80
    assert len(va) == 10
    assert len(te) == 10
    assert len(np.unique(np.concatenate([tr, va, te]))) == 100


def test_chunk_permutation_split_reproducible() -> None:
    a = _chunk_permutation_split(50, split_seed=1, chunk_idx=0, train_frac=0.8, val_frac=0.1)
    b = _chunk_permutation_split(50, split_seed=1, chunk_idx=0, train_frac=0.8, val_frac=0.1)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_chunk_permutation_split_differs_by_chunk_idx() -> None:
    a = _chunk_permutation_split(50, split_seed=1, chunk_idx=0, train_frac=0.8, val_frac=0.1)
    b = _chunk_permutation_split(50, split_seed=1, chunk_idx=1, train_frac=0.8, val_frac=0.1)
    assert not np.array_equal(a[1], b[1])


def test_eval_pool_report_block_structure() -> None:
    pool = _ChunkEvalPool(
        label="nobess_holdout",
        chunk_parent=Path("/data/nobess"),
        chunk_dirs=[Path("/data/nobess/run_001")],
        idx_val_list=[np.array([0, 1])],
        idx_test_list=[np.array([2])],
        selected_ids_list=[None],
        cache_pts=[Path("/tmp/c.pt")],
        bootstrap_cache_pts=[None],
        split_seed=42,
    )
    block = _eval_pool_report_block(
        pool,
        init_baseline_val={"mae_vmag_pu": 0.01},
        init_baseline_test={"mae_vmag_pu": 0.011},
        final_val={"mae_vmag_pu": 0.009},
        final_test={"mae_vmag_pu": 0.010},
        best_epoch=3,
        epoch_history=[{"epoch": 1, "val_metrics": {}, "test_metrics": {}}],
    )
    assert block["label"] == "nobess_holdout"
    assert block["split_seed"] == 42
    assert block["n_chunks"] == 1
    assert "epoch_history" in block
    assert block["best_epoch"] == 3


def test_print_eval_pool_section_headers(capsys) -> None:
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
    _print_eval_pool_section("train_pool_eval", "epoch 1", met, met)
    out = capsys.readouterr().out
    assert "=== train_pool_eval (epoch 1) ===" in out
    assert "Val" in out
    assert "Test" in out
