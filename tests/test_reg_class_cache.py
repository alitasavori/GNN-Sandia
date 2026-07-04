"""Regulator CE class tables and chunk-cache keying (smoke vs full chunk sets)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from train_da_gps_multitask_complex_voltage_gine import (
    _build_reg_class_tables,
    _chunk_cache_path,
    _encode_reg_class_indices,
    _reg_ce_targets_in_range,
    _reg_class_tables_digest,
    _validate_reg_ce_targets,
)


def test_reg_class_digest_differs_for_subset_vs_union():
    reg_cols = ["reg_a", "reg_b"]
    # Single chunk: taps 0.95, 1.0 only
    raw_one = np.array([[0.95, 1.0], [1.0, 1.0]], dtype=np.float64)
    # All chunks: extra tap 1.05 on reg_a
    raw_all = np.vstack([raw_one, np.array([[1.05, 1.0]], dtype=np.float64)])

    tab_one = _build_reg_class_tables(reg_cols, raw_one)
    tab_all = _build_reg_class_tables(reg_cols, raw_all)
    assert tab_one[0]["n_classes"] == 2
    assert tab_all[0]["n_classes"] == 3
    assert _reg_class_tables_digest(tab_one) != _reg_class_tables_digest(tab_all)


def test_encode_and_validate_reg_ce_targets():
    reg_cols = ["reg_a"]
    raw = np.array([[0.95], [1.0], [1.05]], dtype=np.float64)
    tables = _build_reg_class_tables(reg_cols, raw)
    idx = _encode_reg_class_indices(raw, tables)
    y = torch.from_numpy(idx)
    assert _reg_ce_targets_in_range(y, tables)

    logits = [torch.zeros(2, tables[0]["n_classes"])]
    _validate_reg_ce_targets(y, logits, reg_class_tables=tables)

    bad = y.clone()
    bad[0, 0] = tables[0]["n_classes"]
    assert not _reg_ce_targets_in_range(bad, tables)
    with pytest.raises(ValueError, match="out of range"):
        _validate_reg_ce_targets(bad, logits, reg_class_tables=tables)


def test_chunk_cache_path_includes_reg_class_digest():
    p = _chunk_cache_path(
        Path("/tmp/cache"),
        "run_001",
        1.0,
        42,
        0,
        feat_slug="nobess",
        reg_slug="regce",
        reg_classes_digest="abc123def0",
        meta_aux_slug="deadbeef",
    )
    assert "__regce__rcabc123def0__" in p.name


def test_stale_indices_fail_range_check_for_narrower_tables():
    reg_cols = ["reg_a"]
    raw_all = np.array([[0.95], [1.0], [1.05]], dtype=np.float64)
    tables_all = _build_reg_class_tables(reg_cols, raw_all)
    idx_all = _encode_reg_class_indices(raw_all, tables_all)
    y_all = torch.from_numpy(idx_all)

    tables_one = _build_reg_class_tables(reg_cols, np.array([[0.95], [1.0]], dtype=np.float64))
    assert tables_one[0]["n_classes"] == 2
    assert _reg_ce_targets_in_range(y_all, tables_all)
    assert not _reg_ce_targets_in_range(y_all, tables_one)
