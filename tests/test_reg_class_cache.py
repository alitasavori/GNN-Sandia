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
    _filter_sample_ids_unseen_reg_taps,
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


def test_chunk_cache_path_includes_drop_unseen_suffix():
    p = _chunk_cache_path(
        Path("/tmp/cache"),
        "run_withder",
        1.0,
        42,
        0,
        reg_slug="regce",
        reg_classes_digest="abc123def0",
        drop_unseen_reg_taps=True,
    )
    assert "__dropunseen" in p.name


def test_filter_sample_ids_unseen_reg_taps(tmp_path: Path):
    import pandas as pd

    reg_cols = ["reg_feeder_rega_tap_pu"]
    tables = _build_reg_class_tables(reg_cols, np.array([[0.975], [1.0]], dtype=np.float64))
    meta = tmp_path / "gnn_sample_meta.csv"
    pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "reg_feeder_rega_tap_pu": [0.975, 0.98125, 1.0],
        }
    ).to_csv(meta, index=False)
    kept, n_drop = _filter_sample_ids_unseen_reg_taps(meta, [1, 2, 3], reg_cols, tables)
    assert n_drop == 1
    assert kept == [1, 3]
    idx = _encode_reg_class_indices(
        np.array([[0.975], [1.0]], dtype=np.float64), tables, map_unseen_to_nearest=False
    )
    assert _reg_ce_targets_in_range(torch.from_numpy(idx), tables)


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


def test_encode_maps_unseen_tap_to_nearest_init_class():
    reg_cols = ["reg_feeder_rega_tap_pu"]
    tables = _build_reg_class_tables(reg_cols, np.array([[0.975], [1.0], [1.025]], dtype=np.float64))
    raw = np.array([[0.98125], [1.0]], dtype=np.float64)
    idx = _encode_reg_class_indices(raw, tables, map_unseen_to_nearest=True)
    classes = tables[0]["classes"]
    assert float(classes[int(idx[0, 0])]) == pytest.approx(0.975)
    assert int(idx[1, 0]) == tables[0]["class_to_index"]["1.0"]
    assert _reg_ce_targets_in_range(torch.from_numpy(idx), tables)

    with pytest.raises(KeyError, match="Unseen tap value"):
        _encode_reg_class_indices(raw, tables, map_unseen_to_nearest=False)
