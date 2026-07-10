"""Training-pool subset eval helpers for blended fine-tune."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from train_da_gps_multitask_complex_voltage_gine import (
    _ChunkEvalPool,
    _maybe_build_subset_eval_pools_from_train,
    _slice_train_pool_subset,
    _subset_eval_report_key,
    _train_pool_subset_indices,
)


class _Args:
    seed = 42
    eval_subset_nobess_chunk_parent = ""
    eval_subset_nobess_chunk_glob = "run_*"
    eval_subset_nobess_label = "nobess_40"
    eval_subset_withder_chunk_parent = ""
    eval_subset_withder_chunk_glob = "run_*"
    eval_subset_withder_label = "withder_4"


def test_subset_eval_report_key() -> None:
    assert _subset_eval_report_key("nobess_40") == "subset_eval_nobess_40"
    assert _subset_eval_report_key("withder_4") == "subset_eval_withder_4"


def test_slice_train_pool_subset_preserves_indices(tmp_path: Path) -> None:
    parent = tmp_path / "blend"
    parent.mkdir()
    chunks = []
    for nm in ("run_a", "run_b", "run_c"):
        p = parent / nm
        p.mkdir()
        chunks.append(p)
    idx_val = [np.array([0]), np.array([1]), np.array([0, 1])]
    idx_test = [np.array([1]), np.array([0]), np.array([2])]
    caches = [parent / f"{nm}.pt" for nm in ("run_a", "run_b", "run_c")]
    pool = _slice_train_pool_subset(
        label="nobess_40",
        chunk_parent=parent,
        chunk_dirs=chunks,
        indices=[0, 2],
        idx_val_list=idx_val,
        idx_test_list=idx_test,
        selected_ids_list=[None, None, [1, 2]],
        cache_pts=caches,
        bootstrap_cache_pts=[None, None, None],
        split_seed=7,
    )
    assert pool.label == "nobess_40"
    assert [p.name for p in pool.chunk_dirs] == ["run_a", "run_c"]
    assert len(pool.idx_val_list) == 2
    assert np.array_equal(pool.idx_val_list[0], idx_val[0])
    assert np.array_equal(pool.idx_test_list[1], idx_test[2])
    assert pool.cache_pts[1] == caches[2]


def test_maybe_build_subset_eval_pools_from_train(tmp_path: Path) -> None:
    nobess_parent = tmp_path / "nobess"
    withder_parent = tmp_path / "withder"
    blend_parent = tmp_path / "blend"
    for root, names in (
        (nobess_parent, ("run_n1", "run_n2")),
        (withder_parent, ("run_w1",)),
        (blend_parent, ("run_n1", "run_n2", "run_w1")),
    ):
        root.mkdir()
        for nm in names:
            (root / nm).mkdir()

    chunk_dirs = sorted([p for p in blend_parent.iterdir() if p.is_dir()], key=lambda p: p.name)
    n = len(chunk_dirs)
    idx_val = [np.array([0]) for _ in range(n)]
    idx_test = [np.array([1]) for _ in range(n)]
    caches = [blend_parent / f"c{i}.pt" for i in range(n)]

    args = _Args()
    args.eval_subset_nobess_chunk_parent = str(nobess_parent)
    args.eval_subset_withder_chunk_parent = str(withder_parent)

    pools = _maybe_build_subset_eval_pools_from_train(
        args,
        chunk_parent=blend_parent,
        chunk_dirs=chunk_dirs,
        idx_val_list=idx_val,
        idx_test_list=idx_test,
        selected_ids_list=[None] * n,
        cache_pts=caches,
        bootstrap_cache_pts=[None] * n,
    )
    assert len(pools) == 2
    assert pools[0].label == "nobess_40"
    assert len(pools[0].chunk_dirs) == 2
    assert pools[1].label == "withder_4"
    assert len(pools[1].chunk_dirs) == 1


def test_train_pool_subset_indices_comma_glob(tmp_path: Path) -> None:
    parent = tmp_path / "nobess"
    parent.mkdir()
    for nm in ("run_a", "run_b"):
        (parent / nm).mkdir()
    chunks = [parent / "run_a", parent / "run_x"]
    idx = _train_pool_subset_indices(
        chunks,
        subset_parent=parent,
        glob_pat="run_a,run_b",
    )
    assert idx == [0]
