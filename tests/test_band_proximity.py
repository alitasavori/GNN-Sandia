"""Unit tests for warm-start band cloud proximity scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nonunique_da_gps_warmstart_band_daily import (
    _band_proximity_continuous,
    _band_proximity_continuous_per_step,
    _band_proximity_discrete,
    _band_proximity_discrete_per_step,
    _inside_band_fraction,
)


def test_inside_band_all_one_proximity_one():
    y = np.array([0.5, 1.0, 1.5])
    lo = np.array([0.4, 0.9, 1.4])
    hi = np.array([0.6, 1.1, 1.6])
    assert _inside_band_fraction(y, lo, hi) == pytest.approx(1.0)
    assert _band_proximity_continuous(y, lo, hi) == pytest.approx(1.0)


def test_outside_band_decays_continuous():
    y = np.array([1.2])
    lo = np.array([0.8])
    hi = np.array([1.0])
    scores = _band_proximity_continuous_per_step(y, lo, hi, min_scale=1e-4)
    # d=0.2, scale=max(0.1, 1e-4)=0.1 -> exp(-2)
    assert scores[0] == pytest.approx(np.exp(-2.0), rel=1e-6)
    assert 0.0 < scores[0] < 1.0


def test_zero_width_band_uses_min_scale():
    y = np.array([1.01])
    lo = np.array([1.0])
    hi = np.array([1.0])
    scores = _band_proximity_continuous_per_step(y, lo, hi, min_scale=0.01)
    assert scores[0] == pytest.approx(np.exp(-0.01 / 0.01), rel=1e-6)


def test_discrete_inside_band():
    y = np.array([2.0, 3.0])
    lo = np.array([1.0, 3.0])
    hi = np.array([3.0, 4.0])
    assert _band_proximity_discrete(y, lo, hi, step_scale=1.0) == pytest.approx(1.0)


def test_discrete_outside_one_tap_step():
    y = np.array([4.0])
    lo = np.array([1.0])
    hi = np.array([3.0])
    scores = _band_proximity_discrete_per_step(y, lo, hi, step_scale=1.0)
    assert scores[0] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_nan_mask_ignored():
    y = np.array([1.0, np.nan])
    lo = np.array([0.9, 0.9])
    hi = np.array([1.1, 1.1])
    assert _band_proximity_continuous(y, lo, hi) == pytest.approx(1.0)
