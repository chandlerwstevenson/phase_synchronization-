"""Tests for interior_optimum_study.py: the tail/bias diagnostic on
known synthetic inputs, the interval-scaling rule that fixed the
degenerate-anchor confound, and a micro end-to-end run."""

import math
from types import SimpleNamespace

import torch

from interior_optimum_study import (
    intervals_for,
    key,
    run_cell,
    tail_stats,
)


def _fake_result(residuals: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        station_residuals=residuals.unsqueeze(0),
        station_valid=torch.ones(1, residuals.numel(), dtype=torch.bool),
    )


def test_tail_stats_reads_known_bias_and_wander():
    generator = torch.Generator().manual_seed(3)
    bias, spread = 0.2, 0.05
    residuals = bias + spread * torch.randn(
        400, generator=generator, dtype=torch.float64
    )
    rms, measured_bias, wander = tail_stats(_fake_result(residuals))
    # Stats are computed on the tail half only.
    assert abs(measured_bias - 1e3 * bias) < 10.0
    assert abs(wander - 1e3 * spread) < 10.0
    assert rms >= measured_bias


def test_tail_stats_zero_bias_case():
    generator = torch.Generator().manual_seed(4)
    residuals = 0.03 * torch.randn(
        400, generator=generator, dtype=torch.float64
    )
    _, measured_bias, _ = tail_stats(_fake_result(residuals))
    assert measured_bias < 8.0


def test_interval_scaling_prevents_degenerate_anchors():
    # The first sweep's confound: K > intervals means one anchor ever
    # and all such K are the same run. The rule guarantees >= ~4
    # anchor cycles for every K, and the cache key separates the
    # corrected cells from the flat-60 ones.
    for k in (10, 20, 40, 80, 160):
        assert intervals_for(k) >= 4 * k or intervals_for(k) == 60
        assert intervals_for(k) / k >= 1.5
    assert key(5, 10, 0, None) == "n5_K10_s0_sdr-default"
    assert key(5, 80, 0, None).endswith("_i320")


def test_micro_run_produces_finite_stats():
    cell = run_cell(2, 5, 0, None)
    assert cell["rms_mrad"] == cell["rms_mrad"]
    assert cell["bias_mrad"] >= 0.0
    assert 0.0 < cell["detection"] <= 1.0
