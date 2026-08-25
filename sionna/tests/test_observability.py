"""Tests for the oscillator/channel split observability analysis."""

import math

import torch

from observability_analysis import (
    NULL_DIRECTION,
    gramian_rank_and_null,
    jakes_channel_innovation,
    split_uncertainty_cycle,
)
from observability_study import steady_bias_and_rms
from ota_sync import SDRSimulationConfig

FAST = SDRSimulationConfig(num_iterations=20, seed=0, device="cpu")


def test_oneway_gramian_rank_deficient_with_correct_null():
    rank, null = gramian_rank_and_null(0.01, with_anchor=False)
    assert rank == 2
    assert torch.allclose(null.abs(), NULL_DIRECTION.abs(), atol=1e-8)


def test_anchor_restores_full_rank():
    rank, null = gramian_rank_and_null(0.01, with_anchor=True)
    assert rank == 3
    assert torch.allclose(null, torch.zeros(3, dtype=torch.float64))


def test_split_uncertainty_grows_with_cadence_and_channel_motion():
    r1, rf = 1e-4, 1.0
    base = split_uncertainty_cycle(FAST, 5, 10, 0.0, r1, rf)
    longer = split_uncertainty_cycle(FAST, 5, 40, 0.0, r1, rf)
    assert longer.true_split_std > base.true_split_std
    moving = split_uncertainty_cycle(
        FAST, 5, 10, jakes_channel_innovation(0.2, 0.05, 915e6), r1, rf
    )
    assert moving.true_theta_std > base.true_theta_std
    # Believed covariance never sees the true channel innovation, so the
    # mismatch shows up only on the true side.
    assert abs(moving.believed_theta_std - base.believed_theta_std) < 1e-6


def test_ramp_bias_zero_at_rest_and_monotone_in_speed():
    from observability_analysis import los_ramp_bias_cycle

    assert los_ramp_bias_cycle(FAST, 5, 10, 0.0, 1e-4, 1.0) == 0.0
    slow = los_ramp_bias_cycle(FAST, 5, 10, 0.05, 1e-4, 1.0)
    fast = los_ramp_bias_cycle(FAST, 5, 10, 0.2, 1e-4, 1.0)
    assert 0.0 < slow < fast


def test_bias_diagnostic_on_synthetic_offset():
    class Fake:
        station_residuals = torch.full((1, 200), 0.15, dtype=torch.float64)
        station_valid = torch.ones(1, 200, dtype=torch.bool)

    bias, rms = steady_bias_and_rms(Fake())
    assert math.isclose(bias, 0.15, rel_tol=1e-6)
    assert math.isclose(rms, 0.15, rel_tol=1e-6)
