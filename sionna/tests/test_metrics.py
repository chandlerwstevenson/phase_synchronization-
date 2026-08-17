"""Tests for the common metrics layer (metrics.py)."""

import math

import numpy as np
import pytest
import torch

from detection import DetectionParams
from gating_study import run_gated_waveform_detection
from metrics import (
    detection_range_m,
    mean_array_gain,
    net_throughput,
    probability_of_detection,
    spectral_efficiency,
)


def _equidistant_positions(n: int, radius: float = 500.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def test_perfect_sync_gives_exact_n_squared_beamforming_boost():
    # N stations equidistant from a user at the center, zero residuals:
    # received signal-to-noise ratio must be exactly N^2 times one
    # station's.
    n = 6
    positions = _equidistant_positions(n)
    user = np.array([0.0, 0.0])
    residuals = torch.zeros(n, 8, dtype=torch.float64)
    ones = torch.ones_like(residuals)
    single = torch.zeros(1, 8, dtype=torch.float64)
    array_result = spectral_efficiency(
        residuals, ones, positions, user, tx_power_w=1.0
    )
    single_result = spectral_efficiency(
        single, torch.ones_like(single), positions[:1], user, tx_power_w=1.0
    )
    ratio_db = array_result.mean_snr_db - single_result.mean_snr_db
    assert ratio_db == pytest.approx(10.0 * math.log10(n**2), abs=1e-9)


def test_spectral_efficiency_monotone_in_sync_quality():
    n = 5
    positions = _equidistant_positions(n)
    user = np.array([1200.0, 150.0])
    generator = torch.Generator().manual_seed(0)
    tight = 0.1 * torch.randn(n, 200, generator=generator, dtype=torch.float64)
    loose = 1.5 * torch.randn(n, 200, generator=generator, dtype=torch.float64)
    tight[0] = 0.0
    loose[0] = 0.0
    ones = torch.ones_like(tight)
    se_tight = spectral_efficiency(tight, ones, positions, user, 0.5)
    se_loose = spectral_efficiency(loose, ones, positions, user, 0.5)
    assert se_tight.mean_bps_hz > se_loose.mean_bps_hz
    assert se_tight.likely95_bps_hz > se_loose.likely95_bps_hz
    # 95%-likely is never above the mean's distribution top; sanity:
    assert se_tight.likely95_bps_hz <= se_tight.per_draw_bps_hz.max().item()


def test_sic_tier_reduces_to_single_group_when_empty():
    n = 4
    positions = _equidistant_positions(n)
    user = np.array([400.0, 0.0])
    residuals = 0.3 * torch.ones(n, 6, dtype=torch.float64)
    residuals[0] = 0.0
    ones = torch.ones_like(residuals)
    zeros = torch.zeros_like(residuals)
    plain = spectral_efficiency(residuals, ones, positions, user, 0.5)
    with_empty_tier = spectral_efficiency(
        residuals, ones, positions, user, 0.5, noncoherent_weights=zeros
    )
    assert with_empty_tier.mean_bps_hz == pytest.approx(
        plain.mean_bps_hz, abs=1e-12
    )
    # And a two-group split never exceeds using every station coherently
    # with zero residuals, but always beats leaving the demoted group off.
    half = ones.clone()
    half[2:] = 0.0
    tier = 1.0 - half
    tier[0] = 0.0
    split = spectral_efficiency(
        residuals, half, positions, user, 0.5, noncoherent_weights=tier
    )
    off = spectral_efficiency(residuals, half, positions, user, 0.5)
    assert split.mean_bps_hz > off.mean_bps_hz


def test_net_throughput_limits():
    assert net_throughput(8.0, 0.0) == 8.0
    assert net_throughput(8.0, 1.0) == 0.0
    assert net_throughput(8.0, 0.25) == pytest.approx(6.0)


def test_mean_array_gain_perfect_is_one():
    residuals = torch.zeros(5, 7, dtype=torch.float64)
    assert mean_array_gain(residuals, torch.ones_like(residuals)) == (
        pytest.approx(1.0)
    )


def test_detection_range_monotone_in_gain():
    params = DetectionParams(tx_power_w=0.5)
    assert detection_range_m(6, 1.0, params) > detection_range_m(
        6, 0.5, params
    )


def test_detection_wrapper_reproduces_gated_pipeline():
    generator = torch.Generator().manual_seed(3)
    positions = np.array([[0.0, 0.0], [400.0, 100.0], [-300.0, 250.0]])
    residuals = (
        torch.rand(3, 12, generator=generator, dtype=torch.float64) - 0.5
    )
    residuals[0] = 0.0
    targets = np.array([[900.0, 120.0]])
    params = DetectionParams(tx_power_w=0.5)
    kwargs = dict(
        params=params, pulse_length=160, trials=150, h0_trials=3000, seed=5
    )
    direct = run_gated_waveform_detection(
        "direct", positions, residuals, torch.ones_like(residuals),
        targets, **kwargs
    )
    wrapped = probability_of_detection(
        "wrapped", positions, residuals, torch.ones_like(residuals),
        targets, combiner="gated", **kwargs
    )
    assert wrapped.pd_measured == direct.pd_measured
    assert wrapped.measured_pfa == direct.measured_pfa


def test_detection_wrapper_rejects_unknown_combiner():
    residuals = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="combiner"):
        probability_of_detection(
            "x", np.zeros((2, 2)), residuals, torch.ones_like(residuals),
            np.array([[100.0, 0.0]]), combiner="mystery",
        )
